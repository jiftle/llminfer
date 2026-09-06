// Package tensor 提供张量数据结构与基础算子：MatMul、量化反量化、RMSNorm、SiLU、RoPE。
//
// 里程碑：M3。目标：让权重张量可读、可参与矩阵运算，并做常见归一化/激活。
//
// 关键设计决策：
//   - DType 编号与 GGML 完全对齐（见 gguf/types.go 类型表），保证从 GGUF 读出的
//     类型号可直接拿来建张量，无需再做映射。
//   - 所有权重统一走"按行反量化"接口（DequantRow）：量化权重不进完整反量化，
//     而是在矩阵乘时逐行流式反量化，避免一次性把整个几百 MB 的模型展开成 float32。
package tensor

import (
	"encoding/binary"
	"fmt"
	"math"
)

// DType 是张量数据类型，编号与 GGMLType 严格一致。
type DType uint32

const (
	TYPE_F32  DType = 0
	TYPE_F16  DType = 1
	TYPE_Q4_0 DType = 2
	TYPE_Q4_1 DType = 3
	TYPE_Q5_0 DType = 6
	TYPE_Q5_1 DType = 7
	TYPE_Q8_0 DType = 8
	TYPE_Q8_1 DType = 9
	TYPE_Q2_K DType = 10
	TYPE_Q3_K DType = 11
	TYPE_Q4_K DType = 12
	TYPE_Q5_K DType = 13
	TYPE_Q6_K DType = 14
	TYPE_Q8_K DType = 15
)

// IsQuantized 该类型是否为量化格式（非 F32/F16 即量化）。
func (t DType) IsQuantized() bool {
	switch t {
	case TYPE_Q4_0, TYPE_Q4_1, TYPE_Q5_0, TYPE_Q5_1, TYPE_Q8_0, TYPE_Q8_1,
		TYPE_Q2_K, TYPE_Q3_K, TYPE_Q4_K, TYPE_Q5_K, TYPE_Q6_K, TYPE_Q8_K:
		return true
	}
	return false
}

// BlockSize 一个量化块覆盖的元素数；非量化类型为 1。
// 普通量化（Q4_0 等）块=32 ； K 系列块=256。
func (t DType) BlockSize() int {
	switch t {
	case TYPE_Q4_0, TYPE_Q4_1, TYPE_Q5_0, TYPE_Q5_1, TYPE_Q8_0, TYPE_Q8_1:
		return 32
	case TYPE_Q2_K, TYPE_Q3_K, TYPE_Q4_K, TYPE_Q5_K, TYPE_Q6_K, TYPE_Q8_K:
		return 256
	}
	return 1
}

// TypeSize 单个元素字节数；对量化类型是「整个块」的字节数。
// 数值来自 ggml 格式的块大小定义（GGML_BLOCK_TYPE）。
func (t DType) TypeSize() int {
	switch t {
	case TYPE_F32:
		return 4
	case TYPE_F16:
		return 2
	case TYPE_Q4_0:
		return 2 + 16 // scale + 16 字节低位/高位 4bit
	case TYPE_Q4_1:
		return 2 + 2 + 16
	case TYPE_Q5_0:
		return 2 + 4 + 16
	case TYPE_Q5_1:
		return 2 + 2 + 4 + 16
	case TYPE_Q8_0:
		return 2 + 32
	case TYPE_Q8_1:
		return 2 + 2 + 32
	case TYPE_Q2_K:
		return 84
	case TYPE_Q3_K:
		return 110
	case TYPE_Q4_K:
		return 144 // 2(d) + 2(dmin) + 12(scales) + 128(qs)
	case TYPE_Q5_K:
		return 176
	case TYPE_Q6_K:
		return 210 // 2(d) + 16(scales) + 64(qh) + 128(ql)
	case TYPE_Q8_K:
		return 292
	}
	return 4
}

// String 类型名（打印用）。
func (t DType) String() string {
	switch t {
	case TYPE_F32:
		return "F32"
	case TYPE_F16:
		return "F16"
	case TYPE_Q4_0:
		return "Q4_0"
	case TYPE_Q4_1:
		return "Q4_1"
	case TYPE_Q5_0:
		return "Q5_0"
	case TYPE_Q5_1:
		return "Q5_1"
	case TYPE_Q8_0:
		return "Q8_0"
	case TYPE_Q8_1:
		return "Q8_1"
	case TYPE_Q2_K:
		return "Q2_K"
	case TYPE_Q3_K:
		return "Q3_K"
	case TYPE_Q4_K:
		return "Q4_K"
	case TYPE_Q5_K:
		return "Q5_K"
	case TYPE_Q6_K:
		return "Q6_K"
	case TYPE_Q8_K:
		return "Q8_K"
	}
	return fmt.Sprintf("DT(%d)", t)
}

// Tensor 多维张量。
//   - Data: 原始字节（权重从 GGUF 读取后挂这里，量化就是如此）
//   - Floats: 反量化后的 float32 数据（懒加载缓存，勿对量化权重整体调用！）
type Tensor struct {
	Type   DType
	Dims   uint32
	NE     [4]uint32 // 各维元素数
	NB     [4]uint32 // 各维步长（字节），与 ggml 布局一致：ne0 连续
	Data   []byte
	Floats []float32
}

// NewTensor 创建张量，按 ne0 连续布局（与 GGUF 磁盘布局一致，加载零拷贝）。
func NewTensor(dt DType, ne ...uint32) *Tensor {
	t := &Tensor{Type: dt, Dims: uint32(len(ne))}
	for i, n := range ne {
		t.NE[i] = n
	}
	// 步长：ne0 字节为单个元素大小，其余维度累进
	t.NB[0] = uint32(dt.TypeSize())
	for i := 1; i < int(t.Dims); i++ {
		t.NB[i] = t.NB[i-1] * t.NE[i-1]
	}
	return t
}

// Nelements 总元素数。
func (t *Tensor) Nelements() int {
	n := 1
	for i := 0; i < int(t.Dims); i++ {
		n *= int(t.NE[i])
	}
	return n
}

// Nbytes 总字节数：量化类型按块算，非量化按元素数×类型宽。
func (t *Tensor) Nbytes() int {
	n := t.Nelements()
	if t.Type.IsQuantized() {
		return n / t.Type.BlockSize() * t.Type.TypeSize()
	}
	return n * t.Type.TypeSize()
}

// Rows 行数。一维张量（如 norm 权重、bias）在 GGUF 里只写 ne0，视为单行；
// 二维及以上的张量行数 = NE[1]（行主序，ne0 连续）。
func (t *Tensor) Rows() int {
	if t.Dims <= 1 || t.NE[1] == 0 {
		return 1
	}
	return int(t.NE[1])
}

// AsFloat32 返回完整 float32 数据（懒加载反量化并缓存）。
// 注意：大权重请用 DequantRow（流式反量化，省内存）；此接口仅适合小张量或激活。
func (t *Tensor) AsFloat32() []float32 {
	if t.Floats != nil {
		return t.Floats
	}
	n := t.Nelements()
	t.Floats = make([]float32, n)
	per := int(t.NE[0])
	for r := 0; r < t.Rows(); r++ {
		if err := t.DequantRow(uint32(r), t.Floats[r*per:(r+1)*per]); err != nil {
			panic(err)
		}
	}
	return t.Floats
}

// DequantRow 反量化「第 row 行」（宽度=NE[0]）到 dst。
// 权重按 [K, N] 存储，ne0=K 连续，因此行的字节在内存连续，按块流式读。
func (t *Tensor) DequantRow(row uint32, dst []float32) error {
	ne0 := int(t.NE[0])
	if row >= uint32(t.Rows()) {
		return fmt.Errorf("tensor: 行号 %d 越界（共 %d 行）", row, t.Rows())
	}

	// F32 / F16 直接位宽拷贝/转换（0 拷贝开销最低）
	switch t.Type {
	case TYPE_F32:
		off := int(row) * ne0 * 4
		for i := 0; i < ne0; i++ {
			dst[i] = math.Float32frombits(uint32(t.Data[off+4*i]) |
				uint32(t.Data[off+4*i+1])<<8 |
				uint32(t.Data[off+4*i+2])<<16 |
				uint32(t.Data[off+4*i+3])<<24)
		}
		return nil
	case TYPE_F16:
		off := int(row) * ne0 * 2
		for i := 0; i < ne0; i++ {
			dst[i] = F16ToF32(t.Data[off+2*i], t.Data[off+2*i+1])
		}
		return nil
	}

	if !t.Type.IsQuantized() {
		return fmt.Errorf("tensor: 不支持的类型 %s", t.Type)
	}
	bs := t.Type.BlockSize()
	blkBytes := t.Type.TypeSize()
	blocks := ne0 / bs
	if blocks == 0 || ne0%bs != 0 {
		return fmt.Errorf("tensor: 行宽 %d 不是块大小 %d 的整数倍", ne0, bs)
	}
	rowBytes := blocks * blkBytes
	base := int(row) * rowBytes
	for b := 0; b < blocks; b++ {
		if err := DequantBlock(t.Type, t.Data[base+b*blkBytes:base+(b+1)*blkBytes], dst[b*bs:(b+1)*bs]); err != nil {
			return err
		}
	}
	return nil
}

// Print 打印张量信息（调试）。
func (t *Tensor) Print(name string) {
	fmt.Printf("Tensor %s: type=%s ne=[%d %d %d %d] bytes=%d\n",
		name, t.Type, t.NE[0], t.NE[1], t.NE[2], t.NE[3], t.Nbytes())
}

// F16ToF32 把两个小端字节拼成 float16 再转 float32。
// 硬编码位操作，避免引入外部库（高频调用）。
func F16ToF32(lo, hi byte) float32 {
	bits := uint16(lo) | uint16(hi)<<8
	sign := uint32(bits&0x8000) << 16
	exp := uint32(bits>>10) & 0x1F
	mant := uint32(bits) & 0x3FF

	switch {
	case exp == 0:
		if mant == 0 {
			return math.Float32frombits(sign) // ±0
		}
		// 次正规 fp16：mant * 2^-24，落在 fp32 正规区
		v := float32(float64(mant) / (1 << 24))
		if sign != 0 {
			return -v
		}
		return v
	case exp == 31:
		if mant != 0 {
			return float32(math.NaN())
		}
		return math.Float32frombits(sign | 0x7F800000) // ±Inf
	}
	// 正规数：fp32 指数 = fp16 指数 +112，尾数左移 13 位
	return math.Float32frombits(sign | uint32(exp+112)<<23 | mant<<13)
}

// 保证 binary 包被引用（future use）。
var _ = binary.LittleEndian