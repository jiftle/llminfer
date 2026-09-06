package tensor

import (
	"math"
	"math/rand"
	"testing"
)

// refMatMulQuantTransB 参考实现：整行反量化再标量点积（M7.2 之前的旧路径）。
// 用来验证融合反量化点积的数值等价性（允许 <1e-3 相对误差，因融合版先整数累加再乘 scale）。
func refMatMulQuantTransB(a, b, c *Tensor) {
	M, K, N := a.NE[1], a.NE[0], b.NE[1]
	ensureN(c, int(M*N))
	aF := a.AsFloat32()
	rowBuf := make([]float32, K)
	for j := 0; j < int(N); j++ {
		if err := b.DequantRow(uint32(j), rowBuf); err != nil {
			panic(err)
		}
		for i := 0; i < int(M); i++ {
			acc := float32(0)
			base := i * int(K)
			for k := 0; k < int(K); k++ {
				acc += aF[base+k] * rowBuf[k]
			}
			c.Floats[i*int(N)+j] = acc
		}
	}
}

// makeRandomQuantB 生成一个随机的量化权重张量 B（[K, N]，ne0=K 连续）。
// 用手工序列化把随机 float 量化成对应格式的字节（用 DequantBlock 反算不可行，
// 这里反过来：直接构造量化字节流，保证块结构合法即可）。
func makeRandomQuantB(t *testing.T, typ DType, k, n int, seed int64) *Tensor {
	bs := typ.BlockSize()
	blkBytes := typ.TypeSize()
	blocks := k / bs
	raw := make([]byte, n*blocks*blkBytes)

	rng := rand.New(rand.NewSource(seed))
	for b := 0; b < n*blocks; b++ {
		blk := raw[b*blkBytes : (b+1)*blkBytes]
		// 不同量化类型填充不同的假数据，只保证块结构合法、数值在合理范围
		switch typ {
		case TYPE_Q8_0:
			d := float32(rng.Intn(3)+1) / 8 // scale d=0.125..0.5（随机 "合理"）
			blk[0] = byte(0)
			blk[1] = fp16HighByte(d)
			for i := 0; i < bs; i++ {
				blk[2+i] = byte(int8(rng.Intn(7) - 3)) // int8 -3..3
			}
		case TYPE_Q5_0:
			blk[0] = byte(0)
			blk[1] = fp16HighByte(0.5)
			for i := 0; i < 4; i++ {
				blk[2+i] = 0 // qh 全部 0：5bit 值就是 -16..-1 区间低位
			}
			for i := 0; i < bs/2; i++ {
				lo := byte(rng.Intn(6)) & 0x0F      // 低 4bit
				hi := byte(rng.Intn(6)) & 0x0F << 4 // 高 4bit
				blk[6+i] = lo | hi
			}
		case TYPE_Q6_K:
			// 超块 256：ql(128) qh(64) scales(16) d(2)
			d := float32(rng.Intn(3)+1) / 8
			blk[208] = byte(0)
			blk[209] = fp16HighByte(d)
			for i := 0; i < 128; i++ {
				blk[i] = byte(rng.Intn(64)) // ql 6bit 低位
			}
			for i := 0; i < 64; i++ {
				blk[128+i] = byte(rng.Intn(16)) // qh 6bit 高位位
			}
			for i := 0; i < 16; i++ {
				blk[192+i] = byte(int8(rng.Intn(7) - 3)) // scales 子块 scale
			}
		case TYPE_Q4_K:
			d := float32(rng.Intn(3)+1) / 8
			blk[0] = byte(0)
			blk[1] = fp16HighByte(d)
			blk[2] = byte(0)
			blk[3] = fp16HighByte(0.25) // dmin
			for i := 0; i < 12; i++ {
				blk[4+i] = byte(rng.Intn(16)) // scales
			}
			for i := 0; i < 128; i++ {
				blk[16+i] = byte(rng.Intn(256)) // qs 未符号 4bit
			}
		default:
			t.Fatalf("不支持的测试类型 %s", typ)
		}
	}
	return &Tensor{Type: typ, Dims: 2, NE: [4]uint32{uint32(k), uint32(n), 1, 1}, Data: raw}
}

// fp16HighByte 把一个小数写成 fp16 的高字节（低字节为 0），仅测试用。
func fp16HighByte(f float32) byte {
	return byte(F32ToF16Bits(f) >> 8)
}

// F32ToF16Bits 把 float32 转成 fp16 位（测试辅助，仅构造量化数据用）。
// 正常推理的 d 直接读自 GGUF 的 fp16 字节，无需此转换。
func F32ToF16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16(bits>>16) & 0x8000
	exp := int32(bits>>23) & 0xFF
	man := uint16(bits) >> 13 & 0x3FF

	if exp == 0xFF { // Inf/NaN
		if man != 0 {
			return sign | 0x7E00
		}
		return sign | 0x7C00
	}
	e := exp - 127 + 15
	switch {
	case e >= 0x1F: // 上溢 → Inf
		return sign | 0x7C00
	case e <= 0: // 下溢：次正规 fp16 或 0
		if e < -10 {
			return sign
		}
		man2 := uint16(math.Float32bits(f)) >> 13
		return sign | man2>>uint(14-e)&0x0FFF | 0x0000
	default:
		return sign | uint16(e)<<10 | man
	}
}

func TestFusedDotQuant(t *testing.T) {
	types := []DType{TYPE_Q8_0, TYPE_Q5_0, TYPE_Q6_K, TYPE_Q4_K}
	M, K, N := 4, 256, 8 // K=256 满足所有类型的块对齐（Q8_0/Q5_0 32、Q6_K/Q4_K 256）

	for _, typ := range types {
		typ := typ
		t.Run(typ.String(), func(t *testing.T) {
			b := makeRandomQuantB(t, typ, K, N, 42)

			// 激活 A：随机小值
			a := NewTensor(TYPE_F32, uint32(K), uint32(M))
			rng := rand.New(rand.NewSource(7))
			a.Floats = make([]float32, M*K)
			for i := range a.Floats {
				a.Floats[i] = float32(rng.NormFloat64())
			}

			// 融合版（M7.2）
			cFused := NewTensor(TYPE_F32, uint32(N), uint32(M))
			MatMulTransB(a, b, cFused)

			// 参考版（旧路径：反量化 + 标量）
			cRef := NewTensor(TYPE_F32, uint32(N), uint32(M))
			refMatMulQuantTransB(a, b, cRef)

			// 逐元素比较，相对误差 < 1e-3
			for i := range cFused.Floats {
				got, want := cFused.Floats[i], cRef.Floats[i]
				denom := math.Max(math.Abs(float64(want)), 1e-4)
				rel := math.Abs(float64(got-want)) / denom
				if rel > 1e-3 {
					t.Fatalf("%s[%d] fused=%v ref=%v rel=%e", typ, i, got, want, rel)
				}
			}
		})
	}
}

// 备注：fp16HighByte 依赖 F32ToF16Bits（测试辅助，见下）。
// 真实推理里 d 是直接读自 GGUF 的 fp16 字节，这里构造时保持一致即可。
func TestF32ToF16Helper(t *testing.T) {
	// 0.5 -> 0x3800（高字节 0x38）
	if got := F32ToF16Bits(0.5); got != 0x3800 {
		t.Fatalf("F32ToF16Bits(0.5)=0x%04x, want 0x3800", got)
	}
}