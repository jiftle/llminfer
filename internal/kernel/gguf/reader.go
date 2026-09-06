package gguf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

const (
	// magic 是 GGUF 文件头四字节标识 "GGUF"
	magic uint32 = 0x46554747 // little-endian "GGUF"

	// currentVersion 我们支持的 GGUF 版本（v3）
	currentVersion uint32 = 3
)

// TensorInfo 描述一个权重张量的位置与形状。
// 注意：张量数据仍在文件里（Data 为 nil），真正要读时才按 Offset+Type 去取。
type TensorInfo struct {
	Name   string                    // 权重名，如 "model.layers.0.self_attn.q_proj.weight"
	NEE    []uint64                  // 各维长度（行主序存储，ne[0] 连续）
	Type   GGMLType                  // 权重存储类型（F32/F16/量化…）
	Offset uint64                    // Nbytes 后的数据区字节偏移
	Elems  uint64                    // 总元素数（各维乘积）
	Nbytes uint64                    // 该张量按块计算的字节大小
}

// GGUFFile 是解析后的模型文件：头部 + 元数据 KV + 张量信息表。
type GGUFFile struct {
	Version    uint32
	KVs        map[string]any // 元数据：key → 值（string/数值/[]T/[]string…）
	Tensors    []TensorInfo   // 张量信息表（按文件顺序）
	ByName     map[string]*TensorInfo // 名字 → 张量，加载时快速定位
	DataStart  uint64                 // 权重数据区起始偏移（首个张量的 Offset）
	Size       int64                  // 整个文件大小
	f          *os.File               // 持有文件句柄，供按需读取/映射
}

var ErrUnsupportedVersion = errors.New("gguf: 不支持的版本")
var errBadMagic = errors.New("gguf: 不是 GGUF 文件（magic 校验失败）")

// ReadFile 打开并解析 GGUF 文件。成功时持有文件句柄（需 Close）。
func ReadFile(path string) (*GGUFFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	gf := &GGUFFile{f: f, KVs: make(map[string]any), ByName: make(map[string]*TensorInfo)}
	if err := gf.parseHeader(); err != nil {
		f.Close()
		return nil, err
	}
	return gf, nil
}

// parseHeader 依次读完：magic/version/两个 count → KV → TensorInfo 表。
func (g *GGUFFile) parseHeader() error {
	r := newReader(g.f)

	// —— 文件头 ——
	if v, err := r.u32(); err != nil {
		return err
	} else if v != magic {
		return errBadMagic
	}
	if v, err := r.u32(); err != nil {
		return err
	} else {
		g.Version = v
		if v != currentVersion {
			return fmt.Errorf("%w (版本=%d)", ErrUnsupportedVersion, v)
		}
	}
	tensorCount, err := r.u64() // 张量个数
	if err != nil {
		return err
	}
	kvCount, err := r.u64() // 元数据条数
	if err != nil {
		return err
	}

	// —— 元数据区 ——
	for i := uint64(0); i < kvCount; i++ {
		key, val, err := readKV(r)
		if err != nil {
			return fmt.Errorf("gguf: KV[%d] 读取失败: %w", i, err)
		}
		g.KVs[key] = val
	}

	// —— 张量信息表 ——
	// 按文件顺序读；数据区起点 = 张量信息表读取结束后的文件位置（GGUF 规范对齐 32）。
	for i := uint64(0); i < tensorCount; i++ {
		ti, err := readTensorHeader(r)
		if err != nil {
			return fmt.Errorf("gguf: tensor[%d] 读取失败: %w", i, err)
		}
		g.Tensors = append(g.Tensors, ti)
		g.ByName[ti.Name] = &g.Tensors[len(g.Tensors)-1]
	}
	g.DataStart = align32(r.pos)
	if info, err := g.f.Stat(); err == nil {
		g.Size = info.Size()
	}
	return nil
}

// align32 向上取整数到 32 的对齐（GGUF 规范张量数据区按 32 字节对齐）。
func align32(n int64) uint64 {
	return uint64((n + 31) &^ int64(31))
}

// Close 关闭底层文件，释放句柄。
func (g *GGUFFile) Close() error { return g.f.Close() }

// DataReader 返回读取器，可直接取 offset 处的权重字节（Scalar 视图）。
func (g *GGUFFile) DataReader() io.ReaderAt { return g.f }

// String 快速打印一个 KV 键的短串形式。
func (g *GGUFFile) GetString(key string) string {
	if v, ok := g.KVs[key].(string); ok {
		return v
	}
	return ""
}

// GetInt 以 int 返回一个整数 KV（兼容 int32/uint32/int64/uint64）。
func (g *GGUFFile) GetInt(key string) int {
	switch v := g.KVs[key].(type) {
	case int32:
		return int(v)
	case uint32:
		return int(v)
	case int64:
		return int(v)
	case uint64:
		return int(v)
	}
	return 0
}

// GetFloat 以 float64 返回一个浮点 KV（兼容 float32/float64）。
func (g *GGUFFile) GetFloat(key string) float64 {
	switch v := g.KVs[key].(type) {
	case float32:
		return float64(v)
	case float64:
		return v
	}
	return 0
}

// GetStringSlice 取一个字符串数组 KV（如 tokenizer.ggml.tokens/merges）。
func (g *GGUFFile) GetStringSlice(key string) []string {
	v, _ := g.KVs[key].([]string)
	return v
}

// GetInt32Slice 取一个 int32 数组 KV（如 tokenizer.ggml.token_type）。
func (g *GGUFFile) GetInt32Slice(key string) []int32 {
	v, _ := g.KVs[key].([]int32)
	return v
}

// blockReader 基于 *os.File 的顺序小端读取器。
// GGUF 全程 little-endian，字符串长度也是 uint64（v1 是 u32，v3 是 u64——即 GGUF v3 规范）。
type sectionReader struct {
	f      *os.File
	pos    int64
	remain int64
}

// newReader 创建顺序读取器，跟踪当前偏移（sticky read）。
func newReader(f *os.File) *sectionReader {
	st, _ := f.Stat()
	return &sectionReader{f: f, remain: st.Size()}
}

func (r *sectionReader) read(b []byte) error {
	if int64(len(b)) > r.remain {
		return io.ErrUnexpectedEOF
	}
	if _, err := r.f.ReadAt(b, r.pos); err != nil {
		return err
	}
	r.pos += int64(len(b))
	r.remain -= int64(len(b))
	return nil
}

// u8/U16/I32/U32/U64 —— 读一个定长整数（全小端）。
func (r *sectionReader) u8() (byte, error) {
	var b [1]byte
	if err := r.read(b[:]); err != nil {
		return 0, err
	}
	return b[0], nil
}

func (r *sectionReader) u16() (uint16, error) {
	var b [2]byte
	if err := r.read(b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint16(b[:]), nil
}

func (r *sectionReader) u32() (uint32, error) {
	var b [4]byte
	if err := r.read(b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b[:]), nil
}

func (r *sectionReader) i32() (int32, error) {
	v, err := r.u32()
	return int32(v), err
}

func (r *sectionReader) u64() (uint64, error) {
	var b [8]byte
	if err := r.read(b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b[:]), nil
}

func (r *sectionReader) f32() (float32, error) {
	var b [4]byte
	if err := r.read(b[:]); err != nil {
		return 0, err
	}
	return mathFloat32(b[:]), nil
}

func mathFloat32(b []byte) float32 {
	// 复用 binary 避免引入 strconv
	return math.Float32frombits(binary.LittleEndian.Uint32(b))
}

// str 读 GGUF 字符串：uint64 长度 + 原始字节。
func (r *sectionReader) str() (string, error) {
	n, err := r.u64()
	if err != nil {
		return "", err
	}
	buf := make([]byte, n)
	if err := r.read(buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

// readKV 读一条元数据 KV：key → (type, value)。
func readKV(r *sectionReader) (string, any, error) {
	key, err := r.str()
	if err != nil {
		return "", nil, err
	}
	vt, err := r.u32()
	if err != nil {
		return "", nil, err
	}
	switch GGUFValueType(vt) {
	case ValueTypeSTRING:
		s, err := r.str()
		return key, s, err
	case ValueTypeUINT8:
		v, err := r.u8()
		return key, v, err
	case ValueTypeINT8:
		v, err := r.u8()
		return key, int8(v), err
	case ValueTypeUINT16:
		v, err := r.u16()
		return key, v, err
	case ValueTypeINT16:
		v, err := r.u16()
		return key, int16(v), err
	case ValueTypeUINT32:
		v, err := r.u32()
		return key, v, err
	case ValueTypeINT32:
		v, err := r.i32()
		return key, v, err
	case ValueTypeFLOAT32:
		v, err := r.f32()
		return key, v, err
	case ValueTypeBOOL:
		v, err := r.u8()
		return key, v != 0, err
	case ValueTypeUINT64:
		v, err := r.u64()
		return key, v, err
	case ValueTypeINT64:
		v, err := r.u64()
		return key, int64(v), err
	case ValueTypeFLOAT64:
		v, err := r.u64()
		return key, math.Float64frombits(v), err
	case ValueTypeARRAY:
		return readArray(r, key)
	default:
		return key, nil, fmt.Errorf("未支持的 KV 类型 %d", vt)
	}
}

// readArray 读数组型 KV：先读元素类型 + 长度，再逐个读元素。
func readArray(r *sectionReader, key string) (string, any, error) {
	et, err := r.u32() // 元素类型
	if err != nil {
		return key, nil, err
	}
	n, err := r.u64() // 元素个数
	if err != nil {
		return key, nil, err
	}
	// 用一个 switch 展开：string 数组、数值数组、其他数组分别建 slice。
	switch GGUFValueType(et) {
	case ValueTypeSTRING:
		s := make([]string, n)
		for i := uint64(0); i < n; i++ {
			if s[i], err = r.str(); err != nil {
				return key, nil, err
			}
		}
		return key, s, nil
	case ValueTypeINT32:
		s := make([]int32, n)
		for i := uint64(0); i < n; i++ {
			if s[i], err = r.i32(); err != nil {
				return key, nil, err
			}
		}
		return key, s, nil
	case ValueTypeUINT8:
		s := make([]uint8, n)
		for i := uint64(0); i < n; i++ {
			if s[i], err = r.u8(); err != nil {
				return key, nil, err
			}
		}
		return key, s, nil
	default:
		return key, nil, fmt.Errorf("未支持的数组元素类型 %d", et)
	}
}

// readTensorHeader 读一条张量信息：名字、维数、维度、类型、数据偏移。
func readTensorHeader(r *sectionReader) (TensorInfo, error) {
	var ti TensorInfo
	var err error

	if ti.Name, err = r.str(); err != nil {
		return ti, err
	}
	nDims, err := r.u32()
	if err != nil {
		return ti, err
	}
	ti.NEE = make([]uint64, nDims)
	for i := uint32(0); i < nDims; i++ {
		if ti.NEE[i], err = r.u64(); err != nil {
			return ti, err
		}
	}
	typ, err := r.u32()
	if err != nil {
		return ti, err
	}
	ti.Type = GGMLType(typ)
	if !ti.Type.Supported() {
		return ti, fmt.Errorf("不支持的张量类型 %d (%s)", typ, ti.Type)
	}
	if ti.Offset, err = r.u64(); err != nil {
		return ti, err
	}

	// 总元素数 + 按块字节数。量化类型必须整除块元素数（如 Q8_0 的 ne0 是 32 的倍数）。
	ti.Elems = 1
	for _, d := range ti.NEE {
		ti.Elems *= d
	}
	if ti.Type.IsQuantized() {
		n0 := int64(ti.NEE[0])
		be := int64(ti.Type.BlockElems())
		if n0 == 0 || n0%be != 0 {
			return ti, fmt.Errorf("张量 %s 的首维 %d 不是块元素数 %d 的整数倍", ti.Name, n0, be)
		}
		// 每行 = 块数 × 块字节；总字节 = 每行 × 其余所有维
		rowBytes := n0 / be * int64(ti.Type.BlockBytes())
		ti.Nbytes = uint64(rowBytes)
		for i := 1; i < len(ti.NEE); i++ {
			ti.Nbytes *= ti.NEE[i]
		}
	} else {
		ti.Nbytes = ti.Elems * uint64(ti.Type.BlockBytes())
	}
	return ti, nil
}