// Package gguf 解析 GGUF 模型文件：读取头部、元数据 KV、张量信息表。
//
// GGUF 文件布局（顺序三段）：
//
//	┌─ 文件头 ──────────────┐
//	│ magic  "GGUF" 4字节    │
//	│ version  uint32        │
//	│ tensor_count  uint64   │  ← 权重张量个数
//	│ kv_count  uint64       │  ← 元数据条数
//	├─ 元数据区 (kv_count 条) │
//	│ key: int64长度+字节      │
//	│ type: uint32 (值类型)    │
//	│ value: 依类型而定的负载  │
//	├─ 张量信息表 (tensor_count 条) ─┐
//	│ name: string                │
//	│ n_dims: uint32              │
//	│ dims: n_dims 个 uint64      │
//	│ type: uint32 (GGML 类型)     │
//	│ offset: uint64 (数据区偏移)  │
//	└─ 权重数据区                 ┘
//	    offset 到文件尾 = 各张量原始字节（量化/浮点按 type 布局）
package gguf

// GGUFValueType 是元数据 KV 的值类型标签（GGUF v3）。
// 读 KV 时先读 type，才知道后续几个字节怎么解释成值。
type GGUFValueType uint32

const (
	ValueTypeUINT8   GGUFValueType = 0
	ValueTypeINT8    GGUFValueType = 1
	ValueTypeUINT16  GGUFValueType = 2
	ValueTypeINT16   GGUFValueType = 3
	ValueTypeUINT32  GGUFValueType = 4
	ValueTypeINT32   GGUFValueType = 5
	ValueTypeFLOAT32 GGUFValueType = 6
	ValueTypeBOOL    GGUFValueType = 7
	ValueTypeSTRING  GGUFValueType = 8
	ValueTypeARRAY   GGUFValueType = 9
	ValueTypeUINT64  GGUFValueType = 10
	ValueTypeINT64   GGUFValueType = 11
	ValueTypeFLOAT64 GGUFValueType = 12
)

// GGMLType 是张量权重类型（ggml 格式的类型编号）。
// 关键：非浮点类型都是「量化」，数据按 BlockSize 个元素一个块存储，
// 算张量字节数时不能用元素数×字节数，必须按块算。
type GGMLType uint32

const (
	TypeF32  GGMLType = 0
	TypeF16  GGMLType = 1
	TypeQ4_0 GGMLType = 2
	TypeQ4_1 GGMLType = 3
	TypeQ5_0 GGMLType = 6
	TypeQ5_1 GGMLType = 7
	TypeQ8_0 GGMLType = 8
	TypeQ8_1 GGMLType = 9
	TypeQ2_K GGMLType = 10
	TypeQ3_K GGMLType = 11
	TypeQ4_K GGMLType = 12
	TypeQ5_K GGMLType = 13
	TypeQ6_K GGMLType = 14
	TypeQ8_K GGMLType = 15
)

// blockInfo 描述一种量化类型：一个块包含多少元素、占多少字节。
// 这是 Nbytes() 与 DequantRow() 的基础：比如 Q8_0 = 32 个元素共用 + 8 字节（1 个 scale）。
type blockInfo struct {
	// blockElems  一个块覆盖的元素数量
	// blockBytes  一个块占用的字节数
	blockElems int
	blockBytes int
}

// blockTable 只列我们支持的量化/浮点类型；没列出的类型在加载时直接报错，
// 避免用错误的块大小去算偏移导致越界读。
var blockTable = map[GGMLType]blockInfo{
	TypeF32:  {1, 4},
	TypeF16:  {1, 2},
	TypeQ4_0: {32, 2 + 16}, // 1×f16 scale + 16×8bit 量化值
	TypeQ4_1: {32, 2 + 2 + 16},
	TypeQ5_0: {32, 2 + 4 + 16},
	TypeQ5_1: {32, 2 + 2 + 16 + 4},
	TypeQ8_0: {32, 2 + 32}, // 1×f16 scale + 每元素 1 字节
	TypeQ2_K: {256, 2 + 16 + 64},
	TypeQ3_K: {256, 2 + 12 + 32 + 64},
	TypeQ4_K: {256, 2 + 12 + 96 + 16},
	TypeQ5_K: {256, 2 + 12 + 96 + 32 + 32},
	TypeQ6_K: {256, 2 + 16 + 192 + 16},
}

// IsQuantized 判断是否为量化类型（非 F32/F16 即量化）。
func (t GGMLType) IsQuantized() bool {
	return t != TypeF32 && t != TypeF16
}

// BlockElems 返回一个块的元素数量。
func (t GGMLType) BlockElems() int {
	return blockTable[t].blockElems
}

// BlockBytes 返回一个块的字节数。
func (t GGMLType) BlockBytes() int {
	return blockTable[t].blockBytes
}

// Supported 报告该类型是否在支持的块表中。
func (t GGMLType) Supported() bool {
	_, ok := blockTable[t]
	return ok
}

// String 给类型一个可读名（打印模型信息用）。
func (t GGMLType) String() string {
	switch t {
	case TypeF32:
		return "F32"
	case TypeF16:
		return "F16"
	case TypeQ4_0:
		return "Q4_0"
	case TypeQ4_1:
		return "Q4_1"
	case TypeQ5_0:
		return "Q5_0"
	case TypeQ5_1:
		return "Q5_1"
	case TypeQ8_0:
		return "Q8_0"
	case TypeQ8_1:
		return "Q8_1"
	case TypeQ2_K:
		return "Q2_K"
	case TypeQ3_K:
		return "Q3_K"
	case TypeQ4_K:
		return "Q4_K"
	case TypeQ5_K:
		return "Q5_K"
	case TypeQ6_K:
		return "Q6_K"
	default:
		return "?"
	}
}