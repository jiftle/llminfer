// Package cache 提供 KV Cache：存解码时已算好的 K/V，避免每步重算历史注意力。
//
// 为什么需要：注意力公式中，位置 t 的输出只依赖「自己的 Q」和「历史全部 K/V」。
// 生成新词时前文的 K/V 不变，就把它们缓存下来，decode 每步只算新 Q 与新 K/V。
//
// 布局：单块 []float32，K 与 V 各占一半。每半按 [layer][pos][kvHead][headDim] 组织，
// 即一个位置连续存放全部 kvHead 个头、每个头 headDim 维（K/V 各自独立累进）。
package cache

// KVCache 是 KV 缓存：分配、写入、按头读取。
type KVCache struct {
	layers int // Transformer 层数
	step   int // 单层单位置的元素数 = kvHead × headDim
	maxSeq int // 上下文窗口长度（位置上限）
	buf    []float32
	len    int // 当前有效位置数（每个 Write 推进）
}

// New 创建 KV 缓存。内存 = layers × maxSeq × step × 2（K 半区 + V 半区）。
func New(layers, step, maxSeq int) *KVCache {
	c := &KVCache{layers: layers, step: step, maxSeq: maxSeq}
	c.buf = make([]float32, layers*maxSeq*step*2)
	return c
}

// Layers 层数。
func (c *KVCache) Layers() int { return c.layers }

// Step 单层单位置的元素数。
func (c *KVCache) Step() int { return c.step }

// MaxSeq 上下文窗口长度。
func (c *KVCache) MaxSeq() int { return c.maxSeq }

// Len 当前有效位置数。
func (c *KVCache) Len() int { return c.len }

// Bytes 占用的内存字节数。
func (c *KVCache) Bytes() uint64 {
	return uint64(len(c.buf)) * 4
}

// Raw 底层缓冲视图（调试/对比用，勿改）。
func (c *KVCache) Raw() []float32 { return c.buf }

// Write 把位置 pos 的 K/V 行写入第 layer 层。
// k/v 长度必须等于 step，按 [kvHead][headDim] 连续排列。
func (c *KVCache) Write(layer, pos int, k, v []float32) {
	seg := c.step * c.maxSeq
	baseK := layer * seg
	baseV := (layer + c.layers) * seg
	rowK := baseK + pos*c.step
	rowV := baseV + pos*c.step
	copy(c.buf[rowK:rowK+c.step], k)
	copy(c.buf[rowV:rowV+c.step], v)
	if pos+1 > c.len {
		c.len = pos + 1
	}
}

// KHead 返回第 layer 层 pos 位置第 head 个 KV 头的 K 向量（headDim 长切片，引用底层）。
func (c *KVCache) KHead(layer, pos, head, headDim int) []float32 {
	seg := c.step * c.maxSeq
	off := layer*seg + pos*c.step + head*headDim
	return c.buf[off : off+headDim]
}

// VHead 返回第 layer 层 pos 位置第 head 个 KV 头的 V 向量（引用底层）。
func (c *KVCache) VHead(layer, pos, head, headDim int) []float32 {
	seg := c.step * c.maxSeq
	off := (layer+c.layers)*seg + pos*c.step + head*headDim
	return c.buf[off : off+headDim]
}

// Clear 清空缓存并复位有效位置数。
func (c *KVCache) Clear() {
	clear(c.buf)
	c.len = 0
}