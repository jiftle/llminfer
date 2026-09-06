// Package eval 实现前向推理：token 向量 → 逐层 Transformer → logits，并维护 KV Cache。
//
// 里程碑：M4。目标：单次 Forward（含 prefill 一批 / decode 单 token）输出整词表 logits。
//
// 前向顺序（与 Transformer 论文一致）：
//
//	查表：token id → 向量行 x[n, nEmb]
//	逐层：每层做「注意力子块 + 前馈子块」，残差就地累加
//	输出头：最终 RMSNorm + 词表投影，取最后一行作为 logits
package eval

import (
	"github.com/feiyuclaw/llminfer/internal/kernel/cache"
	"github.com/feiyuclaw/llminfer/internal/kernel/model"
	"github.com/feiyuclaw/llminfer/internal/kernel/tensor"
)

// Context 维护一次会话的推理状态。
type Context struct {
	m       *model.LLaMAModel
	nCtx    int // 上下文窗口长度
	kv      *cache.KVCache
	ropeNeox bool // RoPE 配对：qwen2 走 NEOX（前/后半配对），llama 走 NORMAL（相邻对）

	pos uint32 // 当前已处理 token 数（下一个待写位置）

	threads int // 并行 worker 数（M7.3，0=用默认）

	// 复用缓冲（跨 Forward 调用惰性扩容，decode 每步不重复分配）
	qBuf, kBuf, vBuf  []float32
	attBuf, normBuf   []float32
	layerOut          []float32
	ffnA, ffnB        []float32
	outT              *tensor.Tensor // 词表投影输出（复用）
	lastLogits        []float32      // 最近一次 logits 的拷贝
	seq               []uint32       // 已处理的 token 序列（长度==pos）
}

// NewContext 创建推理上下文。nCtx=0 时取模型上下文长度。
// threads 可选：并行 worker 数（0/缺省用默认 8）。
func NewContext(m *model.LLaMAModel, nCtx int, threads ...int) *Context {
	if nCtx == 0 {
		nCtx = m.CtxSize
	}
	// qwen2/gpt-neox 系用 NEOX 配对；llama 系用 NORMAL
	neox := false
	switch m.Arch {
	case "qwen2", "qwen", "qwen2moe", "falcon", "phi2", "phi3", "grok", "stablelm", "olmo2", "bitnet":
		neox = true
	}
	kv := cache.New(m.LayersCount, m.HeadsKVCount()*m.HeadDim(), nCtx)
	thr := 0
	if len(threads) > 0 {
		thr = threads[0]
	}
	return &Context{m: m, nCtx: nCtx, kv: kv, ropeNeox: neox, threads: thr}
}

// Pos 当前已处理 token 数。
func (ctx *Context) Pos() uint32 { return ctx.pos }

// Reset 清空会话状态：重置位置并用新的空 KV Cache 替换旧的。
// 用于多轮对话场景——每轮从完整历史重新 prefill，先清掉上一轮的 K/V。
func (ctx *Context) Reset() {
	ctx.kv = cache.New(ctx.m.LayersCount, ctx.m.HeadsKVCount()*ctx.m.HeadDim(), ctx.nCtx)
	ctx.pos = 0
	ctx.seq = ctx.seq[:0]
}

// KV 暴露 KV Cache（对比/调试用）。
func (ctx *Context) KV() *cache.KVCache { return ctx.kv }

// Forward 前向计算一批 token，返回最后一个 token 的 logits（长度=vocab）。
// ids 位置从 ctx.pos 起连续编号，K/V 写入缓存。
func (ctx *Context) Forward(ids []uint32) []float32 {
	m := ctx.m
	n := uint32(len(ids))
	if n == 0 {
		return nil
	}
	pos0 := ctx.pos
	nEmb := uint32(m.EmbeddingSize)
	nVocab := m.VocabSize
	kvEmb := uint32(m.HeadsKVCount() * m.HeadDim())

	// ① embed：token id → 向量行 x[n, nEmb]
	x := make([]float32, int(n)*int(nEmb))
	for i, id := range ids {
		// token_embd 按 [emb, vocab] 存，取第 id 行 = 该 token 的词向量
		if err := m.TokEmbeddings.DequantRow(uint32(id), x[int(i)*int(nEmb):(int(i)+1)*int(nEmb)]); err != nil {
			panic(err)
		}
	}

	// 准备复用缓冲（惰性扩容）
	ctx.qBuf = ensure(ctx.qBuf, int(n)*int(nEmb))
	ctx.kBuf = ensure(ctx.kBuf, int(n)*int(kvEmb))
	ctx.vBuf = ensure(ctx.vBuf, int(n)*int(kvEmb))
	ctx.attBuf = ensure(ctx.attBuf, int(n)*int(nEmb))
	ctx.normBuf = ensure(ctx.normBuf, int(n)*int(nEmb))
	ctx.layerOut = ensure(ctx.layerOut, int(n)*int(nEmb))
	ctx.ffnA = ensure(ctx.ffnA, int(n)*int(m.FFSize))
	ctx.ffnB = ensure(ctx.ffnB, int(n)*int(m.FFSize))

	// ② 逐层前向（残差就地累加）
	for li := 0; li < m.LayersCount; li++ {
		ctx.forwardLayer(li, x, n, nEmb, kvEmb, pos0)
	}

	// ③ 输出头：最终 RMSNorm → 词表投影，logits 取最后一行
	copy(ctx.normBuf, x)
	tensor.RMSNorm(ctx.normBuf, m.OutputNorm.AsFloat32(), m.RMSNormEpsilon)
	if ctx.outT == nil {
		ctx.outT = tensor.NewTensor(tensor.TYPE_F32, uint32(nVocab), n) // [vocab, n] 兼容 MatMul 输出行序
	}
	// 权重 m.Output 布局 [emb, vocab]，激活 [n, emb] → 输出 [n, vocab]
	tensor.MatMulTransB(actView(ctx.normBuf, n, uint32(nEmb)), m.Output, ctx.outT, ctx.threads)
	logits := ctx.outT.Floats[(int(n)-1)*nVocab : int(n)*nVocab]

	ctx.pos += n
	ctx.seq = append(ctx.seq, ids...)
	// 存 logits：outT 复用会被覆盖，供后续调用与解码比对
	ctx.lastLogits = ensure(ctx.lastLogits, nVocab)
	copy(ctx.lastLogits, logits)
	return logits
}

// forwardLayer 单层前向。注意力子块 + 前馈子块，各自带残差。
func (ctx *Context) forwardLayer(li int, x []float32, n uint32, nEmb, kvEmb, pos0 uint32) {
	m := ctx.m
	L := m.Layers[li]

	// —— 注意力子块 ——
	// 1) RMSNorm（结果放 normBuf，x 保留给残差）
	copy(ctx.normBuf, x)
	tensor.RMSNorm(ctx.normBuf, L.AttentionNorm.AsFloat32(), m.RMSNormEpsilon)
	// 2) QKV 投影（三个线性映射）
	tensor.MatMulTransB(actView(ctx.normBuf, n, nEmb), L.WQ, actView(ctx.qBuf, n, nEmb), ctx.threads)
	tensor.MatMulTransB(actView(ctx.normBuf, n, nEmb), L.WK, actView(ctx.kBuf, n, kvEmb), ctx.threads)
	tensor.MatMulTransB(actView(ctx.normBuf, n, nEmb), L.WV, actView(ctx.vBuf, n, kvEmb), ctx.threads)
	// 3) bias（qwen2 有）
	if L.WQb != nil {
		addBias(ctx.qBuf, L.WQb.AsFloat32())
	}
	if L.WKb != nil {
		addBias(ctx.kBuf, L.WKb.AsFloat32())
	}
	if L.WVb != nil {
		addBias(ctx.vBuf, L.WVb.AsFloat32())
	}
	// 4) RoPE：Q/K 各自按绝对位置旋转
	tensor.RoPE(ctx.qBuf, n, nEmb, uint32(m.HeadsCount), pos0, ctx.ropeNeox, m.RopeFreqBase, m.RopeFreqScale)
	tensor.RoPE(ctx.kBuf, n, kvEmb, uint32(m.HeadsKVCount()), pos0, ctx.ropeNeox, m.RopeFreqBase, m.RopeFreqScale)
	// 5) 写 KV（attention 要读本批的 K/V，必须先入缓存）
	ctx.writeKV(li, n, pos0, kvEmb)
	// 6) 注意力 → attBuf，再 WO 投影 + 残差
	ctx.attention(li, n, nEmb, kvEmb, pos0)
	tensor.MatMulTransB(actView(ctx.attBuf, n, nEmb), L.WO, actView(ctx.layerOut, n, nEmb), ctx.threads)
	addResidual(x, ctx.layerOut)

	// —— 前馈子块（SwiGLU）——
	copy(ctx.normBuf, x)
	tensor.RMSNorm(ctx.normBuf, L.FFNNorm.AsFloat32(), m.RMSNormEpsilon)
	tensor.MatMulTransB(actView(ctx.normBuf, n, nEmb), L.W1, actView(ctx.ffnA, n, uint32(m.FFSize)), ctx.threads)
	tensor.MatMulTransB(actView(ctx.normBuf, n, nEmb), L.W3, actView(ctx.ffnB, n, uint32(m.FFSize)), ctx.threads)
	// SwiGLU：silu(ffnA) ⊙ ffnB，其中 ffnA=W1·x（gate）、ffnB=W3·x（up）
	tensor.SiLU(ctx.ffnA)
	for i := range ctx.ffnA {
		ctx.ffnA[i] *= ctx.ffnB[i]
	}
	tensor.MatMulTransB(actView(ctx.ffnA, n, uint32(m.FFSize)), L.W2, actView(ctx.layerOut, n, nEmb), ctx.threads)
	addResidual(x, ctx.layerOut)
}

// writeKV 把本批 token 的 K/V 行写入 KV Cache。
func (ctx *Context) writeKV(li int, n, pos0, kvEmb uint32) {
	step := int(kvEmb)
	for i := uint32(0); i < n; i++ {
		pos := int(pos0 + i)
		row := int(i) * step
		ctx.kv.Write(li, pos, ctx.kBuf[row:row+step], ctx.vBuf[row:row+step])
	}
}

// ensure 确保 slice 至少 n 容量，扩容则新分配。
func ensure(s []float32, n int) []float32 {
	if cap(s) >= n {
		return s[:n]
	}
	return make([]float32, n)
}

// actView 把一段 F32 缓冲包装成 [rows, cols] 张量视图（零拷贝）。
func actView(floats []float32, rows, cols uint32) *tensor.Tensor {
	return &tensor.Tensor{
		Type:   tensor.TYPE_F32,
		Dims:   2,
		NE:     [4]uint32{cols, rows, 1, 1},
		Floats: floats,
	}
}

// addBias 就地逐行加 bias（bias 长度 = 每行长度）。
func addBias(dst, bias []float32) {
	per := len(bias)
	for i := range dst {
		dst[i] += bias[i%per]
	}
}

// addResidual 就地残差相加：x += y。
func addResidual(x, y []float32) {
	for i := range x {
		x[i] += y[i]
	}
}