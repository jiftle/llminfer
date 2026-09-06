// Package model 表示模型静态结构：从 GGUF 元数据提取超参数，把张量挂到 LLaMAModel。
//
// 里程碑：M3。目标：让「文件里的 290 个张量」变成「内存里按语义组织好的模型」。
//
// 关键决策：
//   - 为简单先把整个模型文件读进内存（0.5B ≈ 398MB，单进程可接受）；真正优化见 M5。
//   - 张量只做「挂载」（Tensor.Data 指向文件字节切片），量化权重不在加载时反量化，
//     等矩阵乘真正用到时再按行反量化（省内存 + 省启动时间）。
package model

import (
	"fmt"
	"os"
	"strings"

	"github.com/feiyuclaw/llminfer/internal/kernel/gguf"
	"github.com/feiyuclaw/llminfer/internal/kernel/tensor"
)

// LLaMALayer 是单个 Transformer 层的一整套权重。
type LLaMALayer struct {
	// qwen2/llama 版注意力：attn_norm → 线性 QKV（含 bias）→ 输出投影 WO
	AttentionNorm *tensor.Tensor
	WQ, WK, WV, WO *tensor.Tensor
	WQb, WKb, WVb  *tensor.Tensor // 可选 bias（qwen2 有）

	// SwiGLU 前馈：ffn_norm → gate(W1)*up(W3) → down(W2)
	FFNNorm        *tensor.Tensor
	W1, W2, W3     *tensor.Tensor // gate / down / up
}

// LLaMAModel 模型内存视图：超参数 + 权重张量 + 词表。
type LLaMAModel struct {
	Arch           string
	LayersCount int
	EmbeddingSize  int
	VocabSize      int
	CtxSize        int
	HeadsCount     int // Q 头数
	HeadsKV        int // KV 头数（GQA；0 表示与 Q 相同）
	FFSize         int
	RopeFreqBase   float32
	RopeFreqScale  float32
	RMSNormEpsilon float32

	// 词表（M2 用）
	VocabTokens []string

	// 权重
	TokEmbeddings *tensor.Tensor // [emb, vocab]
	OutputNorm    *tensor.Tensor // [emb]
	Output        *tensor.Tensor // [emb, vocab]（tied 时复用 TokEmbeddings）
	Layers        []LLaMALayer
}

// Load 读取 GGUF 文件并挂载全部张量。
func Load(path string) (*LLaMAModel, error) {
	ggufFile, err := gguf.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("model.Load read gguf: %w", err)
	}
	defer ggufFile.Close()

	// 整个文件读入内存（M5 再换 mmap 优化加载/内存）
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("model.Load read file: %w", err)
	}

	m, err := FromGGUF(ggufFile, raw)
	if err != nil {
		return nil, err
	}
	return m, nil
}

// FromGGUF 从已解析的 GGUF + 文件字节构造模型（供测试/复用 parser 场景）。
func FromGGUF(gf *gguf.GGUFFile, raw []byte) (*LLaMAModel, error) {
	m := &LLaMAModel{
		Arch:           gf.GetString("general.architecture"),
		RopeFreqBase:   float32(gf.GetFloat("qwen2.rope.freq_base")),
		RMSNormEpsilon: float32(gf.GetFloat("qwen2.attention.layer_norm_rms_epsilon")),
		RopeFreqScale:  1,
	}
	if m.Arch == "" {
		return nil, fmt.Errorf("model: 缺少 general.architecture")
	}
	// 词表：tokens 数组长度即 vocab size（qwen2 不在 metadata 写 vocab_size）
	m.VocabTokens = gf.GetStringSlice("tokenizer.ggml.tokens")
	m.VocabSize = len(m.VocabTokens)
	if m.VocabSize == 0 {
		return nil, fmt.Errorf("model: tokenizer.ggml.tokens 为空")
	}

	// 超参数：块数 = 层数的通用叫法（block_count）
	m.LayersCount = gf.GetInt("qwen2.block_count")
	m.EmbeddingSize = gf.GetInt("qwen2.embedding_length")
	m.HeadsCount = gf.GetInt("qwen2.attention.head_count")
	m.HeadsKV = gf.GetInt("qwen2.attention.head_count_kv")
	m.FFSize = gf.GetInt("qwen2.feed_forward_length")
	m.CtxSize = gf.GetInt("qwen2.context_length")
	if m.CtxSize == 0 {
		m.CtxSize = 4096 // 兜底
	}
	if m.HeadsKV == 0 {
		m.HeadsKV = m.HeadsCount // 无 GQA 时 KV 头等于 Q 头
	}

	// 按名称把张量字节切片挂到模型字段
	if err := m.attachTensors(gf, raw); err != nil {
		return nil, err
	}
	return m, nil
}

// attachTensors 遍历 GGUF 张量表，把每个权重字节引用挂到对应字段。
func (m *LLaMAModel) attachTensors(gf *gguf.GGUFFile, raw []byte) error {
	m.Layers = make([]LLaMALayer, m.LayersCount)

	for _, ti := range gf.Tensors {
		// 张量原始字节区：GGUF 的张量 offset 相对数据区起点，真实文件偏移要加 DataStart
		start := int(gf.DataStart) + int(ti.Offset)
		wt := &tensor.Tensor{
			Type: tensor.DType(ti.Type),
			Dims: uint32(len(ti.NEE)),
			Data: raw[start : start+int(ti.Nbytes)],
		}
		for i, n := range ti.NEE {
			wt.NE[i] = uint32(n)
		}

		name := ti.Name
		switch {
		case name == "token_embd.weight":
			m.TokEmbeddings = wt
		case name == "output_norm.weight":
			m.OutputNorm = wt
		case name == "output.weight":
			m.Output = wt
		default:
			// 形如 blk.N.attn_q.weight / blk.N.attn_output.bias
			if !strings.HasPrefix(name, "blk.") {
				continue
			}
			rest := strings.TrimPrefix(name, "blk.")
			dot := strings.Index(rest, ".")
			if dot < 0 {
				continue
			}
			var li int
			fmt.Sscanf(rest[:dot], "%d", &li)
			if li < 0 || li >= m.LayersCount {
				continue
			}
			field := rest[dot+1:]
			assignLayer(&m.Layers[li], field, wt)
		}
	}

	if m.TokEmbeddings == nil {
		return fmt.Errorf("model: 缺少 token_embd.weight")
	}
	if m.OutputNorm == nil {
		return fmt.Errorf("model: 缺少 output_norm.weight")
	}
	if m.Output == nil {
		m.Output = m.TokEmbeddings // tied embeddings（qwen2 常见）
	}
	return nil
}

// assignLayer 按「张量名后缀 → 层字段」挂载。
func assignLayer(layer *LLaMALayer, field string, wt *tensor.Tensor) {
	switch field {
	case "attn_norm.weight":
		layer.AttentionNorm = wt
	case "attn_q.weight":
		layer.WQ = wt
	case "attn_k.weight":
		layer.WK = wt
	case "attn_v.weight":
		layer.WV = wt
	case "attn_output.weight":
		layer.WO = wt
	case "ffn_norm.weight":
		layer.FFNNorm = wt
	case "ffn_gate.weight":
		layer.W1 = wt
	case "ffn_up.weight":
		layer.W3 = wt
	case "ffn_down.weight":
		layer.W2 = wt
	case "attn_q.bias":
		layer.WQb = wt
	case "attn_k.bias":
		layer.WKb = wt
	case "attn_v.bias":
		layer.WVb = wt
	}
}

// HeadsKVCount 返回实际参与 KV 缓存的头数（无 GQA 时等于 Q 头数）。
func (m *LLaMAModel) HeadsKVCount() int {
	if m.HeadsKV > 0 {
		return m.HeadsKV
	}
	return m.HeadsCount
}

// HeadDim 单头维度。
func (m *LLaMAModel) HeadDim() int {
	return m.EmbeddingSize / m.HeadsCount
}