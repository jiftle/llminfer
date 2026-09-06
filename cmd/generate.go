package cmd

import (
	"fmt"

	"github.com/feiyuclaw/llminfer/internal/kernel/chat"
	"github.com/feiyuclaw/llminfer/internal/kernel/eval"
	"github.com/feiyuclaw/llminfer/internal/kernel/gguf"
	"github.com/feiyuclaw/llminfer/internal/kernel/model"
	"github.com/feiyuclaw/llminfer/internal/kernel/sampler"
	"github.com/feiyuclaw/llminfer/internal/kernel/tokenizer"
)

// GenerateOptions 生成参数。
type GenerateOptions struct {
	NCtx        int     // 上下文窗口
	MaxTokens   int     // 最多生成多少 token
	Temperature float64
	TopK        int
	TopP        float64
	Seed        int64
	Threads     int     // 并行 worker 数（≤0 按核数 2/3 自动）
	Verbose     bool    // 打印模型信息与每步调试
}

// Generate 加载模型、编码 prompt、自回归生成文本。
// 返回 (生成的文本，生成 token 数)。
func Generate(modelPath, prompt string, opt GenerateOptions) (string, int, error) {
	// 加载模型 + tokenizer + 推理上下文 + 采样器
	m, err := model.Load(modelPath)
	if err != nil {
		return "", 0, fmt.Errorf("加载模型失败: %w", err)
	}
	// tokenizer 需要 GGUF 元数据（词表/merges/特殊 token）；model 只留了张量，另行读元数据（轻量）
	gf, err := gguf.ReadFile(modelPath)
	if err != nil {
		return "", 0, fmt.Errorf("读取元数据失败: %w", err)
	}
	defer gf.Close()
	tok, err := tokenizer.NewFromGGUF(gf, m.Arch)
	if err != nil {
		return "", 0, fmt.Errorf("初始化分词器失败: %w", err)
	}

	ctx := eval.NewContext(m, opt.NCtx, opt.Threads)
	sm := sampler.NewSampler(opt.Seed)

	if opt.Verbose {
		fmt.Printf("arch=%s layers=%d emb=%d vocab=%d\n",
			m.Arch, m.LayersCount, m.EmbeddingSize, m.VocabSize)
	}

	// 编码 prompt（仅文本部分；特殊 token 由 Encode 处理）
	ids := tok.Encode(prompt)
	if opt.Verbose {
		fmt.Printf("prompt: %q → %d tokens: %v\n", prompt, len(ids), ids)
	}

	// Prefill：一次前向消化整个 prompt，得到 next-token logits
	logits := ctx.Forward(ids)

	// Decode loop：反复「采样 → 解码 → 前向一个 token」，直到 EOS 或上限
	cfg := sampler.Config{
		Temperature: opt.Temperature,
		TopK:        opt.TopK,
		TopP:        opt.TopP,
	}
	eos := uint32(tok.EOS)
	stopIDs := tok.ChatStopIDs() // 除 EOS 外的特殊停止 token（如 <|im_end|>）

	generated := []uint32{}
	text := ""
	for i := 0; i < opt.MaxTokens; i++ {
		id := sm.Sample(logits, cfg)
		// 命中停止标识即收尾（EOS 或对话标记）
		if stopIDs[id] || (eos > 0 && id == eos) {
			break
		}
		generated = append(generated, id)
		piece := tok.Decode([]uint32{id})
		text += piece
		if opt.Verbose {
			fmt.Printf("  [%3d] token=%d text=%q\n", i, id, piece)
		}

		// 前向刚生成的那个 token（KV cache 自动累积历史），得到下一个 logits
		logits = ctx.Forward([]uint32{id})
	}

	return text, len(generated), nil
}

// ChatSession 一次多轮对话会话：持有模型/tokenizer/推理上下文/采样器 + 消息历史。
// 每轮把完整历史渲染成 prompt 重新 prefill（简单正确）。
type ChatSession struct {
	m         *model.LLaMAModel    // 模型（含权重）
	tok       *tokenizer.Tokenizer // 分词器（含对话模板）
	ctx     *eval.Context        // 推理上下文
	sm        *sampler.Sampler     // 采样器
	template  *chat.Template       // 对话模板（必有，NewChatSession 已校验）
	history   []chat.Message       // 完整对话历史（含之前 assistant 的回复）
	samplerCfg sampler.Config      // 采样参数
	maxTokens int                  // 每轮最多生成 token 数
}

// NewChatSession 打开一个对话会话。modelPath：模型文件路径。
func NewChatSession(modelPath string, opt GenerateOptions) (*ChatSession, error) {
	m, err := model.Load(modelPath)
	if err != nil {
		return nil, fmt.Errorf("加载模型失败: %w", err)
	}
	gf, err := gguf.ReadFile(modelPath)
	if err != nil {
		return nil, fmt.Errorf("读取元数据失败: %w", err)
	}
	defer gf.Close()
	tok, err := tokenizer.NewFromGGUF(gf, m.Arch)
	if err != nil {
		return nil, fmt.Errorf("初始化分词器失败: %w", err)
	}
	if tok.ChatTemplate == nil || tok.ChatTemplate.Type == chat.TemplateUnknown {
		return nil, fmt.Errorf("模型没有可识别的对话模板，无法进入聊天模式")
	}
	return &ChatSession{
		m:         m,
		tok:       tok,
		ctx:     eval.NewContext(m, opt.NCtx, opt.Threads),
		sm:        sampler.NewSampler(opt.Seed),
		template:  tok.ChatTemplate,
		samplerCfg: sampler.Config{
			Temperature: opt.Temperature,
			TopK:        opt.TopK,
			TopP:        opt.TopP,
		},
		maxTokens: opt.MaxTokens,
	}, nil
}

// Chat 输入一条 user 消息，渲染完整历史 → prefill → 生成回复。
// 返回 (模型回复文本, 生成 token 数)。回复自动存入 history 供下一轮使用。
func (s *ChatSession) Chat(userMsg string) (string, int, error) {
	// 本轮完整历史 = 旧历史 + 新 user 消息
	messages := append(append([]chat.Message{}, s.history...),
		chat.Message{Role: "user", Content: userMsg})

	// 渲染成模型输入（addGenerationPrompt=true：末尾加 <|im_start|>assistant\n 引导模型开口）
	prompt := s.template.Render(messages, true)

	// 编码。特殊 token（<|im_end|> 等）由 encodeWithSpecial 整体匹配成单个 id。
	ids := s.tok.Encode(prompt)
	if len(ids) == 0 {
		return "", 0, fmt.Errorf("prompt 分词为空")
	}

	// 每轮重建 KV cache：把完整渲染 prompt 一次性 prefill 进去（简单正确版）。
	// 性能优化（增量续推，跳过历史）留到后续里程碑。
	s.ctx.Reset()
	logits := s.ctx.Forward(ids)

	// 解码循环：采样 → 命中停止标记则停 → 前向下一个 token
	stopIDs := s.tok.ChatStopIDs()
	generated := []uint32{}
	text := ""
	for i := 0; i < s.maxTokens; i++ {
		id := s.sm.Sample(logits, s.samplerCfg)
		if stopIDs[id] {
			break
		}
		generated = append(generated, id)
		text += s.tok.Decode([]uint32{id})
		logits = s.ctx.Forward([]uint32{id})
	}

	// 记入历史，供下一轮（assistant 的回复本身就是下一轮渲染时的 assistant 消息）
	s.history = messages
	s.history = append(s.history, chat.Message{Role: "assistant", Content: text})
	return text, len(generated), nil
}

// History 返回当前消息历史（只读，测试/调试用）。
func (s *ChatSession) History() []chat.Message {
	return append([]chat.Message{}, s.history...)
}