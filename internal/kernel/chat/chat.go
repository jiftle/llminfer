// Package chat 实现对话模板（Chat Template）：把多轮消息列表渲染成模型输入的 prompt 字符串。
//
// 里程碑：M6。目标：让引擎能像 ChatGPT 一样多轮对话。
//
// 关键决策：
//   - 不做 Jinja2 解析（那是完整模板引擎），而是做「关键词检测」——
//     看 chat_template 字符串里有哪些特征标记，就判定是哪种模板（ChatML / LLaMA3...）。
//   - 渲染逻辑用模板字符串里提取的「规则」+ 少量硬编码语义拼出来，覆盖主流模板即可。
//   - qwen2.5 的官方模板有个隐形规则：如果第一轮没有 system 消息，会自动补一条
//     默认系统提示（模板里写死的那句）。这个要提取出来。
package chat

import (
	"regexp"
	"strings"
)

// TemplateType 模板类型枚举。
type TemplateType int

const (
	TemplateUnknown TemplateType = iota
	TemplateChatML               // Qwen2/Qwen2.5/ChatGLM：<|im_start|>role\ncontent<|im_end|>
	TemplateLLaMA3               // Llama 3：<|start_header_id|>role<|end_header_id|>
)

// String 返回模板类型名。
func (t TemplateType) String() string {
	switch t {
	case TemplateChatML:
		return "chatml"
	case TemplateLLaMA3:
		return "llama3"
	default:
		return "unknown"
	}
}

// Message 单条对话消息（与 OpenAI 对齐：role + content）。
type Message struct {
	Role    string
	Content string
}

// Template 对话模板：从 tokenizer.chat_template 字符串检测类型并渲染消息列表。
type Template struct {
	Type TemplateType
	// DefaultSystem 模板自带的默认系统提示。
	// qwen2.5 模板规定：首条消息不是 system 时，先插这句。
	// 从模板字符串里提取（可能为空 = 模板没有默认系统提示）。
	DefaultSystem string

	raw string // 原始模板字符串（调试用）
}

// Detect 从原始 chat_template 字符串检测模板类型。
// 检测逻辑：关键词匹配（非 Jinja2 解析）。
func Detect(raw string) *Template {
	t := &Template{raw: raw}
	if raw == "" {
		return t
	}
	switch {
	// ChatML：含 <|im_start|>，且没有 Phi-4 / SmolVLM 的特有标记
	case strings.Contains(raw, "<|im_start|>") &&
		!strings.Contains(raw, "<|im_sep|>") &&
		!strings.Contains(raw, "<end_of_utterance>"):
		t.Type = TemplateChatML
		t.DefaultSystem = extractDefaultSystem(raw)
	// LLaMA3：含 <|start_header_id|>
	case strings.Contains(raw, "<|start_header_id|>"):
		t.Type = TemplateLLaMA3
	}
	return t
}

// defaultSystemRe 匹配 qwen 系模板的第一轮默认系统提示。
// 模板长这样：{%- if messages[0]['role'] == 'system' %}...{%- else %}
//            {{- '<|im_start|>system\nYou are Qwen...<|im_end|>\n' }}{%- endif %}
// 匹配「system 标记 + 文案 + 结束标记」的单引号整段，再把标记剥掉取中间的文案。
var defaultSystemRe = regexp.MustCompile(`'<\|im_start\|>system[^']*<\|im_end\|>[^']*'`)

// extractDefaultSystem 从模板字符串提取「首条非 system 时插入的默认系统提示文案」。
// 匹配到整个片段后，去掉定界标记（前缀 <|im_start|>system，后缀 <|im_end|>）得到纯文案。
// 匹配不到返回空字符串（模板没有默认系统提示）。
func extractDefaultSystem(raw string) string {
	seg := defaultSystemRe.FindString(raw)
	if seg == "" {
		return ""
	}
	// 剥掉外层单引号再剥两端的定界标记
	seg = strings.Trim(seg, "'")
	const open = "<|im_start|>system"
	const close = "<|im_end|>"
	if !strings.HasPrefix(seg, open) {
		return ""
	}
	seg = seg[len(open):]
	// 模板里 <|im_start|>system 后是字面 \n（反斜杠+n），剥掉
	seg = strings.TrimPrefix(seg, `\n`)
	if i := strings.Index(seg, close); i >= 0 {
		seg = seg[:i]
	}
	return seg
}

// Render 把消息列表渲染成模型输入字符串。
// addGenerationPrompt=true 时在末尾追加 assistant 提示（让模型接着往下写回复）。
func (t *Template) Render(messages []Message, addGenerationPrompt bool) string {
	switch t.Type {
	case TemplateChatML:
		return t.renderChatML(messages, addGenerationPrompt)
	case TemplateLLaMA3:
		return t.renderLLaMA3(messages, addGenerationPrompt)
	default:
		return t.renderFallback(messages)
	}
}

// renderChatML 渲染 ChatML 格式。
// 目标格式：
//
//	[可选默认system]<|im_start|>system\n...<|im_end|>
//	(每条消息)<|im_start|>role\ncontent<|im_end|>\n
//	(若要生成)<|im_start|>assistant\n
//
// 对齐 qwen2.5 官方模板的语义（不含 tools 分支）。
func (t *Template) renderChatML(messages []Message, addGenerationPrompt bool) string {
	var sb strings.Builder

	// 隐形规则：首条不是 system 时，模板会自动插入默认系统提示
	if t.DefaultSystem != "" &&
		(len(messages) == 0 || messages[0].Role != "system") {
		sb.WriteString("<|im_start|>system\n")
		sb.WriteString(t.DefaultSystem)
		sb.WriteString("<|im_end|>\n")
	}

	for _, msg := range messages {
		sb.WriteString("<|im_start|>")
		sb.WriteString(msg.Role)
		sb.WriteString("\n")
		sb.WriteString(msg.Content)
		sb.WriteString("<|im_end|>\n")
	}

	if addGenerationPrompt {
		sb.WriteString("<|im_start|>assistant\n")
	}
	return sb.String()
}

// renderLLaMA3 渲染 LLaMA3 格式（<|begin_of_text|> + 头部标记）。
func (t *Template) renderLLaMA3(messages []Message, addGenerationPrompt bool) string {
	var sb strings.Builder
	sb.WriteString("<|begin_of_text|>")
	for _, msg := range messages {
		sb.WriteString("<|start_header_id|>")
		sb.WriteString(msg.Role)
		sb.WriteString("<|end_header_id|>\n\n")
		sb.WriteString(msg.Content)
		sb.WriteString("<|eot_id|>")
	}
	if addGenerationPrompt {
		sb.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
	}
	return sb.String()
}

// renderFallback 兜底：没有识别到模板时，取最后一条 user 消息直接当 prompt。
func (t *Template) renderFallback(messages []Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return messages[i].Content
		}
	}
	return ""
}