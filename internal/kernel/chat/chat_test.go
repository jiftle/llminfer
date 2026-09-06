package chat

import "testing"

// qwen2.5 官方 ChatML 模板的默认 system 提示要能从模板字符串提取出来。
// 注意用反引号字符串：模板里 <|im_start|>system\n 是字面反斜杠+n，不是换行符。
const qwen25Template = `{%- if tools %}
...
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
...`

func TestDetectChatML(t *testing.T) {
	tmpl := Detect(qwen25Template)
	if tmpl.Type != TemplateChatML {
		t.Fatalf("应识别为 ChatML，得到 %v", tmpl.Type)
	}
	want := "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
	if tmpl.DefaultSystem != want {
		t.Fatalf("默认 system 提取错误\n  想要: %q\n  得到: %q", want, tmpl.DefaultSystem)
	}
}

func TestDetectEmpty(t *testing.T) {
	tmpl := Detect("")
	if tmpl.Type != TemplateUnknown {
		t.Fatalf("空模板应识别为 Unknown，得到 %v", tmpl.Type)
	}
}

func TestDetectLLaMA3(t *testing.T) {
	tmpl := Detect(`{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>{% endfor %}`)
	if tmpl.Type != TemplateLLaMA3 {
		t.Fatalf("应识别为 LLaMA3，得到 %v", tmpl.Type)
	}
}

func TestRenderChatML(t *testing.T) {
	tmpl := Detect(qwen25Template)
	messages := []Message{
		{Role: "user", Content: "你好"},
	}
	got := tmpl.Render(messages, true)
	// 首条非 system → 自动插默认 system；末尾追加 assistant 提示
	want := "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n" +
		"<|im_start|>user\n你好<|im_end|>\n" +
		"<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("ChatML 渲染不符\n  想要: %q\n  得到: %q", want, got)
	}
}

func TestRenderChatMLMultiTurn(t *testing.T) {
	tmpl := Detect(qwen25Template)
	messages := []Message{
		{Role: "system", Content: "你是助手"},
		{Role: "user", Content: "第一条"},
		{Role: "assistant", Content: "回复一"},
		{Role: "user", Content: "第二条"},
	}
	got := tmpl.Render(messages, true)
	want := "<|im_start|>system\n你是助手<|im_end|>\n" +
		"<|im_start|>user\n第一条<|im_end|>\n" +
		"<|im_start|>assistant\n回复一<|im_end|>\n" +
		"<|im_start|>user\n第二条<|im_end|>\n" +
		"<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("多轮渲染不符\n  想要: %q\n  得到: %q", want, got)
	}
}

func TestRenderFallback(t *testing.T) {
	tmpl := Detect("") // 未识别模板 → 兜底取最后一条 user
	got := tmpl.Render([]Message{{Role: "user", Content: "hello"}}, true)
	if got != "hello" {
		t.Fatalf("兜底应返回最后 user，得到 %q", got)
	}
}