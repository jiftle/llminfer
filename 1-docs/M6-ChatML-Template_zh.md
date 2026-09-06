# M6：ChatML 对话模板（多轮聊天）

> 状态：✅ 完成（ChatML 渲染 + 交互聊天，多轮上下文记忆验证通过）
> 代码：internal/kernel/chat/ + cmd/(command.go 交互循环)
> 依赖：M5 生成循环、M2 tokenizer（特殊 token 整体匹配）

## 一句话本质

M5 之前只能「给一句话，补完后面」；M6 让引擎像 ChatGPT 一样**多轮聊天**。核心是「对话模板」——把「谁在什么时候说了什么」（消息列表）翻译成模型认识的 prompt 格式。

```
[system: 你是助手][user: 你好][assistant: 嗨!][user: 今天天气怎么样?]
        ↓ 模板渲染（ChatML）
<|im_start|>system...<|im_end|>
<|im_start|>user...<|im_end|>
<|im_start|>assistant...<|im_end|>
<|im_start|>user...<|im_end|>
<|im_start|>assistant\n      ← 追加这行，引导模型开口
```

## ChatML 格式（大白话）

qwen2.5 用 **ChatML**：每条消息 = `开始标记 + 角色 + 内容 + 结束标记`。

$$\underbrace{\texttt{<|im_start|>}}_{\text{开始}} \underbrace{\texttt{user}}_{\text{角色}} \underbrace{\texttt{\textbackslash n}}_{\text{换行}} \underbrace{\texttt{你好}}_{\text{内容}} \underbrace{\texttt{<|im_end|>}}_{\text{结束}}$$

- 角色只有三个：`system`（设定人设）、`user`（用户）、`assistant`（AI 回复）
- 结尾追加 `<|im_start|>assistant\n` 就是「告诉模型：该你说话了」
- 特殊 token（`<|im_start|>` 等）在词表里是**单独一个 id**，分词时整体匹配不可拆（M2 已实现）

## 隐形坑：默认 system 消息

qwen2.5 的模板字符串里藏着一条规则——**如果第一轮没有 system，就自动插一句**：

```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
```

```go
// internal/kernel/chat/chat.go 提取默认 system 的正则
defaultSystemRe = 匹配 'system<|im_start|>...文案...<|im_end|>' 整段，剥标记取文案
```

违反它的症状：首轮直接以 user 开头，模型会「答非所问」（因为它没见过 `以 user 开始` 的输入，训练时全是 `system先开头`）。

## 检测：不做 Jinja 解析

模型存的是完整 Jinja2 模板（带 if/for），但我们**不解析模板引擎**（太重）。按**关键词检测**判定模板类型：

| 特征关键词 | 判定 |
|---|---|
| 含 `<|im_start|>` 且无 `<|im_sep|>`/`<end_of_utterance>` | ChatML（qwen/chatglm） |
| 含 `<|start_header_id|>` | LLaMA3 |
| 都没有 | 未知 → 兜底取最后一条 user |

## 多轮的关键：历史要一直带着

单轮不需要记忆；多轮必须把**之前所有轮次**都塞进下一次的 prompt：

```
第2轮 prompt = [第0轮 system][第1轮 user 你好][第0轮 assistant 嗨!][第2轮 user 今天天气?]...
                    ↑ 这三条是第1轮的"记忆"，全部原样渲染进第2轮的输入
```

实现：`ChatSession` 持有 `history []Message`，每轮把用户输入 append 进去后整体渲染。KV cache 每轮用 `ctx.Reset()` 重建（简单正确；增量续推优化留作后续）。

## 验收

**1. 模板检测 + 渲染单测全绿**（internal/kernel/chat/，6 个测试）：ChatML/LLaMA3/Unknown 检测、默认 system 提取、单轮/多轮渲染逐字节、兜底。

**2. 真实模型多轮记忆验证**（贪心，qwen2.5-0.5b）：

```
你: The secret word is banana.
AI: The secret word is "banana".
你: What is the secret word?
AI: The secret word is "banana".     ← 记得上一轮!✅
```

**3. 中文闲聊连贯**：

```
你: 你好，介绍一下你自己
AI: 你好！我是Qwen，一个由阿里云开发的超大规模语言模型。我叫通义千问……
```

## 踩过的坑（隐形知识）

1. **Go 正则的 `\n` 是反斜杠+n 两个字符**：模板字符串里 `<|im_start|>system\n` 是字面 `\` 和 `n`，**不是**换行符。正则里要写 `\\n`（转义后匹配字面两位），测试常量用反引号而非双引号字符串。
2. **默认 system 的提取陷阱**：模板里默认系统提示是「system标记+文案+结束标记」的整段单引号字符串（如 `'<|im_start|>system\nYou are...<|im_end|>\n'`），剥标记时要连带剥掉标记后的字面 `\n`，否则渲染多一个换行。
3. **特殊 token 必须整体编码**：渲染后的 prompt 含 `<|im_start|>`，若分词把它拆开，模型永远学不对。M2 的 encodeWithSpecial 已按特殊 token 整段匹配——这正是它能工作的前提。

## 术语表

| 术语 | 中文 | 一句话解释 |
|---|---|---|
| Chat Template | 对话模板 | 把消息列表变成模型输入字符串的规则 |
| ChatML | 对话标记语言 | `<|im_start|>role\ncontent<|im_end|>` 形式的聊天格式（qwen 用） |
| system/user/assistant | 角色 | 人设 / 用户 / AI回复 三种消息类型 |
| 特殊 token | 特殊标记 | 词表里单独占一个 id 的控制符（如 `<|im_start|>`） |
| Default system | 默认系统提示 | 首轮无 system 时模板自动插入的人设语句 |
| add_generation_prompt | 生成提示 | 消息末尾追加 `<|im_start|>assistant\n` 让模型开始回复 |
| Jinja | 模板引擎 | 模型自带的完整模板语言（我们只做关键词检测，不解析） |