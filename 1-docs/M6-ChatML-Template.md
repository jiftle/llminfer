# M6: ChatML Conversation Template (Multi-turn Chat)

> Status: ✅ Complete (ChatML rendering + interactive chat, multi-turn context memory verified)
> Code: internal/kernel/chat/ + cmd/ (command.go interaction loop)
> Depends on: M5 generation loop, M2 tokenizer (special-token whole-token matching)

## The Gist in One Sentence

Before M6, the engine could only "given a sentence, continue what follows"; M6 lets the engine chat **multi-turn** like ChatGPT. The core is the "conversation template" — turning "who said what, when" (a message list) into the prompt format the model understands.

```
[system: You are an assistant][user: 你好][assistant: 嗨!][user: 今天天气怎么样?]
        ↓ template rendering (ChatML)
<|im_start|>system...<|im_end|>
<|im_start|>user...<|im_end|>
<|im_start|>assistant...<|im_end|>
<|im_start|>user...<|im_end|>
<|im_start|>assistant\n      ← append this line to get the model to speak
```

## ChatML Format (Plain Language)

qwen2.5 uses **ChatML**: each message = `start token + role + content + end token`.

$$\underbrace{\texttt{<|im_start|>}}_{\text{start}} \underbrace{\texttt{user}}_{\text{role}} \underbrace{\texttt{\textbackslash n}}_{\text{newline}} \underbrace{\texttt{你好}}_{\text{content}} \underbrace{\texttt{<|im_end|>}}_{\text{end}}$$

- There are only three roles: `system` (sets the persona), `user` (the user), `assistant` (the AI reply)
- Appending `<|im_start|>assistant\n` at the end is how you "tell the model: it's your turn to speak"
- Special tokens (`<|im_start|>` etc.) each occupy a **single id** in the vocabulary and must be matched as a whole, never split, during tokenization (already implemented in M2)

## Hidden Pitfall: The Default System Message

The qwen2.5 template string hides a rule — **if the first turn has no system message, one is inserted automatically**:

```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
```

```go
// internal/kernel/chat/chat.go regex that extracts the default system message
defaultSystemRe = match the whole 'system<|im_start|>...text...<|im_end|>' segment, strip the markers and keep the text
```

Symptom of violating it: if the first turn starts directly with user, the model "answers irrelevantly" (because it never saw inputs that *start with user* — during training they all *start with system*).

## Detection: No Jinja Parsing

The model stores a full Jinja2 template (with if/for), but we **don't parse the template engine** (too heavy). Template type is determined by **keyword detection**:

| Characteristic keywords | Determination |
|---|---|
| Contains `<|im_start|>` and no `<|im_sep|>`/`<end_of_utterance>` | ChatML (qwen/chatglm) |
| Contains `<|start_header_id|>` | LLaMA3 |
| Contains neither | Unknown → fall back to the last user message |

## The Key to Multi-turn: Always Carry the History

A single turn needs no memory; multi-turn requires stuffing **all previous turns** into the next prompt:

```
2nd-turn prompt = [turn-0 system][turn-1 user 你好][turn-0 assistant 嗨!][turn-2 user 今天天气?]...
                      ↑ these three are turn-1's "memory", all rendered verbatim into turn-2's input
```

Implementation: `ChatSession` holds `history []Message`; each turn it appends the user input, then renders everything together. The KV cache is rebuilt each turn with `ctx.Reset()` (simple and correct; incremental continuation optimization is left for later).

## Acceptance

**1. Template detection + rendering unit tests all green** (internal/kernel/chat/, 6 tests): ChatML/LLaMA3/Unknown detection, default-system extraction, byte-for-byte single-/multi-turn rendering, fallback.

**2. Real-model multi-turn memory verification** (greedy, qwen2.5-0.5b):

```
you: The secret word is banana.
AI: The secret word is "banana".
you: What is the secret word?
AI: The secret word is "banana".     ← remembers the previous turn!✅
```

**3. Coherent Chinese small talk**:

```
你: 你好，介绍一下你自己
AI: 你好！我是Qwen，一个由阿里云开发的超大规模语言模型。我叫通义千问……
```

## Pitfalls Hit Along the Way (Hidden Knowledge)

1. **In Go regex, `\n` is backslash-plus-n as two characters**: in the template string, `<|im_start|>system\n` is a literal `\` and `n`, **not** a newline character. In regex you must write `\\n` (escaped, to match the literal two characters); test constants should use backtick strings rather than double-quoted strings.
2. **The default-system extraction trap**: in the template, the default system prompt is a whole single-quoted string of "system marker + text + end marker" (e.g. `'<|im_start|>system\nYou are...<|im_end|>\n'`). When stripping markers, you must also strip the literal `\n` that follows the markers, otherwise rendering produces one extra newline.
3. **Special tokens must be encoded as a whole**: the rendered prompt contains `<|im_start|>`; if tokenization splits it apart, the model can never learn it correctly. M2's encodeWithSpecial already matches special tokens as whole segments — this is precisely the precondition for it to work.

## Glossary

| Term | 中文 | Explanation |
|---|---|---|
| Chat Template | 对话模板 | The rules that turn a message list into the model's input string |
| ChatML | 对话标记语言 | The chat format of the form `<|im_start|>role\ncontent<|im_end|>` (used by qwen) |
| system/user/assistant | 角色 | The three message types: persona / user / AI reply |
| Special token | 特殊标记 | A control symbol that occupies a single id in the vocabulary (e.g. `<|im_start|>`) |
| Default system | 默认系统提示 | The persona statement the template auto-inserts when the first turn has no system message |
| add_generation_prompt | 生成提示 | Appending `<|im_start|>assistant\n` at the end of the messages to get the model to reply |
| Jinja | 模板引擎 | The full template language shipped with the model (we only do keyword detection, not parsing) |
