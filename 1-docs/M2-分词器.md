# M2：tiktoken BPE 分词

> 状态：✅ 完成（三句话 encode 验收通过）
> 代码：`internal/kernel/tokenizer/`
> 依赖：M1 的 gguf 解析（读 tokenizer 相关 KV）

## 一句话本质

让模型"看懂"文本 = 把任意 UTF-8 句子切成"它在训练时见过的子词块"。qwen2 用的是 **字节级 BPE（Byte-level BPE）**，核心三步：

```
文本 → UTF-8 字节 → GPT-2 字节编码（字节→Unicode） → 预分词 → 最长前缀切块 → 按合并规则（merges）不断合并 → token id 序列
```

## 为什么 qwen2 不是"一个字一个 id"

中文看起来是方块字，但 tokenizer 内部全是字节。`"例子：你好"` 会被 `merges` 表里最常出现的字节组合逐步合并成可能跨越多个字的块。**词表里存的是"合并后的块"，不是词典。** 我们的合并依据全部来自 GGUF 的 `tokenizer.ggml.merges`（151387 条）。分词的最终产物是 `ids []uint32`。

## 三个关键数据源（都从 M1 的 KV 里读）

| KV key | 内容 | 用途 |
|---|---|---|
| `tokenizer.ggml.tokens` | 151936 个 token 字符串 | 建立 tokenID 反向映射 + 最长前缀表 |
| `tokenizer.ggml.merges` | 151387 条 `"a b"` 合并规则 | BPE 合并优先级（rank 越小越优先） |
| `tokenizer.ggml.token_type` | 每 token 的类型 | 挑出特殊 token 整体匹配（如 `<\|im_start\|>`） |

## 字节编码表（byte_encode.go）

BPE 合并作用的是"字符串"；但任意字节（如 0x00、0xFF）不适合直接当字符。GPT-2 的做法：建一张 **256 项映射表**——
- 可打印 ASCII / 拉丁字符原样保留（字节值=Unicode 码点）
- 其余别扭字节（控制符、0x7F、0xA0…）映射到 Unicode U+0100 以上空白区

这样任意字节序列都变成合法字符串，`BytesToUnicode`/`UnicodeToBytes` 互为逆运算，**无损**。

## 分词主流程（tokenizer.go）

```
Encode(text)
 ├─ 有特殊 token → encodeWithSpecial（整块匹配特殊 token，其余给 BPE）
 ├─ 有 merges → encodeBPE
 │    每段预分词 → BytesToUnicode → initialTokens(最长前缀) → bpeMerge(循环合并)
 └─ 无 merges → encodeNaive（最长前缀，不合并）

Decode(ids) = 每个 token 的字符串 → UnicodeToBytes → 拼回 []byte → string
```

### bpeMerge 是核心
反复遍历当前 token 块序列，找 `rankMap` 中**优先级最高**（rank 值最小）的相邻对合并，直到没有可合并对。因为初始 tokens 已被"最长前缀"切成整块，一般合并次数不多。

## M2 验收：代表性句子 encode 正确

对三句有代表性的输入，encode 结果如下（值即该句的 token id 序列）：

| 输入 | ids |
|---|---|
| `你好` | `[108386]` |
| `你好，介绍一下你自己` | `[108386 3837 109432 107828]` |
| `[INST]你是谁[/INST]` | `[58 64462 60 105043 100165 24157 64462 60]` |

`[INST]` 这类特殊 token 被整体匹配成一个 id（没被拆散），说明 token_type 识别正确。

## 踩过的坑（隐形知识）

1. **字符串长度是 uint64 不是 u32**：GGUF v3 字符串长度字段是 8 字节，初版 Python 探针当 4 字节读错位。Go 里读 `merges`/`tokens` 全靠它对齐，错一个字节全盘乱。
2. **必须做预分词**：如果不先按 字符类型（字母/数字/空格/标点）切一刀，直接对整句跑合并，跨类合并会把 `你好123abc` 变成奇怪块，token id 和参考实现不一致。
3. **hasMerges 分支**：有些模型（如纯 SentencePiece 的 LLaMA 老版）没有 merges，分词退化为最长前缀匹配；qwen2 有 merges，走完整 BPE。

## 术语表

| 术语 | 中文 | 一句话解释 |
|---|---|---|
| BPE | 字节对编码 | 从最常出现的相邻子词开始反复合并的分词算法 |
| Token | 词片 | 模型实际"看到"的最小单位，一个 id 对应一个字符串块 |
| Vocab | 词表 | id → 字符串的映射，模型的"字典" |
| merges | 合并规则 | "A B" → AB 的规则及优先级，决定怎么组词 |
| Byte-Level BPE | 字节级 BPE | 先转成字节再合并，任何语言都不丢字符 |
| 特殊 token | 特殊标记 | `<\|im_start\|>` 等控制标记，编码时整体匹配不可拆 |
| pretokenize | 预分词 | BPE 前按字符类型粗切，缩小合并区间、避免跨类合并 |