# M2: tiktoken BPE Tokenization

> Status: ✅ Done (three-sentence encode acceptance passed)
> Code: `internal/kernel/tokenizer/`
> Depends on: M1's gguf parsing (reads tokenizer-related KV)

## The Essence in One Sentence

Making the model "understand" text = splitting any UTF-8 sentence into "the subword chunks it saw during training". qwen2 uses **Byte-level BPE**, in three core steps:

```
text → UTF-8 bytes → GPT-2 byte encoding (byte → Unicode) → pretokenize → longest-prefix chunking → repeatedly merge by merge rules (merges) → token id sequence
```

## Why qwen2 Is Not "One Id Per Character"

Chinese looks like block characters, but internally the tokenizer works entirely on bytes. `"例子：你好"` gets gradually merged by the most frequent byte combinations in the `merges` table into chunks that may span multiple characters. **What the vocab stores are "merged chunks", not a dictionary.** All our merge criteria come from GGUF's `tokenizer.ggml.merges` (151,387 entries). The final output of tokenization is `ids []uint32`.

## Three Key Data Sources (all read from M1's KV)

| KV key | Content | Purpose |
|---|---|---|
| `tokenizer.ggml.tokens` | 151,936 token strings | Build tokenID reverse mapping + longest-prefix table |
| `tokenizer.ggml.merges` | 151,387 `"a b"` merge rules | BPE merge priority (lower rank = higher priority) |
| `tokenizer.ggml.token_type` | type of each token | Pick out special tokens for whole-token matching (e.g. `<\|im_start\|>`) |

## Byte Encoding Table (byte_encode.go)

BPE merging operates on "strings"; but arbitrary bytes (e.g. 0x00, 0xFF) don't fit well as characters. GPT-2's approach: build a **256-entry mapping table**—
- Printable ASCII / Latin characters are kept as-is (byte value = Unicode code point)
- The remaining awkward bytes (control chars, 0x7F, 0xA0…) are mapped into the empty region above U+0100

This way any byte sequence becomes a valid string, and `BytesToUnicode`/`UnicodeToBytes` are inverse operations of each other, **lossless**.

## Main Tokenization Flow (tokenizer.go)

```
Encode(text)
 ├─ has special tokens → encodeWithSpecial (match special tokens as whole blocks, rest goes to BPE)
 ├─ has merges → encodeBPE
 │    each pretokenized segment → BytesToUnicode → initialTokens (longest prefix) → bpeMerge (iterative merging)
 └─ no merges → encodeNaive (longest prefix, no merging)

Decode(ids) = string of each token → UnicodeToBytes → join back to []byte → string
```

### bpeMerge Is the Core
Repeatedly scan the current token-block sequence, find the adjacent pair with the **highest priority** (smallest rank value) in `rankMap` and merge it, until no mergeable pair remains. Since the initial tokens are already chunked into whole blocks by "longest prefix", the number of merges is usually small.

## M2 Acceptance: Representative Sentences Encode Correctly

For three representative inputs, the encode results are as follows (values are the token id sequences of that sentence):

| Input | ids |
|---|---|
| `你好` | `[108386]` |
| `你好，介绍一下你自己` | `[108386 3837 109432 107828]` |
| `[INST]你是谁[/INST]` | `[58 64462 60 105043 100165 24157 64462 60]` |

Special tokens like `[INST]` are matched as a whole into one id (not split apart), which confirms token_type recognition is correct.

## Pitfalls Encountered (Hidden Knowledge)

1. **String length is uint64, not u32**: The GGUF v3 string-length field is 8 bytes; the initial Python probe read it as 4 bytes and desynced. In Go, reading `merges`/`tokens` relies entirely on this for alignment—one wrong byte and everything is garbage.
2. **Pretokenization is mandatory**: If you don't first split by character type (letters/digits/spaces/punctuation) and instead run merges directly on the whole sentence, cross-type merges turn `你好123abc` into weird chunks, and the token ids diverge from the reference implementation.
3. **The hasMerges branch**: Some models (e.g. older SentencePiece-based LLaMA) have no merges, and tokenization degrades to longest-prefix matching; qwen2 has merges, so it runs the full BPE.

## Glossary

| Term | 中文 | Explanation |
|---|---|---|
| BPE | 字节对编码 | A tokenization algorithm that repeatedly merges the most frequent adjacent subwords |
| Token | 词片 | The smallest unit the model actually "sees"; one id corresponds to one string chunk |
| Vocab | 词表 | The id → string mapping, the model's "dictionary" |
| merges | 合并规则 | The "A B" → AB rules and their priority, deciding how to group words |
| Byte-Level BPE | 字节级 BPE | Converts to bytes first, then merges, so no characters are lost in any language |
| 特殊 token | 特殊标记 | Control markers like `<\|im_start\|>`, matched as a whole and never split during encoding |
| pretokenize | 预分词 | A coarse split by character type before BPE, shrinking the merge range and avoiding cross-type merges |
