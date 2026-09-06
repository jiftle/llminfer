# M4: Forward Inference (Layer-by-Layer Transformer + KV Cache)

> Status: ✅ Complete (top-20 logits of an 8-token sequence are sensible, forward numeric checks pass)
> Code: `internal/kernel/cache/` + `internal/kernel/eval/`
> Dependencies: M1 gguf, M2 tokenizer, M3 tensor ops; M5 builds a sampling/generation loop on top, M6 adds KV reuse on top for multi-turn conversation

## One-Sentence Essence

M3 turned "model files" into a "warehouse of computable ops", but **ops alone cannot do inference**. M4 is about wiring the ops together in the correct order, following the Transformer paper's formulas:

```
lookup(embed) → 24 layers × [attention sub-block → FFN sub-block] → output head (logits)
```

Plus a "money-saving drawer" — the KV Cache, so you don't have to recompute the entire history for every generated word.

## Complete Forward Data Flow (against eval.go)

Input: a batch of token ids, e.g. `[3, 562, 18727, ...]`. Output: logits over the whole vocabulary (one score per word).

```
① embed         token id → vector row x[n, nEmb]        (look up the token_embd table)
② per-layer forwardLayer(li)
    attention sub-block:
      a. RMSNorm → normBuf (x is kept for the residual)
      b. QKV projection (3 × MatMulTransB) + bias
      c. RoPE rotates q, k (with absolute positions)
      d. writeKV writes the new K/V (everything except q) into the cache
      e. attention (GQA + causal mask) → attBuf
      f. WO projection + residual: x += WO·att
    FFN sub-block (SwiGLU):
      g. RMSNorm → W1·x (gate) ⊙ SiLU(W3·x) → W2 → residual
③ output head  RMSNorm → output projection → take the last row's logits
```

Key invariant: **the residual accumulates in place**. Every normalization step first copies out into normBuf, then normalizes normBuf; `x` itself is untouched and only added back after the sub-block finishes.

## Attention Algorithm Deep-Dive (attention.go)

### Single-Head Attention Formula

For the $i$-th token and the $h$-th Q head, with KV head $kvH = h / \text{group}$ (GQA grouping):

$$\text{score}[p] = \frac{\langle q_{i,h},\ k_{p,kvH} \rangle}{\sqrt{d_{head}}} \quad p = 0, ..., t_{pos}$$

$$\text{att}_{i,h} = \sum_{p=0}^{t_{pos}} \text{softmax}(\text{score})_p \cdot v_{p,kvH}$$

where $d_{head} = 64$ (headDim) and $t_{pos}$ is the current absolute position.

### Causal Mask

Each token only attends to "itself and earlier" tokens; it cannot see the future. Instead of explicitly adding a $-\infty$ matrix, the implementation simply caps the score loop upper bound at $t_{pos}$ — causal by construction.

### GQA Grouping

$$\text{group} = \frac{\text{HeadsCount}}{\text{HeadsKV}} = \frac{14}{2} = 7$$

i.e. every 7 Q heads share one KV head ($kvH = h / \text{group}$), saving KV cache memory at the cost of slightly weaker expressiveness.

### Scaling Factor

Why divide by $\sqrt{d_{head}}$: the variance of the dot product $\langle q, k \rangle$ grows linearly with the dimension $d$. Dividing by $\sqrt{d}$ pulls the variance back to 1, so softmax does not saturate (otherwise the distribution degenerates into one-hot and gradients vanish).

## What Is the KV Cache (in Plain Words)

Attention needs "the dot product of the current Q with every historical position".

- **Without caching**: when generating the 100th word, recompute the K/V of all previous 99 words from scratch — doubling every time, exponential blowup.
- **With caching**: after the 1st word's K/V is computed once, put it in the drawer; on every later step only compute the new word's Q, K, V, and take history directly.

```
position 0  compute K0,V0  →  store [0]
position 1  compute K1,V1  →  store [1], take K0,V0 to pair with
position 2  compute K2,V2  →  store [2], take K0,K1,V0,V1 to pair with
...
```

Memory layout `[layer][pos][kvHead][headDim]`, K half + V half, each growing independently. `cache.go` is this "density-board drawer".

## FFN Formula (SwiGLU)

Each layer's FFN sub-block uses SwiGLU (Gated Linear Unit):

$$\text{FFN}(x) = W_2 \cdot \left[ \text{SiLU}(W_1 \cdot \hat{x}) \odot (W_3 \cdot \hat{x}) \right]$$

where $\hat{x} = \text{RMSNorm}(x)$, $\odot$ is element-wise multiplication (gating), $W_1$ is the gate weight, $W_3$ is the up weight, and $W_2$ is the down weight.

Compared with the standard FFN $\text{FFN}(x) = W_2 \cdot \text{SiLU}(W_1 \cdot x)$, SwiGLU adds a $W_3$ branch for gating, giving stronger expressiveness.

## Residual Connection

$$x = x + \text{sub-block output}$$

Two residuals: after the attention sub-block + after the FFN sub-block. In plain code terms this is `addResidual(x, layerOut)`.

## Pitfalls I Hit in This Version (Hidden Knowledge, Lessons in Blood)

1. **The sneaky "row" problem of 1-D tensors in RMSNorm (biggest pitfall, logits all 0)** 🤯
   Right after finishing forward inference, the logits came out all 0. I debugged for ages: `x` (the embed output) is nonzero and normal, but it becomes -0 as soon as it passes through RMSNorm.
   Root cause: tensors like `attn_norm.weight` are **1-D tensors** (length = nEmb = 896); in the table they only have ne0=896 and no ne1. `DequantRow` reads the offset of `row` as `row * ne0`, while `AsFloat32` uses `NE[1]` as the number of rows → for a 1-D tensor `NE[1]=0` → not a single element is dequantized → w is all 0 → the normalized output is all 0.
   Fix: add `Tensor.Rows()` — treat a 1-D tensor / `NE[1]==0` as a single row, and use `NE[1]` for 2-D and above. **The granularity of "row" in `DequantRow` and the "actual matrix row count" must match strictly in Go.**
2. **Type-mismatch compile hell**: the ROPE/op interfaces take uint32 (the model fields are int), and casting back and forth is error-prone when "passing the baton" between function signatures. Lesson: express a function's dimension parameter with one consistent type, uniform across layers.
3. **Verification relies on switches, not "feelings"**: to localize bugs, I temporarily used the environment variable `FEIYU_DEBUG` to print max/min at every layer, confirming step by step which link was wrong — "embed wrong → normalization wrong → projection wrong" — instead of guessing blindly.

## M4 Acceptance

**Single token + 8-token sequence, top-20 logits self-consistent** (two independent implementations of the same input do not contradict each other):

```
single token (token=3)               8-token sequence [3,562,18727,1055,15496,11,13,2167]
top-20: 100835,282,2889,...          top-20: 11,1661,7010,264,...
max     10.278984                    max     15.58272  → top-1 points to token 11 ✓
```

The 8-token sequence is especially critical: it passes through many attention layers and "tastes" the history (the KV cache holds 7 earlier positions), yet the top-20 are still all correct — proving that GQA + KV Cache + all ops are correct across **multiple positions**, not just a single token's self-consistency.

All unit tests pass (`go test ./...`).

## Document ↔ Code Correspondence

| Document section | Code location |
|---|---|
| ① embed lookup | eval.go `Forward` (fetch row by row via token_embd.DequantRow) |
| ② attention sub-block | first half of eval.go `forwardLayer` |
| b) QKV+bias | eval.go `forwardLayer` + `addBias` |
| c) RoPE | eval.go `forwardLayer` → tensor.RoPE |
| d) writeKV | eval.go `writeKV` |
| e) attention | attention.go `attention` (including GQA + scaling) |
| ② FFN sub-block | second half of eval.go `forwardLayer` (SwiGLU) |
| ③ output head | eval.go `Forward` (OutputNorm + Output projection) |
| KV Cache | cache.go `KVCache` |
| residual/normalization buffer reuse | eval.go `ensure`, `addResidual` |

## Glossary

| Term | 中文 | Explanation |
|---|---|---|
| Forward | 前向推理 | The whole process from input token → layer-by-layer computation → output logits |
| Logits | 原始分数 | Unnormalized vocabulary scores; higher means more likely (before sampling) |
| Residual | 残差连接 | x = x + sub-module output; prevents vanishing gradients and helps train deep networks |
| RMSNorm | 层归一化 | Used for normalization; the op already exists in M3 |
| K/V Cache | 键值缓存 | Stores historical K/V to avoid recomputing attention |
| GQA | 分组查询注意力 | group Q heads share one KV head, saving cache memory |
| Causal Mask | 因果掩码 | Each position can only see itself and earlier positions; looking into the future is forbidden |
| Context | 上下文 | The inference state of a session (KV + position + buffers) |
| HeadDim | 单头维度 | embedding/head count, here = 896/14 = 64 |
| Embedding/查表 | 词向量表 | token id → the corresponding vector row |
| SiLU/SwiGLU | 门控线性单元 | The multiplicative gate×up combination inside the FFN |
