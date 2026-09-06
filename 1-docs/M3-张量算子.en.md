# M3: Tensor Operators (Dequantization / MatMul / Normalization / RoPE)

> Status: ✅ Done (Q8_0 dequantization numerically correct, single-operator unit tests all green)
> Code: `internal/kernel/tensor/` + `internal/kernel/model/`
> Depends on: M1's gguf; M4 will assemble forward inference from these operators

## One-line essence

Model files store **quantized bytes**, which cannot be computed on directly. All of M3's work = three things:

1. **Dequantization**: restore quantized byte blocks to float32 per the format (the correctness lifeline)
2. **Matrix multiplication**: activations × weights = one linear transform per layer (the backbone of attention/FFN)
3. **Activation / normalization**: RMSNorm, SiLU, Softmax, RoPE (so the model can be "nonlinear" and "position-aware")

## What quantization is (plain language)

Storing weights as float32 costs 4 bytes per number, which wastes too much memory. Quantization = compress the **value range** of 32 numbers into 1 scale + 32 low-precision integers; at use time `value = integer × scale` reconstructs an (approximate) value.

Key point: **32 elements share one scale**, so reconstruction must read aligned "block" by "block" — if the byte offset is off by one, the entire layer's weights are garbage.

## Quantization layouts needed by this model (ggml format spec)

| Type | Block size | Layout (byte order) | Dequantization formula |
|---|---|---|---|
| Q8_0 | 32 | `[d:2B fp16][qs:32B int8]` | x = q × d |
| Q5_0 | 32 | `[d:2B][qh:4B][qs:16B]` | x = (q±16bit) × d |
| Q6_K | 256 | `[ql:128B][qh:64B][scales:16B][d:2B]` | x = q(6bit) × d × scale |
| Q4_K | 256 | `[d:2B][dmin:2B][scales:12B][qs:128B]` | x = q×d×sc − min×dmin |

`quant.go` implements this scheme bit by bit. The K-series types Q6_K/Q4_K have cascaded scales (sub-blocks inside the 64/256 blocks have their own independent scales), which is where mistakes are most likely.

## Core operators

### MatMulTransB (transposed matrix multiplication)

Weights are stored row-major as $[K, N]$ (ne0=K is contiguous), so the multiplication is $C = A \times B^T$:

$$C_{ij} = \sum_{k=0}^{K-1} A_{ik} \cdot B_{jk}$$

When B is quantized, dequantize row by row: $B_j = \text{DequantRow}(j)$, avoiding expanding the whole weight matrix at once.

### RMSNorm (root mean square normalization)

Normalize each row $x$ (length $d$), then multiply by a learnable weight $w$:

$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \varepsilon}$$

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot w$$

where $\varepsilon = 10^{-6}$ (qwen2 default), to prevent division by zero. $\odot$ denotes element-wise multiplication.

### SiLU (Swish activation)

$$\text{SiLU}(x) = \frac{x}{1 + e^{-x}} = x \cdot \sigma(x)$$

where $\sigma(x)$ is the sigmoid function. SiLU is the key nonlinearity of the FFN.

### SoftMax (normalized exponential)

Turn a vector $z$ (length $n$) into probabilities; first subtract the max to prevent overflow:

$$\text{SoftMax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^{n} e^{z_j - \max(z)}}$$

The output satisfies $\sum_i p_i = 1$, with each $p_i \in (0, 1)$.

### RoPE (rotary position embedding)

Rotate q/k vectors by the absolute position $\theta$. For each pair of adjacent elements $(q_{2i}, q_{2i+1})$:

$$\begin{pmatrix} q_{2i}' \\ q_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

where the frequency $\theta_i = \text{pos} / \text{base}^{2i/d}$, base $= 10^6$ (qwen2.5 default), and $d$ is the headDim.

qwen2 uses **NEOX pairing**: the first $d/2$ dimensions are paired and rotated with the last $d/2$ dimensions (not adjacent pairs).

## The model package

Maps GGUF's 290 tensors onto `LLaMAModel` by name:
- Hyperparameters: number of layers/dimensions/heads/context length read from KV
- Per-layer 7 weights (Q/K/V/O + FFN gate/up/down) + 3 optional biases
- Tensor bytes are zero-copy references into file memory (no dequantization yet; dequantize row by row when actually computing)

## M3 acceptance

**1. Q8_0 dequantization is numerically correct** (three sampled rows of the same model's token_embd):

```
row 0       -0.01027679 0.04078603 ...   ✓ values sane
row 100     -0.01253176 0.03606701 ...   ✓
row 150000  -0.001326799 -0.002918959... ✓
```

**2. Unit tests**: all 5 green — F16 conversion (8 groups), hand-computed Q8_0 block, hand-computed MatMul, RMSNorm, Softmax.

**3. fp16 conversion** verified: 0x3C00→1.0, 0x399A→0.7, subnormal→6e-8, all correct.

## Pitfalls hit (hidden knowledge)

1. **GGUF tensor offsets are relative**: the tensor Offset is relative to the "start of the data region", not to the file header! When reading you must add `DataStart` (the 32-aligned position after the tensor table is fully read). The first time this was forgotten, header bytes were read directly as weights and all values were garbage (sum = 150 million, NaNs appeared).
2. **Quantization type numbering**: M1 already hit this (Q4_0=2, not 8); if M3 re-copied tensor.DType by hand it would hit it again — reuse the numbering directly from gguf.
3. **Q6_K's 4 sub-blocks of 64 elements indexing**: the `is = l/16` family of subscripts is easy to get wrong; you must check line by line against the ggml dequantization formula.

## Glossary

| Term | 中文 | Explanation |
|---|---|---|
| Quantization | 量化 | Store weights approximately as low-precision integers plus a scale to save memory |
| Dequant | 反量化 | Reconstruction: int × scale = approximate float |
| Block | 块 | The smallest unit of quantized storage (a group of numbers sharing one scale) |
| Scale | 缩放系数 | The reconstruction factor of a block |
| RMSNorm | 均方根归一化 | Divide by the row's RMS to stabilize values |
| SiLU/Swish | 门控激活 | x·σ(x), the key nonlinearity of the FFN |
| Softmax | 归一化指数 | Turn scores into probabilities summing to 1 |
| RoPE | 旋转位置编码 | Rotate q/k vectors by absolute position to inject positional information |
| NEOX | 一种配对方式 | qwen2/GPT-NeoX's RoPE pairing (first/last half), not adjacent pairs |
| MatMulTransB | 转置矩阵乘 | The efficient matrix-multiplication form when weights are row-major [K,N] |
