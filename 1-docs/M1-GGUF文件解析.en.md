# M1: Parsing GGUF Files

> Status: ✅ Done (independently verifiable)
> Code: `internal/kernel/gguf/`
> Milestone docs: this file is the minimal milestone record; the full plain-language version lives in the learn-llm notes (to be added).

## The Essence in One Sentence

GGUF is a **self-describing binary container**. The file header tells you "how many metadata entries and how many tensors follow", and then you read, in fixed order, `metadata region → tensor info table → raw weight data region`, laid out back to back in three segments.

Once you can read it, you've turned 397MB (`qcwen2.5-0.5b.gguf`) into in-memory tensor descriptors you can actually use.

## GGUF v3 File Layout

```
┌─ File header ─────────────┐
│ magic   "GGUF" 4 bytes      │
│ version  uint32             │
│ tensor_count  uint64        │  ← number of weight tensors (qwen2.5-0.5b = 290)
│ kv_count  uint64            │  ← number of metadata entries
├─ Metadata region (kv_count entries) ─┘
│ key: uint64 length + bytes   │
│ type: uint32 (value type)    │
│ value: depends on type      │  ← string/number/array/bool
├─ Tensor info table (tensor_count entries)
│ name: string               │  ← e.g. "model.layers.0.self_attn.q_proj.weight"
│ n_dims: uint32             │
│ dims: n_dims uint64s       │
│ type: uint32 (GGML type)   │  ← F32/F16/Q8_0/Q5_0/Q6_K...
│ offset: uint64             │  ← byte offset of the weight data in the file
└─ Weight data region        ┘
     offset .. end of file = raw bytes of each tensor (laid out per its type)
```

Two key points (tacit knowledge):

**Byte alignment formula**: the data region must start on a 32-byte boundary (SIMD-friendly):

$$\text{DataStart} = \text{align32}(\text{position right after the tensor info table})$$

where $\text{align32}(x) = (x + 31) \ \& \ \sim 31$ (round up to a multiple of 32).

1. **Everything is little-endian**, and string lengths are uint64 (v3; v1/v2 use u32 — this is the easiest trap to fall into).
2. **The weight data region does not directly follow the info table**; instead each tensor carries its own `offset` — because tensor data varies in size, loaders often mmap zero-copy references rather than moving everything into memory.

## How to Compute Tensor Byte Size (Core Algorithm)

Non-quantized (F32/F16): `bytes = element count × bytes per element`

Quantized types (Q8_0/Q5_0/Q6_K...): you **cannot** use "element count × bytes per element", because quantization is stored **in blocks** — 32 or 256 elements share a single scale.

```
blockCount = ne[0] / blockElems   // the first dimension must be a whole multiple of blockElems
rowBytes   = blockCount × blockBytes
totalBytes = rowBytes × all remaining dimensions
```

So `types.go` maintains a `blockTable`; look it up before computing the footprint:

| Type | Elements per block | Bytes per block | Description |
|---|---|---|---|
| F32 | 1 | 4 | float32 |
| F16 | 1 | 2 | float16 |
| Q8_0 | 32 | 2+32 | 1×f16 scale + 32 int8s |
| Q5_0 | 32 | 2+4+16 | scale + qh(4) + qs(16) |
| Q6_K | 256 | 2+16+192+16 | scale/ql/qh/scales |
| Q4_K | 256 | 2+12+96+16 | d + dmin + qs + scales |

## Traps Hit (Real Experiences From This Milestone)

- **GGMLType numbering copied wrong**: at first I marked Q4_0 as 8 from memory, and running against the real model crashed at tensor[3] with "type 6 not supported". Lesson: **type numbers must follow the ggml format spec** (check the ggml_type enum table), never rely on memory.
- **Parsing byte alignment**: the first probe read `tensor_count`/`kv_count` as 4-byte `I`; in GGUF they are both 8-byte `Q`, which shifted everything that followed out of place.

## M1 Acceptance Record

```
go run . run models/qwen2.5-0.5b.gguf "你好"

== Model Info ==
Architecture: qwen2   Type: model(Instruct)    Name: Qwen2.5 0.5B Instruct
Version: GGUF v3   Tensors: 290   File 397807936 bytes
  token_embd.weight  Q8_0  ne=[896 151936] bytes=144643072
  blk.0.attn_norm.weight  F32   ne=[896] bytes=3584
  blk.0.ffn_down.weight  Q6_K  ne=[4864 896] bytes=3847424
  blk.0.ffn_gate.weight  Q5_0  ne=[896 4864] bytes=2996224
  ...
```

✅ All keywords correct: **qwen2 architecture, Qwen2.5 0.5B Instruct, 290 tensors, mixed quantization (Q8_0/Q6_K/Q5_0/F32)**.

Also noticed along the way: `ne=[896 151936]` is **row-major** (ne0=896 is the contiguous dimension). Remember this direction flip when loading weights into MatMul later — a hidden trap for M3, noted down in advance.

## Glossary

| Term | 中文 | Explanation |
|---|---|---|
| GGUF | 通用格式 | The model file format of the ggml ecosystem; a self-describing binary container |
| KV | 键值元数据 | Key-value table of model info (architecture/layer count/vocab size, etc.) |
| GGMLType | 张量类型 | Weight storage type: quantized vs. non-quantized |
| 量化 | 压缩 | Storing weights in low precision (e.g. int8/4bit) to save memory at the cost of precision |
| 块(block) | 量化块 | The atomic unit of quantized storage/access; e.g. Q8_0 shares one scale per 32 elements |
| scale | 缩放系数 | The dequantization factor stored in a quantized block |
| ne | 各维长度 | Per-dimension lengths; ne[0] is the most contiguous (adjacent in memory) dimension |
| offset | 偏移 | Byte position of a tensor's weight data within the file |
