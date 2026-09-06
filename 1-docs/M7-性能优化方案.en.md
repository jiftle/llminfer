# M7: Performance Optimization Plan

> Status: 🚧 In progress (M7.2 fused dot product ✅, M7.3 multithreading ✅, M7.4 buffer reuse ✅, M7.5 KV prefix reuse ✅)
> Code: internal/kernel/tensor/ + eval/ (scope of changes in this milestone)
> Reference: goLLM's ops_quant_dot.go / ops.go / dot_quant_avx2_amd64.s

## The Essence in One Sentence

Running a 0.5B model on the CPU in the 1.5G era, the bottleneck is not "slow dequantization" but the **two memory round-trips plus redundant floating-point multiplies caused by keeping dequantization separate from the dot product**. Optimization = fuse dequantization into the dot product, multiply each block by scale only once, then squeeze bandwidth with multithreading and SIMD.

## Why Optimize Now

The current state (M1-M6) is the "naive version": `MatMulTransB` calls `DequantRow` per **output column** to dequantize the full row into `rowBuf`, then reads it back for a scalar dot product. The problems:

1. B's entire row is dequantized once per column; N columns = N dequantizations (redundant work)
2. `rowBuf` is freshly allocated on every make
3. Scale is multiplied per element (scale is a block-level common factor; pure waste)
4. Purely scalar, no SIMD, no multithreading

## Bottleneck Estimate (0.5B decode, ~2.3 GFLOPs per token)

```
24 layers × (QKV 3×896² + WO 896² + FFN 2×896×4864 + down 4864×896)
```

Breakdown: MatMul takes >95% of the time. Attention/RNorm/RoPE are all small contributors.

## goLLM's Approach: Fused Dequant-Dot Product (Core Insight)

goLLM does **not** pre-dequantize to F32. Instead, it accumulates the dot product directly over quantized integers and multiplies by scale only once per block:

$$\\text{dot}(a_j, \\text{col}_j(B)) = \\underbrace{\\left(\\sum_{k} a[k]\\cdot q[k]\\right)}_{\\text{accumulate over quantized integers}} \\times d$$

Compared with the current state (dequantizing the whole row to a buffer first, then reading it back), this removes the intermediate `rowBuf` round-trip and cuts the number of scale multiplications from O(K) to O(number of blocks). Numerically, integer multiply-accumulate followed by a single multiply by d has a relative error < 1e-4 versus multiplying each element by d, which is acceptable.

It also stacks three more layers on top:

| Layer | Content | Key Code |
|---|---|---|
| Fused dot product | One dedicated dot loop per quant type | `dotQ8_0`/`dotQ5_0`/`dotQ6_K`/`dotQ4_K` |
| fp16 LUT | Scale uses a 65536-entry LUT, O(1) conversion | `fp16ToF32LUT` (256KB) |
| Multithreading | Shards by output column j, worker-private acc/rowBuf | `matmulQuantTransB` |
| SIMD (optional) | Per-block AVX2 assembly dot product | `dot_quant_avx2_amd64.s` |

## Trade-off Analysis: Why "Pre-dequantize to F32" Was Dropped

| Approach | Pros | Cons |
|---|---|---|
| Pre-dequantize to F32 (initial idea) | Minimal MatMul, removes dequant from the hot path | Doubles memory (398MB→800MB), slow startup; a full-F32 dot product wastes the bandwidth saved by quantization |
| **goLLM fused dot product (adopted)** | Memory unchanged, saves scale multiplies, bandwidth-friendly; SIMD version doubles it again | Needs a separate dot loop per quant type (4 sets) |

Conclusion: adopt the goLLM route. Keep the quantized weights as-is and fuse dequantization + dot product into the hot path.

## Implementation Roadmap (ordered by benefit/cost)

| Phase | Content | goLLM Alignment | Expected Gain |
|---|---|---|---|
| **M7.1 Baseline** | `make bench` records prefill/decode tokens/s | — | Comparison baseline ✅ captured |
| **M7.2 Fused dot product (pure Go)** | Write fused loops for the 4 quant types + fp16 LUT | fallback path in `ops_quant_dot.go` | ✅ decode 1.2→1.9 tokens/s (+58%) |
| **M7.3 Multithreaded column split** | Shard by output column j, worker-private acc/rowBuf, `-threads` config (default = proportional to cores 2/3, 12 cores→8) | `matmulQuantTransB` | ✅ decode 1.9→6.6, prefill 2.7→12.8 tokens/s |
| **M7.4 Buffer reuse** | Pool eval's x/acc/rowBuf, drop per-batch make | sync.Pool + Context reuse | ✅ Eliminated high-frequency allocations, throughput unchanged |
| **M7.5 KV prefix reuse** | Skip lcp common prefixes (reuse multi-turn history, forward only the increment) | `ForwardWithCache` lcp | ✅ Multi-turn reuse 60%→82% |
| **M7.6 AVX2 assembly (optional)** | Port the 4 quantized dot products + F32 MatMul to assembly | `dot_quant_avx2_amd64.s` | Another 2-4x |

## M7.1 Baseline Measurements (2026-09-06, this machine's CPU, single-threaded)

```
model load: 279 ms
prefill: 129 tokens in 47712.9 ms  → 2.7 tokens/s
decode:  64 tokens in 53554.3 ms  → 1.2 tokens/s
```

- decode is only **1.2 tokens/s**, slower than estimated—this is exactly the real cost of "dequantize per column + scalar dot product" under pure scalar execution.
- Used as the baseline for comparison at every later phase: M7.2's fused dot product should directly cross 2-3x, and M7.3's multithreading multiplies by the number of cores again.

## M7.2 Measurements (Fused Dequant-Dot Product, Pure Go)

```
prefill: 129 tokens in 48481.5 ms  → 2.7 tokens/s   (on par with baseline ✓)
decode:  64 tokens in 34272.6 ms  → 1.9 tokens/s    (baseline 1.2 → +58% ✓)
```

**Updated 3 things:**

- Added `ops_quant_dot.go`: fused dot products for Q8_0/Q5_0/Q6_K/Q4_K (multiply by scale only once per block) + sub-block LUTs for Q6_K/Q4_K. Unit tests compare against the reference implementation (dequant + scalar) with relative error < 1e-3.
- **Fixed a latent bug along the way**: the block sizes for Q4_K/Q6_K in gguf's `blockTable` were wrong (Q4_K recorded as 126 but actually 144, Q6_K recorded as 226 but actually 210), making the `Data` slice too short and quantized weights read out of bounds—the fused dot product exposed it all at once.
- **Two-path trade-off** (`matmulQuantTransB`): M=1 uses the fused dot product (fast decode); M>1 uses DequantRow + reuse (prefill avoids re-reading quantized weights). Otherwise prefill would drop from 2.7 to 1.5.

**Gotcha: loop order decides prefill speed**—outer loop over i rows, inner loop over j columns, so rows stay resident in cache.

## M7.3 Measurements (Multithreaded Column Split, 8/12 cores)

```
threads: 8 (12 cores on this machine)
prefill: 129 tokens in 10065.5 ms  → 12.8 tokens/s   (M7.2's 2.7 → +374% ✓)
decode:  64 tokens in 9738.5 ms    → 6.6 tokens/s    (M7.2's 1.9 → +247% ✓)
```

**Implementation highlights:**

- `MatMulTransB`/`matmulQuantTransB` take a `threads ...int` variadic; shard by output column j, each worker keeps private acc/rowBuf (zero contention). The F32 branch shards by output row i.
- Default thread count **proportional to core count**: floor(core count × 2/3) (12 cores→8, leaving a 1/3 margin for the system/editor); overridable via the `-threads` flag; automatically clipped down when it exceeds the CPU core count.
- Threads are threaded through all the way from `GenerateOptions.Threads → eval.Context.threads → every MatMulTransB` (variadic keeps old call sites compatible).
- Bonus: `llminfer bench -threads N` can test different core counts.

## M7.4 Measurements (Buffer Reuse)

decode 6.4, prefill 13.3 tokens/s (on par with M7.3).

**Implementation highlights:**
- eval layer: `x` (embed output/residual carrier) and attention's score rows switched to `Context`-reused buffers (`ensure` lazily grows them); attention output accumulates directly into `attBuf`, dropping the per-head `make(vout)`.
- tensor layer: matmul workers' acc/rowBuf go through `sync.Pool` (buffers larger than 1M elements are not pooled, to avoid hogging memory).
- Throughput didn't rise but **allocations disappeared**: decode no longer makes hundreds of small buffers per token, GC pressure drops significantly, and long sessions are more stable.

## M7.5 Measurements (KV Prefix Reuse, Multi-turn Dialogue)

First turn has no history = full prefill; later turns forward only the increment based on lcp:

```
turn 1 (no history):    reuse 0%
turn 2 (after 1 Q&A):   reuse 60%
turn 3 (after 2 Q&As):  reuse 82%   ← the longer the history, the higher the reuse ratio
```

**Implementation highlights:**
- `cache.Truncate`: truncates the number of valid positions (data is retained; the region past pos is overwritten by subsequent Writes).
- `eval.Context` gains `Clear/Rewind/ForwardWithCache`: resent input → directly return last turn's logits; prefix hit → Rewind to lcp and forward only the increment; mismatch → Clear and run full forward.
- `ChatSession.Chat` drops the per-turn Reset and uses `ForwardWithCache` instead—system prompt + history are no longer recomputed.
- Why reuse isn't 100%: assistant reply text is re-encoded after decode, which differs slightly from the token-by-token sequence produced during generation; lcp truncates there (still correct, just saves slightly less).

### M7.5 Multi-turn Measurement Details (greedy, 8 threads, max-tokens=12)

| Turn | Input | Generated tok | Reuse ratio | Total time |
|---|---|---|---|---|
| 1 | My name is Alice. | 10 | 0% (no history) | 5873 ms |
| 2 | What is my name? | 5 | 60% | 3698 ms |
| 3 | Great, and my dog is Rex. | 12 | 74% | 4553 ms |
| 4 | What is my dog's name? | 12 | 74% | 5614 ms |

- Reuse ratio = 1 − (actual forward tokens this turn / full prompt tokens). From turn 2 onward, history is not recomputed; only the increment of "new user segment + leading tokens" is forwarded.
- Turn 4 takes longer than turn 3: this turn generates about as many tokens (12 vs 12) but the prompt is longer → each decode's self-attention must scan a longer history, an unavoidable cost unrelated to reuse.
- With prefix reuse disabled (full prefill after a Reset every turn), turns 2-4 would each recompute the forward of 50-80 extra history tokens, accumulating linearly across turns.

## M7 Full-Phase Summary (prefill at prompt=129 tok / decode 64 tok unless noted)

| Phase | Threads | prefill tok/s | decode tok/s | vs. baseline |
|---|---|---|---|---|
| M7.1 Baseline (DequantRow + scalar) | 1 | 2.7 | 1.2 | 1.0x |
| M7.2 Fused dot product | 1 | 2.7 | 1.9 | decode 1.6x |
| M7.3 Multithreading | 8 | 12.8 | 6.6 | decode 5.5x |
| M7.4 Buffer reuse | 8 | 13.3 | 6.4 | decode 5.3x |
| M7.5 KV prefix reuse (multi-turn gain) | 8 | — | — | multi-turn history not recomputed |

M7.4's throughput being on par with M7.3 is expected: the per-token-decode FLOPS boundary didn't change; buffer reuse saves allocations and GC, not compute.

## M7.3 Thread Scaling Measurements (prompt=49 tok / decode 20 tok)

| Threads | prefill tok/s | decode tok/s | Notes |
|---|---|---|---|
| 1 | 2.6 | 1.8 | baseline |
| 2 | 5.0 | 3.2 | ~linear |
| 4 | 9.2 | 4.3 | prefill superlinear, decode slowing |
| 8 | 12.9 | 6.4 | decode near saturation |
| 12 | 15.0 | 6.3 | decode no longer grows |

**Conclusion**: decode (memory-bandwidth-bound) saturates at 8 threads, and actually drops slightly at 12 (scheduling overhead); prefill (ample parallelism) keeps scaling with thread count. The default of 2/3 of core count (12→8) sits right on decode's plateau, leaving 4 cores of headroom without sacrificing throughput.

**Projected endpoint**: pure-Go portion (fused ×3 + multicore ×4) about 10x; SIMD doubles it again. 0.5B target decode ≥ 20 tokens/s as a starting point.

**Trade-offs**: M7.2-M7.4 are straightforward-first and a must-do (high learning value: understanding "multiply by scale only once per block"); M7.5 has limited gains so defer it; M7.6 assembly violates the "straightforward" positioning, so put it last or skip it.

## Acceptance Criteria

- Every phase compares before/after with `make bench`, archiving the tokens/s improvement
- Numerical correctness: `go test ./...` all green; generation results are semantically unchanged for the same prompt (fused dot product differs by only <1e-4)
- Endpoint: decode throughput ≥5x improvement over the M7.1 baseline

## Glossary

| Term | 中文 | Explanation |
|---|---|---|
| Fused dequant-dot | 融合反量化点积 | Accumulate the dot product directly over quantized integers, multiplying by scale only once per block |
| GFLOPs | 十亿浮点运算 | A unit of compute; 1 GFLOPs = 10⁹ floating-point operations |
| LUT | 查表 | LookUp Table; trades memory for compute (fp16→fp32 conversion) |
| SIMD | 单指令多数据 | Single Instruction, Multiple Data; the CPU processes several data items at once (AVX2 = 256bit) |
| prefill / decode | 预填充 / 生成 | Consuming the whole prompt at once / generating token by token |
| KV prefix reuse | KV cache prefix reuse | Reusing the already-computed common prefix to avoid redundant forwarding |
