# llminfer

A **pure-Go LLM inference engine** built from scratch (learning project, zero external dependencies). It loads a real GGUF model (qwen2.5-0.5b) and generates text token by token, supporting multi-turn chat.

> Positioning: not a production inference service, but a learning project that "explains how an LLM goes from computation to output". Every milestone has internal unit tests and a self-verification record.

## Quick Start

```bash
# Build + single-shot generation
go run . run models/qwen2.5-0.5b.gguf "The capital of France is" -temperature 0

# Interactive multi-turn chat (ChatML template, KV prefix reuse across turns)
go run . run -chat models/qwen2.5-0.5b.gguf

# CLI help
go run . run --help

# Performance benchmark (defaults to 2/3 of your CPU cores; override with -threads)
go run . bench models/qwen2.5-0.5b.gguf
```

> ⚠️ Go flags must come before the MODEL positional arg: `-temperature 0 MODEL` is valid, `MODEL -temperature 0` is not.

### Model File

`models/qwen2.5-0.5b.gguf` (397MB, Qwen2.5 0.5B Instruct) is not committed; provide it yourself and place it under `models/`. The code is currently tuned for the qwen2 architecture + ChatML template.

## Feature Status

| Feature | Status |
|---|---|
| GGUF parsing / byte-level BPE tokenizer | ✅ M1/M2 |
| Dequantization Q4_0~Q6_K + MatMul / RMSNorm / RoPE | ✅ M3 |
| Forward pass (24-layer Transformer + GQA + KV Cache) | ✅ M4 |
| Sampling (greedy / temperature / top-k / top-p) | ✅ M5 |
| Multi-turn chat (ChatML template) | ✅ M6 |
| Perf: fused dequant-dot + multithreading + buffer reuse + KV prefix reuse | ✅ M7 |
| GPTQ/AWQ & other quantizations, tensor parallelism, GPU | ❌ Not done |

## Architecture

```
main.go
├── cmd/                  # CLI dispatch (run/bench) + generation loop + interactive chat
└── internal/kernel/      # pure compute kernel (one-way deps)
    ├── gguf/             # GGUF file parser (metadata + tensor info table)
    ├── tokenizer/        # tiktoken byte-level BPE (Encode/Decode/special tokens/chat template)
    ├── chat/             # ChatML template detection & rendering
    ├── tensor/           # Tensor + MatMul + dequant + RMSNorm/RoPE/SoftMax
    ├── model/            # mount GGUF weights into LLaMAModel
    ├── cache/            # KV Cache + prefix truncation (Truncate)
    ├── eval/             # forward Context (prefill / decode / prefix reuse)
    └── sampler/          # sampler (greedy/temperature/top-k/top-p)
```

## Performance (12-core CPU, qwen2.5-0.5b)

| Stage | decode | prefill |
|---|---|---|
| Baseline before M7 (single thread) | 1.2 tok/s | 2.7 tok/s |
| After M7 (8 threads) | 6.4 tok/s | 13.3 tok/s |

Multi-turn chat uses KV prefix reuse: system prompt + history are not recomputed, only new tokens are forwarded. Details in `1-docs/M7-性能优化方案.md` (Chinese).

## Development

```bash
go build ./... && go vet ./... && go test ./...
make bench    # performance benchmark
```

- **Verification**: `go test ./...` (tensor/chat/sampler/cache unit tests) + `go run . run <model> "<prompt>"` to check output sanity. Milestone acceptance details live in `1-docs/M{n}-*.md`.
- **Doc convention**: one design doc per milestone in `1-docs/M{n}-*.md` (formulas, acceptance records, pitfalls, glossary).
- **Commit convention**: Chinese commit messages, one commit per milestone.

## Docs (Chinese)

- [1-docs/架构设计说明.md](1-docs/架构设计说明.md) — architecture overview, milestone progress, pitfalls
- [1-docs/M1-GGUF文件解析.md](1-docs/M1-GGUF文件解析.md) ~ [M7-性能优化方案.md](1-docs/M7-性能优化方案.md) — detailed milestone notes

## Milestones

```
M1 GGUF parsing → M2 tokenizer → M3 tensor ops → M4 forward + KV cache → M5 generation + sampling → M6 ChatML chat → M7 performance
```

All done ✅. Next candidates: more complete quantized ops (SIMD), a fuller template engine.
