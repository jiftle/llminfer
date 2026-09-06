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

The engine runs **Qwen2.5 0.5B Instruct** (GGUF format, qwen2 architecture + ChatML template). The code is tuned to this architecture; other qwen2-architecture GGUF files (e.g., larger Qwen2.5) load as well.

**How to obtain it (pick one):**

```bash
# 1) Extract from Ollama (the same build used for development/tests)
ollama pull qwen2.5:0.5b
# locate the model blob (~397MB, sha256-prefixed name), copy it as .gguf:
#   ~/.ollama/models/blobs/sha256-xxxxxxxx → models/qwen2.5-0.5b.gguf
# use `ollama show qwen2.5:0.5b --modelfile` to find its source file name

# 2) Download a GGUF from Hugging Face
#   https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF
#   (pick a q4_k_m or q8_0 .gguf and drop it into models/)

# 3) Convert it yourself (requires the original safetensors + a converter)
#   grab weights from https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
#   and follow the official GGUF conversion flow, output into models/
```

`models/qwen2.5-0.5b.gguf` is not committed (large); just place it under `models/` and it will be loaded.

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

Multi-turn chat uses KV prefix reuse: system prompt + history are not recomputed, only new tokens are forwarded. Details in `1-docs/M7-Performance.md`.

## Development

```bash
go build ./... && go vet ./... && go test ./...
make bench    # performance benchmark
```

- **Verification**: `go test ./...` (tensor/chat/sampler/cache unit tests) + `go run . run <model> "<prompt>"` to check output sanity. Milestone acceptance details live in `1-docs/M{n}-*.md`.
- **Doc convention**: one design doc per milestone in `1-docs/M{n}-*.md` (English) / `M{n}-*_zh.md` (Chinese); formulas, acceptance records, pitfalls, glossary.
- **Commit convention**: Chinese commit messages, one commit per milestone.

## Docs

Every doc ships in two languages: `*.md` (English primary) and `*_zh.md` (Chinese).

- [1-docs/Architecture.md](1-docs/Architecture.md) — architecture overview, milestone progress, pitfalls
- [1-docs/M1-GGUF-Parsing.md](1-docs/M1-GGUF-Parsing.md) ~ [M7-Performance.md](1-docs/M7-Performance.md) — detailed milestone notes

## Milestones

```
M1 GGUF parsing → M2 tokenizer → M3 tensor ops → M4 forward + KV cache → M5 generation + sampling → M6 ChatML chat → M7 performance
```

All done ✅. Next candidates: more complete quantized ops (SIMD), a fuller template engine.
