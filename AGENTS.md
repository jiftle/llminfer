# llminfer — Repository Guide

## What this repo is

A **pure-Go LLM inference engine** built from scratch as a learning project: load a real GGUF model (qwen2.5-0.5b) → generate text token by token → multi-turn chat. Not a production codebase — no CI, no lint config, no external deps. Verification relies on `go test` unit tests plus running the scripts and eyeballing the output.

## Common commands

```bash
# Generate once (flags MUST precede MODEL!)
go run . run -max-tokens 30 -temperature 0 models/qwen2.5-0.5b.gguf "The capital of France is"
go run . run -temperature 0.8 -max-tokens 64 models/qwen2.5-0.5b.gguf "Hello"

# Interactive multi-turn chat (ChatML template)
go run . run -chat -max-tokens 80 models/qwen2.5-0.5b.gguf

# Performance benchmark (threads default = 2/3 of CPU cores, e.g. 8 on 12)
go run . bench models/qwen2.5-0.5b.gguf

# All tests
go test ./...

# Compile check
go build ./... && go vet ./...
```

**Flag-order gotcha**: Go `flag.NewFlagSet` stops parsing at the first non-flag arg. `llminfer run MODEL -temp 0.8` does NOT work; use `llminfer run -temp 0.8 MODEL`.

## Dependencies & environment

- Go 1.26 (declared in `go.mod`)
- Zero external dependencies (stdlib only + in-house packages)
- Model file: `models/qwen2.5-0.5b.gguf` (~397MB, not committed — see README for how to obtain)
- Git remotes: `origin` = gitee, `github` = GitHub. Default branch is `main`.

## Directory structure & package responsibilities

```
cmd/              # CLI entry (command.go dispatch, generate.go generation loop)
internal/kernel/
  gguf/           # GGUF file parsing (metadata + tensor info table)
  tokenizer/      # tiktoken byte-level BPE (Encode/Decode/EOS)
  tensor/         # Tensor + MatMul + fused dequant-dot + RMSNorm/RoPE/SoftMax
  model/          # LLaMAModel: mount GGUF weights by name
  cache/          # KV Cache [layer][pos][kvHead][headDim] + Truncate (prefix reuse)
  eval/           # Forward inference: prefill / decode / KV prefix reuse (ForwardWithCache)
  sampler/        # Sampling: temperature/top-k/top-p/greedy
  chat/           # ChatML chat template: detect + render + default-system extraction
  threads.go      # default thread count = NumCPU*2/3 (see tensor/)
1-docs/           # design docs, bilingual: *.md (English) + *_zh.md (Chinese)
```

## Verification approach

- `go test ./...` — unit tests in tensor/chat/sampler/cache (hand-computed references).
- End-to-end sanity: `go run . run <model> "<prompt>"` and check the output reads coherently.
- Per-milestone acceptance + pitfalls are recorded in `1-docs/`.
- When changing fused quantized dot products, run `go test ./internal/kernel/tensor/` and re-benchmark before/after with `go run . bench` (report tokens/s deltas).

## Conventions

- **Commit messages in English** (repo is public / GitHub-oriented). One commit per milestone.
- Code is deliberately straightforward over fast: correctness first, and every line explains *why*. Optimizations that hurt readability must be documented (see M7).
- Docs live in `1-docs/`, one file pair per milestone (English `M{n}-Topic.md` + Chinese `M{n}-Topic_zh.md`), each ending with a glossary.
- `.vscode/` debug configs are added manually (not auto-ignored, not auto-committed).
- GGUF model files are gitignored (too large).

## Milestone status

```
M1 GGUF parsing ✅ → M2 tokenizer ✅ → M3 tensor ops ✅ → M4 forward + KV cache ✅
→ M5 generation + sampling ✅ → M6 ChatML chat ✅ → M7 performance ✅
```

Key perf results (12-core, qwen2.5-0.5b): decode 1.2→6.4 tok/s, prefill 2.7→13.3 tok/s; multi-turn reuses KV prefix. Details in `1-docs/M7-Performance.md`.

## Hard-won pitfalls (avoid re-tripping)

- **GGUF tensor offset is relative to the data section**, not the file header. Always read at `DataStart + Offset`.
- **1-D weight rows**: norm/bias tensors are 1-D (NE=[896,0,0,0]); using `NE[1]` as the row count → 0 → all-zero weights. Use `Rows()` (1-D = 1 row).
- **Flag order**: Go flags must precede positional args.
- **Tokenizer needs its own GGUFFile pass**: `model.Load` closes the file after mounting; re-read metadata for the tokenizer (cheap).
- **M6 template regex `\n`**: in qwen templates `\n` is a literal backslash+n (two chars), not a newline. Match it in regex as `\\n`; write test constants with backquoted strings.
- **M6 default-system extraction**: the default system prompt is one single-quoted segment `'<|im_start|>system\n…<|im_end|>\n'`; strip the markers and then the literal `\n`, or you get a stray newline.
- **M7 gguf block table must match tensor sizes**: Q4_K block = 144 bytes (not 126), Q6_K = 210 (not 226). A mismatch silently truncates `Data` and panics later in dequant.
- **M7 prefill loop order**: outer loop over rows (i), inner over columns (j) so activation rows stay cache-resident; the reverse order drops prefill from 2.7 to 1.5 tok/s.
- **M7 thread default**: don't hardcode 8 — derive `NumCPU()*2/3` so it scales to other machines.
