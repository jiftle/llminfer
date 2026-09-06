# llminfer

从零实现的**纯 Go LLM 推理引擎**（学习项目，无任何外部依赖）。能加载真实 GGUF 模型（qwen2.5-0.5b）并逐字生成文本、进行多轮聊天。

> 定位：不是生产推理服务，而是"把 LLM 从计算到输出拆开讲明白"的学习工程。每个里程碑都有内部单测与自验记录。

## 快速开始

```bash
# 编译 + 单次生成
go run . run models/qwen2.5-0.5b.gguf "The capital of France is" -temperature 0

# 交互式多轮聊天（ChatML 模板，多轮复用 KV 前缀缓存）
go run . run -chat models/qwen2.5-0.5b.gguf

# 参数说明
go run . run --help

# 性能基准（默认按核数 2/3 起线程，可 -threads 覆盖）
go run . bench models/qwen2.5-0.5b.gguf
```

> ⚠️ Go 的 flag 必须在 MODEL 前面：`-temperature 0 MODEL` 合法，`MODEL -temperature 0` 无效。

### 模型文件

引擎使用 **Qwen2.5 0.5B Instruct**（GGUF 格式，qwen2 架构 + ChatML 对话模板）。代码按该架构适配，直接加载同架构的其他 GGUF（如更大 Qwen2.5）也能跑。

**获取方式（任选其一）：**

```bash
# 1) 从 Ollama 提取（与开发测试同款）
ollama pull qwen2.5:0.5b
# 找到模型 blob（约 397MB，名字含 sha256 前缀），复制为 .gguf：
#   ~/.ollama/models/blobs/sha256-xxxxxxxx → models/qwen2.5-0.5b.gguf
# 可用 `ollama show qwen2.5:0.5b --modelfile` 查看其源文件名定位

# 2) 从 Hugging Face 下载 GGUF
#   https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF
#   （选 q4_k_m 或 q8_0 档的 .gguf，放到 models/ 下）

# 3) 自己转换（需要原版 safetensors + 转换工具）
#   从 https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct 下载权重，
#   按 GGUF 官方转换流程转成 .gguf 放入 models/
```

`models/qwen2.5-0.5b.gguf` 未入库（体积大），放入 `models/` 即可被加载。

## 能干什么 / 不能干什么

| 能力 | 状态 |
|---|---|
| GGUF 解析 / 字节级 BPE 分词 | ✅ M1/M2 |
| 量化反量化 Q4_0~Q6_K + MatMul/归一化/RoPE | ✅ M3 |
| 前向推理（24 层 Transformer + GQA + KV Cache） | ✅ M4 |
| 采样生成（贪心/温度/top-k/top-p） | ✅ M5 |
| 多轮聊天（ChatML 对话模板） | ✅ M6 |
| 性能优化：融合反量化点积 + 多线程 + 缓冲复用 + KV 前缀复用 | ✅ M7 |
| GPTQ/AWQ 等其他量化、张量并行、GPU 加速 | ❌ 未做 |

## 架构一览

```
main.go
├── cmd/                  # CLI run/bench 分发 + 生成逻辑 + 交互聊天
└── internal/kernel/      # 纯计算内核（单向依赖）
    ├── gguf/             # GGUF 文件解析（元数据 + 张量信息表）
    ├── tokenizer/        # tiktoken 字节级 BPE（Encode/Decode/特殊token/对话模板）
    ├── chat/             # ChatML 模板检测与渲染
    ├── tensor/           # 张量 + MatMul + 反量化 + RMSNorm/RoPE/SoftMax
    ├── model/            # 从 GGUF 挂载权重为 LLaMAModel
    ├── cache/            # KV Cache + 前缀截断（Truncate）
    ├── eval/             # 前向推理 Context（prefill/decode/前缀复用）
    └── sampler/          # 采样器（贪心/温度/top-k/top-p）
```

## 性能（本机 12 核 CPU，qwen2.5-0.5b）

| 阶段 | decode | prefill |
|---|---|---|
| M7 前基线（单线程） | 1.2 tok/s | 2.7 tok/s |
| M7 后（8 线程） | 6.4 tok/s | 13.3 tok/s |

多轮对话启用 KV 前缀复用：系统提示+历史不重算，只前向新增 token。详见 `1-docs/M7-Performance_zh.md`。

## 开发

```bash
go build ./... && go vet ./... && go test ./...
make bench    # 性能基准
```

- **验证方式**：`go test ./...`（tensor/chat/sampler/cache 单测）+ `go run . run <模型> "<prompt>"` 看生成是否合理，各里程碑验收细节见 `1-docs/M{n}-*.md`。
- **目录约定**：`1-docs/M{n}-*.md`（英文主文件）与 `M{n}-*_zh.md`（中文版）每里程碑一篇（含公式、验收记录、踩坑、术语表）。
- **提交约定**：中文提交信息，一个里程碑一个提交。

## 文档

每份文档双语并存：`*.md` 为英文主文件、`*_zh.md` 为中文版。

- [1-docs/Architecture_zh.md](1-docs/Architecture_zh.md) —— 总体架构、里程碑进度、踩坑经验
- [1-docs/M1-GGUF-Parsing_zh.md](1-docs/M1-GGUF-Parsing_zh.md) ~ [M7-Performance_zh.md](1-docs/M7-Performance_zh.md) —— 各里程碑详细笔记

## 里程碑

```
M1 GGUF解析 → M2 分词 → M3 张量算子 → M4 前向+KV缓存 → M5 生成+采样 → M6 ChatML对话 → M7 性能优化
```

全部完成 ✅。下一步候选：更完整的量化算子（SIMD）、更完整的模板引擎。