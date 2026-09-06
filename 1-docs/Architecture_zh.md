# llminfer 架构与里程碑

> 学习项目：从零重写一个纯 Go 的 LLM 推理引擎。
> 目标：能直接加载 GGUF 模型文件（qwen2.5-0.5b）并逐字生成文本。

## 一句话定位

把「训练好的大模型」跑起来的纯 Go 引擎。相比学习笔记里的 PyTorch 直白代码，这是一个没有框架遮挡、每个字节都自己管的推理实现。

## 为什么从零写

- 学习笔记（learn-llm）用 PyTorch，形状自动管理，掩盖了很多细节。
- PyTorch 之上的推理框架太工程化：缓冲复用、量化、SIMD 全是优化噪音（那些是必要的工程，但不是学习主线）。
- 本工程介于中间：**算法正确、实现直白**（先求对，不求快），每个齿轮单独造出来看它是怎么咬合的。

## 目录结构

```
llminfer/
├── main.go                 # 入口：委托给 cmd
├── cmd/
│   ├── command.go          # CLI 分发：run / bench / help
│   └── generate.go         # 生成循环：encode → prefill → decode loop → EOS
├── internal/kernel/
│   ├── gguf/               # GGUF 文件解析（M1 ✅）
│   ├── tokenizer/          # tiktoken BPE 分词（M2 ✅）
│   ├── tensor/             # 张量结构与算子（M3 ✅）
│   │   ├── tensor.go       # DType/Nelements/DequantRow/AsFloat32
│   │   ├── quant.go        # Q4_0~Q6_K 反量化
│   │   ├── ops.go          # MatMulTransB / MatMul
│   │   ├── norm.go         # RMSNorm / SiLU / SoftMax
│   │   └── rope.go         # RoPE（NEOX / NORMAL）
│   ├── model/              # 模型静态结构（M3 ✅）
│   │   └── model.go        # LLaMAModel + attachTensors
│   ├── cache/              # KV Cache（M4 ✅）
│   │   └── cache.go        # [layer][pos][kvHead][headDim] 布局
│   ├── eval/               # 前向推理（M4 ✅）
│   │   ├── eval.go         # Forward：embed → 24层 → logits
│   │   └── attention.go    # GQA 注意力 + 因果掩码 + 缩放
│   ├── sampler/            # 采样（M5 ✅）
│   │   ├── sampler.go      # 温度/top-k/top-p/贪心
│   │   └── sampler_test.go # 6 个单测全绿
│   └── chat/               # ChatML 对话模板（M6 ✅）
│       ├── chat.go         # 模板检测 + 渲染 + 默认 system 提取
│       └── chat_test.go    # 6 个单测全绿
├── models/                 # 本地 GGUF 模型文件（.gitignore）
└── 1-docs/                 # 设计文档（英文主文件 + _zh 中文版）
    ├── Architecture.md       # 英文 / Architecture_zh.md 本文件
    ├── M1-GGUF-Parsing.md    # 英文 / M1-GGUF-Parsing_zh.md 中文
    ├── M2-Tokenizer.md       # 英文 / M2-Tokenizer_zh.md 中文
    ├── M3-Tensor-Ops.md      # 英文 / M3-Tensor-Ops_zh.md 中文
    ├── M4-Forward-Pass.md    # 英文 / M4-Forward-Pass_zh.md 中文
    ├── M5-Sampling.md        # 英文 / M5-Sampling_zh.md 中文
    ├── M6-ChatML-Template.md # 英文 / M6-ChatML-Template_zh.md 中文
    └── M7-Performance.md     # 英文 / M7-Performance_zh.md 中文
```

## 推理全链路（一遍看懂整个项目在搓什么）

```
输入 "Hello"
  ↓
Encode（BPE 分词）→ [token_ids]
  ↓
Forward Prefill（一次吞掉全部 prompt）
  ↓
24 层 × [RMSNorm → QKV投影 → RoPE → GQA注意力(写KV) → WO→残差 → SwiGLU FFN]
  ↓
输出 logits [151936]（每个候选词的分数）
  ↓
Sample（温度 + top-k + top-p → 挑一个 token id）
  ↓
Decode（token id → 文本）
  ↓
Forward 单 token（KV Cache 自动累积历史）
  ↓
... 循环直到 EOS 或达到上限
```

## 里程碑进度

```
M1 GGUF解析 ✅ → M2 分词 ✅ → M3 张量算子 ✅ → M4 前向+KV缓存 ✅ → M5 生成+采样 ✅ → M6 ChatML模板 ✅
```

| 里程碑 | 做什么 | 验收标准 | 实际结果 | 提交 |
|---|---|---|---|---|
| **M1** GGUF 解析 | 读文件头/元数据/张量信息表 | `run` 打印出 290 个张量、形状正确 | ✅ 全部通过 | `43577e5` |
| **M2** 分词 | tiktoken BPE encode/decode | 中文/英文三句话 encode 结果正确 | ✅ 三句话通过 | `0e8f229` |
| **M3** 张量算子 | Tensor + MatMul + 量化反量化 + RMSNorm/SiLU/RoPE/Softmax | 反量化手算核对 | ✅ 5 个单测全绿 | `d9e86f2` |
| **M4** 前向 | embed → 逐层 → logits + KV Cache | 8-token 序列 top-k 合理 | ✅ top-20 合理 | `c6f8737` |
| **M5** 生成+采样 | temperature/top-k/top-p + 生成循环 | qwen2.5-0.5b 生成连贯文本 | ✅ 贪心/温度均通 | `eacab7f` |
| **M6** ChatML 模板 | `<\|im_start\|>` 包装 + 多轮 | 多轮能记住上下文 | ✅ 多轮记忆验证通过 | `39788c3` |

## 核心模块一句话总结

| 包 | 做什么 | 关键类型 |
|---|---|---|
| `gguf` | 读 GGUF 文件头/元数据/张量信息表 | `GGUFFile`, `TensorInfo` |
| `tokenizer` | tiktoken BPE 分词（Encode/Decode） | `Tokenizer` |
| `tensor` | 张量结构 + MatMul + 反量化 + RMSNorm/RoPE/SoftMax | `Tensor`, `DType` |
| `model` | 从 GGUF 加载权重 → LLaMAModel | `LLaMAModel`, `LLaMALayer` |
| `cache` | KV Cache（存历史 K/V，避免重算注意力） | `KVCache` |
| `eval` | 前向推理：embed → 逐层 Transformer → logits | `Context` |
| `sampler` | 从 logits 挑 token：贪心/温度/top-k/top-p | `Sampler`, `Config` |
| `chat` | ChatML 对话模板：检测 + 渲染 + 默认 system 提取 | `Template`, `Message` |

## 验证方式

- 单算子正确性靠 `go test` 单测（tensor/chat/sampler），手算样本核对数值。
- 端到端靠 `go run . run <模型> "<prompt>"` 观察生成是否连贯合理。
- 各里程碑验收细节与踩坑记在 `1-docs/M{n}-*.md`。

## M1-M6 踩过的坑（经验库）

| 里程碑 | 坑 | 根因 | 修复 |
|---|---|---|---|
| M1 | GGUF 张量 offset 读错 | offset 相对数据区起点，不是文件头 | `DataStart + Offset` |
| M3 | Q6_K 反量化数值错 | 子块索引下标写错 | 逐行核对 ggml 反量化公式 |
| M4 | logits 全 0 | 一维权重（norm）NE[1]=0，AsFloat32 循环 0 次 → w 全 0 | 加 `Rows()` 方法，一维视为 1 行 |
| M5 | flag 参数不生效 | Go flag 必须在位置参数前面 | `llminfer run -temp 0.8 MODEL "prompt"` |
| M6 | 默认 system 提取失败 | 模板 `\n` 是字面两字符 + 整段单引号结构 | 正则剥标记再剥字面 `\n` |

## 下一步

M1-M6 主线完成。当前主线：**M7 性能优化**（融合反量化点积 + 多线程 + 缓冲复用），详见 [M7-Performance_zh.md](M7-Performance_zh.md)。

其他候选方向：

- **增量续推**：多轮对话时跨轮复用 KV cache，只前向新增 token（当前每轮全量 prefill）
- **更完整模板**：支持 LLaMA3 之外更多模板、tools 分支（当前只做关键词检测 + ChatML/LLaMA3 渲染）
