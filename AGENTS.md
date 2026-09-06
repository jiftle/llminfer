# llminfer 仓库指南

## 仓库性质

纯 Go LLM 推理引擎（学习项目）：加载 GGUF 模型 → 逐字生成文本。不是生产代码库——无 CI、无 lint、无 Makefile。验证靠 `go test` 单测 + 运行脚本核对输出。

## 常用命令

```bash
# 跑生成（flags 必须在 MODEL 前面！）
go run . run -max-tokens 30 -temperature 0 models/qwen2.5-0.5b.gguf "The capital of France is"
go run . run -temperature 0.8 -max-tokens 64 models/qwen2.5-0.5b.gguf "Hello"

# 交互式多轮聊天（ChatML 模板）
go run . run -chat -max-tokens 80 models/qwen2.5-0.5b.gguf

# 跑全部测试
go test ./...

# 编译检查
go build ./... && go vet ./...
```

**flag 顺序坑**：Go `flag.NewFlagSet` 遇到非 flag 字符就停止解析。`llminfer run MODEL -temp 0.8` 不生效，必须 `llminfer run -temp 0.8 MODEL`。

## 依赖与环境

- Go 1.26（`go.mod` 声明）
- 无外部依赖（纯标准库 + 自研包）
- 模型文件：`models/qwen2.5-0.5b.gguf`（397MB，未入库，需自行放置）

## 目录结构与包职责

```
cmd/              # CLI 入口（command.go 分发，generate.go 生成循环）
internal/kernel/
  gguf/           # GGUF 文件解析（元数据 + 张量信息表）
  tokenizer/      # tiktoken BPE 分词（Encode/Decode/EOS）
  tensor/         # 张量 + MatMul + 反量化 + RMSNorm/RoPE/SoftMax
  model/          # LLaMAModel：从 GGUF 加载权重，按名挂载张量
  cache/          # KV Cache：[layer][pos][kvHead][headDim] 布局
  eval/           # 前向推理：embed → 24层Transformer → logits
  sampler/        # 采样：温度/top-k/top-p/贪心
  chat/           # ChatML 对话模板：检测 + 渲染 + 默认system提取
1-docs/           # 设计文档（每里程碑一篇，含术语表）
```

## 验证方式

- 验证靠 `go test ./...` + `go run . run <模型> "<prompt>"` 看输出是否合理
- 单算子验证：`go test ./internal/kernel/tensor/`（反量化/MatMul/RMSNorm 手算核对）
- 前向验证：`go test ./internal/kernel/tokenizer/` 等包的单测
- 各里程碑的验收细节记录在 `1-docs/M{n}-*.md`

## 约定

- 中文提交信息，里程碑式提交（一个里程碑一个提交）
- 代码直白优先：先求对，不求快，每行注释"为什么"
- 每个里程碑写 1-docs 文档（含术语表）
- VS Code 调试配置（`.vscode/`）手动 add，不自动忽略
- GGUF 模型文件 `.gitignore` 排除（体积大）

## 里程碑进度

```
M1 GGUF解析 ✅ → M2 分词 ✅ → M3 张量算子 ✅ → M4 前向+KV缓存 ✅ → M5 生成+采样 ✅ → M6 ChatML模板 ✅
```

## 关键踩坑经验

- **GGUF 张量 offset**：相对数据区起点，不是文件头。读取必须 `DataStart + Offset`。
- **一维权重行数**：norm/bias 是一维张量（NE=[896,0,0,0]），`AsFloat32` 用 NE[1] 当行数→0→全0。加 `Rows()` 方法修复。
- **flag 参数顺序**：Go flag 必须在位置参数前面。
- **tokenizer 需要 GGUFFile**：model.Load 读完就 Close，tokenizer 需再读一次元数据（轻量）。
- **M6 模板正则 `\n`**：qwen 模板里的 `\n` 是字面反斜杠+n 两个字符（非换行符），Go 正则要 `\\n` 匹配，测试常量用反引号字符串。
- **M6 默认 system 提取**：模板里默认系统提示是「system标记+文案+结束标记」整段单引号字符串，剥标记后要再剥掉字面 `\n`。
