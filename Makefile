# llminfer —— 纯 Go LLM 推理引擎（学习项目）Makefile
# 常用：make build / make chat / make check

# ---- 可配置变量（可用命令行覆盖，如 make run PROMPT="你好"）----
BINARY = llminfer
MODEL  ?= models/qwen2.5-0.5b.gguf
PROMPT ?= "The capital of France is"

# ---- 默认目标 ----
all: build

# ---- 构建 ----
build:
	go build -o $(BINARY) .

# ---- 运行：编译后单次生成（贪心）----
# 注意 Go flag 必须在 MODEL 前面，所以 -temperature 0 放 $(MODEL) 之前
run: build
	./$(BINARY) run -temperature 0 -max-tokens 64 $(MODEL) $(PROMPT)

# ---- 交互式多轮聊天 ----
chat: build
	./$(BINARY) run -chat -max-tokens 80 $(MODEL)

# ---- 性能测试（prefill/decode 吞吐）----
bench: build
	./$(BINARY) bench -n-tokens 64 -prompt-tokens 128 $(MODEL)

# ---- 免编译直接跑（开发调试快）----
gorun:
	go run . run -temperature 0 -max-tokens 64 $(MODEL) $(PROMPT)

# ---- 检查 ----
test:
	go test ./...

vet:
	go vet ./...

fmt:
	gofmt -w .

# ---- 一次性过全部门禁（M1-M6 每完成一步都跑一遍）----
check: fmt vet build test

# ---- 维护 ----
clean:
	rm -f $(BINARY)
	go clean

help:
	@echo "常用命令："
	@echo "  make build          构建 ./$(BINARY)"
	@echo "  make run            单次生成（贪心）"
	@echo "  make chat           交互式多轮聊天"
	@echo "  make gorun          免编译直接跑"
	@echo "  make test           跑单测"
	@echo "  make check          fmt + vet + build + test"
	@echo "  make clean          删除二进制"