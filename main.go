// llminfer —— 从零重写一个纯 Go 的 LLM 推理引擎（学习项目）。
//
// 目标：能直接加载 GGUF 模型文件（如 qwen2-0.5b）并逐字生成文本。
// 理念：不分层优化，先写最直白的正确实现，跑通了再谈快。
package main

import (
	"fmt"
	"os"

	"github.com/feiyuclaw/llminfer/cmd"
)

// main 把所有工作委托给 cmd 包：run / bench 等子命令。
func main() {
	if err := cmd.Execute(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "llminfer:", err)
		os.Exit(1)
	}
}