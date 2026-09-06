// Package cmd 负责 CLI 子命令分发（run / bench 等）。
package cmd

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"
)

// Execute 解析子命令并执行。
func Execute(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("用法: llminfer <run|bench> [参数]\n提示: 输入 --help 查看子命令详情")
	}
	switch args[0] {
	case "run":
		return run(args[1:])
	case "bench":
		return bench(args[1:])
	case "help", "-h", "--help":
		return usage()
	default:
		return fmt.Errorf("未知子命令 %q", args[0])
	}
}

// usage 打印顶层帮助。
func usage() error {
	fmt.Println("llminfer —— 纯 Go LLM 推理引擎（学习用）")
	fmt.Println()
	fmt.Println("子命令:")
	fmt.Println("  run MODEL [PROMPT]   加载 GGUF 模型并生成文本")
	fmt.Println("  bench MODEL          对模型做推理基准测试")
	return nil
}

// run 子命令：加载模型、编码 prompt、自回归生成文本。
// 用法: llminfer run MODEL "The capital of France is" --temperature 0.8 --max-tokens 64
func run(args []string) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	var nCtx = fs.Int("n-ctx", 0, "上下文窗口长度（0 取模型默认）")
	var maxTokens = fs.Int("max-tokens", 64, "最多生成多少个 token")
	var temperature = fs.Float64("temperature", 0.8, "采样温度（≤0 为贪心）")
	var topK = fs.Int("top-k", 0, "Top-K 截断（≤0 关闭）")
	var topP = fs.Float64("top-p", 0.9, "Top-P 核采样阈值（1 关闭）")
	var seed = fs.Int64("seed", 42, "随机种子")
	var verbose = fs.Bool("verbose", false, "打印模型信息与每步调试")
	var chatMode = fs.Bool("chat", false, "交互式聊天模式（多轮对话，需模型带对话模板）")
	fs.Usage = func() {
		fmt.Println("用法: llminfer run [选项] MODEL \"prompt\"")
		fmt.Println()
		fmt.Println("示例:")
		fmt.Println("  llminfer run models/qwen2.5-0.5b.gguf \"Hello\"")
		fmt.Println("  llminfer run -temperature 0.2 -max-tokens 128 models/qwen2.5-0.5b.gguf \"The capital of France is\"")
		fmt.Println()
		fmt.Println("选项:")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 {
		fs.Usage()
		return fmt.Errorf("用法: llminfer run [--chat] MODEL [\"prompt\"]")
	}

	modelPath := fs.Arg(0)

	// 交互聊天模式：加载一次模型，多轮复用（会话里存完整历史 + 对话模板）
	if *chatMode {
		return runChat(modelPath, GenerateOptions{
			NCtx:        *nCtx,
			MaxTokens:   *maxTokens,
			Temperature: *temperature,
			TopK:        *topK,
			TopP:        *topP,
			Seed:        *seed,
			Verbose:     *verbose,
		})
	}
	if fs.NArg() < 2 {
		fs.Usage()
		return fmt.Errorf("用法: llminfer run MODEL \"prompt\"")
	}

	prompt := fs.Arg(1)

	text, n, err := Generate(modelPath, prompt, GenerateOptions{
		NCtx:        *nCtx,
		MaxTokens:   *maxTokens,
		Temperature: *temperature,
		TopK:        *topK,
		TopP:        *topP,
		Seed:        *seed,
		Verbose:     *verbose,
	})
	if err != nil {
		return err
	}

	fmt.Printf("\n== 生成结果 ==\n")
	fmt.Printf("prompt: %q\n", prompt)
	fmt.Printf("生成: %d tokens\n", n)
	fmt.Printf("输出:\n%s\n", text)
	return nil
}

// runChat 交互式聊天：加载一次会话，循环「读输入 → 生成回复」，直到输入 exit 或 EOF。
func runChat(modelPath string, opt GenerateOptions) error {
	s, err := NewChatSession(modelPath, opt)
	if err != nil {
		return err
	}
	if opt.Verbose {
		fmt.Printf("对话模板: %s\n", s.template)
	}
	fmt.Println("进入聊天模式（输入 exit 退出）")
	fmt.Println("----------------------------------------")

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("你: ")
		line, err := reader.ReadString('\n')
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			return err
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if line == "exit" || line == "quit" {
			break
		}

		text, n, err := s.Chat(line)
		if err != nil {
			fmt.Printf("生成失败: %v\n", err)
			continue
		}
		fmt.Printf("AI: %s\n", text)
		if opt.Verbose {
			fmt.Printf("  [本轮 %d tokens]\n", n)
		}
	}
	fmt.Println("bye")
	return nil
}

// humanBytes 把字节数转成人能读的大小：< KB 显示 B，否则递进 KB/MB/GB。
// 只关心首位 ~3 位有效数字，尾数截取一下即可。
func humanBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for m := n / unit; m >= unit; m /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(n)/float64(div), "KMGTPE"[exp])
}