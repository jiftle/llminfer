// Package cmd 负责 CLI 子命令分发（run / bench 等）。
package cmd

import (
	"flag"
	"fmt"
	"time"

	"github.com/feiyuclaw/llminfer/internal/kernel/eval"
	"github.com/feiyuclaw/llminfer/internal/kernel/gguf"
	"github.com/feiyuclaw/llminfer/internal/kernel/model"
	"github.com/feiyuclaw/llminfer/internal/kernel/sampler"
	"github.com/feiyuclaw/llminfer/internal/kernel/tokenizer"
)

// bench 子命令：对模型做简单性能基准。
// 测两块速度：prefill（一次性吞 prompt）和 decode（逐个生成），都以 tokens/s 计。
// 用法: llminfer bench [选项] MODEL
func bench(args []string) error {
	fs := flag.NewFlagSet("bench", flag.ContinueOnError)
	var nTok = fs.Int("n-tokens", 32, "decode 阶段生成多少个 token")
	var ppLen = fs.Int("prompt-tokens", 128, "prefill 阶段用多少个 token 当 prompt")
	var seed = fs.Int64("seed", 42, "随机种子")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 {
		return fmt.Errorf("用法: llminfer bench [选项] MODEL")
	}
	modelPath := fs.Arg(0)

	// 加载模型 + tokenizer（与 Generate 相同的两段式：模型一次、元数据一次）
	m, err := model.Load(modelPath)
	if err != nil {
		return fmt.Errorf("加载模型失败: %w", err)
	}
	t0 := time.Now()
	gf, err := gguf.ReadFile(modelPath)
	if err != nil {
		return fmt.Errorf("读取元数据失败: %w", err)
	}
	defer gf.Close()
	tok, err := tokenizer.NewFromGGUF(gf, m.Arch)
	if err != nil {
		return fmt.Errorf("初始化分词器失败: %w", err)
	}
	loadDur := time.Since(t0)

	// 生成固定长度的测试 prompt：反复编码一句开场白直到够长
	var prompt string
	for i := 0; len(tok.Encode(prompt)) < *ppLen; i++ {
		prompt += "The quick brown fox jumps over the lazy dog. "
	}
	promptIDs := tok.Encode(prompt)[:*ppLen]
	if tok.BOS >= 0 {
		promptIDs = append([]uint32{uint32(tok.BOS)}, promptIDs...) // 加 BOS 让模型进入状态
	}
	fmt.Printf("模型: %s（%d 层, emb=%d）\n", m.Arch, m.LayersCount, m.EmbeddingSize)
	fmt.Printf("加载耗时: %d ms\n", loadDur.Milliseconds())

	// prefill：一次前向吞掉整个 prompt（warmup 一次避免首跑含分配开销）
	ctx := eval.NewContext(m, 0)
	ctx.Forward(promptIDs)
	t0 = time.Now()
	ctx.Reset()
	ctx.Forward(promptIDs)
	ppDur := time.Since(t0)
	ppSpeed := float64(len(promptIDs)) / ppDur.Seconds()

	// decode：逐 token 生成，问引擎 rate
	sm := sampler.NewSampler(*seed)
	cfg := sampler.Config{Temperature: 0, TopK: 0, TopP: 1}
	stopIDs := tok.ChatStopIDs()
	ctx.Reset()
	logits := ctx.Forward(promptIDs)
	start := time.Now()
	nGen := 0
	for i := 0; i < *nTok; i++ {
		id := sm.Sample(logits, cfg)
		if stopIDs[id] {
			break
		}
		nGen++
		logits = ctx.Forward([]uint32{id})
	}
	tgDur := time.Since(start)
	tgSpeed := float64(nGen) / tgDur.Seconds()

	fmt.Printf("prefill: %d tokens in %.1f ms  → %.1f tokens/s\n", len(promptIDs), ppDur.Seconds()*1000, ppSpeed)
	fmt.Printf("decode:  %d tokens in %.1f ms  → %.1f tokens/s\n", nGen, tgDur.Seconds()*1000, tgSpeed)
	fmt.Printf("（本机 CPU 单线程，无 SIMD 加速）\n")
	return nil
}