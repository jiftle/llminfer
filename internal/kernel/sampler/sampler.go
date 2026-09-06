// Package sampler 提供自回归生成时的 token 采样。
//
// 一句话：前向输出一排"词表分数"（logits），采样器按策略挑一个 token 作为下一个词。
// 挑法可以是贪心（永远取分数最高），或带随机性（温度/top-k/top-p 让输出更多样）。
//
// 本版直白优先：用全量排序 + 累计概率，逻辑最清晰（vocab=151936 排序略慢，优化留 M6）。
package sampler

import (
	"math"
	"math/rand"
	"sort"
)

// LogitData 单个 token 的 logit 与 softmax 概率。
type LogitData struct {
	ID    uint32
	Logit float64
	P     float64
}

// Config 采样超参数。
type Config struct {
	Temperature float64 // ≤0 时取 argmax（贪心）；>0 时为温度缩放
	TopK        int     // 保留前 k 个最高分 token；≤0 关闭
	TopP        float64 // 核采样阈值 (0,1]；1 关闭
}

// Sampler 递增随机源，保证同 seed 输出可复现。
type Sampler struct {
	rng *rand.Rand
}

// NewSampler 创建采样器。
func NewSampler(seed int64) *Sampler {
	return &Sampler{rng: rand.New(rand.NewSource(seed))}
}

// Sample 从 logits 采样一个 token id，返回其 id。
// 流程：
//
//	温度 → softmax → top-k 截断 → top-p 截断 → 按累计概率抽签
func (s *Sampler) Sample(logits []float32, cfg Config) uint32 {
	if len(logits) == 0 {
		return 0
	}

	// 0) 组装 (id, logit) 表
	data := make([]LogitData, len(logits))
	for i, v := range logits {
		data[i] = LogitData{ID: uint32(i), Logit: float64(v)}
	}

	// 1) 贪心：温度 ≤0 直接取 argmax，不走随机
	if cfg.Temperature <= 0 {
		best := data[0]
		for _, d := range data[1:] {
			if d.Logit > best.Logit {
				best = d
			}
		}
		return best.ID
	}

	// 2) 温度缩放：score' = score / T（T>1 拉平分布更多样，T<1 更尖锐）
	for i := range data {
		data[i].Logit /= cfg.Temperature
	}

	// 3) softmax（数值稳定：先减行内 max 防 e 溢出）
	maxL := data[0].Logit
	for _, d := range data[1:] {
		if d.Logit > maxL {
			maxL = d.Logit
		}
	}
	var sum float64
	for i := range data {
		data[i].P = math.Exp(data[i].Logit - maxL)
		sum += data[i].P
	}
	for i := range data {
		data[i].P /= sum
	}

	// 4) 按概率降序排序（后续 top-k/top-p 都基于此序）
	sort.Slice(data, func(i, j int) bool { return data[i].P > data[j].P })

	// 5) top-k：只保留前 k 个
	if cfg.TopK > 0 && cfg.TopK < len(data) {
		data = data[:cfg.TopK]
	}

	// 6) top-p：累积概率达到阈值即截断（允许至少 1 个 token）
	if cfg.TopP > 0 && cfg.TopP < 1.0 {
		cum := 0.0
		keep := 1
		for keep < len(data) {
			cum += data[keep-1].P
			if cum >= cfg.TopP {
				break
			}
			keep++
		}
		data = data[:keep]
	}

	// 7) 抽签：均匀随机数落在哪个累计区间就选谁
	r := s.rng.Float64()
	acc := 0.0
	for _, d := range data {
		acc += d.P
		if r <= acc {
			return d.ID
		}
	}
	// 兜底（浮点累积误差导致 r 略大时）
	return data[len(data)-1].ID
}
