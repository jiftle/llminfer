package sampler

import (
	"math"
	"testing"
)

func TestGreedySelectsMax(t *testing.T) {
	s := NewSampler(42)
	logits := []float32{1.0, 5.0, 3.0, 2.0}
	id := s.Sample(logits, Config{Temperature: -1})
	if id != 1 {
		t.Fatalf("贪心应选 index=1 (logit=5.0)，实际选了 %d", id)
	}
}

func TestTemperatureZeroIsGreedy(t *testing.T) {
	s := NewSampler(42)
	logits := []float32{-10, 10, -10}
	id := s.Sample(logits, Config{Temperature: 0})
	if id != 1 {
		t.Fatalf("温度 0 = 贪心，应选 index=1，实际选了 %d", id)
	}
}

func TestTopKReducesCandidates(t *testing.T) {
	s := NewSampler(1)
	// logit[100] 最大，top-k=3 应只保留它和两个次大
	logits := make([]float32, 1000)
	for i := range logits {
		logits[i] = -100
	}
	logits[100] = 100

	// 跑 50 次，top-k=3 应只选 index=100（因为其他两个 -100，softmax 后几乎为 0）
	count := 0
	for i := 0; i < 50; i++ {
		id := s.Sample(logits, Config{Temperature: 1.0, TopK: 3})
		if id == 100 {
			count++
		}
	}
	if count < 45 {
		t.Fatalf("Top-K=3 时几乎只应选 index=100，实际命中 %d/50", count)
	}
}

func TestTopPReducesCandidates(t *testing.T) {
	s := NewSampler(1)
	// logit[0] 占 99%，top-p=0.8 应只保留它
	logits := []float32{100, -100, -100}
	count := 0
	for i := 0; i < 50; i++ {
		id := s.Sample(logits, Config{Temperature: 1.0, TopP: 0.8})
		if id == 0 {
			count++
		}
	}
	if count < 45 {
		t.Fatalf("Top-P=0.8 时几乎只应选 index=0，实际命中 %d/50", count)
	}
}

func TestSeedReproducible(t *testing.T) {
	logits := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	ids1 := make([]uint32, 100)
	ids2 := make([]uint32, 100)

	s1 := NewSampler(123)
	for i := range ids1 {
		ids1[i] = s1.Sample(logits, Config{Temperature: 0.8, TopK: 5, TopP: 0.9})
	}
	s2 := NewSampler(123)
	for i := range ids2 {
		ids2[i] = s2.Sample(logits, Config{Temperature: 0.8, TopK: 5, TopP: 0.9})
	}
	for i := range ids1 {
		if ids1[i] != ids2[i] {
			t.Fatalf("同种子应输出同序列，第 %d 位不同: %d vs %d", i, ids1[i], ids2[i])
		}
	}
}

func TestHighTempDistributes(t *testing.T) {
	s := NewSampler(1)
	logits := []float32{1, 1, 1, 1, 1} // 均匀
	hits := make([]int, 5)
	for i := 0; i < 1000; i++ {
		id := s.Sample(logits, Config{Temperature: 100.0})
		hits[id]++
	}
	// 均匀分布下每个应接近 200±50
	for i, c := range hits {
		if c < 150 || c > 250 {
			t.Fatalf("高温度下应近均匀分布，index %d 命中 %d 次（期望 ~200）", i, c)
		}
	}
	_ = math.Float32bits // 保编译
}
