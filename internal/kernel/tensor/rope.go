package tensor

import "math"

// RoPE 对激活热路旋转（Rotary Position Embedding）。
//
// 输入布局：buf 长度 = n(行数/token数) × rowLen(每行向量长度)，
// 每行内部按 head 切块：[heads][headDim]，headDim = rowLen / heads。
//
// 位置由 pos0 起递增：第 i 个 token 的绝对位置 = pos0 + i。
// freqBase 默认 10000（qwen2 也是），freqScale 默认 1。
//
// 两种旋转配对（隐形知识，极易踩坑）：
//   - NORMAL：相邻对 (2j, 2j+1) 旋转 — LLaMA 系列
//   - NEOX  ：前/后半在 (j, j+headDim/2) 配对旋转 — Qwen2/GPT-NeoX 等
//
// qwen2 走 NEOX！这决定注意力 score 完全对不上，错误很隐蔽。
//
// 频率：theta_j = freqBase^(-2j/headDim)，角度 = pos * theta
// 旋转：x' = (x0*cos - x1*sin, x0*sin + x1*cos)
func RoPE(buf []float32, n, rowLen, heads, pos0 uint32, ropeNeox bool, freqBase, freqScale float32) {
	headDim := int(rowLen / heads)
	// 预计算 base 的幂：第 j 维对应 theta_j。headDim 一般不超过 128，提前算好一次。
	half := headDim / 2
	thetas := make([]float32, half)
	for j := 0; j < half; j++ {
		// 频率随维度上升递减：NEOX 用 -2j/headDim；NORMAL 用 -j/headDim
		var e float64
		if ropeNeox {
			e = -2 * float64(j) / float64(headDim)
		} else {
			e = -float64(j) / float64(headDim)
		}
		thetas[j] = float32(math.Pow(float64(freqBase), e))
	}

	for i := uint32(0); i < n; i++ {
		pos := float64(pos0+i) * float64(freqScale)
		row := int(i) * int(rowLen)
		for h := 0; h < int(heads); h++ {
			off := row + h*headDim
			if ropeNeox {
				// NEOX：槽 j 与槽 j+half 配对
				for j := 0; j < half; j++ {
					theta := float64(thetas[j]) * pos
					c := float32(math.Cos(theta))
					s := float32(math.Sin(theta))
					x0 := buf[off+j]
					x1 := buf[off+j+half]
					buf[off+j] = x0*c - x1*s
					buf[off+j+half] = x0*s + x1*c
				}
			} else {
				// NORMAL：槽 2j 与 2j+1 配对
				for j := 0; j < half; j++ {
					theta := float64(thetas[j]) * pos
					c := float32(math.Cos(theta))
					s := float32(math.Sin(theta))
					x0 := buf[off+2*j]
					x1 := buf[off+2*j+1]
					buf[off+2*j] = x0*c - x1*s
					buf[off+2*j+1] = x0*s + x1*c
				}
			}
		}
	}
}