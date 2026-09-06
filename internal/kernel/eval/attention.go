package eval

import (
	"math"

	"github.com/feiyuclaw/llminfer/internal/kernel/tensor"
)

// attention 计算注意力子块输出，写回 attBuf[n, nEmb]。
//
// 核心公式（对第 i 个 token、第 h 个头）：
//
//	score[p] = <q_i,h , k_p,kvH> / sqrt(headDim)    p = 0..pos0+i（因果掩码只往前）
//	att_i,h  = Σ_p softmax(score)_p · v_p,kvH
//
// GQA：group = HeadsCount / HeadsKV，即一个 KV 头服务 group 个 Q 头。
// kvH = h / group —— 多个 Q 头共享同一个 KV 头（省 KV 缓存显存）。
//
// 缓冲复用（M7.4）：score 行长度随位置增长，复用 ctx.scoresBuf；vout 直接累加
// 到 attBuf 对应段，免去临时缓冲。
func (ctx *Context) attention(li int, n, nEmb, kvEmb, pos0 uint32) {
	m := ctx.m
	headDim := m.HeadDim()
	group := m.HeadsCount / m.HeadsKVCount()
	invSqrt := float32(1.0 / math.Sqrt(float64(headDim)))

	for i := uint32(0); i < n; i++ {
		tPos := int(pos0 + i) // 该 token 的绝对位置
		qRow := int(i) * int(nEmb)

		// score 行复用：最长 = tPos+1，随位置增长惰性扩容
		ctx.scoresBuf = ensure(ctx.scoresBuf, tPos+1)
		scores := ctx.scoresBuf[:tPos+1]

		for h := 0; h < m.HeadsCount; h++ {
			kvH := h / group
			qp := qRow + h*headDim

			// 1) score = q·k / √d
			for p := 0; p <= tPos; p++ {
				k := ctx.kv.KHead(li, p, kvH, headDim)
				var acc float32
				for d := 0; d < headDim; d++ {
					acc += ctx.qBuf[qp+d] * k[d]
				}
				scores[p] = acc * invSqrt
			}
			// 2) softmax（数值稳定版：先减行内 max）
			tensor.SoftMax(scores, scores, tPos+1)
			// 3) 加权求和 V，直接累加进 attBuf 的 (i, h) 段（免临时 vout）
			ap := qRow + h*headDim
			for d := 0; d < headDim; d++ {
				ctx.attBuf[ap+d] = 0
			}
			for p := 0; p <= tPos; p++ {
				v := ctx.kv.VHead(li, p, kvH, headDim)
				w := scores[p]
				for d := 0; d < headDim; d++ {
					ctx.attBuf[ap+d] += w * v[d]
				}
			}
		}
	}
}