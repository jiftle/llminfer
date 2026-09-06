package tensor

import "math"

// RMSNorm 对 [rows, cols] 矩阵按行做 RMS 归一化，再乘以可学习权重 w（长度=cols）。
//
// 公式（why 拆解）：
//
//	RMS(x) = sqrt( mean(x²) + eps )
//	y = x / RMS(x) * w
//
// 与 LayerNorm 的差别：不做减均值（LLM 里均值为 0 时信息冗余，省一次统计）。
// `w` 是逐通道缩放，训练学出来的。就地写回 x 或输出到 dst。
//
// 注意 n 关系：x 一维扁平（len=rows*cols）。行内 cols 个是连续的。
func RMSNorm(x []float32, w []float32, eps float32) {
	if len(x)%len(w) != 0 {
		panic("tensor.RMSNorm: 行宽与权重长度不匹配")
	}
	rows := len(x) / len(w)
	cols := len(w)
	for r := 0; r < rows; r++ {
		base := r * cols
		// 1) 行内平方均值
		var sum float32
		for c := 0; c < cols; c++ {
			v := x[base+c]
			sum += v * v
		}
		// 2) RMS = sqrt(mean + eps)，取倒数再乘（省一次除法）
		inv := 1 / float32(math.Sqrt(float64(sum/float32(cols)+eps)))
		// 3) 归一化并套上可学习权重
		for c := 0; c < cols; c++ {
			x[base+c] = x[base+c] * inv * w[c]
		}
	}
}

// SiLU 逐元素 Swish 激活：silu(x) = x / (1 + e^-x)
// 向量化应用于整段数据。
func SiLU(x []float32) {
	for i, v := range x {
		x[i] = v / (1 + float32(math.Exp(-float64(v))))
	}
}

// SoftMax 按行做 softmax（行宽 = cols），数值稳定版（先减行内 max 防止 e 溢出）。
// 输入输出都可原地（dst 可指向 src 同一段）。
func SoftMax(src, dst []float32, cols int) {
	// 防误用：一行最后余数丢弃（见尾部注释）
	rows := len(src) / cols
	tail := len(src) % cols

	for r := 0; r < rows; r++ {
		base := r * cols
		// 1) 行内 max（指数前的数值稳定）
		mx := src[base]
		for c := 1; c < cols; c++ {
			if s := src[base+c]; s > mx {
				mx = s
			}
		}
		// 2) e^(x-mx) 累加
		var sum float32
		for c := 0; c < cols; c++ {
			dst[base+c] = float32(math.Exp(float64(src[base+c] - mx)))
			sum += dst[base+c]
		}
		// 3) 除以和，得概率
		inv := 1 / sum
		for c := 0; c < cols; c++ {
			dst[base+c] *= inv
		}
	}
	// 尾行不完整数据（正常不会出现），清零避免读到旧值。
	if tail != 0 {
		for i := rows * cols; i < len(dst); i++ {
			dst[i] = 0
		}
	}
}