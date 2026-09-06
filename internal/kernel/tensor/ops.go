package tensor

// MatMulTransB 计算 C = A × B^T。
//
// 布局约定（与 GGUF 一致）：所有权重矩阵 B 按 [K, N] 存储，ne0=K 连续。
// 即 B 的一行是一个向量；A×B^T 等价于「A 的每行点乘 B 的每行」，便于逐行反量化。
//
//   - A 是激活（F32），形状 [M, K]
//   - B 是权重（F32/F16/量化），形状 [K, N]
//   - C 输出 [M, N]
//
// 性能：本实现是"M3 直白版"，每条输出行对整行 B 反量化再点乘，无异步、无 SIMD。
// 正确性优先；速度优化留到 M5。
func MatMulTransB(a, b, c *Tensor) {
	M, K, N := a.NE[1], a.NE[0], b.NE[1]
	ensureN(c, int(M*N))
	aF := a.AsFloat32()

	// 输出 j 列由 B 的第 j 行反量化后与 A 每行点乘得到。
	// 对量化 B：走融合反量化点积（M7.2）——直接在量化块上累加整数，每块只乘一次 scale。
	if b.Type.IsQuantized() {
		matmulQuantTransB(a, b, c)
		return
	}

	// 非量化 B（F32/F16）：整体反量化后按行乘，B 的每一行作 [N,K]
	bF := b.AsFloat32()
	for i := 0; i < int(M); i++ {
		base := i * int(K)
		for j := 0; j < int(N); j++ {
			acc := float32(0)
			brow := j * int(K)
			for k := 0; k < int(K); k++ {
				acc += aF[base+k] * bF[brow+k]
			}
			c.Floats[i*int(N)+j] = acc
		}
	}
}

// MatMul 计算 C = A × B（A:[M,K] × B:[K,N]，标准矩阵乘）。
// 本引擎的前向最常用 MatMulTransB（因为权重按 [K,N] 行主序）；此接口主要给通用算子测试用。
func MatMul(a, b, c *Tensor) {
	M, K, N := a.NE[1], a.NE[0], b.NE[1]
	ensureN(c, int(M*N))
	aF := a.AsFloat32()
	bF := b.AsFloat32()
	for i := 0; i < int(M); i++ {
		for j := 0; j < int(N); j++ {
			acc := float32(0)
			for k := 0; k < int(K); k++ {
				acc += aF[i*int(K)+k] * bF[k*int(N)+j]
			}
			c.Floats[i*int(N)+j] = acc
		}
	}
}

// ensureN 确保输出张量有至少 n 个 float32 槽位。
func ensureN(c *Tensor, n int) {
	if len(c.Floats) < n {
		c.Floats = make([]float32, n)
	}
}