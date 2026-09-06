package tensor

import "sync"

// MatMulTransB 计算 C = A × B^T。
//
// 布局约定（与 GGUF 一致）：所有权重矩阵 B 按 [K, N] 存储，ne0=K 连续。
// 即 B 的一行是一个向量；A×B^T 等价于「A 的每行点乘 B 的每行」，便于逐行反量化。
//
//   - A 是激活（F32），形状 [M, K]
//   - B 是权重（F32/F16/量化），形状 [K, N]
//   - C 输出 [M, N]
//   - threads（可选）并行 worker 数，缺省 8（12 核留余量）；1=纯串行。
//
// 并行策略（M7.3）：量化与 F32 都按「输出列/行」分片，每 worker 私有缓冲零竞争。
func MatMulTransB(a, b, c *Tensor, threads ...int) {
	nT := numThreads(threads)
	M, K, N := a.NE[1], a.NE[0], b.NE[1]
	ensureN(c, int(M*N))
	aF := a.AsFloat32()

	// 量化权重：融合反量化点积（M7.2）
	if b.Type.IsQuantized() {
		matmulQuantTransB(a, b, c, nT)
		return
	}

	// 非量化 B（F32/F16）：整体反量化后按行乘
	bF := b.AsFloat32()
	if nT <= 1 || int(M) <= 1 {
		// 串行
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
		return
	}

	// 并行：按输出行 i 分片，每 worker 私有一段行区间
	var wg sync.WaitGroup
	chunk := (int(M) + nT - 1) / nT
	for w := 0; w < nT; w++ {
		i0 := w * chunk
		i1 := i0 + chunk
		if i1 > int(M) {
			i1 = int(M)
		}
		if i0 >= int(M) {
			break
		}
		wg.Add(1)
		go func(i0, i1 int) {
			defer wg.Done()
			for i := i0; i < i1; i++ {
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
		}(i0, i1)
	}
	wg.Wait()
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