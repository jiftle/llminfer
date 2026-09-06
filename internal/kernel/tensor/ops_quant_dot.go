// 融合反量化点积：量化权重矩阵乘的快速路径。
//
// 里程碑：M7.2 / M7.3。
//
// 思路（对齐 goLLM）：MatMulTransB 在量化权重上做 C = A × B^T，
// 传统做法先 DequantRow 把 B 整行反量化进 rowBuf 再读回乘累加（两次内存往返），
// 且每个元素乘一次 scale。融合版直接在量化块上累加整数 q：
//
//	dot = Σ_k a[k]*q[k]      （先对整数累加，每块只乘一次 scale）
//
// M7.3 起支持多线程：按输出列 j 分片，每 worker 私有 acc/rowBuf。
package tensor

import "sync"

// dotQuantCol 计算 B 的第 j 列与 aF（M×K）逐行点积，写入 c[i*N+j]。
// 仅对 Nelements≥1 且 M≥1 生效；acc 为调用方复用的 M 长累加缓冲，避免每列分配。
// rowBuf 仅在 fallback 分支使用（未融合类型反量化中转）。
func dotQuantCol(aF []float32, m, k uint32, b *Tensor, j uint32, cF []float32, n uint32, acc, rowBuf []float32) {
	switch b.Type {
	case TYPE_Q8_0:
		dotQ80(aF, m, k, b, j, cF, n, acc)
	case TYPE_Q5_0:
		dotQ50(aF, m, k, b, j, cF, n, acc)
	case TYPE_Q6_K:
		dotQ6K(aF, m, k, b, j, cF, n, acc)
	case TYPE_Q4_K:
		dotQ4K(aF, m, k, b, j, cF, n, acc)
	default:
		dotQuantFallback(aF, m, k, b, j, cF, n, acc, rowBuf)
	}
}

// Q8_0 布局：[d 2B fp16][qs 32B int8]，块 32，x = q*d
// 融合：整块先对 q 累加，最后一块乘 d。
func dotQ80(aF []float32, m, k uint32, b *Tensor, j uint32, cF []float32, n uint32, acc []float32) {
	const blk = 32
	const blkBytes = 2 + blk // 34 字节/块
	rowOff := int(j) * (int(k) / blk) * blkBytes
	data := b.Data

	for i := uint32(0); i < m; i++ {
		acc[i] = 0
	}
	for kb := 0; kb < int(k)/blk; kb++ {
		base := rowOff + kb*blkBytes
		d := F16ToF32(data[base], data[base+1])
		qs := data[base+2 : base+blkBytes]
		for i := uint32(0); i < m; i++ {
			var s float32
			ab := aF[int(i)*int(k)+kb*blk:]
			for l := 0; l < blk; l++ {
				s += ab[l] * float32(int8(qs[l]))
			}
			acc[i] += s * d // 整块只乘一次 scale
		}
	}
	for i := uint32(0); i < m; i++ {
		cF[int(i)*int(n)+int(j)] = acc[i]
	}
}

// Q5_0 布局：[d 2B fp16][qh 4B][qs 16B]，块 32，x = q*d，q 有符号 5bit(-16..15)
func dotQ50(aF []float32, m, k uint32, b *Tensor, j uint32, cF []float32, n uint32, acc []float32) {
	const blk = 32
	const blkBytes = 2 + 4 + blk/2 // 22 字节/块
	rowOff := int(j) * (int(k) / blk) * blkBytes
	data := b.Data

	for i := uint32(0); i < m; i++ {
		acc[i] = 0
	}
	for kb := 0; kb < int(k)/blk; kb++ {
		base := rowOff + kb*blkBytes
		d := F16ToF32(data[base], data[base+1])
		qh := uint32(data[base+2]) | uint32(data[base+3])<<8 | uint32(data[base+4])<<16 | uint32(data[base+5])<<24
		qs := data[base+6 : base+6+blk/2]
		for i := uint32(0); i < m; i++ {
			var s float32
			ab := aF[int(i)*int(k)+kb*blk:]
			for l := 0; l < 16; l++ {
				xh0 := ((qh >> (uint(l) + 0)) << 4) & 0x10
				xh1 := (qh >> (uint(l) + 12)) & 0x10
				q0 := int32(qs[l]&0x0F | byte(xh0)) - 16
				q1 := int32(qs[l]>>4 | byte(xh1)) - 16
				s += ab[l]*float32(q0) + ab[l+16]*float32(q1)
			}
			acc[i] += s * d
		}
	}
	for i := uint32(0); i < m; i++ {
		cF[int(i)*int(n)+int(j)] = acc[i]
	}
}

// Q6_K 布局（超块 256）:[ql 128B][qh 64B][scales 16B][d 2B fp16]，块 256
// 16 组 16 元素子块，每组 scale = d * sc（sc 在 scales 里按 2 元素间隔取）。
func dotQ6K(aF []float32, m, k uint32, b *Tensor, j uint32, cF []float32, n uint32, acc []float32) {
	const blk = 256
	const blkBytes = 210
	rowOff := int(j) * (int(k) / blk) * blkBytes
	data := b.Data

	for i := uint32(0); i < m; i++ {
		acc[i] = 0
	}
	for kb := 0; kb < int(k)/blk; kb++ {
		base := rowOff + kb*blkBytes
		d := F16ToF32(data[base+208], data[base+209])
		ql := data[base : base+128]
		qh := data[base+128 : base+192]
		sc := data[base+192 : base+208]

		for n128 := 0; n128 < 2; n128++ {
			qlo := ql[n128*64 : n128*64+64]
			qho := qh[n128*32 : n128*32+32]
			sco := sc[n128*8 : n128*8+8]
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := int32(int8(qlo[l]&0x0F | ((qho[l]>>0)&3)<<4)) - 32
				q2 := int32(int8(qlo[l+32]&0x0F | ((qho[l]>>2)&3)<<4)) - 32
				q3 := int32(int8(qlo[l]>>4 | ((qho[l]>>4)&3)<<4)) - 32
				q4 := int32(int8(qlo[l+32]>>4 | ((qho[l]>>6)&3)<<4)) - 32
				s1 := d * float32(int8(sco[is+0]))
				s2 := d * float32(int8(sco[is+2]))
				s3 := d * float32(int8(sco[is+4]))
				s4 := d * float32(int8(sco[is+6]))
				for i := uint32(0); i < m; i++ {
					ab := aF[int(i)*int(k)+kb*blk+n128*128+l:]
					acc[i] += ab[0]*float32(q1)*s1 + ab[32]*float32(q2)*s2 +
						ab[64]*float32(q3)*s3 + ab[96]*float32(q4)*s4
				}
			}
		}
	}
	for i := uint32(0); i < m; i++ {
		cF[int(i)*int(n)+int(j)] = acc[i]
	}
}

// Q4_K 布局（超块 256）:[d 2B][dmin 2B][scales 12B][qs 128B]，块 256
// 8 组 32 元素子块，每子块 scale = d*sc，偏移 = -dmin*min：
// x = q*d*sc - min*dmin。用 16 项 LUT 预计算每个子块的值 = q*d - min。
func dotQ4K(aF []float32, m, k uint32, b *Tensor, j uint32, cF []float32, n uint32, acc []float32) {
	const blk = 256
	const blkBytes = 144
	rowOff := int(j) * (int(k) / blk) * blkBytes
	data := b.Data

	for i := uint32(0); i < m; i++ {
		acc[i] = 0
	}
	for kb := 0; kb < int(k)/blk; kb++ {
		base := rowOff + kb*blkBytes
		d := F16ToF32(data[base], data[base+1])
		dmin := F16ToF32(data[base+2], data[base+3])
		scales := data[base+4 : base+16]
		qs := data[base+16 : base+144]

		for seg := 0; seg < 4; seg++ {
			is := 2 * seg
			sc0, m0 := getScaleMinK4(is, scales)
			sc1, m1 := getScaleMinK4(is+1, scales)
			d1, m1f := d*float32(sc0), dmin*float32(m0)
			d2, m2f := d*float32(sc1), dmin*float32(m1)
			// LUT 预计算：值 = q*d - min，每子块 16×2 次，内层 M 行循环复用
			var lut0, lut1 [16]float32
			for q := 0; q < 16; q++ {
				lut0[q] = d1*float32(q) - m1f
				lut1[q] = d2*float32(q) - m2f
			}
			qsOff := 32 * seg  // qs 解码索引（每 seg 32 字节）
			dstOff := 64 * seg // 反量化输出位置（每 seg 64 值）
			for i := uint32(0); i < m; i++ {
				var s float32
				ab := aF[int(i)*int(k)+kb*blk+dstOff:]
				for l := 0; l < 32; l++ {
					qb := qs[qsOff+l]
					s += ab[l]*lut0[qb&0x0F] + ab[l+32]*lut1[qb>>4]
				}
				acc[i] += s
			}
		}
	}
	for i := uint32(0); i < m; i++ {
		cF[int(i)*int(n)+int(j)] = acc[i]
	}
}

// dotQuantFallback 回退：未融合类型整行反量化后标量内积。rowBuf 为复用缓冲。
func dotQuantFallback(aF []float32, m, k uint32, b *Tensor, j uint32, cF []float32, n uint32, acc, rowBuf []float32) {
	if err := b.DequantRow(j, rowBuf); err != nil {
		panic(err)
	}
	for i := uint32(0); i < m; i++ {
		var s float32
		ab := aF[int(i)*int(k):]
		for kk := uint32(0); kk < k; kk++ {
			s += ab[kk] * rowBuf[kk]
		}
		cF[int(i)*int(n)+int(j)] = s
	}
}

// matmulQuantTransB 量化权重专用矩阵乘：C = A × B^T。
// A:[M,K] F32，B:[K,N] 量化，C:[M,N]。threads 为并行 worker 数（M7.3）。
//
// 两条路径权衡（M 是批量行数）：
//   - M=1（decode 单 token）：每次点积只有一行 a，融合版直接在量化块上累加整数、
//     每块只乘一次 scale，省掉 rowBuf 内存往返。按列 j 分片并行。
//   - M>1（prefill 批量）：每列 DequantRow 一次进 rowBuf，M 行复用（只读一遍量化权重）。
//     按列 j 分片并行，每 worker 私有 rowBuf 零竞争。
func matmulQuantTransB(a, b, c *Tensor, threads int) {
	M, K, N := a.NE[1], a.NE[0], b.NE[1]
	ensureN(c, int(M*N))
	aF := a.AsFloat32()
	cF := c.Floats

	// 每个 worker 处理一段输出列 [j0, j1)，私有缓冲。
	work := func(j0, j1 uint32, acc, rowBuf []float32) {
		switch {
		case M == 1:
			// decode：融合点积，每列单行 dot
			for j := j0; j < j1; j++ {
				dotQuantCol(aF, 1, K, b, j, cF, N, acc, rowBuf)
			}
		default:
			// prefill：每列反量化一次进 rowBuf，M 行复用
			for j := j0; j < j1; j++ {
				if err := b.DequantRow(j, rowBuf); err != nil {
					panic(err)
				}
				for i := uint32(0); i < M; i++ {
					s := float32(0)
					base := int(i) * int(K)
					for k := uint32(0); k < K; k++ {
						s += aF[base+int(k)] * rowBuf[k]
					}
					cF[int(i)*int(N)+int(j)] = s
				}
			}
		}
	}

	// 并行或串行
	if threads <= 1 || N <= 1 {
		acc := poolGet(int(M))
		rowBuf := poolGet(int(K))
		work(0, N, acc, rowBuf)
		poolPut(acc)
		poolPut(rowBuf)
		return
	}

	var wg sync.WaitGroup
	chunk := (N + uint32(threads) - 1) / uint32(threads)
	bufs := allocWorkerBufs(threads, int(M), int(K))
	defer freeWorkerBufs(bufs)
	for w := 0; w < threads; w++ {
		j0 := uint32(w) * chunk
		j1 := j0 + chunk
		if j1 > N {
			j1 = N
		}
		if j0 >= N {
			break
		}
		wg.Add(1)
		go func(w int, j0, j1 uint32) {
			defer wg.Done()
			work(j0, j1, bufs[w].acc, bufs[w].rowBuf)
		}(w, j0, j1)
	}
	wg.Wait()
}

// workerBuf 每 worker 私有的累加/行缓冲。
type workerBuf struct {
	acc    []float32
	rowBuf []float32
}

// allocWorkerBufs 从 sync.Pool 取线程数个 worker 缓冲（M7.4：避免每次 matmul 都 make）。
func allocWorkerBufs(n, m, k int) []workerBuf {
	bufs := make([]workerBuf, n)
	for w := 0; w < n; w++ {
		bufs[w].acc = poolGet(m)
		bufs[w].rowBuf = poolGet(k)
	}
	return bufs
}

// freeWorkerBufs 归还缓冲到池。
func freeWorkerBufs(bufs []workerBuf) {
	for _, b := range bufs {
		poolPut(b.acc)
		poolPut(b.rowBuf)
	}
}

// float32Pool 复用可变长 float32 工作缓冲（各 matmul 内部临时用）。
var float32Pool = sync.Pool{New: func() any { return make([]float32, 0) }}

// poolGet 取长度 n 的缓冲：池里有足够容量的就复用，否则新建。
func poolGet(n int) []float32 {
	if s, ok := float32Pool.Get().([]float32); ok && cap(s) >= n {
		return s[:n]
	}
	return make([]float32, n)
}

// poolPut 归还缓冲（长度清零，保留容量待下次扩容）。超大缓冲不入池。
func poolPut(s []float32) {
	if cap(s) > 1<<20 { // >1M 元素（4MB）不回收，避免霸占内存
		return
	}
	float32Pool.Put(s[:0])
}