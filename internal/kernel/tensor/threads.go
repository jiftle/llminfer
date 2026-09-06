package tensor

import "runtime"

// DefaultThreads 返回默认并行线程数：本机核数的 2/3（向下取整，至少 1）。
// 原则：推理吃满所有核会拖慢系统/编辑器响应，留 1/3 余量。
// 例：12 核 → 8；8 核 → 5；4 核 → 2。
func DefaultThreads() int {
	n := runtime.NumCPU() * 2 / 3
	if n < 1 {
		return 1
	}
	return n
}

// numThreads 解析线程参数；未指定或非法时用默认（核数的 2/3）。
// 显式传的线程数超过 CPU 核数时按核数裁剪（goroutine 太多反而互相争抢）。
func numThreads(threads []int) int {
	if len(threads) == 0 || threads[0] < 1 {
		return DefaultThreads()
	}
	if n := runtime.NumCPU(); threads[0] > n {
		return n
	}
	return threads[0]
}