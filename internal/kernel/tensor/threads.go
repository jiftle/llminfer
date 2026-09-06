package tensor

import "runtime"

// 默认并行线程数：本机 12 核，留 4 核给系统/编辑器，用 8。
// 想改可加 -threads flag 覆盖（见 cmd）。
const DefaultThreads = 8

// numThreads 解析线程参数；未指定或非法时用默认。
// 规则：超过 CPU 核数的按核数裁剪（goroutine 太多反而互相争抢）。
func numThreads(threads []int) int {
	if len(threads) == 0 || threads[0] < 1 {
		return DefaultThreads
	}
	if n := runtime.NumCPU(); threads[0] > n {
		return n
	}
	return threads[0]
}