package tensor

import (
	"math"
	"testing"
)

// TestF16 验证 fp16→fp32 转换（含次正规数与非标准值）。
func TestF16(t *testing.T) {
	cases := []struct {
		lo, hi byte
		want   float32
	}{
		{0x00, 0x3C, 1.0},     // ±1.0
		{0x00, 0x40, 2.0},     // 2.0
		{0x00, 0xC0, -2.0},    // -2.0
		{0x00, 0x38, 0.5},     // 0.5
		{0x9A, 0x39, 0.7002},  // 0.7 
		{0x33, 0x3B, 0.89990}, // 常见压测值
		{0x00, 0x00, 0.0},     // 0
		{0x01, 0x00, 6e-8},    // 最次正规数 5.96e-8
	}
	for _, c := range cases {
		got := F16ToF32(c.lo, c.hi)
		if math.Abs(float64(got-c.want)) > 1e-4 {
			t.Errorf("F16ToF32(0x%02X%02X) = %v, want≈%v", c.hi, c.lo, got, c.want)
		}
	}
}

// TestQ8_0RoundTrip 捏造一个 Q8_0 块，验证反量化公式 x = q*d。
func TestQ8_0RoundTrip(t *testing.T) {
	// 手工构造：d=0.5，qs=[1..32]（有符号）
	blk := make([]byte, 34)
	blk[0], blk[1] = 0x00, 0x38 // fp16 0.5
	for i := 0; i < 32; i++ {
		blk[2+i] = byte(i - 16) // -16..15
	}
	dst := make([]float32, 32)
	if err := DequantBlock(TYPE_Q8_0, blk, dst); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 32; i++ {
		want := float32(i-16) * 0.5
		if math.Abs(float64(dst[i]-want)) > 1e-6 {
			t.Errorf("Q8_0[%d] = %v, want %v", i, dst[i], want)
		}
	}
}

// TestMatMulTransB 手算一个小矩阵乘对比。
// 布局说明：B 按 [K, N] 行主序，B 的第 j 行（长 K）点乘 A 的每行得到 C 的第 j 列。
func TestMatMulTransB(t *testing.T) {
	// A: [2,3] = [[1,2,3],[4,5,6]]
	// B: [3,2]，B 行0=[1,0,1]，行1=[0,1,1]
	// C[i][j] = A行i · B行j
	// C[0][0]=1*1+2*0+3*1=4  C[0][1]=0+2+3=5
	// C[1][0]=4+0+6=10        C[1][1]=0+5+6=11
	a := NewTensor(TYPE_F32, 3, 2)
	a.Floats = []float32{1, 2, 3, 4, 5, 6}
	b := NewTensor(TYPE_F32, 3, 2)
	b.Floats = []float32{1, 0, 1, 0, 1, 1}
	c := NewTensor(TYPE_F32, 2, 2)
	MatMulTransB(a, b, c)
	want := []float32{4, 5, 10, 11}
	for i := range want {
		if math.Abs(float64(c.Floats[i]-want[i])) > 1e-6 {
			t.Fatalf("C[%d] = %v, want %v", i, c.Floats[i], want[i])
		}
	}
}

// TestRMSNorm 验证归一化零均值特性：sum(x^2)=维度数（规范化后）。
func TestRMSNorm(t *testing.T) {
	// 输入全 2，w 全 1：x_hat = 2/2 = 1，norm(1)=1，sum = cols
	x := []float32{2, 2, 2, 2}
	w := []float32{1, 1, 1, 1}
	RMSNorm(x, w, 1e-6)
	for _, v := range x {
		if math.Abs(float64(v-1)) > 1e-6 {
			t.Fatalf("RMSNorm 结果 %v", x)
		}
	}
}

// TestSoftmax 验证 softmax 和为 1。
func TestSoftmax(t *testing.T) {
	src := []float32{3, 1, 0.2}
	dst := make([]float32, len(src))
	SoftMax(src, dst, len(src))
	var sum float32
	for _, v := range dst {
		sum += v
	}
	if math.Abs(float64(sum-1)) > 1e-6 {
		t.Fatalf("softmax sum = %v", sum)
	}
}