package cache

import "testing"

// TestWriteTruncate 验证 Write/Truncate 语义：Truncate 后 len 缩短，后续 Write 覆盖旧位置。
func TestWriteTruncate(t *testing.T) {
	c := New(1, 2, 8) // 1 层, step=2, 最多 8 位置
	k := []float32{1, 2}
	v := []float32{3, 4}
	c.Write(0, 0, k, v)
	c.Write(0, 1, k, v)
	c.Write(0, 2, k, v)
	if c.Len() != 3 {
		t.Fatalf("Len=%d, want 3", c.Len())
	}

	// 截断到 1：位置 1、2 失效，但数据仍在（等待覆盖）
	c.Truncate(1)
	if c.Len() != 1 {
		t.Fatalf("Truncate 后 Len=%d, want 1", c.Len())
	}

	// 从位置 1 重写新值，应覆盖旧数据
	c.Write(0, 1, []float32{9, 9}, []float32{9, 9})
	if c.Len() != 2 {
		t.Fatalf("重写后 Len=%d, want 2", c.Len())
	}
	if got := c.KHead(0, 1, 0, 2)[0]; got != 9 {
		t.Fatalf("pos1 K[0]=%v, want 9（应被新值覆盖）", got)
	}
	if got := c.KHead(0, 0, 0, 2)[0]; got != 1 {
		t.Fatalf("pos0 K[0]=%v, want 1（截断前数据应保留）", got)
	}
}

// TestTruncateBounds Truncate 越界/负值不 panic。
func TestTruncateBounds(t *testing.T) {
	c := New(1, 2, 8)
	k := []float32{1, 2}
	v := []float32{3, 4}
	c.Write(0, 0, k, v)
	c.Write(0, 1, k, v)
	c.Truncate(-5) // 负值 → 0
	if c.Len() != 0 {
		t.Fatalf("Truncate(-5) 后 Len=%d, want 0", c.Len())
	}
	c.Truncate(99) // 超过当前 len → 不增长
	if c.Len() != 0 {
		t.Fatalf("Truncate(99) 不应增长，Len=%d", c.Len())
	}
}