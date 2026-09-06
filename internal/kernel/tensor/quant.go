package tensor

import "fmt"

// DequantBlock 把一个量化块（blk，长度=TypeSize 字节）解码成 BlockSize 个 float32。
// 这是本包最重要的函数——权重从 GGUF 取来后全靠它变成可算的浮点数。
func DequantBlock(typ DType, blk []byte, dst []float32) error {
	switch typ {
	case TYPE_Q4_0:
		dequantQ4_0(blk, dst)
	case TYPE_Q4_1:
		dequantQ4_1(blk, dst)
	case TYPE_Q5_0:
		dequantQ5_0(blk, dst)
	case TYPE_Q5_1:
		dequantQ5_1(blk, dst)
	case TYPE_Q8_0:
		dequantQ8_0(blk, dst)
	case TYPE_Q4_K:
		dequantQ4_K(blk, dst)
	case TYPE_Q5_K:
		dequantQ5_K(blk, dst)
	case TYPE_Q6_K:
		dequantQ6_K(blk, dst)
	default:
		return fmt.Errorf("tensor: DequantBlock 暂不支持 %s", typ)
	}
	return nil
}

// ============ 普通量化（每 32 个元素一个块，1 个 fp16 scale） ============

// Q4_0 布局：[d fp16 2B][qs 16B]  共享 d，qs 为有符号 4bit(-8..7)
// 低 nibble 是元素 0..15，高 nibble 是元素 16..31。
// 还原：x = q * d
func dequantQ4_0(blk []byte, dst []float32) {
	const qk = 32
	d := F16ToF32(blk[0], blk[1])
	for i := 0; i < qk/2; i++ {
		x0 := int8(blk[2+i]&0x0F) - 8 // 低 4bit 是 -8..7
		x1 := int8(blk[2+i]>>4) - 8    // 高 4bit 是 -8..7
		dst[i] = float32(x0) * d
		dst[i+qk/2] = float32(x1) * d
	}
}

// Q4_1 布局：[d fp16][m fp16][qs 16B]  无符号 4bit，表达式 x = q*d + m
func dequantQ4_1(blk []byte, dst []float32) {
	const qk = 32
	d := F16ToF32(blk[0], blk[1])
	m := F16ToF32(blk[2], blk[3])
	for i := 0; i < qk/2; i++ {
		x0 := float32(blk[4+i] & 0x0F)
		x1 := float32(blk[4+i] >> 4)
		dst[i] = x0*d + m
		dst[i+qk/2] = x1*d + m
	}
}

// Q5_0 布局：[d fp16][qh 4B][qs 16B]  有符号 5bit(-16..15)
// qh 每字节的低 2 bit 是 "高 1 bit" 的补充位；qs 存低 4 bit。
func dequantQ5_0(blk []byte, dst []float32) {
	const qk = 32
	d := F16ToF32(blk[0], blk[1])
	qh := uint32(blk[2]) | uint32(blk[3])<<8 | uint32(blk[4])<<16 | uint32(blk[5])<<24
	for i := 0; i < qk/2; i++ {
		// 第 i 个元素的两个"第5bit"分别取 qh 的第 i 位和 i+12 位（即错出 16 个元素）
		xh0 := ((qh >> (uint(i) + 0)) << 4) & 0x10
		xh1 := (qh >> (uint(i) + 12)) & 0x10
		x0 := int32(blk[6+i]&0x0F|byte(xh0)) - 16
		x1 := int32(blk[6+i]>>4|byte(xh1)) - 16
		dst[i] = float32(x0) * d
		dst[i+qk/2] = float32(x1) * d
	}
}

// Q5_1 布局：[d fp16][m fp16][qh 4B][qs 16B] 无符号 5bit，x = q*d + m
func dequantQ5_1(blk []byte, dst []float32) {
	const qk = 32
	d := F16ToF32(blk[0], blk[1])
	m := F16ToF32(blk[2], blk[3])
	qh := uint32(blk[4]) | uint32(blk[5])<<8 | uint32(blk[6])<<16 | uint32(blk[7])<<24
	for i := 0; i < qk/2; i++ {
		xh0 := ((qh >> (uint(i) + 0)) << 4) & 0x10
		xh1 := (qh >> (uint(i) + 12)) & 0x10
		x0 := int32(blk[8+i]&0x0F | byte(xh0))
		x1 := int32(blk[8+i]>>4 | byte(xh1))
		dst[i] = float32(x0)*d + m
		dst[i+qk/2] = float32(x1)*d + m
	}
}

// Q8_0 布局：[d fp16][qs 32B]  有符号 8bit，x = q * d（最简单，精度最高）
func dequantQ8_0(blk []byte, dst []float32) {
	const qk = 32
	d := F16ToF32(blk[0], blk[1])
	for i := 0; i < qk; i++ {
		dst[i] = float32(int8(blk[2+i])) * d
	}
}

// ============ K 系列量化（超块 256 元素，8 组 32 元素子块） ============

// getScaleMinK4 解析 Q4_K/Q5_K 的 8 组 scale/min（每组 6 bit 紧凑存储）。
// scales[12] 字节里把 8 个 6bit 值塞进去，低 4 组直接取低 6 bit，高 4 组拆在两处再拼。
func getScaleMinK4(j int, scales []byte) (sc, m uint8) {
	if j < 4 {
		sc = scales[j] & 63          // 低 4 组：scales[j] 低 6bit
		m = scales[j+4] & 63          // min 存在后面 4 个字节
	} else {
		// 高 4 组：scale 拆在 scales[j+4] 的低4bit 与 scales[j-4] 的顶2bit
		sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
		m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
	}
	return
}

// Q4_K 布局：[d fp16][dmin fp16][scales 12B][qs 128B]
// 8 组子块，每组独立 scale/min，x = q*d*sc - m*dmin（q 无符号 4bit）
func dequantQ4_K(blk []byte, dst []float32) {
	d := F16ToF32(blk[0], blk[1])
	dmin := F16ToF32(blk[2], blk[3])
	scales := blk[4:16]
	qs := blk[16:144]

	// 4 轮，每轮覆盖 64 元素（2 组 32 元素子块）
	for seg := 0; seg < 4; seg++ {
		is := 2 * seg
		sc0, m0 := getScaleMinK4(is, scales)
		sc1, m1 := getScaleMinK4(is+1, scales)
		d1, mf1 := d*float32(sc0), dmin*float32(m0)
		d2, mf2 := d*float32(sc1), dmin*float32(m1)
		off := 32 * seg
		for l := 0; l < 32; l++ {
			qb := qs[off+l]
			dst[64*seg+l] = d1*float32(qb&0x0F) - mf1
			dst[64*seg+l+32] = d2*float32(qb>>4) - mf2
		}
	}
}

// Q5_K 布局：[d fp16][dmin fp16][scales 12B][qh 32B][qs 128B]
// 同 Q4_K 结构，但每个元素有独立第 5 bit（存 qh）。
func dequantQ5_K(blk []byte, dst []float32) {
	d := F16ToF32(blk[0], blk[1])
	dmin := F16ToF32(blk[2], blk[3])
	scales := blk[4:16]
	qh := blk[16:48]
	qs := blk[48:176]

	u1, u2 := uint8(1), uint8(2)
	for seg := 0; seg < 4; seg++ {
		is := 2 * seg
		sc0, m0 := getScaleMinK4(is, scales)
		sc1, m1 := getScaleMinK4(is+1, scales)
		d1, mf1 := d*float32(sc0), dmin*float32(m0)
		d2, mf2 := d*float32(sc1), dmin*float32(m1)
		off := 32 * seg
		for l := 0; l < 32; l++ {
			qb := qs[off+l]
			h := qh[l]
			var h1, h2 uint8
			if h&u1 != 0 {
				h1 = 16
			}
			if h&u2 != 0 {
				h2 = 16
			}
			dst[64*seg+l] = d1*float32(qb&0x0F+h1) - mf1
			dst[64*seg+l+32] = d2*float32(qb>>4+h2) - mf2
		}
		u1 <<= 2 // qh 每 2 元素一行，16 元素后换下一行 bit
		u2 <<= 2
	}
}

// Q6_K 布局：[ql 128B][qh 64B][scales 16B][d fp16]（4 个 64 元素半超块）
// 16 个子组（scales 中 int8），每个元素 6bit：ql 低 4bit + qh 2bit，x = q*d*sc
func dequantQ6_K(blk []byte, dst []float32) {
	d := F16ToF32(blk[208], blk[209])
	ql := blk[0:128]
	qh := blk[128:192]
	sc := blk[192:208]

	// ql/qh/sc 各推进 2 次（第二次从 64/32/8 偏移读）
	for n := 0; n < 256; n += 128 {
		for l := 0; l < 32; l++ {
			// 每字节含两个 q 的低 4 bit；qh 用 2 bit/元素给第 5、6 位
			is := l / 16
			q1 := int8(ql[l]&0x0F|((qh[l]>>0)&3)<<4) - 32
			q2 := int8(ql[l+32]&0x0F|((qh[l]>>2)&3)<<4) - 32
			q3 := int8(ql[l]>>4|((qh[l]>>4)&3)<<4) - 32
			q4 := int8(ql[l+32]>>4|((qh[l]>>6)&3)<<4) - 32
			dst[n+l] = d * float32(int8(sc[is+0])) * float32(q1)
			dst[n+l+32] = d * float32(int8(sc[is+2])) * float32(q2)
			dst[n+l+64] = d * float32(int8(sc[is+4])) * float32(q3)
			dst[n+l+96] = d * float32(int8(sc[is+6])) * float32(q4)
		}
		ql = ql[64:]
		qh = qh[32:]
		sc = sc[8:]
	}
}