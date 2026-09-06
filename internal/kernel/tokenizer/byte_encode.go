package tokenizer

// byteToUnicode 是 GPT-2 风格的 字节→Unicode 映射表，256 项。
//
// 规则（GPT-2 字节编码表）：
//   - 可打印 ASCII 0x21~0x7E、扩展拉丁 0xA1~0xAC、0xAE~0xFF 原样映射（本身是合法字符）
//   - 其余"别扭"的字节（控制符 0x00~0x20、0x7F、0xA0、0xAD）顺序映射到 U+0100+
//
// 反向（UnicodeToByte）靠遍历该表反查。
var byteToUnicode [256]rune

func init() {
	// 直接映射段：这些字节本身就是可打印/拉丁字符
	for i := 0x21; i <= 0x7E; i++ {
		byteToUnicode[i] = rune(i)
	}
	for i := 0xA1; i <= 0xAC; i++ {
		byteToUnicode[i] = rune(i)
	}
	for i := 0xAE; i <= 0xFF; i++ {
		byteToUnicode[i] = rune(i)
	}

	// 剩余字节（控制器/边界符/空格类）从 U+0100 起顺序编号
	n := 0
	for i := 0; i < 256; i++ {
		if isMappedByte(i) {
			continue
		}
		byteToUnicode[i] = rune(256 + n)
		n++
	}
}

// isMappedByte 判断字节 i 是否已在"原样映射段"里。
func isMappedByte(i int) bool {
	switch {
	case i >= 0x21 && i <= 0x7E:
		return true
	case i >= 0xA1 && i <= 0xAC:
		return true
	case i >= 0xAE && i <= 0xFF:
		return true
	}
	return false
}

// BytesToUnicode 把任意字节序列转成"GPT-2 编码字符串"：
// 每个字节 → 一个 Unicode 码点。这是 BPE 合并前的一步。
func BytesToUnicode(b []byte) string {
	runes := make([]rune, len(b))
	for i, c := range b {
		runes[i] = byteToUnicode[c]
	}
	return string(runes)
}

// UnicodeToBytes 把 GPT-2 编码字符串反解码回原始字节序列。
// 遇到不在映射表里的字符说明不是合法编码，返回 ok=false。
func UnicodeToBytes(s string) ([]byte, bool) {
	out := make([]byte, 0, len(s))
	for _, r := range s {
		b, ok := unicodeToByte(r)
		if !ok {
			return nil, false
		}
		out = append(out, b)
	}
	return out, true
}

// unicodeToByte 反向查表：找哪个字节映射到了 rune r。
func unicodeToByte(r rune) (byte, bool) {
	for i, u := range byteToUnicode {
		if u == r {
			return byte(i), true
		}
	}
	return 0, false
}