package tokenizer

import "unicode"

// PreTokenizer 在 BPE 合并之前，把连续文本按"字符类型"粗切一刀。
//
// 为什么要预分词：BPE 的合并规则大多只针对"同一类字符"（字母/数字/空格/标点）。
// 先按类切分，可以避免比如"abc123"里 'c'+'1' 这种跨类合并，也大幅缩小每次要
// 跑的合并区间长度（GPT-2 字节级 BPE 的预分词规则）。
type PreTokenizer struct {
	arch string
}

// NewPreTokenizer 创建预分词器。qwen2 走 GPT-2 风格，llama 走另一套，其余默认 GPT-2。
func NewPreTokenizer(arch string) *PreTokenizer {
	return &PreTokenizer{arch: arch}
}

// Pretokenize 把 text 按字符类型切成若干段。
func (pt *PreTokenizer) Pretokenize(text string) []string {
	var out []string
	var cur []rune

	for _, r := range text {
		if len(cur) > 0 && shouldSplit(cur[len(cur)-1], r) {
			out = append(out, string(cur))
			cur = cur[:0]
		}
		cur = append(cur, r)
	}
	if len(cur) > 0 {
		out = append(out, string(cur))
	}
	return out
}

// shouldSplit 判断 last 和 next 是否属于不同类型（应切开）。
// 规则：字母/数字/空格/标点各是一类，跨类即切。
func shouldSplit(last, next rune) bool {
	return charClass(last) != charClass(next)
}

// charClass 给字符一个粗糙的"类标号"：字母/数字/空格/标点/其他。
func charClass(r rune) int {
	switch {
	case unicode.IsLetter(r):
		return 0
	case unicode.IsDigit(r):
		return 1
	case unicode.IsSpace(r):
		return 2
	default:
		return 3 // 标点与其他归一类（空格已在上面排除）
	}
}