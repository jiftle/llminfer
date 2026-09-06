package tokenizer

import (
	"fmt"
	"sort"
	"strings"

	"github.com/feiyuclaw/llminfer/internal/kernel/chat"
	"github.com/feiyuclaw/llminfer/internal/kernel/gguf"
)

// Tokenizer 基于 GGUF 词表 + merges 的字节级 BPE 分词器。
type Tokenizer struct {
	vocab    []string          // 词表：token id -> GPT-2 字节编码字符串
	tokenIDs map[string]uint32 // 反向：字节编码字符串 -> token id
	rankMap  map[string]int    // 合并规则："A B" -> 优先级（值越小越优先）
	// sortedTokens 按字节长度降序，用于"最长前缀匹配"。
	// 编码时从当前位置尝试最长的词表 token，这等价于一次性跳过多个字节。
	sortedTokens   []sortedToken
	specialTokens  []specialToken // 特殊 token（<|im_start|> 等），整体匹配不可拆
	EOS, BOS       int32
	Unknown        uint32
	HasMerges      bool
	preTokenizer   *PreTokenizer
	// ChatTemplate 从 GGUF tokenizer.chat_template 检测出的对话模板（M6）。
	// 多轮对话时把消息列表渲染成模型输入字符串。nil 表示模型无/未识别模板。
	ChatTemplate *chat.Template
}

type sortedToken struct {
	id   uint32
	data string
}

type specialToken struct {
	id   uint32
	text string
}

// NewFromGGUF 从 GGUFFile 构造分词器。
func NewFromGGUF(gf *gguf.GGUFFile, arch string) (*Tokenizer, error) {
	t := &Tokenizer{
		vocab:    gf.GetStringSlice("tokenizer.ggml.tokens"),
		tokenIDs: make(map[string]uint32),
		rankMap:  make(map[string]int),
		EOS:      -1,
		BOS:      -1,
		Unknown:  0,
	}
	if len(t.vocab) == 0 {
		return nil, fmt.Errorf("tokenizer: 词表为空")
	}

	// 反向映射 + 预排序（最长优先，编码时用）
	for i, s := range t.vocab {
		t.tokenIDs[s] = uint32(i)
		t.sortedTokens = append(t.sortedTokens, sortedToken{id: uint32(i), data: s})
	}
	sort.Slice(t.sortedTokens, func(i, j int) bool {
		return len(t.sortedTokens[i].data) > len(t.sortedTokens[j].data)
	})

	// merges：每一条是 "被合并的前片 被合并的后片"（两段之间空格分隔）
	for rank, s := range gf.GetStringSlice("tokenizer.ggml.merges") {
		if strings.Contains(s, " ") {
			t.rankMap[s] = rank
		}
	}
	t.HasMerges = len(t.rankMap) > 0

	// 特殊 token 识别：token_type=3(CONTROL) 或 4(USER_DEFINED) 视为特殊 token
	tTypes := gf.GetInt32Slice("tokenizer.ggml.token_type")
	for i, tt := range tTypes {
		if tt == 3 || tt == 4 {
			t.specialTokens = append(t.specialTokens, specialToken{
				id:   uint32(i),
				text: t.vocab[i],
			})
		}
	}

	// 边界 token
	if v := gf.GetInt("tokenizer.ggml.bos_token_id"); v >= 0 {
		t.BOS = int32(v)
	}
	if v := gf.GetInt("tokenizer.ggml.eos_token_id"); v >= 0 {
		t.EOS = int32(v)
	}

	t.preTokenizer = NewPreTokenizer(arch)

	// M6：检测对话模板。从 GGUF 读原始 chat_template 字符串，检测类型（关键词匹配，非 Jinja 解析）
	t.ChatTemplate = chat.Detect(gf.GetString("tokenizer.chat_template"))
	return t, nil
}

// Encode 文本 → token id 序列。普通文本走 BPE，特殊 token 整体匹配。
func (t *Tokenizer) Encode(text string) []uint32 {
	if len(t.specialTokens) > 0 {
		return t.encodeWithSpecial(text)
	}
	if t.HasMerges {
		return t.encodeBPE(text)
	}
	return t.encodeNaive(text)
}

// encodeWithSpecial 处理含特殊 token 的文本：整段匹配特殊 token，普通片段交给 BPE。
func (t *Tokenizer) encodeWithSpecial(text string) []uint32 {
	var ids []uint32
	rest := text
	for rest != "" {
		// 找当前开头处最长的特殊 token 前缀
		bestID, bestLen := -1, 0
		for _, st := range t.specialTokens {
			if len(st.text) > bestLen && strings.HasPrefix(rest, st.text) {
				bestID, bestLen = int(st.id), len(st.text)
			}
		}
		if bestID >= 0 {
			ids = append(ids, uint32(bestID))
			rest = rest[bestLen:]
			continue
		}
		// 不命中：一直推进到下一个特殊 token 出现之前，普通片段编码
		next := len(rest)
		for _, st := range t.specialTokens {
			if idx := strings.Index(rest, st.text); idx >= 0 && idx < next {
				next = idx
			}
		}
		if next == 0 {
			next = 1 // 防御死循环
		}
		if t.HasMerges {
			ids = append(ids, t.encodeBPE(rest[:next])...)
		} else {
			ids = append(ids, t.encodeNaive(rest[:next])...)
		}
		rest = rest[next:]
	}
	return ids
}

// encodeBPE 完整 BPE：预分词 → 字节编码 → 合并。
func (t *Tokenizer) encodeBPE(text string) []uint32 {
	var ids []uint32
	for _, piece := range t.preTokenizer.Pretokenize(text) {
		ids = append(ids, t.bpeMerge([]byte(piece))...)
	}
	return ids
}

// bpeMerge 对一个（预分词后的）字节序列执行 BPE 合并。
//
// 步骤：
//  1. 把字节序列转成 GPT-2 编码字符串，用"最长前缀匹配"切出初始 token 序列。
//  2. 反复找 rankMap 中优先级最高（值最小）的可合并相邻对，合并之，直到无对可合并。
func (t *Tokenizer) bpeMerge(data []byte) []uint32 {
	// ① 最长前缀匹配出初始 token
	init := t.initialTokens(data)
	if len(init) <= 1 {
		return init
	}
	texts := make([]string, len(init))
	for i, id := range init {
		texts[i] = t.vocab[id]
	}

	// ② 迭代合并
	for {
		bestPos, bestRank := -1, int(^uint(0)>>1)
		for i := 0; i+1 < len(texts); i++ {
			pair := texts[i] + " " + texts[i+1]
			if rank, ok := t.rankMap[pair]; ok && rank < bestRank {
				bestPos, bestRank = i, rank
			}
		}
		if bestPos < 0 {
			break
		}
		texts[bestPos] = texts[bestPos] + texts[bestPos+1]
		texts = append(texts[:bestPos+1], texts[bestPos+2:]...)
	}

	// ③ 结果回查 token id（合并产物必然在词表，防万一逐字节兜底）
	out := make([]uint32, 0, len(texts))
	for _, text := range texts {
		if id, ok := t.tokenIDs[text]; ok {
			out = append(out, id)
			continue
		}
		// 合并产物不在词表（正常不会发生）：退化为逐字节
		for i := 0; i < len(text); i++ {
			if id, ok := t.tokenIDs[string(text[i])]; ok {
				out = append(out, id)
			} else {
				out = append(out, t.Unknown)
			}
		}
	}
	return out
}

// initialTokens 用最长前缀匹配把字节序列切成初始 token 序列。
// 因为词表里每个单字节都有对应 token，所以一定能匹配完整个序列。
func (t *Tokenizer) initialTokens(data []byte) []uint32 {
	encoded := BytesToUnicode(data)
	var ids []uint32
	i := 0
	for i < len(encoded) {
		bestID, bestLen := -1, 0
		// sortedTokens 已按长度降序，遇到第一个可匹配的就是最长前缀
		for _, st := range t.sortedTokens {
			l := len(st.data)
			if l <= bestLen || i+l > len(encoded) {
				continue
			}
			if encoded[i:i+l] == st.data {
				bestID, bestLen = int(st.id), l
			}
		}
		if bestID < 0 {
			ids = append(ids, t.Unknown)
			i++
		} else {
			ids = append(ids, uint32(bestID))
			i += bestLen
		}
	}
	return ids
}

// encodeNaive 无 merges 时的退化实现：直接用最长前缀匹配（不合并）。
func (t *Tokenizer) encodeNaive(text string) []uint32 {
	var ids []uint32
	encoded := BytesToUnicode([]byte(text))
	i := 0
	for i < len(encoded) {
		bestID, bestLen := -1, 0
		for _, st := range t.sortedTokens {
			l := len(st.data)
			if l <= bestLen || i+l > len(encoded) {
				continue
			}
			if encoded[i:i+l] == st.data {
				bestID, bestLen = int(st.id), l
			}
		}
		if bestID < 0 {
			ids = append(ids, t.Unknown)
			i++
		} else {
			ids = append(ids, uint32(bestID))
			i += bestLen
		}
	}
	return ids
}

// Decode token id 序列 → 文本。词表里存的是字节编码字符串，需反向映射回字节。
func (t *Tokenizer) Decode(ids []uint32) string {
	var raw []byte
	for _, id := range ids {
		if int(id) >= len(t.vocab) {
			continue
		}
		token := t.vocab[id]
		b, ok := UnicodeToBytes(token)
		if ok {
			raw = append(raw, b...)
		} else {
			// 个别 token 可能是拼接的 UTF-8（比如合并了多字节），逐字节兜底
			raw = append(raw, token...)
		}
	}
	return string(raw)
}

// Validate 校验词表非空。
func (t *Tokenizer) Validate() error {
	if len(t.vocab) == 0 {
		return fmt.Errorf("tokenizer: 词表为空")
	}
	return nil
}

// ChatStopIDs 返回对话结束 token 集合：EOS + 常见对话终止标记。
// qwen 系为 <|im_end|>/<|endoftext|>；llama3 系为 <|eot_id|>/<|end_of_text|>。
func (t *Tokenizer) ChatStopIDs() map[uint32]bool {
	set := map[uint32]bool{}
	if t.EOS >= 0 {
		set[uint32(t.EOS)] = true
	}
	for _, st := range t.specialTokens {
		switch st.text {
		case "<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>", "</s>":
			set[st.id] = true
		}
	}
	return set
}