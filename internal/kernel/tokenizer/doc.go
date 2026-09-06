// Package tokenizer 实现词表加载与文本↔token id 互转（qwen2/gpt2 风格字节 BPE）。
//
// 里程碑：M2（词表）。目标：中文 prompt 正确分词，Decode 还原文字。
//
// 为什么需要"字节编码"这层：BPE 合并规则（merges）作用在 token 字符串上，
// 而任意 UTF-8 文本的字节都可能是不可见/不方便的字节。GPT-2 的思路是——
// 把每个字节 b 映射到一个 Unicode 码点（可打印字符原样、其余映射到 U+0100+ 私有区），
// 得到一个"看起来是字符串"的 token 序列，再在上面跑合并。反向时再映射回字节。
package tokenizer