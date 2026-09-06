// Package eval 实现前向推理核心：token 向量 → 逐层 Transformer → logits。
// 里程碑：M3（前向）目标——单步 Forward 输出整词表 logits。
package eval