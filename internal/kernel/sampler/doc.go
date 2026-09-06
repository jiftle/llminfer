// Package sampler 在 logits 上选下一个 token（温度 / top-k / top-p / 重复惩罚）。
// 里程碑：M4（生成）目标——让同一条 prompt 能随机但合理地续写。
package sampler