// Package gguf 负责解析 GGUF 模型文件：读取头部、元数据 KV 与张量信息表。
// 里程碑：M1（进料）目标——能把 .gguf 读成内存里的张量描述。
package gguf