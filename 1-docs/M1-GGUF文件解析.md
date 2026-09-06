# M1：GGUF 文件解析

> 状态：✅ 完成（可独立验收）
> 代码：`internal/kernel/gguf/`
> 里程碑文档目录：本文件按里程碑最简记录，完整大白话版见 learn-llm 笔记（待补）。

## 一句话本质

GGUF 是一个**自描述的二进制容器**。文件头告诉你「之后有多少个元数据、多少个张量」，然后按固定顺序读到 `元数据区 → 张量信息表 → 原始权重数据区`，三段挨着排开。

把它读懂，你就把 397MB（qcwen2.5-0.5b.gguf）变成了内存里能用的张量描述。

## GGUF v3 文件布局

```
┌─ 文件头 ──────────────┐
│ magic   "GGUF" 4字节    │
│ version  uint32         │
│ tensor_count  uint64    │  ← 权重张量个数（qwen2.5-0.5b = 290）
│ kv_count  uint64        │  ← 元数据条数
├─ 元数据区(kv_count 条) ──┘
│ key: uint64长度 + 字节    │
│ type: uint32 (值类型)     │
│ value: 依 type 而定        │  ← string/数值/数组/bool
├─ 张量信息表(tensor_count 条)
│ name: string             │  ← 如 "model.layers.0.self_attn.q_proj.weight"
│ n_dims: uint32           │
│ dims: n_dims 个 uint64    │
│ type: uint32 (GGML 类型)  │  ← F32/F16/Q8_0/Q5_0/Q6_K...
│ offset: uint64           │  ← 权重数据在文件里的字节偏移
└─ 权重数据区               ┘
     offset .. 文件尾 = 各张量原始字节（按类型布局）
```

两个关键点（隐形知识）：

**字节对齐公式**：数据区起点必须对齐到 32 字节边界（SIMD 友好）：

$$\text{DataStart} = \text{align32}(\text{张量信息表读完后的位置})$$

其中 $\text{align32}(x) = (x + 31) \ \& \ \sim 31$（向上取整到 32 的倍数）。

1. **全程 little-endian**，字符串长度是 uint64（v3；v1/v2 是 u32 —— 这是最容易踩的坑）。
2. **权重数据区不跟在信息表后面**，而是每个张量自己带一个 `offset`——因为张量数据有大有小，加载器经常 mmap 零拷贝引用，不整体搬进内存。

## 张量字节大小怎么算（核心算法）

非量化（F32/F16）：`字节 = 元素个数 × 每元素字节`

量化类型（Q8_0/Q5_0/Q6_K…）：**不能**用「元素数 × 字节」，因为量化是**分块**存的——32 或 256 个元素共享一个 scale（缩放系数）。

```
块数   = ne[0] / blockElems      // 首维必须是块元素数的整数倍
行字节 = 块数 × blockBytes
总字节 = 行字节 × 其余所有维度
```

所以 `types.go` 里维护一张 `blockTable`，算占位前先查它：

| 类型 | 一个块覆盖元素 | 一个块字节数 | 说明 |
|---|---|---|---|
| F32 | 1 | 4 | float32 |
| F16 | 1 | 2 | float16 |
| Q8_0 | 32 | 2+32 | 1×f16 scale + 32 个 int8 |
| Q5_0 | 32 | 2+4+16 | scale + qh(4) + qs(16) |
| Q6_K | 256 | 2+16+192+16 | scale/ql/qh/scales |
| Q4_K | 256 | 2+12+96+16 | d + dmin + qs + scales |

## 踩过的坑（本里程碑真实经历）

- **GGMLType 编号照抄错了**：一开始我按记忆把 Q4_0 标成了 8，结果跑真模型在 tensor[3] 崩溃报"类型 6 不支持"。教训：**类型编号必须以 ggml 格式规范为准**（对照 ggml_type 枚举表），别凭记忆。
- **解析字节对齐**：首版探针把 `tensor_count`/`kv_count` 用 4 字节 `I` 读了，GGUF 里它们都是 8 字节 `Q`，导致后面全错位。

## M1 验收记录

```
go run . run models/qwen2.5-0.5b.gguf "你好"

== 模型信息 ==
架构: qwen2  类型: model(Instruct)   名称: Qwen2.5 0.5B Instruct
版本: GGUF v3   张量: 290 个   文件 397807936 字节
  token_embd.weight  Q8_0  ne=[896 151936] bytes=144643072
  blk.0.attn_norm.weight  F32   ne=[896] bytes=3584
  blk.0.ffn_down.weight  Q6_K  ne=[4864 896] bytes=3847424
  blk.0.ffn_gate.weight  Q5_0  ne=[896 4864] bytes=2996224
  ...
```

✅ 关键词全部正确：**qwen2 架构、Qwen2.5 0.5B Instruct、290 张量、混合量化（Q8_0/Q6_K/Q5_0/F32）**。

顺带发现：`ne=[896 151936]` 是**行主序**（ne0=896 为连续维）。后面加载权重到 MatMul 时要记住这个方向翻转 —— 这是 M3 的隐形坑，先记下。

## 术语表

| 术语 | 中文 | 一句话解释 |
|---|---|---|
| GGUF | 通用格式 | ggml 生态的模型文件格式，自描述二进制容器 |
| KV | 键值元数据 | 模型信息的 key-value 表（架构/层数/词表大小等） |
| GGMLType | 张量类型 | 权重存储类型，量化 vs 非量化 |
| 量化 | 压缩 | 用低精度(如 int8/4bit)存权重，省内存换精度 |
| 块(block) | 量化块 | 量化存取的原子单位，如 Q8_0 每 32 元素共用一个 scale |
| scale | 缩放系数 | 量化块里存的反量化因子 |
| ne | 各维长度 | ne[0] 是最连续（内存中相邻）的那一维 |
| offset | 偏移 | 张量权重数据在文件中的字节位置 |