# M3：张量算子（反量化 / MatMul / 归一化 / RoPE）

> 状态：✅ 完成（Q8_0 反量化数值正确，单算子单元测试全绿）
> 代码：`internal/kernel/tensor/` + `internal/kernel/model/`
> 依赖：M1 的 gguf；M4 将用这里的算子组装前向推理

## 一句话本质

模型文件里存的是**量化后的字节**，不能直接算。M3 的全部工作 = 三件事：

1. **反量化**：把量化字节块按格式还原成 float32（正确性生死线）
2. **矩阵乘**：激活 × 权重 = 一层线性变换（注意力/FFN 的骨架）
3. **激活/归一化**：RMSNorm、SiLU、Softmax、RoPE（让模型能"非线性"和"带位置"）

## 量化是什么（大白话）

权重用 float32 存 4 字节一个数太费内存。量化 = 把 32 个数的**取值范围**压成 1 个 scale + 32 个低精度整数，用的时候 `值 = 整数 × scale` 还原（近似）。

关键：**32 个元素共用一个 scale**，所以还原时必须按"块"对齐读——差分错一个字节，整层权重全废。

## 本模型需要的量化布局（ggml 格式规范）

| 类型 | 块大小 | 布局（字节序） | 反量化公式 |
|---|---|---|---|
| Q8_0 | 32 | `[d:2B fp16][qs:32B int8]` | x = q × d |
| Q5_0 | 32 | `[d:2B][qh:4B][qs:16B]` | x = (q±16bit) × d |
| Q6_K | 256 | `[ql:128B][qh:64B][scales:16B][d:2B]` | x = q(6bit) × d × scale |
| Q4_K | 256 | `[d:2B][dmin:2B][scales:12B][qs:128B]` | x = q×d×sc − min×dmin |

`quant.go` 逐位实现了这套，Q6_K/Q4_K 的 K 系列是级联 scale（64/256 块内的子块还有独立缩放），最容易错。

## 核心算子

### MatMulTransB（转置矩阵乘）

权重按 $[K, N]$ 行主序存储（ne0=K 连续），所以矩阵乘是 $C = A \times B^T$：

$$C_{ij} = \sum_{k=0}^{K-1} A_{ik} \cdot B_{jk}$$

量化 B 时逐行反量化：$B_j = \text{DequantRow}(j)$，避免一次性展开整个权重矩阵。

### RMSNorm（均方根归一化）

对每行 $x$（长度 $d$）做归一化，再乘以可学习权重 $w$：

$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \varepsilon}$$

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot w$$

其中 $\varepsilon = 10^{-6}$（qwen2 默认值），防止除零。$\odot$ 表示逐元素乘。

### SiLU（Swish 激活）

$$\text{SiLU}(x) = \frac{x}{1 + e^{-x}} = x \cdot \sigma(x)$$

其中 $\sigma(x)$ 是 sigmoid 函数。SiLU 是 FFN 的关键非线性。

### SoftMax（归一化指数）

对向量 $z$（长度 $n$）做概率化，先减 max 防溢出：

$$\text{SoftMax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^{n} e^{z_j - \max(z)}}$$

输出 $\sum_i p_i = 1$，每个 $p_i \in (0, 1)$。

### RoPE（旋转位置编码）

对 q/k 向量按绝对位置 $\theta$ 旋转。对每对相邻元素 $(q_{2i}, q_{2i+1})$：

$$\begin{pmatrix} q_{2i}' \\ q_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

其中频率 $\theta_i = \text{pos} / \text{base}^{2i/d}$，base $= 10^6$（qwen2.5 默认），$d$ 是 headDim。

qwen2 用 **NEOX 配对**：前 $d/2$ 维与后 $d/2$ 维配对旋转（不是相邻对）。

## model 包

把 GGUF 的 290 个张量按名字挂到 `LLaMAModel`：
- 超参数：层数/维度/头数/上下文长度从 KV 读
- 每一层的 7 个权重（Q/K/V/O + FFN 的 gate/up/down）+ 3 个可选 bias
- 张量字节零拷贝引用文件内存（不反量化，等真正计算时按行反量化）

## M3 验收

**1. Q8_0 反量化数值正确**（同一模型 token_embd 三行抽样）：

```
row 0       -0.01027679 0.04078603 ...   ✓ 数值合理
row 100     -0.01253176 0.03606701 ...   ✓
row 150000  -0.001326799 -0.002918959... ✓
```

**2. 单元测试** 5 个全绿：F16 转换（8 组）、Q8_0 手算块、MatMul 手算、RMSNorm、Softmax。

**3. fp16 转换**验证：0x3C00→1.0、0x399A→0.7、次正规数→6e-8 全对。

## 踩过的坑（隐形知识）

1. **GGUF 的 tensor offset 是相对的**：张量 Offset 相对"数据区起点"，不是文件头！读取时必须加 `DataStart`（张量表读完后对齐 32 的位置）。头一回忘了加,直接读到文件头字节当权重,数值全乱（sum=1.5亿、出现 NaN）。
2. **量化类型编号**：M1 已踩过（Q4_0=2 不是 8），M3 的 tensor.DType 若自己再抄一遍还会踩——直接从 gguf 复用编号。
3. **Q6_K 的 4 个 64 元素子块索引**：`is = l/16` 那组下标很容易写错，必须对照 ggml 反量化公式逐行核对。

## 术语表

| 术语 | 中文 | 一句话解释 |
|---|---|---|
| Quantization | 量化 | 用低精度整数加一个 scale 近似存权重，省内存 |
| Dequant | 反量化 | 还原：int × scale = 近似 float |
| Block | 块 | 量化存取的最小单位（共享一个 scale 的一组数） |
| Scale | 缩放系数 | 块的还原因子 |
| RMSNorm | 均方根归一化 | 除以行的 RMS，稳定数值 |
| SiLU/Swish | 门控激活 | x·σ(x)，FFN 的关键非线性 |
| Softmax | 归一化指数 | 把分数变成和为 1 的概率 |
| RoPE | 旋转位置编码 | 按绝对位置旋转 q/k 向量，注入位置信息 |
| NEOX | 一种配对方式 | qwen2/GPT-NeoX 的 RoPE 配对（前/后半），不是相邻对 |
| MatMulTransB | 转置矩阵乘 | 权重 [K,N] 行主序下的高效矩阵乘形态 |