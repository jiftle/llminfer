# M5：采样生成循环（Logits → Token → 文本）

> 状态：✅ 完成（贪心/温度/top-k/top-p 全通，qwen2.5-0.5b 生成连贯文本）
> 代码：internal/kernel/sampler/ + cmd/generate.go
> 依赖：M4 前向推理；M6 已加 ChatML 对话模板

## 一句话本质

M4 的前向输出了「每个词的分数」（logits），但 logits 不是文本。M5 就是把 logits 变成文字的三步：

1. **采样**：从 151936 个候选词中按策略挑一个
2. **解码**：把 token id 翻译成人能读的字符串
3. **循环**：把刚生成的 token 再喂回去，重复 1-2 直到 EOS 或上限

## 采样策略（大白话）

### 贪心（Temperature ≤ 0）

永远选分数最高的词：

$$\text{id} = \arg\max_i \ \text{logits}[i]$$

确定性强，但容易重复、无趣。像考试只选最保险的答案。

### 温度缩放（Temperature > 0）

把 logits 除以 $T$ 再 softmax：

$$p_i = \frac{\exp(\text{logit}_i / T)}{\sum_j \exp(\text{logit}_j / T)}$$

- $T = 0.2$：分布很尖，几乎只选最高的（接近贪心）
- $T = 0.8$：适度随机，平衡多样性和质量（默认值）
- $T = 100$：分布很平，几乎均匀随机（胡说八道）

### Top-K 截断

保留概率最高的 $K$ 个 token，其余概率置零：

$$\text{candidates} = \text{TopK}(p, K)$$

防止选到完全不相关的词。

### Top-P 核采样

按概率降序累加，累积到 $P$ 阈值就截断：

$$\text{keep} = \min\left\{n : \sum_{i=1}^{n} p_{(i)} \geq P \right\}$$

其中 $p_{(i)}$ 是降序排列后的概率。比 Top-K 更灵活——分布很尖时只留 1-2 个词，分布很平时留几百个。

**组合使用**：Top-K 和 Top-P 可以同时用，先砍 K 再砍 P，取交集。

### 抽签（最终采样）

从截断后的概率分布中按累计概率抽签：

$$\text{id} = t \quad \text{where} \quad \sum_{i=1}^{t-1} p_i < r \leq \sum_{i=1}^{t} p_i$$

其中 $r \sim U(0, 1)$ 是均匀随机数。

## 生成循环（generate.go）

    prompt → Encode → [token_ids] → Forward(prefill) → logits
                                                                ↓
                                                      Sample(logits) → token_id
                                                                ↓
                                                      Decode(token_id) → 字
                                                                ↓
                                                      Forward([token_id]) → next_logits
                                                                ↓
                                                      ... 重复直到 EOS

关键不变量：**每步只 Forward 一个 token**。KV Cache 保证历史不会重算。

## 停止条件

- **EOS**：tokenizer.ggml.eos_token_id（qwen2.5 = 151645）
- **对话标记**：im_end、im_start 等特殊 token（ChatStopIDs）

## M5 验收

**1. 贪心模式输出连贯英文**：

    $ llminfer run -temperature 0 -max-tokens 30 models/qwen2.5-0.5b.gguf "The capital of France is"
    输出: Paris. It is the most populous city in France, with a population of over 2 million.

**2. 温度采样输出连贯中文**：

    $ llminfer run -temperature 0.8 -max-tokens 64 models/qwen2.5-0.5b.gguf "Hello"
    输出: 一下，我是来自中国的一名留学生...

**3. Sampler 单测** 6 个全绿：贪心、温度=0、Top-K、Top-P、种子可复现、高温度均匀分布。

## 踩过的坑

1. **Go flag 参数顺序**：flag 必须在位置参数前面。 不生效， 才行。Go 的 flag.NewFlagSet 按顺序解析，遇到非 flag 字符就停。
2. **tokenizer 需要 GGUFFile**：model.Load 读完文件就 Close 了，tokenizer 需要词表等元数据。生成循环里再读一次（轻量，只读 metadata 不读张量数据）。
3. **flag 未使用警告**：去掉旧的  后，新 flag 必须传给 Generate，否则编译警告。

## 文档 ↔ 代码对应

| 文档章节 | 代码位置 |
|---|---|
| 采样策略 | sampler.go Sample() |
| 贪心/温度/softmax | strategies.go（在 sampler 包内） |
| 生成循环 | cmd/generate.go Generate() |
| run 命令入口 | cmd/command.go run() |
| 停止条件 | generate.go ChatStopIDs + EOS |

## 术语表

| 术语 | 中文 | 一句话解释 |
|---|---|---|
| Sampling | 采样 | 从概率分布中随机选一个 token |
| Greedy/Argmax | 贪心 | 永远选分数最高的（温度=0） |
| Temperature | 温度 | 控制分布的尖锐程度（T<1 尖，T>1 平） |
| Top-K | 前K截断 | 只保留分数最高的K个候选词 |
| Top-P/Nucleus | 核采样 | 累积概率达到阈值就截断 |
| EOS | 序列结束 | End Of Sequence，表示生成完成 |
| Prefill | 预填充 | 一次 Forward 吞掉整个 prompt |
| Decode | 解码循环 | 每步 Forward 一个 token |
| Logit | 原始分数 | 未归一化的词表分数 |
| ChatStopIDs | 对话停止符 | im_end 等特殊 token，遇到即停止 |
