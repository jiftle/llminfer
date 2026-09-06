# M4：前向推理（逐层 Transformer + KV Cache）

> 状态：✅ 完成（8-token 序列的 logits top-20 合理，前向数值校验通过）
> 代码：`internal/kernel/cache/` + `internal/kernel/eval/`
> 依赖：M1 gguf、M2 tokenizer、M3 tensor 算子；M5 在之上做采样生成循环，M6 在其上加 KV 复用做多轮对话

## 一句话本质

M3 把"模型文件"变成了"能算的算子仓库"，但**光有算子不会推理**。M4 就是照着 Transformer 论文的公式，把算子按正确顺序串起来：

```
查表(embed) → 24 层×[注意力子块 → 前馈子块] → 输出头(logits)
```

外加一个"省钱的抽屉"——KV Cache，让每生成一个词不用把所有历史重新算一遍。

## 前向的完整数据流（对照 eval.go）

输入：一批 token id，比如 `[3, 562, 18727, ...]`。输出：整词表的 logits（每个词一个分数）。

```
① embed         token id → 向量行 x[n, nEmb]        （查 token_embd 表）
② 逐层 forwardLayer(li)
    注意力子块：
      a. RMSNorm → normBuf（x 保留给残差）
      b. QKV 投影（3 个 MatMulTransB）+ bias
      c. RoPE 旋转 q、k（带绝对位置）
      d. writeKV 把 q 之外的新 K/V 入 cache
      e. attention（GQA + 因果掩码）→ attBuf
      f. WO 投影 + 残差：x += WO·att
    前馈子块（SwiGLU）：
      g. RMSNorm → W1·x(门) ⊙ SiLU(W3·x) → W2 → 残差
③ 输出头        RMSNorm → output投影 → 取最后一行的 logits
```

关键不变量：**残差就地累加**。每一步规范化都先拷贝出 normBuf 再对 normBuf 归一化，`x` 本身原封不动，等子块算完再加回去。

## 注意力算法精读（attention.go）

### 单头注意力公式

对第 $i$ 个 token、第 $h$ 个 Q 头，KV 头 $kvH = h / \text{group}$（GQA 分组）：

$$\text{score}[p] = \frac{\langle q_{i,h},\ k_{p,kvH} \rangle}{\sqrt{d_{head}}} \quad p = 0, ..., t_{pos}$$

$$\text{att}_{i,h} = \sum_{p=0}^{t_{pos}} \text{softmax}(\text{score})_p \cdot v_{p,kvH}$$

其中 $d_{head} = 64$（headDim），$t_{pos}$ 是当前绝对位置。

### 因果掩码

每个 token 只跟"自己和更早的"比，看不到未来。实现上不是显式加 $-\infty$ 矩阵，而是直接把 score 循环上界设成 $t_{pos}$——天然因果。

### GQA 分组

$$\text{group} = \frac{\text{HeadsCount}}{\text{HeadsKV}} = \frac{14}{2} = 7$$

即每 7 个 Q 头共用一个 KV 头（$kvH = h / \text{group}$），省 KV 缓存显存，代价是表达力稍弱。

### 缩放因子

除以 $\sqrt{d_{head}}$ 的原因：点积 $\langle q, k \rangle$ 的方差随维度 $d$ 线性增长。除以 $\sqrt{d}$ 把方差拉回 1，softmax 不会饱和（否则分布退化成 one-hot，梯度消失）。

## KV Cache 是什么（大白话）

注意力要"当前 Q 点和历史上每个位置的点积"。

- **不缓存**：生成第 100 个词时，把前 99 个词的 K/V 全重新算一遍——每次翻倍，指数爆炸。
- **缓存**：第 1 词算过一次 K/V 后放抽屉里，后面每步只算"新词的 Q、K、V"，历史的直接取。

```
位置 0  算 K0,V0  →  存 [0]
位置 1  算 K1,V1  →  存 [1]，取 K0,V0 配对
位置 2  算 K2,V2  →  存 [2]，取 K0,K1,V0,V1 配对
...
```

内存布局 `[layer][pos][kvHead][headDim]`，K 半区 + V 半区，各自独立累进。`cache.go` 就是这张"密度板抽屉"。

## FFN 公式（SwiGLU）

每层的前馈子块用 SwiGLU（Gated Linear Unit）：

$$\text{FFN}(x) = W_2 \cdot \left[ \text{SiLU}(W_1 \cdot \hat{x}) \odot (W_3 \cdot \hat{x}) \right]$$

其中 $\hat{x} = \text{RMSNorm}(x)$，$\odot$ 是逐元素乘（门控），$W_1$ 是 gate 权重，$W_3$ 是 up 权重，$W_2$ 是 down 权重。

对比标准 FFN $\text{FFN}(x) = W_2 \cdot \text{SiLU}(W_1 \cdot x)$，SwiGLU 多了一个 $W_3$ 分支做门控，表达力更强。

## 残差连接

$$x = x + \text{子块输出}$$

两处残差：注意力子块后 + FFN 子块后。直白版就是 `addResidual(x, layerOut)`。

## 我这版踩过的坑（隐形知识，血泪）

1. **RMSNorm 一维张量的"行"暗坑（最大坑，logits 全 0）** 🤯
   刚写完前向，跑出来 logits 全是 0。查了半天：`x`（embed 结果）正常非零，但一过 RMSNorm 就变 -0。
   根因：像 `attn_norm.weight` 这种是**一维张量**（长度 = nEmb = 896），它在表里只有 ne0=896，没有 ne1。而 `DequantRow` 读 `row` 行的偏移用的是 `row * ne0`，`AsFloat32` 却用 `NE[1]` 当行数 → 一维张量 `NE[1]=0` → 一个元素都没反量化 → w 全 0 → 归一化输出全 0。
   修复：加 `Tensor.Rows()`——一维/`NE[1]==0` 视为单行，二维以上用 `NE[1]`。**`DequantRow` 的"行"和"实际矩阵行数"在 Go 里的粒度必须严格一致。**
2. **类型不齐的编译地狱**：ROPE/算子接口要 uint32（模型字段是 int），来回强转容易在函数签名之间"接力"错。经验：一个函数的维度参数用同一种类型表达，跨层统一。
3. **验证不靠"感觉"靠开关**：为了定位，临时用环境变量 `FEIYU_DEBUG` 在每层打印 max/min，一步步确认是"embed 错 → 归一化错 → 投影错"哪一环，而不是瞎猜。

## M4 验收

**单 token + 8-token 序列，logits top-20 自洽**（同一输入两份独立实现互不矛盾）：

```
单 token (token=3)               8-token 序列 [3,562,18727,1055,15496,11,13,2167]
top-20: 100835,282,2889,...       top-20: 11,1661,7010,264,...
max     10.278984                 max     15.58272  → top-1 指向 token 11 ✓
```

8-token 序列尤其关键：它经过多层注意力、尝到了"历史"（KV cache 里存了 7 个早前位置），top-20 仍全对 → 证明 GQA + KV Cache + 全部算子在**多位置**都正确，而不只是单个 token 的自圆其说。

单元测试全绿（`go test ./...`）。

## 文档 ↔ 代码对应

| 文档章节 | 代码位置 |
|---|---|
| ① embed 查表 | eval.go `Forward`（token_embd.DequantRow 逐行取） |
| ② 注意力子块 | eval.go `forwardLayer` 前半 |
| b) QKV+bias | eval.go `forwardLayer` + `addBias` |
| c) RoPE | eval.go `forwardLayer` → tensor.RoPE |
| d) writeKV | eval.go `writeKV` |
| e) attention | attention.go `attention`（含 GQA + 缩放） |
| ② 前馈子块 | eval.go `forwardLayer` 后半（SwiGLU） |
| ③ 输出头 | eval.go `Forward`（OutputNorm + Output 投影） |
| KV Cache | cache.go `KVCache` |
| 残差/归一化缓冲复用 | eval.go `ensure`、`addResidual` |

## 术语表

| 术语 | 中文 | 一句话解释 |
|---|---|---|
| Forward | 前向推理 | 输入 token → 逐层算 → 输出 logits 的整个过程 |
| Logits | 原始分数 | 未归一化的词表分数，越大越可能（采样前） |
| Residual | 残差连接 | x = x + 子模块输出，防梯度消失、帮深网络训练 |
| RMSNorm | 层归一化 | 归一化用的，M3 已有算子 |
| K/V Cache | 键值缓存 | 存历史 K/V 避免重复计算注意力 |
| GQA | 分组查询注意力 | group 个 Q 头共用一个 KV 头，省缓存显存 |
| Causal Mask | 因果掩码 | 每个位置只能看自己及之前，禁止看未来 |
| Context | 上下文 | 一次会话的推理状态（KV + 位置 + 缓冲） |
| HeadDim | 单头维度 | embedding/头数，这里 = 896/14 = 64 |
| Embedding/查表 | 词向量表 | token id → 对应向量行 |
| SiLU/SwiGLU | 门控线性单元 | FFN 里 gate×up 的乘性组合 |
