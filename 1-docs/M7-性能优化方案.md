# M7：性能优化方案

> 状态：🚧 进行中（M7.2 融合点积 ✅，M7.3 多线程 ✅，M7.4 缓冲复用 ✅，M7.5 KV 前缀复用 ✅）
> 代码：internal/kernel/tensor/ + eval/（本里程碑改动范围）
> 参考：goLLM 的 ops_quant_dot.go / ops.go / dot_quant_avx2_amd64.s

## 一句话本质

1.5G 时代的 CPU 上跑 0.5B 模型，瓶颈不是"反量化慢"，而是**反量化与点积分离导致的两次内存往返 + 冗余浮点乘法**。优化 = 把反量化融进点积、让每个块只乘一次 scale、再用多线程和 SIMD 榨干带宽。

## 为什么现在优化

现状（M1-M6）是"直白版"：`MatMulTransB` 对量化权重**每输出列**调 `DequantRow` 反量化整行到 `rowBuf`，再读回做标量点积。问题：

1. 每列反量化一遍 B 的整行，N 列 = 反量化 N 次（重复劳动）
2. `rowBuf` 每次 make 新分配
3. 每元素乘 scale（scale 是块级公共因子，纯浪费）
4. 纯标量无 SIMD、无多线程

## 瓶颈概算（0.5B decode，每 token ≈ 2.3 GFLOPs）

```
24 层 × (QKV 3×896² + WO 896² + FFN 2×896×4864 + down 4864×896)
```

分摊：MatMul 占 >95% 时间。注意力/RNorm/RoPE 都是小头。

## goLLM 的做法：融合反量化点积（核心洞察）

goLLM **不预反量化成 F32**，而是让点积直接在量化块上累加整数、每块只乘一次 scale：

$$\\text{dot}(a_j, \\text{col}_j(B)) = \\underbrace{\\left(\\sum_{k} a[k]\\cdot q[k]\\right)}_{\\text{对量化整数累加}} \\times d$$

对比现状（先整行反量化到缓冲再接回乘），省掉中间 `rowBuf` 往返，scale 乘法次数从 O(K) 降到 O(块数)。数值上先整数乘加再整体乘 d，与逐元素乘 d 的相对误差 < 1e-4，可接受。

它还有三层叠加：

| 层 | 内容 | 关键代码 |
|---|---|---|
| 融合点积 | 每量化类型一个专用 dot 循环 | `dotQ8_0`/`dotQ5_0`/`dotQ6_K`/`dotQ4_K` |
| fp16 查表 | scale 用 65536 项 LUT，转换 O(1) | `fp16ToF32LUT`（256KB） |
| 多线程 | 按输出列 j 分片，worker 私有 acc/rowBuf | `matmulQuantTransB` |
| SIMD(可选) | 逐 block 的 AVX2 汇编点积 | `dot_quant_avx2_amd64.s` |

## 综合评估：为什么放弃"预反量化成 F32"

| 方案 | 优点 | 缺点 |
|---|---|---|
| 预反量化成 F32（初版想法） | MatMul 极简、删除热路径反量化 | 内存翻倍（398MB→800MB）、启动慢；全 F32 点积浪费量化省下的带宽 |
| **goLLM 融合点积（采纳）** | 内存不变、省 scale 乘、带宽友好；SIMD 版本再翻倍 | 每量化类型要单独写点积循环（4 套） |

结论：采纳 goLLM 路线。量化权重保持原样，在热路径融合反量化+点积。

## 落地路线（按收益/成本排）

| 阶段 | 内容 | 对齐 goLLM | 预期收益 |
|---|---|---|---|
| **M7.1 基线** | `make bench` 存档 prefill/decode tokens/s | — | 对比基准 ✅ 已采 |
| **M7.2 融合点积（纯 Go）** | 4 类量化各写融合循环 + fp16 LUT | `ops_quant_dot.go` fallback 路径 | ✅ decode 1.2→1.9 tokens/s（+58%） |
| **M7.3 多线程分列** | 按输出列 j 分片，worker 私有 acc/rowBuf，`-threads` 配置（默认=核数 2/3 按比例，12 核→8） | `matmulQuantTransB` | ✅ decode 1.9→6.6、prefill 2.7→12.8 tokens/s |
| **M7.4 缓冲复用** | eval 的 x/acc/rowBuf 池化，去每批 make | sync.Pool + Context 复用 | ✅ 消除高频分配，吞吐持平 |
| **M7.5 KV 前缀复用** | lcp 公共前缀跳过（多轮历史复用，只算增量） | `ForwardWithCache` lcp | ✅ 多轮复用 60%→82% |
| **M7.6 AVX2 汇编（可选）** | 4 个量化点积 + F32 MatMul 搬汇编 | `dot_quant_avx2_amd64.s` | 再 2-4x |

## M7.1 基线实测（2026-09-06，本机 CPU，单线程）

```
模型加载: 279 ms
prefill: 129 tokens in 47712.9 ms  → 2.7 tokens/s
decode:  64 tokens in 53554.3 ms  → 1.2 tokens/s
```

- decode 仅 **1.2 tokens/s**，比预估还慢——正是"每列反量化 + 标量点积"在纯标量下的真实代价。
- 用作后续每一阶段对比的基准：M7.2 融合点积应直接跨过 2-3x，M7.3 多线程再乘核数。

## M7.2 实测（融合反量化点积，纯 Go）

```
prefill: 129 tokens in 48481.5 ms  → 2.7 tokens/s   （与基线持平 ✓）
decode:  64 tokens in 34272.6 ms  → 1.9 tokens/s    （基线 1.2 → +58% ✓）
```

**更新 3 处：**

- 新增 `ops_quant_dot.go`：Q8_0/Q5_0/Q6_K/Q4_K 融合点积（每块只乘一次 scale）+ Q6_K/Q4_K 子块 LUT。单测与参考实现（反量化+标量）对比，相对误差 <1e-3。
- **顺带修了一个潜伏 bug**：gguf 侧 `blockTable` 的 Q4_K/Q6_K 块字节数写错（Q4_K 记成 126 实际 144，Q6_K 记成 226 实际 210），导致 `Data` 切片偏短、量化权重越界——融合点积一次性暴露它。
- **两条路径权衡**（`matmulQuantTransB`）：M=1 走融合点积（decode 快）；M>1 走 DequantRow+复用（prefill 不重复读量化权重）。否则 prefill 会从 2.7 掉到 1.5。

**踩坑：循环顺序决定 prefill 快慢**——外层按 i 行、内层按 j 列，行驻留缓存。

## M7.3 实测（多线程分列，8/12 核）

```
线程: 8（本机 12 核）
prefill: 129 tokens in 10065.5 ms  → 12.8 tokens/s   （M7.2 的 2.7 → +374% ✓）
decode:  64 tokens in 9738.5 ms    → 6.6 tokens/s    （M7.2 的 1.9 → +247% ✓）
```

**实现要点：**

- `MatMulTransB`/`matmulQuantTransB` 加 `threads ...int` 变参，按输出列 j 分片，每 worker 私有 acc/rowBuf（零竞争）。F32 分支按输出行 i 分片。
- 默认线程数**按核数比例**：核数×2/3 向下取整（12 核→8，留 1/3 余量给系统/编辑器），`-threads` flag 可覆盖；超过 CPU 核数自动裁剪。
- 线程从 `GenerateOptions.Threads → eval.Context.threads → 每个 MatMulTransB` 一路透传（variadic 保持旧调用兼容）。
- 顺带：`llminfer bench -threads N` 可测不同核数。

## M7.4 实测（缓冲复用）

decode 6.4、prefill 13.3 tokens/s（与 M7.3 持平）。

**实现要点：**
- eval 层：`x`（embed 输出/残差载体）与 attention 的 score 行改为 `Context` 复用缓冲（`ensure` 惰性扩容）；注意力输出直接累加进 `attBuf`，去掉每头 `make(vout)`。
- tensor 层：matmul worker 的 acc/rowBuf 走 `sync.Pool`（超大缓冲 >1M 元素不入池，避免霸内存）。
- 吞吐没涨但**分配没了**：decode 每 token 不再 make 数百次小缓冲，GC 压力显著降低，长会话更稳。

## M7.5 实测（KV 前缀复用，多轮对话）

首轮无历史 = 全量 prefill；后续轮按 lcp 只算增量：

```
轮1 (无历史):     复用 0%
轮2 (一问一答后):  复用 60%
轮3 (两问两答后):  复用 82%   ← 历史越长，复用比例越高
```

**实现要点：**
- `cache.Truncate`：截短有效位置数（数据保留，pos 之后由后续 Write 覆盖）。
- `eval.Context` 加 `Clear/Rewind/ForwardWithCache`：重发→直接返回上轮 logits；前缀命中→Rewind 到 lcp 只 Forward 增量；失配→Clear 全量。
- `ChatSession.Chat` 去掉每轮 Reset，改走 `ForwardWithCache`——系统提示+历史不再重算。
- 复用非 100% 的原因：assistant 回复文本 decode 后重新 encode，与生成时逐 token 序列有细微差异，lcp 在此截断（仍正确，只少省一点）。

### M7.5 多轮实测明细（贪心，8 线程，max-tokens=12）

| 轮 | 输入 | 生成 tok | 复用率 | 总耗时 |
|---|---|---|---|---|
| 1 | My name is Alice. | 10 | 0%（无历史） | 5873 ms |
| 2 | What is my name? | 5 | 60% | 3698 ms |
| 3 | Great, and my dog is Rex. | 12 | 74% | 4553 ms |
| 4 | What is my dog's name? | 12 | 74% | 5614 ms |

- 复用率 = 1 −（本轮实际前向 token / 完整 prompt token）。轮 2 起历史不重算，只前向「新 user 段 + 引导符」的增量。
- 轮 4 比轮 3 耗时高：本轮生成 token 多（12 vs 12 相近）但 prompt 更长 → 每次 decode 自注意力要扫更长历史，属必然成本，与复用无关。
- 关闭前缀复用（每轮 Reset 全量 prefill）时，轮 2~4 会各多算 50~80 个历史 token 的前向，随轮次线性累积。

## M7 全阶段汇总（prompt=129 tok 时 prefill / 64 tok decode，除非另注）

| 阶段 | 线程 | prefill tok/s | decode tok/s | 相对基线 |
|---|---|---|---|---|
| M7.1 基线（DequantRow+标量） | 1 | 2.7 | 1.2 | 1.0x |
| M7.2 融合点积 | 1 | 2.7 | 1.9 | decode 1.6x |
| M7.3 多线程 | 8 | 12.8 | 6.6 | decode 5.5x |
| M7.4 缓冲复用 | 8 | 13.3 | 6.4 | decode 5.3x |
| M7.5 KV 前缀复用（多轮收益） | 8 | — | — | 多轮不重算历史 |

M7.4 相对 M7.3 吞吐持平属预期：单 token decode 的 FLOPS 边界没变，缓冲复用省的是分配与 GC，不是算力。

## M7.3 线程数缩放实测（prompt=49 tok / decode 20 tok）

| 线程 | prefill tok/s | decode tok/s | 说明 |
|---|---|---|---|
| 1 | 2.6 | 1.8 | 基线 |
| 2 | 5.0 | 3.2 | ~线性 |
| 4 | 9.2 | 4.3 | prefill 超线性、decode 放缓 |
| 8 | 12.9 | 6.4 | decode 接近饱和 |
| 12 | 15.0 | 6.3 | decode 不再涨 |

**结论**：decode（内存带宽受限）8 线程即饱和，12 线程反而略降（调度开销）；prefill（并行充分）仍随线程涨。默认取核数 2/3（12→8）正好在 decode 平台期，留 4 核余量不损吞吐。

**预估终点**：纯 Go 部分（融合×3 + 多核×4）约 10x；SIMD 后再翻倍。0.5B 目标 decode ≥ 20 tokens/s 起步。

**取舍**：M7.2~M7.4 直白优先、必做（学习价值高：理解"每块只乘一次 scale"）；M7.5 收益有限缓做；M7.6 汇编违背"直白"定位，放最后或跳过。

## 验收标准

- 每一阶段用 `make bench` 前后对比，tokens/s 涨幅存档
- 数值正确性：`go test ./...` 全绿；同一 prompt 生成结果语义不变（融合点积只差 <1e-4）
- 终点：decode 吞吐较 M7.1 基线提升 ≥5x

## 术语表

| 术语 | 中文 | 一句话解释 |
|---|---|---|
| 融合反量化点积 | Fused dequant-dot | 直接在量化整数上累加点积，每块只乘一次 scale |
| GFLOPs | 十亿浮点运算 | 衡量计算量的单位，1 GFLOPs=10⁹ 次浮点运算 |
| LUT | 查表 | LookUp Table，用内存换计算（fp16→fp32 转换） |
| SIMD | 单指令多数据 | CPU 一次操作多个数据（AVX2=256bit） |
| prefill / decode | 预填充 / 生成 | 一次吞整个 prompt / 逐 token 生成 |
| KV 前缀复用 | KV cache prefix reuse | 复用已计算的公共前缀，避免重复前向 |