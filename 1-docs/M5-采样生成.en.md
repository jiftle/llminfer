# M5: Sampling & Generation Loop (Logits → Token → Text)

> Status: ✅ Complete (greedy / temperature / top-k / top-p all pass, qwen2.5-0.5b generates coherent text)
> Code: internal/kernel/sampler/ + cmd/generate.go
> Depends on: M4 forward inference; M6 adds the ChatML conversation template

## The Essence in One Sentence

M4's forward pass outputs a "score per word" (logits), but logits are not text. M5 is the three steps that turn logits into words:

1. **Sampling**: pick one candidate from the 151936 candidate words according to a strategy
2. **Decoding**: translate the token id into a human-readable string
3. **Looping**: feed the just-generated token back in and repeat steps 1-2 until EOS or the limit is reached

## Sampling Strategies (Plain Language)

### Greedy (Temperature ≤ 0)

Always pick the word with the highest score:

$$\text{id} = \arg\max_i \ \text{logits}[i]$$

Deterministic, but easily repetitive and dull. Like always choosing the safest answer on an exam.

### Temperature Scaling (Temperature > 0)

Divide the logits by $T$ and then apply softmax:

$$p_i = \frac{\exp(\text{logit}_i / T)}{\sum_j \exp(\text{logit}_j / T)}$$

- $T = 0.2$: very peaked distribution, almost always picks the highest (close to greedy)
- $T = 0.8$: moderately random, balances diversity and quality (default value)
- $T = 100$: very flat distribution, almost uniform random (nonsense)

### Top-K Truncation

Keep the $K$ tokens with the highest probability and zero out the rest:

$$\text{candidates} = \text{TopK}(p, K)$$

Prevents picking completely irrelevant words.

### Top-P Nucleus Sampling

Sort probabilities in descending order and accumulate; truncate once the cumulative sum reaches the $P$ threshold:

$$\text{keep} = \min\left\{n : \sum_{i=1}^{n} p_{(i)} \geq P \right\}$$

where $p_{(i)}$ is the probability in descending order. More flexible than Top-K — when the distribution is very peaked it keeps only 1-2 words, and when it is very flat it keeps hundreds.

**Combined use**: Top-K and Top-P can be applied together — first trim by K, then trim by P, taking the intersection.

### Drawing Lots (Final Sampling)

Draw lots from the truncated probability distribution by cumulative probability:

$$\text{id} = t \quad \text{where} \quad \sum_{i=1}^{t-1} p_i < r \leq \sum_{i=1}^{t} p_i$$

where $r \sim U(0, 1)$ is a uniform random number.

## Generation Loop (generate.go)

    prompt → Encode → [token_ids] → Forward(prefill) → logits
                                                                ↓
                                                      Sample(logits) → token_id
                                                                ↓
                                                      Decode(token_id) → character
                                                                ↓
                                                      Forward([token_id]) → next_logits
                                                                ↓
                                                      ... repeat until EOS

Key invariant: **only Forward one token per step**. The KV Cache guarantees history is never recomputed.

## Stopping Conditions

- **EOS**: tokenizer.ggml.eos_token_id (qwen2.5 = 151645)
- **Conversation markers**: special tokens such as im_end, im_start (ChatStopIDs)

## M5 Acceptance

**1. Greedy mode outputs coherent English**:

    $ llminfer run -temperature 0 -max-tokens 30 models/qwen2.5-0.5b.gguf "The capital of France is"
    Output: Paris. It is the most populous city in France, with a population of over 2 million.

**2. Temperature sampling outputs coherent Chinese**:

    $ llminfer run -temperature 0.8 -max-tokens 64 models/qwen2.5-0.5b.gguf "Hello"
    Output: 一下，我是来自中国的一名留学生...

**3. Sampler unit tests**: all 6 pass: greedy, temperature=0, Top-K, Top-P, reproducible seed, uniform distribution at high temperature.

## Pitfalls Encountered

1. **Go flag argument order**: flags must come before positional arguments. Putting them after does not take effect; only leading placement works. Go's flag.NewFlagSet parses sequentially and stops as soon as it hits a non-flag argument.
2. **Tokenizer needs GGUFFile**: model.Load closes the file after reading it, but the tokenizer needs the vocabulary and other metadata. Read it once more in the generation loop (lightweight — only reads metadata, not tensor data).
3. **Unused flag warning**: after removing the old flag, the new flag must be passed to Generate, otherwise there is a compile warning.

## Documentation ↔ Code Mapping

| Doc Section | Code Location |
|---|---|
| Sampling strategies | sampler.go Sample() |
| Greedy / temperature / softmax | strategies.go (inside the sampler package) |
| Generation loop | cmd/generate.go Generate() |
| run command entry | cmd/command.go run() |
| Stopping conditions | generate.go ChatStopIDs + EOS |

## Glossary

| Term | 中文 | Explanation |
|---|---|---|
| Sampling | 采样 | Randomly pick one token from a probability distribution |
| Greedy/Argmax | 贪心 | Always pick the highest score (temperature=0) |
| Temperature | 温度 | Controls how peaked the distribution is (T<1 peaked, T>1 flat) |
| Top-K | 前K截断 | Keep only the K highest-scoring candidate words |
| Top-P/Nucleus | 核采样 | Truncate once the cumulative probability reaches the threshold |
| EOS | 序列结束 | End Of Sequence, marks that generation is complete |
| Prefill | 预填充 | Consume the entire prompt in a single Forward |
| Decode | 解码循环 | Forward one token per step |
| Logit | 原始分数 | Unnormalized vocabulary score |
| ChatStopIDs | 对话停止符 | Special tokens such as im_end; generation stops when one is hit |
