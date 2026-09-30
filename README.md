# Speculative Decoding (Go, local deterministic simulation)

一个完全本地运行、无外部依赖的投机解码（speculative decoding）推理生成服务。
草稿模型（draft）一次提议多个候选 token，目标模型（target）对整段候选做
**一次并行验证**，按目标模型的真实概率接受连续前缀；在首个被拒位置用残差
分布重新采样修正。输出分布与「只用目标模型逐 token 串行生成」**严格一致**。

所有模型均为确定性模拟：给定种子与上下文，目标/草稿分布是纯函数，不访问
网络、不加载真实权重。

## 目录结构

- `model/` — 确定性模拟的 `DraftModel`（提议）与 `TargetModel`（验证/单步）。
- `sampling/` — 采样、接受规则 `min(1, p/q)`、残差重采样、单步投机逻辑、随机流。
- `cache/` — 每请求序列缓存：临时追加 / 快照 / 回滚 / 提交，回滚即丢弃尾部。
- `engine/` — 单请求生成主循环、每步验证超时与退回。
- `server/` — 多请求并发提交、批调度语义、panic/取消隔离。
- `stats/` — 线程安全的诊断计数。
- `main.go` — 固定种子的并发演示。

## 采样语义与接受规则

设位置 `i` 的草稿候选为 `x`，草稿分布 `q = P_draft(.|prefix)`，目标分布
`p = P_target(.|prefix)`。

1. 草稿按 `q` 自回归提议最多 `k` 个 token；目标对 `prefix + candidates`
   做一次批量前向，返回 `k+1` 个位置的目标分布 `p_i`（多出的最后一个
   分布用于全部接受时的 bonus token）。
2. 逐位置以概率

   ```
   a_i = min(1, p_i[x_i] / q_i[x_i])
   ```

   决定是否接受候选 `x_i`。这里**没有任何固定阈值或贪心截断**：接受与否
   完全由目标模型在该 token 上的概率决定。
3. 第一个被拒位置 `j`：接受 `x_0..x_{j-1}`（连续前缀），拒绝 `x_j`，并从
   调整后的残差分布

   ```
   p'(v) = max(0, p_j(v) - q_j(v)) / Σ_u max(0, p_j(u) - q_j(u))
   ```

   重新采样一个修正 token。**被拒绝的候选不会进入输出**。
4. 若 `k` 个候选全部接受，再从最后一个目标分布免费采样一个 bonus token。

### 为什么分布与目标模型完全一致

标准投机采样恒等式。对任意输出 token `y`，单位置边际为

```
Pr_spec(y) = q[y]·min(1,p[y]/q[y])
           + (Σ_x q[x]·(1 - min(1,p[x]/q[x]))) · p'(y)
           = p[y]
```

即接受项与「拒绝后残差修正」项恰好合成原始目标分布。测试以两种方式覆盖：

- 精确枚举边际（`TestMarginalEquivalence`）逐分量断言等于 `p`；
- 端到端卡方检验（`TestDistributionEquivalenceEndToEnd`）对投机与纯目标
  生成按位置比较经验直方图。

## 随机流与可复现性

每个请求从自己的种子派生**两条相互独立的确定性 PCG 流**：

- draft 流：仅用于采样草稿候选；
- target 流：用于接受判定、残差修正、bonus 与非投机退回采样。

纯目标生成只消费 target 流，因此一次投机轮无论是否消耗过 draft 流，都不会
改变后续目标采样的对齐——这保证了**超时退回时该位置的 token 与逐 token
目标生成完全相同**。固定种子 + 相同输入 ⇒ 逐条一致（`TestFixedSeedReproducibility`）。

## 拒绝回滚与内存有界

每轮开始记录序列快照；验证失败/超时/拒绝后，接受点之后的全部临时缓存状态
被丢弃（`cache.Rollback` 会重建底层切片，尾部数组不残留）。缓存只保留单一
底层数组，内存随最长序列有界增长，不随投机轮数累积。随后生成只能看到已
接受的前缀。

## 超时、取消与三类路径

- **投机成功**：候选被接受（可能含修正/bonus），计入 `steps`。
- **退回（fallback）**：单步验证超过 `VerifyTimeoutMS` 或出现瞬时验证错误时，
  回滚整轮、丢弃所有候选，改用一次非投机目标单步产出恰好一个 token，并记录
  退回原因（`verify_timeout` / `verify_error`）。**绝不产出部分候选**，序列
  也不会停在中间位置。
- **失败（failure）**：请求级 context 被取消，或非瞬时错误且退回也失败时，
  请求终止并计入失败。

每个请求在独立 goroutine 中运行，拥有独立缓存与随机流；panic 被
 `recover` 转换为错误响应。单个请求的失败或取消不影响其他请求。

验证批次如何划分只影响墙钟时间：模拟模型是输入的纯函数，批大小
（`server.New` 的 `batchSize`）不改变任何请求的输出分布。

## 统计指标

`stats.Counters`（线程安全）记录：

- 接受率 `acceptance_rate = accepted / proposed`；
- 每步平均产出 `avg_tokens_per_step = output / (投机步数 + 退回步数)`；
- 回滚次数、退回总次数及按原因的细分、失败次数。

当草稿与目标分布接近（小 `Noise`）时接受率接近 1，`k=3` 时每步平均产出
接近 4（3 候选 + bonus），`TestTokensPerStepGreaterThanOneWhenClose` 断言
该值严格大于 1。

## 配置项

`model.Config`

- `VocabSize`：词表大小（默认 16）。
- `Seed`：模拟模型的基础种子（决定确定性权重）。
- `Temperature`：softmax 温度，越小分布越尖锐。
- `Noise`：草稿相对目标的 logit 扰动幅度。`0` 时草稿分布与目标严格相同；
  越大草稿越差、接受率越低。
- `LatencyMS`：模拟单次前向的耗时（0 表示不 sleep，可被 context 取消）。

`engine.Config`

- `MaxDraftTokens`：每轮草稿提议数 `k`。
- `MaxTokens`：序列目标总长度（含 prompt）。
- `VerifyTimeoutMS`：单轮验证超时，超时退回非投机单步；0 表示不设超时。

`server.New(draft, target, batchSize, defaultCfg)`：`batchSize` 声明验证批次
容量（纯函数模拟下仅影响调度，不影响分布）。

## 运行

```bash
go run .            # 固定种子并发演示 + 诊断输出
go test ./...       # 全部测试
go test -race ./... # 并发竞态检测
```

## 测试覆盖

- 分布等价性（精确边际 + 端到端卡方）；
- 接受规则与残差分布单元测试；
- 拒绝回滚、拒绝 token 不泄漏、防御性拷贝、内存重置；
- 多请求并发状态隔离、固定种子复现；
- 请求取消不影响同伴、panic 隔离；
- 每轮验证超时退回且输出与纯目标生成逐 token 一致、退回原因记录；
- 草稿接近目标时每步平均产出 > 1；
- 统计计数在并发下正确。
