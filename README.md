# speculative-decoding-go

一个可在本地运行的、支持**投机解码（speculative decoding）**的确定性模拟推理生成服务。
草稿模型一次提议多个候选 token，目标模型对整段候选做一次并行验证；
在**输出分布与目标模型逐 token 单独生成严格一致**的前提下，提高每步产出的 token 数。

模型全部为仓库内的确定性模拟实现（见 `model/`），不依赖任何外部服务或真实模型，
无第三方依赖，仅需 Go 1.25+。

## 快速开始

```bash
go run ./cmd/specserver   # 并发演示 + 统计 + 固定种子复现校验
go test ./...              # 全部自动化测试
go test -race ./...       # 并发竞态检测
```

## 目录结构

- `model/`：确定性草稿/目标模型对。分布只由（模型种子 + 最近 `ContextWindow=8`
  个历史 token）决定；草稿分布 = 目标 logits 加确定性高斯噪声（`DraftNoise`）。
  `DraftNoise == 0` 时草稿与目标在每个上下文上完全同分布，用于验证“每步产出 > 1”。
- `sampling/`：固定种子（PCG + splitmix64 种子扩展）的分类采样器，
  `Sample(p)` 与 `Float64()` 各恰好消费一次均匀抽样。
- `kv/`：每请求独立的有界前缀缓存，支持 `Truncate(acceptPoint)` 回滚；
  超容量仅淘汰最旧 token，而模型只依赖最近 `ContextWindow` 个 token，
  因此淘汰不改变任何分布（容量默认取 `max(配置, ContextWindow)`）。
- `spec/`：投机解码核心算法（提议 → 并行验证 → 概率接受 → 拒绝重采样/bonus）。
- `service/`：多请求并发调度、验证批处理（batcher）、统计聚合、panic/取消隔离。
- `cmd/specserver/`：本地演示入口。

## 采样语义（与目标模型逐 token 生成严格同分布）

设草稿候选为 `x1..xk`，草稿在各位置的分布为 `qi`，目标并行验证返回
`k+1` 个分布：`pi = p_target(· | prefix, x1..x{i-1})`（i=1..k），
以及末尾位置 `p{k+1}`。

1. 从 i=1 起依次判断候选 `xi`：以概率 `a(xi) = min(1, pi[xi] / qi[xi])`
   接受该 token，其中接受判定所用的均匀随机数来自该请求自己的固定种子采样流。
2. 在**首个被拒绝位置 j**：丢弃 `xj..xk`，从归一化残差分布
   `p'(·) = norm(max(pj(·) - qj(·), 0))` 重采样一个修正 token；
   拒绝点之后不再采样任何 bonus，本步提交 `x1..x{j-1}` + 修正 token。
3. 若 k 个候选全部接受：再从 `p{k+1}` 采样一个 bonus token，
   本步最多产出 k+1 个 token。
4. 若接受过程中采到 EOS（token 0）：序列立即终止，EOS 之后不采样 bonus。

接受规则 `min(1, p/q)` 与拒绝残差重采样是标准投机解码构造，
其数学结果是：**每个输出位置的边际分布恰好等于目标模型在该上下文下的分布**。
没有任何固定阈值或贪心截断；所有拒绝都只依据目标模型概率。
测试 `TestDistributionEquivalentToOneStepTarget` 用 20000 个独立种子做
Pearson 卡方检验（df=V-1，阈值 χ²≈40）验证两种生成方式同分布。

> 边界：`pi[xi]=0` 时接受概率为 0（目标不可能输出该 token，必拒）；
> `qi[xi]=0 且 pi[xi]>0` 时接受概率为 1。

## 拒绝回滚与有界内存

- 每步开始时记录缓存长度快照；候选 token 先“暂存”进缓存再验证。
- 拒绝：缓存截断到 `快照 + 已接受数`，再追加修正 token；
  被拒候选及其后任何状态不会进入输出，后续生成只能看到已接受前缀。
- 每次拒绝计入 `Rollbacks`。
- 缓存按容量淘汰最旧 token，配合模型的有限上下文窗口保证内存长期有界。

## 超时、验证错误与取消

- `spec.Config.VerifyTimeoutMS > 0` 时，单次并行验证有独立截止时间。
  超时则**丢弃全部候选、缓存回滚到步前快照、退回不投机的逐 token 生成一次**，
  该步记为 `fallback / verify_timeout`，`TimeoutFallbacks` 与 `Rollbacks` 各 +1，
  不产出部分候选，序列不留在中间位置。
- **非超时的验证失败**（目标/批处理返回 `context.DeadlineExceeded` 以外的任何错误，
  含内部 panic 被批处理恢复后上报）走同一条退回路径，但原因记为
  `fallback / verify_error`，由独立的 `ErrorFallbacks` 计数，`Rollbacks` 同样 +1。
  两种原因在 `StepResult.FallbackCause` 与汇总统计中严格区分、互斥；
  单次退回只影响这一步并继续完成本请求，绝不影响其他并发请求。
  退回时产生的 token 与“直接从目标模型逐 token 生成”完全一致（退回前的草稿
  提议只消耗该请求自己的随机流，不改变目标分布）。
- 请求自身 ctx 取消：当前提议/验证立即返回错误，已暂存候选被回滚，
  该请求不产出 token，计为 `failure`，不影响任何其他请求。
- 三类路径分别统计：`success`（投机成功步）、`fallback`（超时或非超时错误退回步）、
  `failure`（取消/错误步）。

## 统计口径（关键不变式）

所有统计在“每一步实际交给调用方的 token”这一口径上记账，核心不变式：

> **`EmittedTokens` 恒等于调用方实际收到的 token 数**，任何请求、任何长度都成立。

- 一步投机最多可产出 `DraftK+1` 个 token（全部接受 + bonus），但当**剩余生成预算
  不足**时（例如 `maxNewTokens` 只剩 2，而本步可产出 5），超出预算的 token 不会交付：
  它们既不出现在返回切片里，也**不计入 `EmittedTokens`**，并且从工作缓存中剔除，
  不会进入任何后续上下文。`TokensPerStep = EmittedTokens / Steps` 因此永远不被高估。
- `ProposedTokens` / `AcceptedTokens` 仍按“提议/接受”这一行为口径计数（接受率
  `AcceptedTokens / ProposedTokens` 是投机质量指标，独立于交付口径），因此接受率
  恒在 `[0,1]`；`EmittedTokens` 才与输出 token 数严格相等。
- 四条路径记账点唯一、不漏不重：
  1. **候选全部接受**：交付 `k` 个候选 + 1 个 bonus，超预算部分只从交付与缓存中剔除；
  2. **候选被拒绝**：交付已接受前缀 + 1 个残差修正 token，超预算部分同样剔除；
  3. **采到 EOS 提前结束**（候选中 EOS 或 bonus 为 EOS）：只交付到 EOS 为止
     （预算更紧时只交付预算内前缀），之后不再采样；
  4. **验证失败退回**：丢弃全部候选，恰好交付 1 个 vanilla 目标 token，记 1 次退回。
- 单请求（`spec.Decoder.Stats()`）与服务聚合（`service.Stats()`）共用同一套字段与
  口径。每个 `service.Result` 自带该请求的 `Stats` 快照；服务聚合值**精确等于各请求
  统计值之和**（`InFlight` 除外），与 `BatchSize / BatchWindowMS / Workers` 无关
  （有测试在多组配置下逐字段断言相等）。
- `Rollbacks` 同时统计“拒绝回滚”和“失败退回回滚”；超时与非超时退回另由
  `TimeoutFallbacks`、`ErrorFallbacks` 分别计数。
- 长时间运行下内存有界：每请求缓存容量固定（`CacheCapacity ≥ ContextWindow`），
  统计仅为定长计数器；批处理 channel 定长、关闭时 worker 全部退出（有 goroutine
  泄漏检测与长运行精确记账测试）。

## 并发与批处理

- 每个请求拥有独立 `Decoder`：独立种子、独立采样流、独立 KV 缓存，
  请求间不共享可变状态。
- service 内有 N 个 worker，每 worker 一个 batcher；并发验证调用按
  时间窗（`BatchWindowMS`）或容量（`BatchSize`）分组。批处理只控制**调度**：
  每个请求的概率行都由无状态目标模型用该请求自己的 history/candidates 独立计算，
  因此批次划分不可能改变任一请求的输出分布（有测试专门对比 BatchSize=1
  与大批次下同种子输出逐条一致）。
- 单个请求 panic 在自己的 goroutine 内 recover 并落到它自己的 Result，
  取消只通过该请求的 ctx 传播。

## 统计与诊断

`service.Stats()`（单请求可用 `spec.Decoder.Stats()`）：

| 指标 | 含义 |
| --- | --- |
| `AcceptRate` | `AcceptedTokens / ProposedTokens`（投机质量口径，∈[0,1]，与交付数无关） |
| `TokensPerStep` | `EmittedTokens / Steps`，分母为 success+fallback+failed 步数 |
| `EmittedTokens` | **实际交给调用方的 token 总数，恒等于各请求返回切片长度之和** |
| `ProposedTokens / AcceptedTokens` | 提议/被接受候选数（按投机行为口径，可能超过交付数） |
| `SuccessSteps / FallbackSteps / FailedSteps` | 三类路径计数，三者之和等于 `Steps` |
| `Rollbacks` | 回滚次数（拒绝回滚 + 超时/非超时错误退回回滚） |
| `TimeoutFallbacks` | 验证**超时**（`verify_timeout`）退回次数 |
| `ErrorFallbacks` | **非超时**验证错误（`verify_error`）退回次数，与超时互斥 |
| `RequestsOK / RequestsFailed / InFlight` | 请求级状态 |

每个 `service.Result.Stats` 是该请求的逐字段快照；除 `InFlight` 外，
`service.Stats()` 中每个字段都精确等于已完成请求该字段之和。

草稿与目标越接近接受率越高；`DraftNoise=0` 时接受率为 1，
每步稳定产出 `DraftK+1` 个 token（`TokensPerStep > 1`）。有噪声时接受率落入
(0,1) 的合理区间，且每步平均产出仍大于 1。

## 配置项

`model.Config`：

| 字段 | 说明 |
| --- | --- |
| `VocabSize` | 词表大小（token 0 为 EOS），默认 32 |
| `Seed` | 模型确定性种子（决定分布形状） |
| `DraftNoise` | 草稿相对目标 logits 的高斯扰动幅度；0 表示完全一致 |
| `EOSBias` | EOS logit 偏置（负值使序列更长） |
| `TargetLatencyMS / DraftLatencyMS / VerifyLatencyMS` | 模拟延迟，均可被 ctx 快速中断（≤10ms 粒度） |

`spec.Config`：

| 字段 | 说明 |
| --- | --- |
| `DraftK` | 每步最多提议 token 数（默认 4）；全部接受时一步最多交付 `DraftK+1` 个 |
| `VerifyTimeoutMS` | 单次验证超时；超时以 `verify_timeout` 退回逐 token 生成，<=0 表示不设超时。其他验证错误以 `verify_error` 独立计数 |

`service.Config`：

| 字段 | 说明 |
| --- | --- |
| `CacheCapacity` | 每请求 KV 缓存容量，自动下限为 `ContextWindow` |
| `BatchSize` | 单个验证批次最大请求数 |
| `BatchWindowMS` | 批次等待窗口（ms） |
| `Workers` | 并发 worker/batcher 数（默认 4） |

## 复现性

输出由 `(模型种子, 请求种子, 输入 prompt, 配置)` 完全决定。
`sampling.New(seed)` 使用 PCG 并以 splitmix64 扩展种子，同种子采样流逐条一致；
批处理调度不消耗任何请求的随机数。固定种子与相同输入下，
无论是否投机、是否与他人并发，输出逐条一致（见复现性测试）。

## 测试覆盖

- 分布等价性（卡方）、残差恒等式 `norm(max(p-q,0))+min(p,q)=p`、接受规则边界
- 拒绝回滚（被拒候选不进输出、缓存只留接受前缀+修正 token）
- **统计口径边界**（全部断言均为精确数值，不使用宽松下界）：
  - 剩余预算小于单步候选数（预算 1..5）时 `EmittedTokens == 返回长度`，
    且输出是大预算参考输出的逐 token 前缀；
  - 拒绝修正后恰好用尽预算：修正 token 必须交付、计数精确为 3、缓存精确；
  - bonus 采到 EOS 但预算更紧：EOS 被裁掉，交付数与统计仍精确相等；
  - 每步 `Emitted` 之和等于请求总交付数。
- **退回原因区分**：非超时验证错误记 `verify_error / ErrorFallbacks`（含“先错后恢复”
  与“持续失败”两种），超时记 `verify_timeout / TimeoutFallbacks`，二者互斥；
  退回输出与纯目标逐 token 生成逐 token 一致（复放提议随机流后严格比对）
- 验证超时退回（逐 token 产出、计数与原因正确、不等待完整验证）
- 请求取消不产出 token、不留中间状态
- **并发汇总精确性**：长短请求（预算 1..21）混合、多组 `BatchSize/Workers` 配置下，
  聚合逐字段等于各请求 `Result.Stats` 之和，`EmittedTokens == 总返回长度`
- **超时/错误并存隔离**：同一服务中慢请求超时、注入错误请求走 `verify_error`、
  健康请求零退回，三类计数互不串扰
- **长时间运行**：24 个长请求（每请求 60 token）后聚合仍精确、`InFlight` 归零、
  `Close` 后无 goroutine 残留
- 多请求并发隔离、单请求取消不影响他人、批处理不改变输出
- 固定种子逐条复现、不同种子不同
- 有界缓存淘汰与淘汰后回滚

## 本轮行为的验证方法

```bash
go build ./...
go vet ./...
go test ./...          # 全部精确断言用例
go test -race ./...    # 并发竞态 + 隔离性
go run ./cmd/specserver
```

本轮在开发期还通过“修复前版本 vs 修复后版本”的跨版本对拍确认**输出序列零变化**：
在噪声 `{0,0.3,0.8,1.5}` × EOS 偏置 `{-6,-2.5,0}` × 词表 `{8,32,64}` ×
种子 40 个 × 预算 7 档 × 4 种提示长度（共 30240 例），以及 5400 例缓存容量扫描、
120 例超时退回扫描上，两个版本对每个 `(配置,种子,预算)` 生成的 token 流哈希
**逐位相同**；分布一致性仍由 20000 种子卡方检验
（`TestDistributionEquivalentToOneStepTarget`）保证。修复仅改变统计记账与缓存
收敛，不改变任何接受/拒绝/采样判定。
