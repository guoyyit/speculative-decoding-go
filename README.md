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

## 超时、验证失败与取消

- `spec.Config.VerifyTimeoutMS > 0` 时，单次并行验证有独立截止时间。
  超时则**丢弃全部候选、缓存回滚到步前快照、退回不投机的逐 token 生成一次**，
  该步记为 `fallback / verify_timeout`，`TimeoutFallbacks` 与 `Rollbacks` 各 +1，
  不产出部分候选，序列不留在中间位置。
- 验证也可能因超时之外的原因失败（如批处理通道关闭、验证器内部错误）。
  只要请求自身 ctx 仍然存活，处理与超时完全一致：丢弃全部候选、回滚、
  退回逐 token 生成一次，该步记为 `fallback / verify_error`，
  `VerifyErrorFallbacks` 与 `Rollbacks` 各 +1。**两种原因分别计数、互不影响**，
  且单次失败不影响该请求继续完成生成，也不影响其他并发请求。
- 请求自身 ctx 取消：当前提议/验证立即返回错误，已暂存候选被回滚，
  该请求不产出 token，计为 `failure`，不影响任何其他请求。
- 三类路径分别统计：`success`（投机成功步）、`fallback`（验证失败退回步，
  按 `verify_timeout` / `verify_error` 细分）、`failure`（取消/错误步）。

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
| `EmittedTokens` | **实际交付给调用方的 token 总数**；任何请求、任何长度下都等于 `Generate` 返回的 token 数 |
| `AcceptRate` | `AcceptedTokens / ProposedTokens`，恒在 `[0,1]` |
| `TokensPerStep` | `EmittedTokens / (success+fallback+failed steps)`，由实际交付数量计算 |
| `SuccessSteps / FallbackSteps / FailedSteps` | 三类路径计数 |
| `Rollbacks` | 拒绝回滚次数（含验证失败退回的回滚） |
| `TimeoutFallbacks` | 验证超时（`verify_timeout`）退回次数 |
| `VerifyErrorFallbacks` | 非超时验证失败（`verify_error`）退回次数 |
| `RequestsOK / RequestsFailed / InFlight` | 请求级状态 |

**产出 token 口径**：一步最多可提交 `DraftK+1` 个 token；当请求的剩余生成长度
小于该值时，只有实际交付的那部分 token 被计入 `EmittedTokens`（缓存中也只保留
已交付前缀）。该口径覆盖全部接受、候选拒绝、EOS 提前结束、验证失败退回所有
路径，不漏记也不重复记。服务端聚合值精确等于各请求统计值之和，
与批次大小、等待窗口、工作协程数量无关（有专门测试固定这三种配置断言逐字段相等）。

草稿与目标越接近接受率越高；`DraftNoise=0` 时接受率为 1，
每步稳定产出 `DraftK+1` 个 token（`TokensPerStep > 1`）。

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
| `DraftK` | 每步最多提议 token 数（默认 4） |
| `VerifyTimeoutMS` | 单次验证超时；超时退回逐 token 生成，<=0 表示不设超时 |

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
- 产出 token 口径：剩余长度小于/恰好等于单步产出上限、拒绝修正恰好用尽预算、
  预算截断修正 token、EOS 提前结束——`EmittedTokens` 与每步 `Emitted` 之和
  均精确等于实际交付 token 数（全部精确数值断言）
- 验证超时退回（逐 token 产出、计数与原因正确、不等待完整验证）
- 非超时验证失败与超时退回的区分：`verify_error` / `verify_timeout` 分别计数，
  失败不影响请求继续生成，退回 token 与纯目标模型逐 token 生成逐条一致
- 请求取消不产出 token、不留中间状态
- 多请求并发隔离、单请求取消不影响他人、批处理不改变输出
- 并发长短请求混合时服务聚合统计精确等于各请求（直接生成路径）统计之和，
  且与 `BatchSize` / `BatchWindowMS` / `Workers` 无关
- 固定种子逐条复现、不同种子不同
- 有界缓存淘汰与淘汰后回滚
