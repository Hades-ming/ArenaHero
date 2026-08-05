# 当前优化迭代计划

本文件是当前轮次的唯一任务板。长期流程见 `OPTIMIZATION_LOOP.md`，最终目标见 `GOAL.md`。

## 轮次元数据

| 字段 | 当前值 |
|---|---|
| 轮次 | `2026-08-06-R1` |
| 基线提交 | `4ed9c8d393e0b03e6fa0c147d5d926912deba28e` |
| 规则 / SDK | gameplay `v0.13` / Python SDK `0.2.8` |
| 官方版本复核 | 2026-08-06，官方 source/version 与本地 bundle 一致 |
| 最终目标 | 长期得到最多资源 |
| 主代理 | 当前 Codex 主任务 |
| 专家模型 | 用户要求 Luna；当前接口未暴露，首轮临时降级为 `gpt-5.6-terra` |
| live 服务 | `io.arenahero.tactic`，只允许一个 `play.py` |

## 已完成的前置处置

### `OPS-001` 清除多客户端计划覆盖

- **状态**：`STATIC_VERIFIED -> LIVE_CANARY`
- **证据**：曾同时存在 PID `58672/58678/59898`，同一 Tick 被写入 2-5 次，且不同客户端
  生成了不同路线。所有客户端共享同一个 Agent 槽，后提交会完整覆盖前提交。
- **处置**：终止两个非 LaunchAgent 孤儿进程，重启受管服务以加载 `4ed9c8d`。
- **结果**：2026-08-06 00:18 +08:00 起仅 PID `67272`；新 `game.log` 从 `t57607`
  单条记录起算，并看到 `DEPOSIT_SUCCEEDED[1]`，无认证或协议错误。
- **限制**：旧日志含重复 Tick，不能作为 A/B 基线，只能用于故障取证。

## 当前任务队列

| 优先级 | 任务 | 状态 | 目的 | 执行顺序 |
|---:|---|---|---|---:|
| P0 | `OBS-001` 决策耗时与唯一 Tick 观测 | `APPROVED` | 建立可信性能/链路基线，行为零变化 | 1 |
| P1 | `RULE-001` 修复 `4ed9c8d` 规则与策略缺陷 | `PROPOSED` | 防止错误目标优先、资源争用和不可达军备目标 | 2 |
| P1 | `EXP-001` 历史资源与探索名额竞争 | `DISCOVERED` | 避免所有 Worker 被陈旧历史点占满 | 3 |
| P2 | `PERF-001` 前 K 候选 A* / EV | `DISCOVERED` | 降低全量搜索，升级往返收益估计 | 4 |

## 已批准任务：`OBS-001`

### 合同

- **负责人**：隔离执行 Agent；主代理验收与上线。
- **base SHA**：`4ed9c8d393e0b03e6fa0c147d5d926912deba28e`。
- **分支 / worktree**：`codex/obs-001-timing` / 仓库外独立目录。
- **允许文件**：`play.py`、`meta/monitor.py`、`tests/test_tactic.py`。
- **禁止文件**：`.env`、`tactic.py`、`tactic_state.json`、所有运行日志和 LaunchAgent。
- **问题**：当前只有动作/事件计数，没有 `decide`、`submit`、总本地耗时和唯一 Tick 口径，
  无法证明持久地图增长后仍有足够的 15 秒窗口余量。
- **假设**：仅增加计时与解析不会改变计划；可用 500 个唯一 Tick证明 P95 决策耗时是否安全。
- **非目标**：不改变 Worker 分配、路径、战斗、生产、持久化或任何阈值。

### 实现与测试边界

1. 用单调时钟分别测量 `decide_ms`、`submit_ms` 和本地总耗时。
2. 日志只增加紧凑数值字段，不记录 API key、请求头、完整响应或异常对象。
3. monitor 按唯一 Tick 聚合，显式报告重复 Tick 数，不把重连重复状态当独立样本。
4. 输出 P50/P95/P99、窗口错误数、样本数；旧日志无新字段时保持兼容。
5. 新增解析、分位数、重复 Tick 和敏感信息测试；全量测试不得退化。

### 验收

- 静态：全量 pytest、compileall、锁文件检查、依赖检查、diff check 全通过。
- canary：唯一进程连续 20 个新 Tick；无认证、协议、提交和窗口错误。
- 基线：累计 500 个唯一成功 Tick，覆盖至少一次资源刷新；
  `plan_ms P50 < 250ms`、`P95 < 1000ms`、`P99 < 2000ms`。
- 停止：任一 `COMMAND_WINDOW_CLOSED`，或计时日志改变原计划内容，立即拒绝。

## 待批准修复：`RULE-001`

首轮专家在 `4ed9c8d` 发现三项需连续 Tick 集成测试确认的问题：

1. `_select_ranger_target` 仍先按“敌 Core”排序，新增的“距我方 Core <=2”并未压过敌核，
   与提交说明不一致。
2. Unit `HEAL` 在 Core 动作之前花资源；当前 Vanguard/Ranger 与修盾/生产没有共享资源预算，
   会排队可预见的 `INSUFFICIENT_RESOURCES`，并可能占住 Core 格影响交付和生产。
3. `_standing_army_targets(18/19)` 返回 `V1/R1`，但人口到 19 后 Core 停产；纯函数测试不能证明
   连续 Tick 真能重建两个兵种。

批准前必须补齐：多目标 Ranger 测试、Unit HEAL + Core action 资源账本测试、`W18/W19`
连续 Tick 生产测试，以及“带货 Worker 距 Core <=2 + Core 格低血战斗单位”的交付压力测试。

## 候选资源实验：`EXP-001`

- **假设**：没有当前可见资源、空载 Worker 至少 4 个时，最多使用 `N-1` 个 Worker 追历史
  资源，保留一个稳定前沿探索者，会提高长期入核率。
- **为什么暂不执行**：旧日志被多客户端重复污染，Worker 级链路时延尚不可观测。
- **进入条件**：`OBS-001` 完成；唯一进程基线至少 200 Tick、50 次历史资源分配和
  50 个完整采集到入核样本。
- **主指标**：入核资源 / 唯一 Tick。
- **次指标**：新探索格 / Tick、历史分配到采集 P50/P95、历史资源失败率。
- **回滚**：连续两个 100-Tick 窗口入核率下降 10%，或历史分配 P95 上升 25%，或失败率
  上升 5 个百分点。

## 暂缓方向

- `PERF-001`：只对 Worker 的前 K 个 Manhattan 候选计算/缓存 A*，再用
  `confidence * yield / (outbound + harvest + return + deposit_wait)` 排序。
- 人口 20+：除非边际 Worker 的实测增量入核率覆盖 upkeep=1/Tick、生产成本和防御风险，
  否则保持人口 19 以下。
- Core 满仓缓冲队列：先取得“距 Core 两格到入核”的 P50/P95，再决定是否实现。

## 本轮完成定义

1. 所有批准任务由隔离执行 Agent 交付、主代理验收；执行 Agent 不直接 push/live。
2. 规则与 SDK 版本一致，唯一 live 进程，日志按唯一 Tick 统计。
3. 每项行为变化都有基线、样本量、主指标、安全护栏和回滚条件。
4. 验收结果更新本文件和 `LESSONS.md`，由主代理精准 commit、push 并核验 GitHub SHA。
