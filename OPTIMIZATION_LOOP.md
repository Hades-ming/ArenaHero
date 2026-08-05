# Arena Hero 优化迭代机制

本文件是项目的长期优化治理合同。`GOAL.md` 定义唯一最终目标，本文件定义如何安全、
可复现地朝目标迭代，`ITERATION_PLAN.md` 记录当前唯一有效的执行计划。

## 1. 最终目标与判断顺序

最终目标只有一个：**长期得到最多的资源**。

任何提案必须按以下顺序判断，不能用局部指标替代最终目标：

1. **运行正确性**：只有一个 Agent 客户端；计划按时提交；无认证、协议和窗口错误。
2. **资源生存性**：不丢 Core、不发生库存溢出销毁、不让带货 Worker 无谓死亡。
3. **资源净收益**：提高每个唯一 Tick 的入核资源与敌核实际捕获资源，同时扣除 upkeep、
   生产、治疗、修盾及可归因的失败开销。
4. **长期吞吐**：提高 `发现 -> 分配 -> 采集 -> 入核` 完整链路的成功率，降低 P50/P95
   时延，而不是只提高某一个中间计数。
5. **防御与攻击**：只有能够降低资源损失或带来可验证的敌核资源捕获时，军备开销才成立。

用户的 Manual 行为优先于 Agent。发生手动迁核、生产、治疗或其他资源消费时，对应窗口必须
标记为 `MANUAL_INTERVENTION`，不能把资源变化错误归因给算法。

## 2. 角色与权限

### 主代理（唯一决策者）

- 完整阅读基础规则、SDK、`GOAL.md`、本文件和当前计划。
- 核验 Git、规则/SDK 版本、唯一 live 进程、日志连续性和当前战况。
- 从代码与日志提出可证伪假设，召集专家，决定是否批准实验。
- 编写任务合同，审查由用户安排的执行 Agent 所交付的 diff 和证据。
- 不派发执行 Agent，不代替执行 Agent 编写任务代码；是否开始执行由用户决定。
- 独占 live 重启、静态验收、canary、commit、merge 和 push 权限。
- 验收失败时拒绝或回滚，不以“测试通过”代替真实战况验证。

### 专家 Agent（只读探针）

- 规则专家：只依据当前发布规则与兼容 SDK，返回规则原文和 `file:line`。
- 资源/经济专家：分析入核率、资源成本、库存风险和 Manual 干预混杂。
- 算法专家：分析匹配、A*、探索、复杂度和 15 秒窗口预算。
- 对抗专家：分析护核、交付通道、攻击机会成本和敌核捕获条件。
- 工程专家：分析日志、测试、持久化、进程、Git 和发布隔离。
- 默认不得修改文件；结论只是线索，主代理按定位抽查后才能采纳。

专家模型固定使用用户指定的 **Luna**。若当前协作接口没有暴露 Luna，主代理必须在
`ITERATION_PLAN.md` 记录实际降级模型；降级输出只能作为临时会诊，禁止标记成 Luna 结果。

### 外部执行 Agent（由用户安排）

- 主代理只提供任务合同；由用户决定何时、交给哪个 Agent 执行。
- 每个任务使用独立 `codex/<task-id>` 分支和仓库外 worktree。
- 只能修改任务合同 `允许文件` 中的路径，只运行离线测试。
- 不得读取或复制 `.env`，不得运行 `play.py`、操作 `launchctl`、提交、推送或合并。
- 交付未提交 diff、测试结果、风险、未完成项和精确 `file:line`。
- 主代理验收前，执行结果不进入 `main`，也不影响 live。

## 3. 触发条件

满足任一条件就进入一轮审查：

- 每累计 200 个**唯一成功 Tick**，或距离上次完整审查 6 小时；
- `meta/monitor.py` 出现瓶颈、命令窗口错误、协议错误或提交错误；
- 任一 `CORE_DESTROYED`、库存溢出、无法解释的大额资源下降；
- 连续两个 100-Tick 窗口的入核率下降 10% 以上；
- 用户手动迁核、批量生产或显著消耗资源；
- 规则、官方文档或 SDK 版本变化；
- 持久地图、候选资源或决策耗时跨过任务计划中的规模门槛。

定时触发不代表自动改代码。它只启动“取证与专家会诊”；没有被主代理批准的任务不能执行。

## 4. 每轮状态机

```text
DISCOVERED -> PROPOSED -> APPROVED -> IN_PROGRESS -> STATIC_VERIFIED
           -> LIVE_CANARY -> OBSERVING -> ACCEPTED -> COMMITTED -> PUSHED

任一阶段可进入 REJECTED；已上线实验可进入 ROLLED_BACK。
```

- `DISCOVERED`：日志、用户观察或审查发现问题。
- `PROPOSED`：有假设、基线、指标、样本量和回滚条件。
- `APPROVED`：主代理确认收益与风险，锁定 `base_sha` 和文件范围。
- `IN_PROGRESS`：用户已安排外部执行 Agent 在隔离 worktree 实现。
- `STATIC_VERIFIED`：主代理完成 diff、测试、编译和凭据检查。
- `LIVE_CANARY`：主代理重启唯一服务，先跑小窗口。
- `OBSERVING`：样本不足，禁止提前宣布优化成功。
- `ACCEPTED`：达到样本量且主指标改善、护栏不退化。
- `COMMITTED/PUSHED`：由主代理精准提交并核验 GitHub 一致性。

## 5. 任务合同

每个执行任务必须在 `ITERATION_PLAN.md` 包含：

```text
任务 ID / 状态 / 负责人 / base SHA / 分支 / worktree
问题证据 / 可证伪假设 / 非目标
允许文件 / 禁止文件
实现边界 / 离线测试
基线窗口 / canary 窗口 / 最小样本量
主指标 / 次指标 / 安全护栏
停止条件 / 回滚条件
交付证据 / 最终提交 / GitHub 状态
```

不同任务的允许文件有交集时必须串行。合并前若 `main` 的目标文件已从 `base_sha` 变化，
任务回到 `REJECTED`，重新基线化和验收，禁止静默覆盖。

## 6. 指标与实验规则

### 数据有效性门槛

- 同一账号只能有一个 `play.py`；所有 Agent 客户端共享一个 Agent 计划槽，后提交会完整覆盖。
- KPI 必须按唯一 Tick 计算；重复 Tick、重连重复状态和无提交 Tick 单独计数。
- 行为实验前至少有 200 个唯一成功 Tick；Worker 链路实验至少 50 个完整采集到入核样本。
- 规则正确性、安全缺陷和行为零变化的可观测性改动不必等待收益样本，但仍必须测试和 canary。

### 核心指标

- 主指标：`(DEPOSIT amount + CORE_RESOURCES_CAPTURED amount) / unique_tick`。
- 净收益：主指标减去 Agent 可归因的 upkeep、生产、治疗、修盾和溢出损失。
- 链路：分配到采集、采集到入核、进入 Core 两格到入核的 P50/P95。
- 探索：每 Tick 新增探索格、当前可见/历史资源分配、历史提示失败/冷却/不可达。
- 性能：`decide_ms`、`submit_ms`、本地总耗时 P50/P95/P99 和窗口关闭次数。
- 安全：Core HP/盾、Core 死亡、单位/货物损失、Core 四邻满占率和移动失败率。

### A/B 原则

- 一次只改变一个假设；参数、代码和环境同时变化时不能归因。
- baseline 与 candidate 使用相同的唯一进程、规则/SDK 版本和统计口径。
- 用户 Manual 干预窗口从主比较中剔除，但保留在报告中。
- 任一 `CORE_DESTROYED`、`COMMAND_WINDOW_CLOSED` 或规则不兼容立即停止扩大实验。
- 不能因为局部指标变好就接受；主指标至少不退化，所有安全护栏必须通过。

## 7. Git、live 与回滚

1. 开始前：主代理确认干净 `main`、`HEAD == origin/main == ls-remote main`，并提供基线 SHA。
2. 执行时：由用户安排的 Agent 只在隔离 worktree 产生 diff，主代理不派发实现，也不并发修改
   同一文件。
3. 静态验收：定向测试、全量测试、`compileall`、`uv sync --locked --dry-run`、
   `uv pip check`、`git diff --check` 和敏感信息扫描。
4. live 验收：主代理执行
   `launchctl kickstart -k gui/$(id -u)/io.arenahero.tactic`，核验唯一 PID、连续唯一 Tick、
   无认证/协议/提交错误。
5. 仅暂存合同允许文件；commit 后 push，并再次核验本地、tracking ref 与 GitHub SHA。
6. 回滚使用明确的反向提交，不使用破坏性 reset，不删除用户运行产物。

以下内容永不提交：`.env`、`game.log*`、`play.out*`、`monitor.live.log`、`tactic_state.json`
以及 `.agent-runs/`。

## 8. 规则与 SDK 复核

每轮开始记录：规则版本、SDK 本地版本、PyPI 最新版本、官方 source/version 页面和复核日期。
当前基线是 gameplay v0.13、SDK 0.2.8。若 live 合同更新或不兼容，立即冻结规则相关改动，
先更新文档与 SDK；不得从旧经验猜测新规则。
