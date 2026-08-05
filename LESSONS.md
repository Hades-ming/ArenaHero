# 经验总结（LESSONS）

踩过的坑、对应的根因与修复。每次迭代后追加，避免重犯。**任何改动若修复了
某个坑，都要在此归档**，并保持 `GOAL.md` 的"行为→目标"映射同步更新。

---

## L1 — 单位拥堵（UNIT_CLUMPING）— 误诊修正
- **原误判**：早期监控用 `CELL_UNIT_LIMIT 次数 / tick 数`，算出 121% 的"严重拥堵"。
- **真因（专家评审修正）**：分母应为「移动事件总数」而非 tick 数；真实阻塞率约 9%。
  阻塞来自少数「僵尸 Worker」（如 `b91dec`/`286c14` 长期占 Core 邻格死锁），并非全队拥堵。
- **修复（已落地）**：`meta/monitor.py` 改为 `move_failed / (failed+succeeded)`，阈值 0.10。
  监控不再误报；具体死锁 Worker 的解除留待后续迭代（见 L6）。
- **状态**：✅ 监控已修正（💡 教训：统计口径必须量纲一致，先验证再改战术）

## L2 — 采集停滞（LOW_HARVEST 后期）— 已修复
- **现象**：后期仅 ~0.026 harvests/tick；附近资源采空后 Worker 无法有效扩散。
- **根因**：`MAX_HARVEST_FROM_CORE=30` 把 d=41-46 的可见节点全部硬过滤；且全员外派扫图，
  Core 正下方一格资源都无人采。
- **修复（已落地）**：新增饥饿感知 `_harvest_radius(turn)`——健康经济用 30，饥饿（>50 tick
  无 harvest 或 `_known_resources` 为空）放宽到 55，三处采集过滤点统一调用。保证有收入而非空转。
- **状态**：✅ 已修复

## L3 — 有敌不突袭（NO_RAID）— 部分修复 + 发现死代码
- **现象**：敌人可见 37 ticks，0 个敌核被摧毁；常备军长期 `V2 R1`，且攻击单位从未出现在 plan 里。
- **根因（专家评审关键发现）**：
  1. `decide` 用 `e.unit_type == "CORE"` 检测敌核，但 `CoreView` 只有 `.kind=="CORE"`、
     无 `unit_type` → **整段敌核突袭逻辑是死代码**（`enemy_core_visible` 恒为 False）。
  2. `_chase_target` 猎核门槛 `resources>=60 且 dist<40`，当前 d=44 被卡死。
  3. 唯一 Ranger（index 0）永远走 `_guard_step`，从不猎核。
- **修复（已落地）**：检测改为 `getattr(e,'kind','')=="CORE"`，并复用 `_known_enemy_cores`
  集合（敌核仅在视野闪现几 tick，靠持久记忆才能组队突袭）；见敌把造兵地板降到 2、见敌核
  把目标攻击数提高到 `DEFENSE+2`。后续还需放开猎核门槛、新增 `_raid_squad`、让 index 0 的
  Ranger 也能猎核（见 L6）。
- **状态**：🟡 检测与造兵已修；突袭寻路/编队待续

## L4 — 丢核即清零（CORE_DEFENSE）— 已修复
- **现象**：历史曾因经济卡在 ~5 资源、买不起还击单位而丢核，库存全失。
- **根因**：经济地板之上仍缺军备时继续造 Worker，把资源耗光，无法还击。
- **修复（已落地）**：`army_short` 时（有敌或达经济地板）不再造 Worker，把资源攒给
  攻击单位；护盾仅在受威胁时修。
- **状态**：✅ 已修复

## L5 — 资源闲置（IDLE_GOLD）— 监控修正
- **现象**：资源长时间顶在 capacity。
- **原则**：手里的资源不生息。把结余投入经济、防御或军备，让资本始终运转。
- **修复**：监控改为「连续」计数（idle_gold_streak），阈值降到 15，避免把全程累计误报。
- **状态**：🟡 持续监控

## L6 — 资源凭空蒸发（RESOURCE_LOSS）— 新增检测
- **现象**：`t55135 r95/95` → `t55136 r0/95`，单 tick 掉 95 且无任何 spawn/事件。
- **根因（待查）**：疑似 capacity 溢出销毁（`CORE_RESOURCE_OVERFLOW`）或用户手动消耗。
  原监控只记 res min/max，完全漏报这个最致命的单次损失。
- **修复（已落地）**：`meta/monitor.py` 新增 `RESOURCE_LOSS` 告警（单 tick 跌幅 >10 且无
  spawn），并跟踪 `largest_drop`。
- **状态**：🟡 检测已加；根因/应对待续（若确认是溢出，需在满员且 res 高位时改投修盾/生产/换兵）

## L7 — 迭代方法本身（元教训）
- 监控口径必须量纲一致、先验证再改战术（L1 误诊）。
- 专家子代理并行评审能发现单看代码发现不了的死代码与统计 bug（L3、L1）。
- 每次评审结论都要落到 `tactic.py` / `meta/monitor.py` 的具体改动并跑测试，再 commit/push。
- **状态**：✅ 已建立闭环

## L8 — 改完战术必须重启 live 进程 + 资源"看得见才采得到"
- **现象（用户实测）**：改了 `tactic.py` 后 worker 仍在核心旁闲转，战况没变。
- **根因 1（最坑）**：`play.py` 是长驻进程（启动时把 tactic 载入内存），改文件**不会热加载**。
  不重启就永远是旧逻辑。→ 任何战术改动后必须 `kill` 旧 `play.py` 再重启，否则"改了等于没改"。
- **根因 2（worker 闲转的直接 bug）**：旧锁定逻辑要求 worker 离"已知资源"≤30 格才锁定，
  16 个 worker 铺在整块地，绝大多数时刻离任何已知资源都 >30 → 永不锁定 → 只在核心旁扫荡。
  → 已修复：对"已知资源"取消 worker 距离上限（用采集半径 40/65 代替），任意 worker 都会
  锁定并 A* 走向最近的已知资源；记忆保留半径也对齐到 65，避免会被锁的资源被提前遗忘。
- **根因 3（更深层）**：worker 视野很小（`obs`≈29-39），地图资源大多不在其视野/扫荡带内，
  所以 `_known_resources` 常常为空 → 只能盲扫。用户在网页/大视图看到"满地图资源"，
  worker 却几乎看不见 → 采集率极低（实测 ~0.02 harvest/tick）。
- **应对（根因 3）**：战术侧已让"已知资源必采"；但若资源真的远在扫荡带外，正解是**把 Core
  迁到资源簇附近**（用户可手动迁移；战术支持迁移并会重新扫新区块）。 worker 视野小是规则限制，
  加宽扫荡带会令 worker 跑太远（净亏），不如迁核。
- **状态**：🟡 锁定/记忆 bug 已修并重启生效；发现率取决于资源是否进入视野，必要时迁核。

## L9 — 地图持久化"存了不用"（_explored_cells 死存）
- **现象（用户质问）**："已经做了地图持久化为什么不使用？"
- **根因**：`tactic_state.json` 持久化了 4 张图，其中 3 张**真的在用**——
  `_known_resources`（采集）、`_known_obstacles`（A* 寻路）、`_known_enemy_cores`（猎核）；
  唯独 `_explored_cells`（已探索区域集）**只写不读**：`_observe_terrain` 每 tick 往里塞视野格、
  有障碍变动才落盘，但 `_explore_step` 完全不读它，探索纯靠 chunk 牛耕式 + 每 worker 列偏移。
  docstring 里写的"避免重扫已点亮区 / 定期回访重生资源"从未落地——典型的"建了持久化但没消费"。
- **修复（已落地）**：
  1. 新增 `_next_explore_col_off()`：在 y 边界推进列时，若下一列已被点亮则跳到**最近的**未探索列；
     全部列都探索过才退回牛耕式（兼顾重生回访）。无探索记忆时与 `_advance_col_off` 完全一致。
  2. `_observe_terrain` 按 `_EXPLORED_SAVE_STEP=1000` 批量落盘，重启不再丢失整张图。
  3. `_clear_exploration_state`（Core 迁移触发）现在一并清空 `_explored_cells` 并写回磁盘，
     避免重启读到旧地图。
- **遗留**：SDK 的 `Turn` 无 `game_id`，无法区分"同局崩溃恢复"与"开新局"——目前持久化对崩溃恢复
  有利、但开新局可能带入上局地图。需 SDK 暴露局标识才能根治（已记入待查清单）。
- **状态**：✅ 探索已真正消费持久化地图；40 测试通过。

## L10 — 击毁敌核 ≠ 固定 +6（CORE_RESOURCES_CAPTURED 真实机制）
- **现象（用户纠正）**："击毁敌方 core 不一定获取 +6 资源"。
- **真相（查 SDK `arena_hero/models.py` 的 `CoreResourceCapture` 类型）**：
  击毁敌核触发 `CORE_RESOURCES_CAPTURED` 事件，携带 4 个字段且约束 `amount + destroyed == available`：
  - `available`：敌核被摧毁时的库存资源（**变量**，取决于对方攒了多少、是否花光）
  - `amount`：**你实际获得**的资源（= available 中未被销毁的部分）
  - `destroyed`：被销毁/溢出的资源（available - amount）
  - `capacity`：敌核容量上限
  即回报 = 敌核当时**库存中未被销毁的那部分**，**不是固定 +6**；且受我方核容量约束
  （超出我方容量的溢出会被销毁）。若敌方把资源全花在生产上（库存≈0），击毁它**几乎得不到资源**。
  （当前打包规则与 SDK 已对齐 v0.13 / 0.2.8，规则和 `CoreResourceCapture` 模型均支持该结论。）
- **已改（tactic.py）**：删除 5 处"+6 resource jackpot / pays +6 / shipped home"的错误假设，
  改为准确描述 `CORE_RESOURCES_CAPTURED`（变量回报 + 战略价值=清场+不确定战利品）。
  突袭敌核的**战略价值仍在**（消灭对方舰队、削弱竞争者），但不再被当成"稳赚 +6 的资源经济"来过度加权。
  `play.py` 日志现在也会打印真实 `CORE_RESOURCES_CAPTURED[amount]`，便于验证真实数额。
- **后续可优化（未做，避免范围蔓延）**：`_known_enemy_cores` 一旦见过就永久保留，
  会导致"记得的敌核"持续触发额外突击编队——在回报不确定的情况下可能浪费产能。
  可改为仅在**当前可见**敌核时才超编。留待后续评审。
- **状态**：✅ 错误假设已修正并记入；40 测试通过。

## L11 — 更新 tactic.py 后必须重启 play.py（无热加载）
- **纪律（用户要求）**：程序（tactic.py）更新后**及时重启 play.py 进程**，否则改动不生效。
- **原因**：`play.py` 是长驻进程、不热加载（L8）。本地持久化地图 `tactic_state.json` 会在
  重启后自动重载，故重启安全、状态连续；服务器侧也保留对局状态，重连即可续上。
- **安全重启命令**（括号技巧 `[p]lay\.py` 避免 pkill 误杀自身；`nohup ... & disown` 完全脱离）：
  `cd /Users/hx/myself/ArenaHero && pkill -f "[p]lay\.py"; sleep 2; nohup .venv/bin/python3 play.py > play.out 2>&1 < /dev/null & disown`
  验证：`game.log` 的 tick 继续推进、且 `play.out` 无 error。
- **当前运行方式**：`launchctl` 用户服务 `io.arenahero.tactic` 托管 `play.py`；改动后用
  `launchctl kickstart -k gui/$(id -u)/io.arenahero.tactic` 重启，并验证唯一 PID 与连续新 Tick。
- **自动化边界**：Codex heartbeat `arena-hero` 已创建，每 6 小时复核 SDK、规则文档、live KPI，
  瓶颈时调用经济/战斗/路径/运行专家；仅失败运行通知。人工审查仍需实时核验，不能只信历史配置。

## L12 — 全量复审：规则真相必须压过历史启发式
- **对角火力漏判**：v0.13 Ranger 支持横、竖和 45 度对角线 1-3 格射击。旧实现先按
  Manhattan 距离过滤，导致 `(2,2)`、`(3,3)` 等合法目标永远不射；现改为八方向射程，
  射击障碍只检查实际横、竖或对角 shot cells；对角线旁的障碍不阻挡。视野遮挡仍使用
  supercover，两者不能混用。
- **历史敌核冻结经济**：`_known_enemy_cores` 是永久坐标集合，过去却与“当前可见敌核”共用
  超编信号，使 4V/4R 的不可达目标长期保持 `army_short`，阻止继续造 Worker。现仅当前可见
  敌核触发超编；历史坐标仍可供有界 Ranger 追踪，不再支配生产。
- **资源认领排序**：最近资源优先排序现在先筛空载 Worker，载货返航者不会影响可认领单位的
  优先级；同距时保留原始顺序，结果可重复；旧的远端锁不能覆盖近 Worker 的新认领。
- **当前 state 胜过旧事件**：上一 Tick 的 `HARVEST_SUCCEEDED` 不能删除当前 Turn 仍明确存在的
  resource cell；剩余 cargo pile 或同坐标 refill 都要求先处理事件、再由当前集合覆盖。
- **禁止为省 upkeep 盲目自毁**：pop21 自毁到19会先损失两个生产资产，再因容量从105降到95
  立即销毁最多10库存，只为省1/tick；且会撤销用户手动扩军。现不再自动自毁。
- **Beacon 所有权与低风险获取**：`CARRIED` 不代表我方持有，必须用友方 `carrier_id` 判断10盾
  上限；Worker/Core 与地面 Beacon 同格时可拾取，但不做高成本跨图追逐。
- **探索必须尊重遮挡**：障碍后的格子不在当前视野，不能写入 `_explored_cells`，否则后续扫描
  会永久跳过真正未知区域。
- **文档漂移**：同步修正 `GOAL.md` 的固定 +6 假设和 `README.md` 的 v0.7 / SDK 0.2.4。
- **重启必须保证单实例**：把多个 PID 拼成带空格字符串交给 zsh `kill` 会被当成一个非法 PID，
  新旧客户端随后竞争同一个 Agent plan 槽。现在直接以 `.venv/bin/python3 play.py` 启动并把真实
  Python PID 写入 `play.pid`；停止时逐个 PID 传参，重启后检查唯一进程和连续新 Tick。
- **状态**：✅ 失败测试复现后修复；以 v0.13 / SDK 0.2.8 为准。
