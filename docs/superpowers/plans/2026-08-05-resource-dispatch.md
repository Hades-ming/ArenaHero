# Arena Hero Resource Dispatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让空闲 Worker 优先采集全部可信已知资源，并在无资源时分散到不同未探索前沿，以提高长期资源吞吐。

**Architecture:** 每 Tick 从当前完整 Turn 和持久化资源提示生成一次确定性全局资源匹配，资源任务覆盖探索任务。未匹配 Worker 使用稳定的分散前沿目标；A* 继续只负责把高层目标转成当前 Tick 的一步移动。

**Tech Stack:** Python 3.11、arena-hero SDK 0.2.8、pytest、标准库最小费用流/A*、JSON 持久化

## Global Constraints

- 最终目标是得到最多的资源，所有 Worker 行为必须能追溯到资源采集、发现或安全运输。
- 当前完整 Turn 始终高于持久化历史提示。
- 不引入第三方依赖，不修改官方 SDK，不保存跨 Tick 控制器对象。
- 仅修改本轮实现、测试和经验文档；不提交 `.env`、`monitor.live.log`、`play.out` 或 `tactic_state.json`。

---

### Task 1: 锁定资源记忆与全局分配合同

**Files:**
- Modify: `tests/test_tactic.py`
- Modify: `tactic.py`

**Interfaces:**
- Consumes: `Turn.workers`、`Turn.resource_cells`、`_known_resources`
- Produces: `_worker_resource_assignments(turn) -> dict[str, tuple[int, int]]`

- [x] **Step 1: 写失败测试**

新增用例验证：远端资源不被 `_observe_resources` 裁剪；可见资源优先；贪心反例得到最小总距离；Worker 输入顺序不影响 UUID 到坐标的映射；载货 Worker 排除且资源唯一认领。

- [x] **Step 2: 运行定向测试确认失败**

Run: `.venv/bin/pytest tests/test_tactic.py -k 'global_resource or distant_known or visible_resource_priority' -v`

Expected: FAIL，原因是 `_worker_resource_assignments` 尚不存在或远端资源仍被裁剪。

- [x] **Step 3: 实现最小费用分配并移除距离裁剪**

在 `tactic.py` 中实现确定性一对一全局匹配。目标权重必须严格保证“分配数量 > 当前可见数量 > 总距离 > 稳定平局”；删除 `_observe_resources` 的 Core 距离裁剪和 `_control_workers` 的采集半径硬门槛。

- [x] **Step 4: 运行定向测试确认通过**

Run: `.venv/bin/pytest tests/test_tactic.py -k 'resource or harvest or worker' -v`

Expected: PASS。

### Task 2: 用分散前沿替换固定扫列作为空闲策略

**Files:**
- Modify: `tests/test_tactic.py`
- Modify: `tactic.py`

**Interfaces:**
- Consumes: `_explored_cells`、`_known_obstacles`、空闲 Worker 坐标、Core 坐标
- Produces: `_assign_explore_targets(...) -> dict[str, tuple[int, int]]`、`_explore_targets`

- [x] **Step 1: 写失败测试**

新增用例验证：四个无任务 Worker 获得跨不同方向的目标；目标之间满足间距；输入顺序不影响结果；目标未探索时保持稳定；障碍阴影后的未知前沿仍可分配。

- [x] **Step 2: 运行定向测试确认失败**

Run: `.venv/bin/pytest tests/test_tactic.py -k 'frontier or dispersed_exploration' -v`

Expected: FAIL，原因是前沿分配接口尚不存在。

- [x] **Step 3: 实现候选前沿和稳定目标**

从已探索区域边界和 Core 周围退化环生成候选，按新增视野、已选目标间距、Worker 路程和坐标稳定排序逐个分配。资源任务出现、目标被探索或 Worker 消失时清除对应探索目标。

- [x] **Step 4: 接入 Worker 控制流程**

在处理载货、即时采集和资源匹配后，使用目标导向 A* 为剩余 Worker 排队移动；保留现有卡死逃生与 Core 拥塞保护。

- [x] **Step 5: 运行探索及全量测试**

Run: `.venv/bin/pytest tests/test_tactic.py -k 'explor or worker' -v && .venv/bin/pytest -q`

Expected: 全部 PASS。

### Task 3: 文档、静态验证与实战观察

**Files:**
- Modify: `GOAL.md`
- Modify: `LESSONS.md`

**Interfaces:**
- Consumes: 本轮测试与 live Tick 证据
- Produces: 可复用的资源调度经验、下一轮优化方向和已验证运行结果

- [x] **Step 1: 更新目标与经验文档**

记录全局匹配、持久化提示边界、分散前沿设计、失败模式和后续以数据校准远端资源价值的方向。

- [x] **Step 2: 运行完整验证**

Run: `.venv/bin/pytest -q && .venv/bin/python -m compileall tactic.py play.py meta tests && uv sync --locked --dry-run && git diff --check`

Expected: 测试全过、编译成功、锁文件不变、无空白错误。

- [x] **Step 3: 重启并观察连续 Tick**

Run: `launchctl kickstart -k gui/$(id -u)/io.arenahero.tactic`

Expected: 常驻进程恢复运行；新发现资源由对应 Worker 领取；无资源阶段的 Worker 坐标开始向多个方向扩散；无认证、协议或提交错误。

- [ ] **Step 4: 精准提交并推送**

只暂存 `tactic.py`、`tests/test_tactic.py`、`GOAL.md`、`LESSONS.md` 和本轮 `docs/superpowers/` 文档，创建提交并推送 `main`。

- [ ] **Step 5: 核对远端一致性**

Run: `git status --short --branch && git rev-parse HEAD && git rev-parse origin/main && git ls-remote origin refs/heads/main && git rev-list --left-right --count origin/main...HEAD`

Expected: local、tracking ref 和 GitHub `main` 指向同一提交，ahead/behind 为 `0 0`，仅保留用户运行产物。

## Self-Review

- 规格覆盖：资源持久化、全局匹配、当前可见优先、载货排除、分散探索、稳定性、文档与 live 验证均有对应任务。
- 占位符检查：没有 `TBD`、`TODO` 或未定义接口。
- 类型一致性：资源分配键统一为 `str(worker.id)`，目标统一为 `(int, int)` 坐标；所有控制器仅从当前 Turn 获取。
