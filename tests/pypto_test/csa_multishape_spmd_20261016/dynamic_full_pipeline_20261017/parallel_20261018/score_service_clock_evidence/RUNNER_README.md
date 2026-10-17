# 正式668长128K/B16 Score服务诊断入口

本目录从既有 `parallel_20261018/attention_tick_clock_evidence/run_tick_diagnostic.py` 和 `run_device.sh` 最小改造，**只接受正式668、H131072/B16/S6、baseline-only**。源目录 `source_snapshot/` 的33个 Python文件直接取自提交 `66882634`，准备时与正式工作区逐文件直接字节相同，未混入 N128、Attention geometry 或其它实验。全部 `pl.dynamic`、device Q/L/position/page 及生产数学保持原状。

本轮只准备入口、冻结源和标准库审计，没有导入 torch/PyPTO，没有完整CPU lowering、DFX性能窗或设备提交。只有主会话后续授权的设备执行才能给出正确性与时序结果；插桩时钟不能用于认定性能收益。

## 复用的验证范围

`run_score_diagnostic.py` 继续使用原 fixture、正式模型权重、原分task PTO oracle、相同 Native 根地址和独立 workspace/signal。A、B 输入分别生成原task精确参考；候选入口先 eager，再 capture/replay 预热，最后三轮图重放 A→B→A。每次检查输出、TopK indices/scores、state/cache，精确容差仍为0；Native根保护区、工作区首尾保护区、全72核 phase0 当前代及保留位检查均保留。这里的 oracle 是原PTO，不是 Native forward 严格验收。

设备同步、CPU读回、完整验证通过后才导出时钟。重放不清 signal，不靠清零掩盖跨代问题；诊断仍使用100 tick GM wait-backoff。没有 L1/DFX 性能采样循环，只有三轮诊断。

## Score hook与padding边界

运行时导入同目录 `score_service_clock.py`，由 `hc_tail_scalar_candidate` 独立提供。调用接口为 `compilation_probe(directory)`、`export_clock(signal, destination, phases)`、`check_probe_padding(signal, generation)`。运行输出快照复制该模块，已移除原 `attention_tick_clock.py` 的导入及快照名单。

`checked()` 把完整 signal 与当前代交给 hook 检查：原 phase_clock 在所有phase/核的列16…19；Score服务记录仅占 phase34、AIC0…23、列32…63。其它原保留位必须为0，特别是其它phase/AIV的列32…63不能像旧Attention tick入口那样普遍放行。phase0全核代际检查仍单独保留。具体记录字段、端点含义、固定规格计数与插桩可逆性由 hook 自己的审计说明。

## 生成物检查按活动guard执行

现有 AIC/AIV wrapper 均保留 Cube 和 Vector 两个函数文本，再由 `__DAV_CUBE__` / `__DAV_VEC__` 条件编译选择。因此 **AIV整个.cpp中存在Cube禁用分支的标记不代表AIV执行了Score探针**，也不应为满足文本断言删除禁用函数。

`check_active_probe()` 分别检查两个函数活动guard：Cube区包含 `CSA_SCORE_PROBE_BEGIN/END`，Vector区没有此标记；两区原 `CSA_PROFILE_NOTIFY` 均存在，并拒绝残留Attention逐tick标记。每个wrapper的角色、两区标记/通知数量写入 `result.json:active_probe_audits`。这只是生成C++检查，不代替最终机器码审查，也不声称插桩无开销。

## 文件与审计

- `run_score_diagnostic.py`、`run_device.sh`：本次入口和隔离依赖包装；包装只改调用的脚本名。
- `source_snapshot/`：正式668完整33源，不作源码插桩。
- `prepare_runner.py`、`runner.diff`：精确改造记录；准备脚本遇到已冻结源/入口即拒绝覆盖。
- `audit_runner.py`、`RUNNER_STATIC_AUDIT.json`：逐字节基线检查、逆变换全文/AST、hook接口与标准库 `py_compile` 语法结果。`syntax_check/` 中的.pyc只是Python语法检查，不是内核编译。
- `SOURCE_PREPARATION.json`：来源路径、文件数和大小，不使用摘要校验。
- `audit_hook_interface.py`、`HOOK_INTERFACE_AUDIT.json`：hook作者冻结后只补接口/语法检查，直接复制同名字节，不重新执行runner审计或任何模型。hook自身的插桩/保留位反例审计由作者独立目录提供，本入口不重复它。

主会话归档目标为 `parallel_20261018/score_service_clock_evidence`。运行须通过 `task-submit --device auto`，包装继续拒绝未分配设备或已存在输出目录；本目录没有自行执行该包装。默认 `--baseline-source` 为同目录 `source_snapshot/`，传入其它位置时也必须与该33源直接字节一致，防止混入并行候选。仍须显式 `--baseline-only`。
