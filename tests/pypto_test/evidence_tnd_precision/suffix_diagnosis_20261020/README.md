# 后缀计算、搬运、同步及单因素诊断证据

本轮正式精度运行源仍为48c58ce7c；旧常驻后缀a31a78f8与全部独立候选均保留。
仅保留phase1八owner方向供后续实验，不改正式路由。
完整结论见[性能诊断报告](../../DSV4_FLASH_CSA_TND_RESIDENT_SUFFIX_DIAGNOSIS_20261020.md)。

## 文件

| 文件 | 内容 |
| --- | --- |
| raw-results.tar.gz、RAW_SHA256.json | 原始结果、各候选proof、逐窗口统计、任务清单和离线复核；不改raw字段 |
| reproduction-tools.tar.gz | 原样DSL候选、各历史probe版本、运行和CPU检查脚本、父候选与proof |
| suffix-diagnosis-run-receipts-v1.tar.gz | 本轮14项任务最终状态、原始result、run/task日志；均exit0 |
| suffix-dfx-t126-v1-evidence.tar.gz | 六阶段及整后缀24个L1窗口 |
| suffix-ab-dfx-phase1-t126-v1-evidence.tar.gz | phase1同任务三臂9个L1窗口 |
| suffix-ab-dfx-phase5-t126-v1-evidence.tar.gz | phase5同任务三臂9个L1窗口 |
| suffix-pmu-t126-v1-evidence.tar.gz | CANN原始设备CSV及PMU汇总；host-flow关联告警保留 |
| suffix-stage-cpu-v1-evidence.tar.gz | 六阶段CPU完整编译结果与WO-A/B生成物 |
| suffix-combo-cpu-v1-evidence.tar.gz | 组合CPU完整编译与实际PTO生成物，不含二进制 |
| wait-isa-source-20261020.tar.gz | 实际TWAIT ISA头文件及源码身份 |
| independent-reviews.tar.gz | 设计、harness、owner、组合、完整接入方案和最终结果独立审查 |
| REPRODUCTION.md | 逐项复现、源版本选择、前置门禁和停止条件 |
| WHY_SLOW.md | 证据支持的解释、未证明的机制及后续方向 |
| CHECK_HISTORY.md | 原始未完成快照和离线检查假设修正，未删失败历史 |
| SHA256.json | 本目录发布文件完整性，不包含自身 |

## 验证范围

真实第4层WO-A/B权重，其余后缀输入合成。金值为冻结PTO数学，并非新Native整模型验收。
五边界eager/FULL A/B/A、只读权重、工作区guard与双代际检查通过；无静默回退。
正式延迟、L1、PMU和已就绪wait微基准分开任务。计数可重叠，阶段延迟不可相加成为总耗时。

组合相对phase1在T126回退2.50%，不再扩大组合T240。未接完整CSA，未新增8K/128K整模型token资格。
前序工具链与冻结版本资格见[上一轮证据](../resident_suffix_20261020/README.md)。

早期设计审查保留当时尚未编译/尚未运行的状态，最终结论以归档内FINAL_RESULT_REVIEW.md及原始结果为准。
