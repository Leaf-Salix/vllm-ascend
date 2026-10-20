# Sinkhorn 与投影重叠：原始证据与复现材料

对应主报告： [DSV4_FLASH_CSA_TND_PRECISION_SINKHORN_OVERLAP_20261020.md](../../DSV4_FLASH_CSA_TND_PRECISION_SINKHORN_OVERLAP_20261020.md)。

| 文件 | 内容 |
| --- | --- |
| micro-full-evidence.tar.gz | 三档前缀、探索与最终单层 FULL、编译收据、启动参数失败日志 |
| whole-overlap-v1-8k-evidence.tar.gz | 16 rank 原始报告、Native gold/历史计时、审计、77份运行源码、3399文件 SHA 清单 |
| whole-overlap-v1-128k-evidence.tar.gz | 同上，128K/B24；测量实际21条并发，保留 CPU 哨兵观测限制 |
| reproduction-tools.tar.gz | 最终诊断/编译/提交脚本、新6源、完整CSA三路对拍工具及整模型辅助脚本 |
| SDK_RUNTIME_MANIFEST.json | 实际 Python 包、运行时和编译二进制的 SHA 身份；不包含其二进制载荷 |
| FORMAL_CANDIDATE_SOURCES.json | 最终6份候选源码 SHA，与两档FULL及整模型快照一致 |
| SCHEDULE_AST_AUDIT.json | 仅移动原 comb 调用的静态审计，不单独作为设备正确性证明 |
| WHOLE_*/ FULL_MICRO_* | 从归档原始载荷独立重算的结论及脚本输出 |
| RUNS.json / CHECKLIST.md | 版本、任务、精度门禁和复测顺序 |
| SHA256.json | 此目录所有其他文件的 SHA256 |

## 复现边界

工具中路径固定为227既有隔离环境，具体环境、模型/bank/gold、plan SHA见配置和preflight。
先恢复主报告指定的 Git 基线和既有SDK；展开工具到实验根，将 `src/` 的6源和冻结71源保持一致。
冻结71源可从任一整模型证据包的`runtime-sources/`恢复到SOURCE_ROOT；将其中SPMD14、prepack6、BF16/QA1268源再按原相对目录放到实验根`src/`。
微测试`compile_full_overlap.py`需要的`BASELINE.json`、父编译收据`compile-full-csa-packaged.json`及`prepack_candidate.py`已随工具归档；
`micro-environment-reference.sh`保存真实引用的环境脚本，原提交脚本source的固定路径仍需恢复或按身份等价映射。
`whole-tools/`为已执行快照的辅助脚本；完整原版同时存于两档整模型包的`tools/`下。
`prepare_whole_overlap.py`还依赖旧已验收的`whole-bf16qa126-v2`快照及helper，不能在缺少这些输入时跳过源码/编译门禁。
Native fixtures还依赖冻结Git源码的`tests/pypto_test`及原delivery PYTHONPATH下的`dsv4_csa_native_case`和`dsv4_csa_single_layer`；这不是仅复制新6源就能独立运行的包。
`recheck_overlap_whole_raw.py`是本机复核助手，依赖原workspace目录结构和旧QA126的8K原始归档；不是展开后可直接在远端运行的工具。
远端正式审计使用job的`qualification-audit`与`audit-case.sh`。
本包不包含权重、大型cache bank、全部3399文件载荷或SDK二进制；包含77份实际运行源码及完整输入身份。

按主报告次序运行：无设备编译 → 单层T126/T240 → 新名字建立整模型快照 → 8K预检/设备任务/严格审计 → 128K。
任务上限780秒、内部750秒；发现首差停止，不修改运行中源码。请用新输出名称，避免覆盖原始证据。
`audit-case.sh`固定本次manifest；改变源文件后应重新做资格审核，不能保留旧SHA来宣称同一源码通过。
性能均保留所有样本；整模型和历史基准跨任务，不能将差值完全归因于本次优化。
