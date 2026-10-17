# 2026-10-17：统一 vllm-ascend 工作区并同步远端

按用户要求，本会话后续统一在以下工作区继续：

`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1`

当前分支：`dsv4-flash-pto-v0.25.1rc1`。

## 代码位置

- 本轮完整 CSA 单 SPMD 的区间入口、生成器、调度改写和实验核体：本目录 `csa_multishape_spmd_20261016/`。主要入口为 `range_policy.py`、`shape_specialization.py`、`prepare_range_case.py`、`device_compare.py` 和 `run_device_compare.sh`。
- 静态原型与冻结的原动态 PTO 参考：同级 `csa_static_spmd_feasibility_20261015/`，原始实现与既有结果保留。
- 生产算子目录仍为工作区内的 `vllm_ascend/ops/pypto/`。本轮单 SPMD 实验尚未接入生产目录。
- PyPTO 与 Simpler 的独立依赖 worktree 保留原位置，不更改共享安装。

## 远端同步

实际获取 `nalinaly/vllm-ascend` 的 `dsv4-flash-pto-v0.25.1rc1` 分支，远端为 `a919f119`。
此前本地与远端分别有 20 个、2 个独有提交，无同名已提交文件冲突，也不触及原有未提交修改。
以合并提交 `643d8039` 保留双方历史，合入：

- `f7ee8dda`：HC_pre 的 RMS 输入复用及 Sinkhorn 简化。
- `a919f119`：CSA 静态单 SPMD 原型、同步修复和性能对照。

补齐该分支的 fetch refspec 和 upstream，避免后续默认 fetch 仅更新旧的 `dsv4-flash-pto` 分支。
本次仅更新本地，未 push，也未提交本轮动态实验。原有三个已跟踪文件的未提交补丁逐内容保持一致。

## 迁移与验证

完整复制原动态实验目录，包含本地编译产物、泳道图及报告；没有删除目标目录的独有文件。
迁移前的静态目录及目标目录少数被更新的过程文件已在外部备份，具体位置见 JSON 记录。
既有日志中的绝对路径表示当时实际运行位置，不重写历史。旧动态工作树只作历史留存，后续不在其中开发。
本目录不再向自身执行 `mirror_process_files.py`；过程记录直接写在这里。

只在新位置运行一次 B17/H65536 的 CPU 模板生成，确认使用当前目录的 IR、绑定与源代码 DAG，
布局及同步协议检查通过；结果在 `workspace_migration_cpu_check_20261017/`。
这不是新的 lowering、设备精度或性能结果。本次未重跑 NPU，未计算文件 hash。
文件数、大小、链接目标、路径与版本的记录见 `WORKSPACE_MIGRATION_20261017.json`。

`hc_pre_reference_20261017.py` 是同步远端之前的参考快照；后续研究应读取现生产目录的最新 `hc_pre.py`。
冻结动态基线未跟随这次远端 HC_pre 更新改变，避免把历史对照的来源混淆。
所有档位降低至少 30% 的目标仍然有效，当前尚未达到。
