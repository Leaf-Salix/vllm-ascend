# Heads布局128K精度补验证据

## 结论

Direct、Packed和冻结版对同输入Native的A/B八边界共48项逐位通过。
四arm的FULL/raw、毒化恢复、ABA及12次额外重放检查通过。
这补齐了此前因性能门槛跳过的128K/B24精度；不是新的性能或整模型token测试。

## CPU复核

Python3.11.8以上，在仓库根运行：

```bash
python tests/pypto_test/evidence_tnd_precision/heads_layout_precision_20261022/verify.py
```

仅标准库，不导入Torch/vLLM、不运行NPU。校验封包、15文件助手manifest、候选及原编译收据身份，
检查实际48项Native边界、FULL/64Tensor/PTO路由、ABA和12次额外重放/保护区结果。
该验证器重读结果资格，不等同于重新执行硬件。

## 内容

- `heads-precision-128k-b24-v1-evidence.tar.gz`：实际result、preflight、运行日志及当时的完整助手/候选/编译收据。
- `tools.tar.gz`：准备、精度助手、提交/观察/收集、CPU核验、原始提交及终态记录。
- `RECHECK.json`：本轮CPU重算汇总。
- `INDEPENDENT_REVIEW.md`：独立原始证据复核。
- `SHA256SUMS`：封包和验证器SHA绑定。

不含模型权重、cache payload、失败Tensor或构建二进制。候选与原性能测试逐字节相同。

## 重跑

原隔离环境、模型及canonical快照身份沿用[上轮证据说明](../heads_layout_perf_20261022/README.md)。
新工具目录为`/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/heads-precision-20261022-codex`。
工具中的env.sh引用现有环境；更换目录时一致更新task.py、collect.py和相关引用并记录新manifest。
先用preflight.sh做无设备预检，确认原3399/77源、QA及编译收据一致，不重装SDK。

本机工具目录中，使用新case标签：

```bash
python task.py submit heads-precision-128k-b24-v2 24 131072
python task.py observe heads-precision-128k-b24-v2
python collect.py heads-precision-128k-b24-v2
```

单卡task-submit，外限780秒、入口750秒；提交后持续观察，首差立即停止。
未知提交状态不自动重试，不新开后台watch或定时自动化。不得绕过TASK_DEVICE直接运行硬件。

## 口径

真实第4层权重、合成hidden/history131072、B24/S6/T144、TP1/page32/NZ2/seed1024、FULL raw graph。
Native Torch/NPU level1/HCCL确定性开启，SK0、13静态包实际reselect及DSA多流。
未改数学、SDK、runtime或编译生成的C++。本轮DFX/dep-gen关闭，无性能门槛和计时。
两个候选的整模型192-token、短query或混合1～6行验收不由此报告新增。
