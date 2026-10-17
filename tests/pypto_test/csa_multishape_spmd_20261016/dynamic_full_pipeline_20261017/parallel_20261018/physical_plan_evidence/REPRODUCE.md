# 复现与搬运说明

基线 `593c5d25`，独立worktree `/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-physical-plan-20261018`。`source.patch` 仅改五个正式源码文件，`source/` 有同次完整快照。不要用本目录生成冻结entry；继续保留正式pl.dynamic与device Q/L/position/pages。

首次完整CPU编译（不占卡，输出目录须全新）：

```bash
bash /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/reproduce_cpu.sh <全新绝对CPU目录> --attention-query-metadata off --spmd-source /data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-physical-plan-20261018/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd --controls-json /data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-physical-plan-20261018/tests/pypto_test/ultra_physical_plan_20261018/controls_on.json
```

CPU同源oracle：

```bash
python /data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-physical-plan-20261018/tests/pypto_test/ultra_physical_plan_20261018/check_oracle.py
```

实际设备唯一有效组合任务 `task_20261018_010403_360472313010`：

```bash
task-submit --device auto --max-time 1200 "bash /data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-physical-plan-20261018/tests/pypto_test/ultra_physical_plan_20261018/run_device.sh"
```

脚本是本轮执行快照，输出路径已存在，因此要重跑必须先给脚本换全新probe/build/cache及matrix输出目录。不能覆盖已有证据；本轮没有进一步重跑授权，也不建议为0.81%名义收益扩测。先私有device probe通过，再主驱动完整B16 matrix；全部NPU通过task-submit。

单独归档 `handoff/` 可整体复制。源码工厂/设备脚本使用原worktree路径结构，不能把归档内脚本的位置变化误认成新的运行入口。大生成物/二进制/泳道留本地，路径和大小见LOCAL_ARTIFACTS；未提交/push，未改主仓/依赖/共享安装。

五窗旧SPMD减候选分别为 6.96 / 8.80 / 6.80 / 17.94 / 2.90 μs，均只是本轮有序窗口实测；不做不必要显著性统计。中位数仅降0.811%，相对原动态task仍慢1.945%，目标未达。
