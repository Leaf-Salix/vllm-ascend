# CSA TND v122 之后的九档优化

目标与约束见 `vllm_ascend/ops/pypto/CSA_SPMD_TND_NEXT_10PCT_GOAL_20261022.md`。
过程见同目录 `CSA_SPMD_TND_NEXT_10PCT_LOG_20261022.md`。

固定比较完整 HC_pre/RMSNorm/CSA/HC_post 的 72 核 L1 首尾。八档 S6 加固定
128K/B16 混长，九档等权几何平均 candidate/v122 <= 0.90。

## 基线重建

`frozen_v122.json` 固定 Git 提交、41 个源码 hash 和两个覆盖文件。
`reference_overlay/*.py.source` 保存这两个文件的原始字节，避免格式器修改固定基线。
它们不作为当前默认实现或独立导入包。
重建只读 Git 和覆盖文件，不依赖旧 `runs/shared_combined_candidate_v122`。

在 worktree 根目录运行，输出目录必须不存在：

```bash
python3 tests/pypto_test/csa_spmd_tnd_opt_20261022/materialize_reference.py \
  tests/pypto_test/csa_spmd_tnd_opt_20261022/runs/frozen_v122
python3 tests/pypto_test/csa_spmd_tnd_20261020/prepare_incremental_harness.py \
  --reference tests/pypto_test/csa_spmd_tnd_opt_20261022/runs/frozen_v122 \
  --output tests/pypto_test/csa_spmd_tnd_opt_20261022/runs/frozen_v122_harness \
  --unused-score-arena
python3 tests/pypto_test/csa_spmd_tnd_opt_20261022/prepare_control_candidates.py
```

两个单变量候选为 `output_balance_off` 和 `short_half_ub96`；准备并不代表采用。
准备器生成各自 descriptor，CPU 编译使用旧实验目录的 `run_compile_cpu.sh`。
NPU 仅通过冻结 harness 的 `submit_probe.py`；沿用 `--bsh-probe --native-toolchain`
是历史 CLI 兼容，报告 `comparison_reference.kind=frozen_tnd_v122` 明确实际分母。
旧八档 BSH 验收器仍拒绝这些结果。任务写入旧目录的 `jobs.json`，完整命令可复现。

## 验收

`evaluate_nine_cases.py --jobs ../csa_spmd_tnd_20261020/jobs.json --output REPORT TASK1 ... TASK9`
在本目录执行；选择最终同一候选的九个成功任务。它检查固定输入、容量、两侧
原生150 tick代码、代际检查、至少五组交替窗口、72核单MIX、源码和SDK一致性。
每一侧从全部窗口重新取中位数，不使用最快窗口或历史拼接分母。
性能门禁不替代动态 metadata、连续状态和真实服务的受影响验收。

CPU 定向检查：

```bash
python3 -m pytest -q --confcutdir=tests/pypto_test/csa_spmd_tnd_opt_20261022 \
  tests/pypto_test/csa_spmd_tnd_opt_20261022/test_nine_cases.py
```

`runs/` 保存本地编译产物和原始设备输出，不提交二进制、权重和完整编译目录。
清单、准备脚本、必要结果摘要和过程日志提交。没有用户新指令不 push。
