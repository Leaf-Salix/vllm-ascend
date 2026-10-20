# 仅本地保留的构建产物

下列目录/文件不入Git；只记录路径与现有大小，不计算hash。完整CSA任务仍在运行，后续产物可能增加。

- `tests/pypto_test/csa_roofline_optimization_20261020/sdma_refresh_20261020/cpu_ob_b28_v1/kernel`：当前 90974654 字节。
- `tests/pypto_test/csa_roofline_optimization_20261020/sdma_refresh_20261020/full_ob_b16_v1`：当前 67826031 字节。
- `tests/pypto_test/csa_roofline_optimization_20261020/sdma_refresh_20261020/full_ob_b28_v1`：当前 54226325 字节。
- `/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-sdma-refresh-20261020`：独立源码及其本地构建；依赖提交尚未push。
- `/data/pyptouser/qinchuanyu/pto-eager/.worktrees/simpler-csa-sdma-refresh-20261020`：独立源码及其本地构建；依赖提交尚未push。

正式模型权重沿用 `/data/model/DeepSeek-V4-Flash-0731-w8a8`，不复制或提交。
PMU原始分析目录与完整trace保留本地；Git仅保存摘要、完整检查JSON和对应kernel_details.csv。
