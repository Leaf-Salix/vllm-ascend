# v11：单leaf融合Top-K发布，未保留

状态：编译失败，无设备计时或精度结果。正式入口已恢复到v10（0ed4f926）。
该目录不是可用性能结果，不混入既有七档v4/v7矩阵。

目标是让两个AIV各自处理一个query，直接在Score任务内排序并发布Top-K，
省去8K路径的半leaf pair写回和独立归并任务；仍有score暂存GM，并非Native的完整UB流式方案。

初版动态分支把idx_topk的Out ABI派生为InOut，KernelArtifact检查拒绝编译。
改为constexpr区分单leaf/多leaf、在各分支完整生产输出后，ABI检查通过，
但生成的AICPU C++把scope内部Tensor别名用于scope外消费者，编译报
`idx_topk__ssa_v2 was not declared in this scope`。
去除显式输出别名绑定仍不能消除该问题。未禁用ABI检查、未修改生成C++或编译器。

保存[未合并候选补丁](unmerged_candidate.patch)和[最终CPU编译错误](compile_error.txt)，
用于恢复后继续解决scope与输出传递；不要把补丁当作已验证实现。
在0ed4f926独立checkout应用该补丁，并复制本记录对应版本的改进`../compile_contiguous.py`后，
设`VLLM_ASCEND_ENABLE_NZ=2`，传一个空输出目录复现；该检查不初始化NPU。
CPU检查现在既检查KernelArtifact ABI，也编译生成的调度C++。
既有CPU fixture来自`csa_baseline_20260926/nz_native_b16_timing/mode2/case`。

两次正式入口任务均在设备执行前失败：
`task_20260927_130109_3593909330`（ABI）、
`task_20260927_131614_41763712433`（AICPU调度C++）。
未形成计时或泳道，不为该失败候选追加其他档位测试。
