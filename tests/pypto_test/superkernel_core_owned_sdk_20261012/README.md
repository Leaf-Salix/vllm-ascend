# 过程109：本核控制行优化已接入私有正常SDK，单卡正常入口确认通过

2026-10-12。采用107，撤下108叠加门铃入口；不采用106内联Wait。
活动私有Simpler已从3d0b9c820快进到b5a358b88，包含先前普通入口修复1c56bcd3e。
PyPTO 4a2651638同步KernelABI pin及runtime gitlink，13ae621f7保存失败样本落盘修复。
源码、binding、runtime、torch_npu适配器都在同一私有工作区重建，导入版本一致。
没有替换共享安装，没有忽略ABI校验，没有修改旧冻结产物的manifest。

正常编译两路计算Python和AIV C++与此前冻结产物相同，orchestration仅ABI描述串随SDK版本改变。
不再依靠实验驱动临时替换runtime或编译器；实际运行入口就是task_boundary_acceptance.py。
正常源码重建的19项CPU检查通过。单卡任务`task_20261012_163414_33212415229`在卡0完成，退出0，
顺序旧SDK→当前SDK→旧SDK；固定17×48、Scalar256、work5/167、8次图内调用、30样本，
逐级/换输入/保护区全部精确，三项局部AIC/AIV/MIX检查通过。

此状态只代表实现与版本已落地，不代表295/250μs双目标完成；此前候选证据见107和108。
旧3d0b9c820产物需要匹配的旧私有SDK，不能用新环境强行加载或修改ABI描述来通过校验。
[正常构建记录](runtime_build.log)、[CPU结果](cpu.log)、[源代码比较](source_comparison.json)、
[运行脚本](repro/run.sh)、[Simpler提交](simpler.patch)、[PyPTO版本提交](pypto.patch)。

## 正常 SDK 的同卡结果

单位μs；17级×48 AIV，work5/167、Scalar256、8次图内调用、30样本，未开启profiler。

| 路径 | 5μs min/mean/max | 50μs min/mean/max |
| --- | --- | --- |
| 旧SDK前 | 285.34 / 288.75 / 292.37 | 1012.91 / 1016.45 / 1019.90 |
| 当前SDK，不加Set/Wait | 286.21 / 291.23 / 298.39 | 1016.86 / 1019.60 / 1023.67 |
| 当前SDK，加Set/Wait | 295.51 / 300.11 / 304.77 | 996.67 / 1001.43 / 1005.40 |
| 旧SDK后 | 286.54 / 289.55 / 293.72 | 1013.86 / 1016.53 / 1021.69 |

本轮普通路径mean291.23μs低于295μs；标记mean300.11μs仍未低于250μs。
短链标记比同轮旧SDK前后中点289.15μs慢约3.79%，不能算early_start净收益。
长链标记相比旧SDK中点1016.49μs快约1.48%，相比当前普通路径快约1.78%。
记录完整1/17级样本；没有改变工作量、派发策略或allow_early_resolve表达。
[原始样本与检查](device/)、[提取摘要](device_summary.json)。
