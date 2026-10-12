# 过程109：本核控制行优化已接入私有正常SDK，单卡确认排队

2026-10-12。采用107，撤下108叠加门铃入口；不采用106内联Wait。
活动私有Simpler已从3d0b9c820快进到b5a358b88，包含先前普通入口修复1c56bcd3e。
PyPTO 4a2651638同步KernelABI pin及runtime gitlink，13ae621f7保存失败样本落盘修复。
源码、binding、runtime、torch_npu适配器都在同一私有工作区重建，导入版本一致。
没有替换共享安装，没有忽略ABI校验，没有修改旧冻结产物的manifest。

正常编译两路计算Python和AIV C++与此前冻结产物相同，orchestration仅ABI描述串随SDK版本改变。
不再依靠实验驱动临时替换runtime或编译器；实际运行入口就是task_boundary_acceptance.py。
正常源码重建的19项CPU检查通过。单卡任务`task_20261012_163414_33212415229`仍排队，
顺序旧SDK→当前SDK→旧SDK；固定17×48、Scalar256、work5/167、8次图内调用、30样本，
保留逐级/换输入/保护区检查及三项局部设备功能。当前没有该任务的设备结果。

此状态只代表实现与版本已落地，不代表295/250μs双目标完成；此前候选证据见107和108。
旧3d0b9c820产物需要匹配的旧私有SDK，不能用新环境强行加载或修改ABI描述来通过校验。
[正常构建记录](runtime_build.log)、[CPU结果](cpu.log)、[源代码比较](source_comparison.json)、
[运行脚本](repro/run.sh)、[Simpler提交](simpler.patch)、[PyPTO版本提交](pypto.patch)。
