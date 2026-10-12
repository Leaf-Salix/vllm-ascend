# 过程111：固定5μs链的原批次内下一块预取

2026-10-12。用户要求后续先聚焦5μs，暂停50μs。沿用原work=5、Scalar256、
1/17级×48 AIV、8次图内调用、30样本；calibration_5us.json只筛选原标定记录，未重新标定或减少计算。

过程110即时预取失败后撤下；本候选独立从Simpler b5a358b88构建。
只在stage_consumer_blocks原来的idle/pending桶内前视下一块，对CoreExecState、派发头和
非普通成员的描述行发预取。gated路径不写args，不预取该行。sync_start原调用默认不变。
位图只复制用于查看下一核，不提前修改真实位图；核选择、block顺序、同一wmb/发布批次、
完整输入等待、ACK后续流程和early-resolve保持。普通成员不预取交接描述行。

独立源码+25/-3行，编译来源确认指向候选；19项原有CPU检查、clang-format及Ruff通过。
初始从非仓库目录加载环境退出128，未进入编译；切到PyPTO仓后正常构建。
复用109的原计算产物，私有活动SDK和公共安装都不修改。

任务`task_20261012_190920_192615822351`：当前SDK前→候选→当前SDK后，单卡，结果待回收。
仅预取提示变化，固定链仍做逐级输出、换输入重放和保护区精确检查；不增加额外设备用例。
295/250μs双目标尚未达成，不把预取提前描述为收益。

[补丁](candidate.patch)、[运行脚本](repro/run.sh)、[构建入口](repro/build.sh)、
[构建驱动](repro/driver.py)、[CPU结果](cpu.log)、[编译源路径](build_source_check.json)。
