# 过程110：预取新增交接描述缓存行，单卡对照待回收

2026-10-12。基于Simpler b5a358b88，在独立detached源码副本中，仅对非普通入口预取
submitted_overlap所在缓存行；原代码只预取任务头和参数。该行是CPU每次必写的不可变描述，
预取置于build_payload之前，用原参数准备窗口提前申请写权限。其余字段、分支、发布顺序不变。
普通入口不发出预取；不改AICore或算子计算，不更改布局、派发、选核、ACK或allow_early_resolve。

代码+5/-2行。已有强制内联试验098没有净收益，本轮未重复它。
首次editable模块仍指向活动源码，路径断言失败后外层shell却继续构建，未修改版本的产物未上卡。
修正构建脚本的源码根并加断言，外层使用set -e；重新完整构建成功。
compile_commands确认目标文件来自候选目录，Ruff、diff --check通过。公共安装和活动SDK未改变。

任务`task_20261012_165632_136120626957`，当前正常版→候选→当前正常版，结果待回收。
复用109的计算产物；保持原1/17级×48 AIV、Scalar256、work5/167、8次图内调用、30样本。
逐级数值、换输入和保护区检查保持；仅CPU预取提示变化，不重复额外的局部AIC/AIV/MIX测试。
没有设备结果前，不认定预取收益或295/250μs双目标达成。

[候选补丁](candidate.patch)、[源码来源检查](build_source_check.json)、[构建日志](build.log)、
[首次构建问题](initial_preflight_failure.txt)、[运行脚本](repro/run.sh)、[构建驱动](repro/driver.py)。
