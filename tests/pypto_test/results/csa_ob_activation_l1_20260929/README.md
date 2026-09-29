# NZ O-B激活驻留L1，跨N分块复用

基底为已采用单根Indexer的2ed8ae2e；从它已测私有包再完整复制baseline/candidate，
不修改刚发布的七档源码、原泳道或生产代码。

参考最新ops-nn19614968 A3可用的QuantBatchMatmulV3Tiling::GetIteratorOrder
（quant_batch_matmul_v3_tiling.cpp:1056–1087）：A的K全载、B非全载时选择N优先，
使多个N分块复用A。这里吸收的是条件策略，不声称实测Native的K8192也能整A载入L1。

当前PTO每组K1024、每worker两个N256块，ROW32/96仍逐N、逐K256加载A和B。
候选只把A的K1024一次载入L1并跨两个N块复用；B保持K256流水，所有INT8→INT32乘加、
组内量化、组间反量化相加、最终舍入、权重指针和任务依赖均保持。ROW128沿原权重常驻路径，ND不改。
这是A复用，不是重做验证日志§110.2中已撤回的K1024完整权重B常驻。

pypto-lib2164563按ROW32/96/128及N256选择分块，小中档同样逐K256读取输入，
大档驻留B；候选保留其算术和分块，但用显式Mat tile保证A生命周期跨N，避免仅写Tensor视图被折叠回GM读取。
Native权重仍为二维[G*K,N] NZ，直接按组K偏移借用，无转置或设备复制。

先CPU解析/完整编译，检查A确实在N循环外加载且片上容量可用，再正常auto排队：
128K/B16是ROW96受影响档，8K/B24是ROW128不变路径控制；各5预热20次正式设备事件及四窗DFX。
比较64份O-B核时、四窗范围和完整CSA/P95，八类完整状态先用零容差，不因整数优化放宽标准。
若没有真实收益停止扩大测试；若有收益再验证ROW32/96尾行及padding后采用。
旧七档Native基线保持，精度版和真实EP16验收后置；当前没有性能结论。

两入口依赖解析、candidate完整PTOAS/CCE/link/load、Ruff和shell通过。
最终IR确认ROW96的[96,1024]激活在N循环外一次加载，Mat共224KiB、Left48KiB、Right64KiB、Acc96KiB；
权重K256双槽保持。编译后只移除基底中已未使用的BF16_WEIGHT_LAYOUT导入，算子正文未再变动。
2026-09-29 12:31正常auto提交task_20260929_123150_61152813936，最长5400秒，已确认running。
见[静态依据](static_evidence.json)；少加载是候选依据，不能替代核内或完整CSA实测。

[冻结及变换](prepare.py)、[补丁](candidate.patch)、[来源](source.json)、
[编译](compile.py)、[设备入口](run.sh)、[收集](collect.py)。
