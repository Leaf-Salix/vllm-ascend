# 长档三query只预取Key到L1

独立554b3bca候选，工作树`.cache/csa-score-key-l1-only-554b3bca`；主要覆盖128K/B8。
不含尚在单卡对照的B4双槽候选；不影响冻结的七档EP16。

## 实际差异与取舍

本地最新ops-transformer b5b33e14的QLI V2 Cube：
`ProcessWs`按query的gSize=64个head循环，`LoadSToL0b`取64×N128的FP16面板（16KiB），
`ComputeWs`使用K=gSize。Key/Score在4个16KiB L0B槽中轮转。
当前PTO三query把权重放在分块对角矩阵，用一次K192的Cube完成三路规约，Score Right占48KiB；
这减少WS调用次数，同时限制了Key L0B预取容量，不能把Native四槽直接照搬。

七档单卡中128K/B8的Score AIC核内均值189.48μs、包络229.50μs，是长档核内热点之一。
本候选优先保留M192/N128、每N1024八次QK/WS、原量化/规约和任务数；
每轮先把当前面板提取到一个16KiB L0B槽，再把下一面板放入独立Key L1双槽，
避免当前Key提取在表达上依赖下一Key的DMA完成；实际同步仍须核查生成码。
目标是在不超过L0B容量的前提下允许MTE2前行。能否重叠须看生成地址/等待和单卡数据，不能按省等待推断收益。

pypto-lib采用自己的连续cache/query分组，当前Native分页物理视图和分组策略保留；
本项只改变读取生命周期，不改vllm-ascend cache布局或用入口复制换连续地址。

[CPU入口](compile.py)已在B4设备对照和状态汇总结束后完成完整编译/链接/load，
编译期间七档模型仍pending；没有与本线程设备计时重叠。
[生成码核查](codegen.py)、[地址和等待证据](codegen_evidence.json)：Key L1池基址0、32KiB，
Score双槽基址32768/81920、各48KiB，Mat末端128KiB；Key L0B首块16384、稳态0，WS基址16384起48KiB，
稳态合计64KiB。仍8次Key/QK/WS，MTE1→FIX等待保持0；其他四组Score AIC/AIV和该组AIV共9份执行bin一致。
这些只证明生成码与容量符合表达，不代表DMA一定隐藏或实际收益。

[单卡入口](run_layer.sh)复用已执行的B4对照流程，明确传入本候选源码、结果根和128K/B8；
保留8K/B16控制，每侧20次无profiler计时/四DFX窗口、八类状态精确比较及图重放。
[离线汇总](summarize.sh)通过实际任务句柄核查结束后再读取状态，不在设备作业内查询队列。
尚未取得本候选设备结果，未合入生产。
已提交单卡`task_20260928_190508_19636832316`；[实际句柄](layer_task.txt)。
