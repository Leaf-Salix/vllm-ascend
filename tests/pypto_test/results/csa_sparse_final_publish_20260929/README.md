# Sparse最后PV块融合输出

基线为生产算子4ffccb7b；完整私有副本使用`pkg:dsv4_csa_sparse_final_publish_4ffccb7b`。
本候选尚未合入生产。只改变Sparse最终数据交接，不重试superkernel或更换Native cache布局。

## AscendC依据与PTO差异

参考ops-transformer28f40354的
`experimental/attention/sparse_attn_sharedkv/op_kernel/arch22/sparse_attn_sharedkv_scfa_block_vector.h`，
`DealBmm2ResBaseBlock`末块分支（约934行）在PV更新后直接做RowDivs和最终输出。
这里仍使用Div；不是倒数乘替代除法，也不将最新源码冒称当前release Native二进制。

当前PTO先在QK/PV内完成累计规约，然后把FP32 mi/li/oi写到GM；独立merge_norm再读回，
执行sink分母、除法、逆RoPE和O_A分组发布。候选直接消费最后PV块的累加值完成这些操作，
去除中间mi/li/oi及48个merge_norm worker，O_A直接依赖最终发布的QK/PV任务。
rope_cs移到QK/PV之前声明并加入其显式依赖；保留现有QK/PV早派发属性。
因此消费者的任务边界发生必要变化，不能把全部CSA变化都归因为算术或搬运时间。

pypto-lib2164563的`models/deepseek_v4_flash_dspark/decode_sparse_attn_csa.py`
同样保留独立merge_norm。本候选偏离上游的原因是借鉴AscendC末块发布以省去GM交接，
不是输入格式差异，也没有增加入口拆分、cache重排或写回。

保留性能版当前块内softmax最大值、跨块alpha/beta、sink分母和逆RoPE的算术顺序；
不是此前被否定的累计softmax/PV算术候选。精度版不修改。
最终候选保持原16-head发布粒度、lane索引初始化及BF16转换顺序。
softmax规约临时区从全循环常驻改为只在softmax子阶段存活，不携带跨块状态。

## 缓冲与验证边界

16-head融合初版UB用量197632字节，超过188416字节上限；8-head后仍曾超1024字节。
8-head仍超限后，MemoryReuse IR确认16KiB softmax临时区不必要地全循环常驻。
改为局部声明，恢复原16-head发布；不修改PTOAS/PTO-ISA或绕过容量检查。
中间编译限制与修正见[记录](compile_initial_error.txt)；编译成功不代表设备收益。

先128K/B16和8K/B24：每侧5预热20次编译CSA图事件，独立四窗level-4泳道，
复用现有八类完整状态跨版本零容差、图/eager、Top-K及保护区检查。
先比较Sparse加原merge的完整AIV核·μs、AIC核时及设备发布跨度，再看完整CSA 8:2和P95/max。
融合后Sparse单任务增加了工作，不能仅看该任务均值上升就判定失败；同样不能以少一个任务预认收益。
本阶段不新增七档或16卡。若保留，再补受影响尾块/padding及阶段出口。

[冻结和改造](prepare.py)、[候选补丁](candidate.patch)、[CPU编译](compile.py)、
[设备入口](run.sh)、[CPU收集器](collect.py)、[来源](source.json)。

## CPU通过并开始真机

完整PyPTO/PTOAS/CCE编译及KernelArtifact.load通过，两根依赖图均可解析。
最终UB为181248字节（上限188416），生成程序没有独立merge_norm；原16-head算术和转换顺序保持。
使用同一组冻结源码正常auto单卡排队，不再编辑算子/公共依赖/设备runner；只做两代表档。
设备结果尚待任务结束，不能把编译通过或少一次GM交接当作性能收益。
[CPU结果](compile_candidate.json)、[缓冲分配](lowering.json)、[任务](task.txt)。
