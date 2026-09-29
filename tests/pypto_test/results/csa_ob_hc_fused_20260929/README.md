# O-B反量化与HC_post融合：冻结候选

基底为已采用O-B激活复用的9a868d26，独立复制两份私有整包，尚未修改生产。
上轮七档尾部分析显示前置FIN到收尾首次start仅0.58–0.84μs，
不再把提前派发时的数据等待当作可删除调度。本轮改动真正的数据交接和任务边界。

性能版原来先启动proj_b_act，再将BF16 attn_out落GM交给独立HC_post。
候选合为proj_b_act_hc_post，任务T16/N512、内部T8。长128K/B16为48份worker，
短8K/B24为72份；基线两级分别24+24、40+36份，不能直接比较单worker均值。
组内量化、INT32累加、逐组FP32反量化相加及权重scale顺序均保持，
仍先RINT为BF16、再转FP32执行post*x和四路残差逐项mul/add，最终RINT为BF16。

HC四路残差常驻策略继续沿用最新ops-transformer28f40354的Permanent-X参考，
详见[来源与原采用证据](../../DSV4_FLASH_CSA_ASCENDC_REFERENCES.md)。
此次跨O-B反量化/HC任务融合是针对PTO任务组织的实验，不声称Native已有同样融合。
与pypto-lib2164563相比，PTO原按组反量化算术保持，但本接入的Native BF16残差、
BF16中间输出及HC末处理必须保留；不能直接拿历史725μs图的单任务均值归因。
共享HC函数抽出同一helper：原单行使用标量门控，融合多行使用块加载/UB转置取列。

两个CSA入口依赖解析、完整PTOAS/CCE/link/load及共享HC单行入口CPU编译通过。
最终三条行块特化均为117504字节Vec上界；每个T8块门控只加载/转置各一次，
两次TTRANS、零TMOV；反量化BF16往返保留于UB，无attn_out GM分配和独立HC任务。
输出四次TSTORE按实际token行数裁剪。门控加载保持实际有效行/列，
仅在加载到UB后放宽物理块范围以便转置/reshape；padding行不会写出。

早期CPU草案分别遇到Tensor/Tile混用、分支变量类型、门控ND列加载、
32字节物理对齐及部分有效视图reshape限制，均在算子侧调整；没有修改工具链。
生成码又发现门控放在输出循环内重复读取/转置，已提到循环外：8次TTRANS降至2次。
所有这些只发生在未排队私有副本；旧编译失败日志保留本地，不作为成功证据。

显式依赖保留八组O-B。融合任务在manual scope外，post/comb仍为input，
运行时会合并显式和自动依赖；设备DFX须证明八组O-B、split_pre_post及comb_sinkhorn前置完整。
共享原HC入口CPU编译通过不等于精度版设备验收；若值得采用再补这一受影响入口和尾行。

13:56按正常auto队列提交task_20260929_135654_65664815311，已确认设备1上running。
同卡测试128K/B16、8K/B24，长档baseline先、短档candidate先；两份私有Python源已设为只读。
固定CANN9.2、mode2、atomic0、det0、正式第二个CSA层权重与合成历史。
每侧5预热20次正式计时及独立四窗DFX，保存八类完整状态，跨版本零容差比较。
报告完整CSA/P95、两级或融合任务的总核内工作量、收尾跨度及真实依赖。
按长短8:2评估，有收益才补尾行/padding；不新增Native、七档或EP16采集。
当前没有设备性能、状态或采用结论。

[候选差异](candidate.patch)、[来源](source.json)、[生成码证据](static_evidence.json)、
[CSA编译](compile_candidate.json)、[共享HC编译](compile_shared_hc.json)、
[设备入口](run.sh)、[收集器](collect.py)。
