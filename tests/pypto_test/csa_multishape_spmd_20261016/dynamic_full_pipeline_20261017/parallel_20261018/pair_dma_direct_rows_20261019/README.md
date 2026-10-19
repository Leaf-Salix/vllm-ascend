# 双行 DMA 直接写最终采集 Tile

从正式289cc072的动态物理plan/MGATHER路径出发，延续相邻双行DMA实验。
现有生成代码显示，通用正逆序合并需要两行临时Tile、MTE2→V等待和TMOV，
降序还有额外行切片。该成本可能抵消减少的小DMA，不能仅凭burst数少就
推断整层收益。本候选仅合并原序已递增的合法相邻两项，并用gather_row
直接落入最终64×512 Tile；其它情况沿原序逐行加载，不做额外UB重排。

仍要求整段span在真实allocation内、正delta及A3 uint32 byte gap可编码；
不通过扩张shape越界。非法候选保留零，数学/事件/Top-K次序/分档不改。
少合并一部分地址可能优于通用重排，需要真机精度及完整CSA L1决定。
不固化batch、长度或页表；所有地址判断仍在device。

使用独立gm_view工作区及FlattenTileNdTo2D修复（PyPTO #3042/#3046），
三方统一PTOAS v3。正式源已恢复，任务只读冻结源码，未采用本候选。

首项128K/B16任务task_20261019_095520_2364018490 completed/exit0，全部
冻结精确/状态/保护/metadata/padding/图/代际通过。五窗多task853.14、
MGATHER参考812.14、候选817.60μs，仍回退0.672%，不采用、不扩大七档。
生成C++确认双行TLOAD直接落最终Tile，没有该段额外TMOV；源码简化成功
并不等于整层获益，剩余成对判断/描述符与DMA收益的净结果按真机结算。
