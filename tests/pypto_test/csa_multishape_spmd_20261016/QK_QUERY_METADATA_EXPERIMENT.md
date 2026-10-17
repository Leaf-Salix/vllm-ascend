# 2026-10-17：QK每query地址准备候选

候选通过完整设备检查，但未显示相对现选的明确收益，不采用、不扩测七档。

实现位于`qk_query_metadata.py`，开关`--qk-query-metadata`默认为关闭。
在当前query的tick%5==0时批量计算512个压缩KV物理行，四个压缩块复用。
原始索引与物理行各占一个1×512 INT32 Tile，共4096B，由循环状态跨tick携带。
页表每query最多4次加载减至1次；每AIV处理的地址总数则从256增到512，
这是明确的额外工作，不能只按加载次数推断性能。

保留原负索引判断、KV逐行gather、所有数学、三槽交接和FFTS。提前准备不依赖
第一个压缩块有效；尾部drain tick不加载越界query。原plan写满512项，把不合法
候选变为-1。CPU检查B1..40/S6共2519040个lane/行映射，所有query恰好覆盖一次，
每次使用前已更新元数据。该检查不代替设备数值验证。

prepare/cpu_v1通过；实际驻留UB地址135168与137216，末端139264，小于188416。
任务`task_20261017_070649_214406126149`终态exit0，正式第4层权重、128K/B16/S6，
输出/TopK/六cache-state、eager/图、30代信号及未写区域保护检查均通过。
只固定fixture恢复重放，不能称为连续decode状态轨迹。未运行Native forward。

| 路径 | L1三窗 μs | 中位 μs |
| --- | --- | ---: |
| 单SPMD候选 | 811.82 / 815.08 / 805.18 | 811.82 |
| 同轮原动态PTO | 904.92 / 906.44 / 893.10 | 904.92 |

计时为完整L1 incore首尾，未插桩。现选历史807.16、同进程复核817.88均来自其他运行，
本轮没有现选同进程A/B，不把几微秒历史差值作为精确收益或回退。
结果`qk_query_metadata_result.json`；完整记录`b16_row_deps_qk_query_metadata_v1/`；
地址合同与物理审计分别为`qk_query_metadata_contract.json`、`qk_query_metadata_codegen_audit.json`。
