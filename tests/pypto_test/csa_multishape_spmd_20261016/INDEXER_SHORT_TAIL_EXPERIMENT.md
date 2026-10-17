# 2026-10-17：Indexer设备端短尾分支

本候选未替换现选七档。B16已完整通过但没有明确性能收益；B24也已完整通过，但没有明确性能收益。
目标仍是多档位分别编译、128K/B16完整无插桩L1低于700μs，不把局部计算量当性能。

## 改动与边界

`indexer_short_tail.py`在原N64的1024候选块循环里，读取已有设备计算的
`valid_count - score_begin`。剩余1..64项时只执行第一面板；其余情况保留
原完整路径AST。运行时长度、位置和页表仍来自device Tensor；没有新增Host取值、
GM工作区或硬编码131072。开关`--indexer-short-tail`默认关闭。

保留QK M384/N64/K128、WS M16/N64/K384、FP16 FIX缩放/ReLU及原归约。
每个1024块的FFTS READY/CONSUMED次数、两槽复用和末尾排空不变。
只改Cube；AIV仍按原逻辑读分数、逐元素乘scale、按query有效长度fillpad、再Top-K。

短路径没有写`score_transfer`的64..1023列。这些是内部无效尾部，不能被当作
新计算出的有效值；原AIV在进入排序前排除它们。沿用原S6组内位置有序的前提，
最后query提供该组最大可见范围。未扩大到不满足原输入合同的乱序position场景。

## CPU证据

- `indexer_short_tail_prepare_v1`、`indexer_short_tail_cpu_v1`完整通过。
- `indexer_short_tail_contract.py/json`核对原九条fillpad分支，并枚举叶内1..8192
  有效数、184314个query/step组合，其中2554个短尾组合。检查未写区不进入有效排序前缀。
- `indexer_short_tail_codegen_audit.json`核对实际C++：短/完整路径矩阵乘2/32，
  FIX写L1为1/16，GM store为1/16，两路径内部PIPE_ALL均0。
- B24的`indexer_short_tail_b24_prepare_v1`准备通过；设备任务自行完整编译。

`indexer_short_tail_workload.json`保留原叶分配、按物理核统计面板数量：

|128K/S6档位|最繁忙核原面板数|短尾后|解释|
|---|---:|---:|---|
|B4|96|96|仅少数核有短尾，最繁忙核不变|
|B8|176|176|同上|
|B16|352|352|16核省15面板，另8核不变|
|B24|528|513|每核都有一次短尾|

此模型不含FFTS、Top-K、缓存及调度成本，不是耗时预测。它说明不能用B16某些核
减少计算推断整层提速，也说明为何只额外验证B24，没有重跑B4/B8或七档全量。

## 设备结果

使用正式W8A8权重、S6、原完整输出/Top-K/六状态exact门槛，以及图重放、30代信号
和保护区。固定fixture每次恢复，不代表连续decode轨迹；不执行Native forward。
计时为完整无插桩L1 incore首尾，保留等待和阶段间空隙。

|档位|任务|单SPMD三窗/μs|中位/μs|同轮原动态PTO/μs|结论|
|---|---|---|---:|---:|---|
|128K B16|task_20261017_074327_274989723191|820.12 / 820.32 / 819.08|820.12|902.04|全部检查PASS，不采用|
|128K B24|task_20261017_074803_283067829725|1146.14 / 1127.42 / 1127.60|1127.60|1175.66|全部检查PASS，不采用|

B16证据为`indexer_short_tail_b16_result.json`和`b16_row_deps_indexer_short_tail_v1/`。
没有同进程现选SPMD配对，不把历史807.16与本轮820.12的差值称为精确回退。

复现沿用`run_device_compare.sh`，追加`--indexer-short-tail`；B16用现选earliest_ready、
QK地址缓存和row deps；B24用现选critical_path、自己的成本文件，不加row deps或QK地址缓存。
所有设备执行经`task-submit --device auto`，依赖仓、冻结静态版本与生产入口均未修改。

B24任务exit0，检查范围与B16相同。证据`indexer_short_tail_b24_result.json`及
`b24_indexer_short_tail_v1/`。工作量减少没有转成可确认的整层收益；没有同进程现选
配对，不能把历史1121.12与本轮1127.60之差当精确回退。保留候选，停止扩测。
