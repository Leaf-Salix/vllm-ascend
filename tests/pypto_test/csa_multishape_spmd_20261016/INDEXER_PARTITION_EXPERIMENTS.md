# 2026-10-17：128K/B16 Indexer分片与Acc流水实验

两项候选均通过完整CSA固定fixture的精度和图重放检查，均未替换当前选择。
使用正式`/data/model/DeepSeek-V4-Flash-0731-w8a8`第4层权重、B16/S6和历史131072。
性能口径始终是完整CSA的L1 incore首尾，包含核内等待和阶段间空隙，不执行Native forward。

|方案|完整L1三窗/μs|中位/μs|同轮原动态PTO中位/μs|采用|
|---|---|---|---|---|
|既有选择：QK地址缓存|见原结果|815.84|902.80|保持|
|Indexer每叶16384候选|829.22 / 829.42 / 813.78|829.22|897.84|否|
|Indexer M192/N64双Acc|968.92 / 965.56 / 972.00|968.92|904.72|否|

现选SPMD与新候选没有同一进程直接A/B，历史差值不能当作精确的回退幅度。
尤其不能用第一项的单窗最小值813.78替代三窗中位数宣布提升。700μs目标尚未达到。

## 每叶容量8192→16384

代码为`indexer_leaf_capacity.py`，入口开关`--indexer-leaf16384`。
原配置每叶最多8192候选，131072历史加S6对应最大32769个压缩候选；
按3的倍数均衡后是6叶。此前NEXT_STEP中“9叶”的推测错误，现已修正。
新容量16384对应3叶；Cube、Vector与最终Top-K merge同时采用同一计划。

- 每个Cube仍执行22个1024分数块；query和系数的加载轮次4→2。
- 保留原分数数学、分页读取、1024块交接、FFTS协议和pair_arena布局。
- 最终merge的有效叶数随device长度计算，不把131072写入核内候选范围。
- Top-K的分组合并改变，数学等价不足以证明tie行为相同，因此仍做原exact检查。

`indexer_leaf_prepare_v1`的CPU模型覆盖4133种长度，每种最多5个有效前缀，
检查完整覆盖、无重叠、生产者与merge有效叶数一致、叶容量和arena根数上限。
该模型覆盖范围不等于当前图的页表和KV allocation支持范围。
`indexer_leaf_cpu_v1`完整编译通过。

任务`task_20261017_034636_30078467438`退出0，结果位于
`b16_cached_indexer_leaf16k_v1/result.json`与`indexer_leaf_result.json`。
输出、Top-K、六份cache/state、图重放、30代信号及保护区均通过；
只验证固定fixture，不代表连续decode或该候选的动态metadata图变体已经验证。

## M384/N64→两组M192/N64

代码为`indexer_score_pingpong.py`，入口开关`--indexer-score-pingpong`。
原INT32分数Tile为384×64，占96KiB；本候选分两组三query，每份192×64占48KiB。
先提交两份QK，再写回第一份，使两份结果的生命周期相交，阻止分配器将其别名。
保持原6叶计划、query可见范围、分页地址、分数缩放/Relu、GM交接及Vector Top-K。

完整CPU编译`indexer_pingpong_cpu_v1`通过。逐16组核对实际IR：
两份Acc分别位于0、49152字节处，均不重叠；生成C++的两次TMATMUL
也都在第一份TINSERT/FIX前。没有依靠固定MemRef、修改编译器或删除同步。
这些证据证明地址和提交顺序，不证明实际硬件重叠幅度。

首次审计错误地按TMOV寻找量化写回，断言失败；实际指令是TINSERT。
只修正审计匹配，算子未因此改变。复现工具为`indexer_score_codegen_audit.py`。

任务`task_20261017_035437_31474978052`退出0，结果位于
`b16_cached_indexer_pingpong_v1/result.json`与`indexer_pingpong_result.json`。
完整输出/Top-K/状态、图重放、30代信号及保护区均通过，但整层明显没有提速。

|Indexer AIC生成代码中的静态位点|现选|双Acc候选|
|---|---:|---:|
|TMATMUL|32|64|
|TINSERT/FIX|16|32|
|TSTORE|16|32|
|set_flag / wait_flag|166 / 166|187 / 187|
|PIPE_ALL|1|1|

缩小Tile使两份Acc能同时容纳，也增加了矩阵指令、写回和同步提交次数。
该权衡没有在整层上产生收益；没有阶段探针，不能把静态指令数量换算成具体耗时，
也不能声称已测得Cube/FIX重叠程度。此候选不扩到其他档位，不加入默认选择。

## 复现入口与约束

在本目录使用`run_device_compare.sh 16 131072 新输出目录`，共同参数为：

```text
--balanced-schedule --schedule-policy earliest_ready
--schedule-cost-profile 本目录/b16_dynamic_l1_cost.json
--cached-wait --wait-backoff-ticks 100 --qk-vector-indices
```

两项实验分别追加`--indexer-leaf16384`或`--indexer-score-pingpong`，不能组合。
设备任务必须通过`task-submit --device auto --max-time 1800`提交。
wrapper会保存实际driver；首次精度检查失败就停止，不继续采性能。
运行时metadata仍是device Tensor，未增加host回读。

实际编译产物的地址审计可以单独执行，不需要设备：

```bash
python indexer_score_codegen_audit.py \
  --artifact indexer_pingpong_cpu_v1/kernel/ptoas \
  --output 新审计结果.json
```

以上所有改动只在本实验目录，冻结静态版本、生产入口和PyPTO/Simpler仓保持原状。
