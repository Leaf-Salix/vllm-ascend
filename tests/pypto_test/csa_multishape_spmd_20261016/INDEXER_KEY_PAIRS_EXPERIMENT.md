# N64循环版两面板Key预取（2026-10-17）

完整128K/B16/S6验证通过，L1中位813.42μs，没有优于现选约807μs的证据。
不采用、不扩其他档、不重复本配置。原七档选择不变，700μs目标仍未达到。

## 变化及边界

`indexer_l1_key_pairs.py`保留已通过的N64 score双槽循环，在每次循环中计算两个面板。
完整顺序为首块0、循环(1,2)…(13,14)、末块15、排空WS15。每一对在第一个QK前
提交第二块Key的GM→L1，第二块Key的Right提取仍在第一个QK/WS之后。

只有score双槽跨循环携带，Key/Right都局部创建和消费，避免之前Key分支phi跨回边的
别名问题。Key池物理地址复用受MTE1→MTE2完成事件保护；它不是两个Key同时保存在同一区域。

保持QK M384/N64/K128 INT8、WS M16/N64/K384 FP16，FIX 2^-10与ReLU转换不变。
保持每个request的原leaf集合、动态长度/位置/页表读取、1024分数交接、双槽FFTS、
AIV Top-K、归并和最终排空。没有新增Host读取、Native forward或依赖仓改动。

## CPU与生成代码

`prepare_indexer_key_pairs.py`、`indexer_key_pairs_prepare_v1`、`indexer_key_pairs_cpu_v1`
完整codegen通过。集成`build_variant(indexer_key_pairs=True)`后生成入口与已编译入口
文本完全相同，没有重复完整CPU编译。设备开关为`--indexer-key-pairs`，默认关闭。

`indexer_l1_ring_codegen_audit.py --key-pairs`核对实际L1、Right、Acc分配及FIX转换：

- score为98304B双槽，地址65536；Key池地址163840，全部分支/循环原位。
- 三个QK(current)/WS(previous)静态对的Right与Acc均分开，没有不支持的同层搬运。
- 静态QK/WS各4处，FIX与GM写各4处；双面板循环内4次matmul、2次FIX、2次GM写，无PIPE_ALL。
- 以循环C++行号计：第二Key静态加载位点269/302/356/389；391发出MTE2完成事件。
  第一QK415、第一WS427；484才等待第二Key，第二QK在513。搬运提交提前且完成等待推迟。

这证明源编排在C++中保留了对应提交顺序，不等于已经证明设备重叠时长或性能收益。
证据见`indexer_key_pairs_codegen_v1.json`及`.cpp.txt`/`.loop.cpp.txt`。
扩展审计后，已有单面板ring设备产物仍通过默认审计路径。

## 设备结果

任务`task_20261017_111540_229620229969`，exit0。
输出目录`b16_row_deps_indexer_key_pairs_v1`，机器摘要`indexer_key_pairs_result_v1.json`。
实际设备产物执行前通过同样地址/提交顺序检查，并确认AIC/AIV各74处自定义等待，
DCCI后读取及100tick退避真正写入CCEC输入源码。

| 实现 | 三个完整L1窗口 μs | 中位数 μs |
|---|---|---:|
| 两面板Key预取单SPMD | 813.42 / 814.14 / 811.40 | 813.42 |
| 同轮原动态PTO | 907.18 / 898.80 / 892.24 | 898.80 |

Event为926.35/950.66μs，仅作附录，不参与700μs验收。
完整eager/graph/final的输出、Top-K与六份cache/state零差异，30代通知和全部保护区通过。
每次调用恢复fixture，不代表连续decode轨迹。与此前现选没有同进程配对，
不把跨轮小差值定量归因于预取；本轮结果不支持更换现选。

命令使用`run_device_compare.sh 16 131072 <新目录>`，追加：

```text
--balanced-schedule --schedule-policy earliest_ready
--schedule-cost-profile <本目录>/b16_dynamic_l1_cost.json
--cached-wait --wait-backoff-ticks 100 --qk-vector-indices --row-dependencies
--indexer-key-pairs
```

必须通过`task-submit --device auto`运行。此前三项等待配置错误及更正结果见
[编译上下文更正](COMPILE_CONTEXT_CORRECTION.md)，不要把旧约1000μs结果用作本候选的有效比较基线。
