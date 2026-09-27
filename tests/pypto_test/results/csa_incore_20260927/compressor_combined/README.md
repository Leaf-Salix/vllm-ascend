# Native式KV/gate合并投影：核内退化，已撤回

2026-09-27。基底算子 `da2e2368`（测试前HEAD为文档提交 `5b1464ac`）。
仅改性能版两个Compressor的投影函数，任务数和依赖不变，精度版不动。

## 候选及编译

Native `compressor_block_cube_perf.h` 将wkv/wgate写入同一L1面板，以一次较宽的Mmad计算，
再分两次Fixpipe写回。PTO基底每个K512块独立执行KV、gate两次matmul。
候选通过两次gather_row在L1拼接权重，将输出N翻倍，一次矩阵乘后切分Acc写回两张原输出。
GM权重没有重排，没有新增Torch算子；逻辑FLOPs和权重字节数不变。

先在隔离副本做完整根算子的CPU编译，不改当时还在补七档的生产源码。
Tile级转置显式使用transpose_view；第一段K在循环外计算，使Acc继承尾块的compact行距；
Acc切片显式给出valid_shape，解决当前PTOAS对动态有效行的校验要求。
最后CPU根编译及AICPU调度C++编译通过，`compile.log`含SPLIT_CACHE_CPU_COMPILE_OK。
未修改PyPTO/Simpler/PTOAS/PTO-ISA。

[compile_candidate.py](compile_candidate.py)从固定基底生成两个隔离模块与[candidate.patch](candidate.patch)，
通过真实模块名载入后调用既有CPU编译入口。大编译产物不加入Git。

## B40先导

任务 `task_20260927_171130_243676420225` 退出0。
单卡8K/B40、正式layer4权重、S6/TP1/mode2/atomic1/确定性0、EPLB关、第二层metadata复用。
5次warmup/20次无profiler计时，另采4个DFX窗口；两轮独立，不剔除慢样本。

| 项目 | 基底 | 候选 |
| --- | ---: | ---: |
| CSA本体均值 μs | 1342.68 | 1340.21 |
| 本体p50 / p95 μs | 1335.63 / 1401.54 | 1334.13 / 1387.88 |
| 完整PTO均值 μs | 1683.07 | 1680.50 |
| 同轮Native均值 μs | 1427.96 | 1403.69 |
| Attention投影，24 block均值范围 μs | 33.21–38.00 | 42.95–44.40 |
| Indexer投影，24 block均值范围 μs | 23.36–25.52 | 29.48–35.95 |

本体只差约2.5μs，不能认定收益；两个投影任务的核内范围均明显变差，候选撤回，不扩测其他档位。
没有将1340.21μs替换进保留实现的七档表。

生成代码的L0B tile也发生变化：Attention的K256/N64变为K128/N128，
Indexer的K512/N32变为K256/N64。扩大N同时缩小了K分块，
源码级两次matmul合为一次，不等于硬件Mmad次数减半，更没有完整复现Native的流水。
这能解释为何不能从高层调用数推算收益；尚无PMU证据将退化全部归因于K分块。
[两侧tile形状和生成文件路径](codegen_tiles.json)。

保护区失败0、Top-K结构错误0、非有限值0；输出max_abs=0.03125，
RMSE=0.003291107（基底0.003292603），Top-K集合替换900（基底901）。
对Native零容差仍FAIL，未做16卡token/DSpark；这些检查不冒充完整精度验收。

证据：[原始计时、逐项误差、四窗口路径](report.json)、[设备脚本](run.sh)、
[离线提取脚本](summarize.py)。生产源码已恢复至保留实现。
