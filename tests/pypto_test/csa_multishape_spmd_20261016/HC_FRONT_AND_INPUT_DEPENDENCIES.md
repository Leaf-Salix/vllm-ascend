# HC 三阶段融合与 Indexer 输入依赖：2026-10-17

本轮三个真机候选都通过完整用例，但未显示优于现选行依赖版本的明确收益，均未纳入
`selected_shape_policies.json`。128K/B16/S6 现选首轮为 807.16 μs，同进程复核为
817.88 μs；不同运行不能直接据相差几微秒判定新候选更快或更慢。

计时使用**完整、无内部插桩的 L1 incore 首尾**，包括所有等待和阶段间隙。每侧三个窗口，
只比较原动态 PTO 和单 SPMD，正式权重，不运行 Native forward。目标 <700 μs 尚未达到。

| 候选 | 单 SPMD 三窗 / μs | 中位 / μs | 同轮动态 PTO 中位 / μs | 任务 |
| --- | --- | --- | --- | --- |
| HC reduce→split→mix/norm 融合，保留 GM 传递 | 804.84 / 810.80 / 806.76 | 806.76 | 900.14 | `task_20261017_055741_99590920590` |
| 上述融合内改用 Tile 直传 | 819.74 / 810.44 / 810.44 | 810.44 | 891.96 | `task_20261017_061445_12767489743` |
| 原数学及分核，进一步细化 Indexer 输入依赖 | 812.62 / 815.48 / 810.34 | 812.62 | 882.64 | `task_20261017_061600_130106629188` |

三项任务均 exit=0；输出、Top-K、六份 cache/state、图重放、信号代际及保护区检查通过。
这是固定 fixture 恢复后的算子用例，不代表连续 decode 轨迹或整网验证。结构化结果见
[hc_front_and_input_deps_results.json](hc_front_and_input_deps_results.json)。

## HC 三阶段融合

实现：[hc_front_fusion.py](hc_front_fusion.py)，开关 `--hc-front-fusion`。

- reducer 的六个 16 行成员改为十二个 8 行成员，与后续 split/mix 的行分工一致。
- 保持四份投影的累加顺序、四路混合 `(y0+y1)+(y2+y3)`、BF16 舍入点、
  16 个 K256 RMS 部分和的原顺序及 rsqrt 路径。
- 复用三个原函数的 AST 数学体，在一个成员里依次执行。首版保留原 GM 读写，内部加两处
  本核 fence；原阶段外的发布、等待及缓存维护保留。
- Sinkhorn 独立保留，改为等待融合阶段。工作区和根参数不变。
- 数学 task 从 66 个减至 64 个；发布从 722 减至 704，但重排后的等待从 12183 变为
  12189，不能声称所有控制项都减少。离线融合成本使用原三段成本之和，仅为估计。

CPU 证据：`hc_front_fusion_prepare_v1/`、`hc_front_fusion_cpu_v1/`。
真机证据：`b16_row_deps_hc_front_fusion_v1/`。

## HC 内部 Tile 直传及编译失败记录

实现：[hc_front_forward_tiles.py](hc_front_forward_tiles.py)，额外开关 `--hc-front-tiles`。

T96 的每个成员只有一次 8 行 split，因此展开该单次循环，将 reduce 的 8×32 Tile
先转置为 32×8，提取行 0～7 和 4～11，再转置回 pre/post 的 8×8 Tile。
这样避免非对齐的列 4 UB 视图。mix 直接使用 split 产生的 pre Tile。

保留所有 GM 写出、原数学、编译器自动事件及阶段外同步，删除内部三次 GM 读取和两处
显式 fence。PTO IR 数学区域静态 TLOAD 从 22 处降至 19 处，`pto.fence` 从 4 处降至 0 处。
这是静态指令数，不能当作执行次数、时间或最终性能收益。

CPU 编译记录全部保留：

1. `hc_front_tiles_cpu_v1`：显式 transpose scratch 的形状错误，编译失败。
2. `hc_front_tiles_cpu_v2`：修正 scratch 后，slice 的静态 valid 与 scratch 的动态 valid
   在 PTO IR 中不一致，编译失败。
3. `hc_front_tiles_cpu_v3`：改用标准三参 transpose，让编译器生成 scratch，仍遇到该
   valid 类型不一致：`!pto.tile_buf<vec,8x8xf32>` 与 `valid=?x?`。
4. `hc_front_tiles_cpu_v4`：将对齐行 slice 改为 extract，使用有实体输出的 UB 提取，
   CPU/PTOAS 通过，随后真机完整通过。没有改 PyPTO、PTOAS 或 PTO-ISA。

实际脚本、生成源码和错误分别位于对应 `prepare_v*/` 与 `cpu_v*/`。
IR 检查见两个成功 CPU 目录中的 `hc_front_codegen_audit.json` 和
`hc_front_codegen_phase.pto.txt`。真机结果在 `b16_row_deps_hc_front_tiles_v1/`。

## Indexer 输入行依赖

实现：[indexer_input_dependencies.py](indexer_input_dependencies.py)，
[row_dependencies.py](row_dependencies.py) 的可选扩展，开关 `--indexer-input-dependencies`。
保持现选版的数学体、物理分核、核内先后、工作区、通知实现和缓存维护，只放宽已证明的三条整任务边。

- quant 成员 `w` 写 query `w`、`w+48` 的全部 64 个头，每 query 的 scale 为独占 256 B。
- 系数成员 `r` 只读该请求的六个 query，写独占的 12288 B 系数块。
- T96 的长 Indexer 按 `item=leaf*16+request` 分到 24 组。覆盖 1～32 叶后，每组只读
  两个请求，需要六个 quant 成员和两个系数成员；原来等待全部 48/16 个成员。
- 同一 MIX 组的三成员保守地使用相同输入要求。原 QK 复用 FFTS 的 216 条组内资源边保留。
- scale 的 `[24576,1]` 与 `[384,64]` 视图按元素地址检查等价，不只检查 allocation 名称。

CPU 检查 32 叶数×24 组共 768 个覆盖情形；原未修改边、同核先后和进度表转换的祖先关系
全部检查通过。等待从 12183 减到 10674，但发布从 722 增到 729。等待项减少不能直接
推导性能提升，本轮实测尚无明确收益，保持可选。

证据：`indexer_input_deps_prepare_v1/`、`indexer_input_deps_cpu_v1/`、
`b16_row_deps_indexer_inputs_v1/`。当前仅验证 B16/S6 长策略，不能套用其他档位。

## 未上卡的 scale 页 owner 候选

[indexer_scale_page_owner.py](indexer_scale_page_owner.py) 的八成员版本按物理页分配写者，
同一页的 compact 行仍由同一个成员按原序提交，避免 64 B 整行回写互相覆盖。
`indexer_scale_owners_prepare_v1/` 和 `indexer_scale_owners_cpu_v1/` 已通过 CPU 检查和编译。

历史探针中 scale 约 204 μs 完成，Indexer 还要等待约 235 μs 的 query 系数；离线模型也
没有缩短完整时长。因此暂不占卡测该候选。见
[indexer_scale_owner_priority_analysis.json](indexer_scale_owner_priority_analysis.json)。

## 复现参数及下一步

沿用 `run_device_compare.sh 16 131072 新结果目录`，共同参数为：

```text
--balanced-schedule --schedule-policy earliest_ready
--schedule-cost-profile <本目录>/b16_dynamic_l1_cost.json
--cached-wait --wait-backoff-ticks 100 --qk-vector-indices --row-dependencies
```

再加表中对应候选开关。通过 `task-submit --device auto` 提交，不裸跑占卡，不覆盖旧结果。
新候选均未扩测其他六档，因为尚未显示值得推广的收益。

现选行依赖版阶段探针 `b16_row_deps_stage_clock_v1` 已完整通过。Indexer AIC 的数学及核内等待段
中位约 211.03 μs，QK/PV AIV 约 158.02 μs；后投影还存在同核三个 O-A、两个 O-B 的串行链。
按最晚显式前驱回溯，数学/内部同步/发布段累计 724.36 μs，外部就绪至开工间隙 71.88 μs。
这不是硬件理论下限，不能把所有间隙都称为软同步成本，更不能用插桩 span 796.24 μs
替代正式 L1 成绩。证据见 `row_deps_stage_summary.json` 和 `row_deps_stage_critical_chain.json`。
此前各档结果和选择不变。
