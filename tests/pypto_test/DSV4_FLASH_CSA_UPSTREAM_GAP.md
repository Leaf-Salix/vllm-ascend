# 当前性能版与 pypto-lib 泳道差距

更新：2026-09-28。当前性能版71153bb3；下方旧表保留其历史采样版本，不代表当前结果。

## 当前与上游的差别及保留原因

源码参考为已核对的pypto-lib官方main2164563；历史725μs图实际Worker首尾727.98μs，
缺完整配置和对应源码，只作调度参考，不拿它计算严格加速比。

| 部位 | 上游参考 | 当前接入 | 原因与实测边界 |
| --- | --- | --- | --- |
| Indexer历史cache | 连续key/scale入参 | 直接读写Native的4160字节物理页 | 已删外部拆分/写回；源头分离候选未给出合格EP16收益，当前保留原布局 |
| query复用 | 逐query处理 | 长档按输入选双query、三query或S6；短档按最忙核工作量选双query或S6 | 吸收Native同请求复用Key的策略；本轮三query/S6分支输出状态固定规约一致 |
| Score核内 | 上游direct/buffered路径及自己的启用门槛 | FIXPIPE FP16中间值＋第二次Cube完成head规约；S6为M384/N64 | 源于Native的有效核内策略，不能为形式一致退回逐query；仍有分页地址处理和独立系数任务 |
| 长短调度 | 历史上游图的时序及默认派发 | 长档Score整组准入、禁止提前释放，短档保留原标志 | 长档已有100次P95保护证据；短档不能仅凭长档结果启用相同调度 |
| 正式输入与验收 | 上游参考合成权重/FP32残差与scale、容量16 | 正式layer4及全模型权重、Native BF16残差/FP16 scale、容量40、动态metadata | 上游约810μs新采单卡与接入不属于同输入对照；最终看EP16正式forward |

核内收益分别记录：128K/B8 Score AIC238.08→188.99μs（−20.62%），完整CSA884.77→854.86μs；
8K/B40 Score AIC95.91→46.69μs（−51.32%），完整CSA1320.76→1296.41μs。
短档Native控制出现明显波动，不用它宣称更大的净加速比；核内四窗口与正式20次计时独立。
[分组策略、原始结果与必要状态对照](results/csa_indexer_adaptive_20260928/README.md)。

基底9a01a276真实EP16两档仅快0.81%/0.54%；其模型profile显示CSA有收益、后续FFN抵消部分收益，
不能继续将所有整网差距解释成PTO调度。71153bb3统一源码七档EP16已采集：B8长档仍慢4.77%，
B16短档有2rank DSpark差异，其余五档快0.35%～2.37%；尚未通过整体验收。
[模型CSA/FFN分解及边界](results/csa_indexer_six_20260928/model/MODEL_GAP.md)。

## 此前十轮调度结束：历史算子07365e52

前十轮已完成，仅保留轻Indexer Compressor与Q_A交叠，较重Attention Compressor仍等待Q_A；其他九项撤回。
用户随后追加5轮（11–15），每轮长短上下文共同判断，完成后才回incore；本表仍是追加阶段固定起点。
[逐轮台账与证据](DSV4_FLASH_CSA_SCHEDULING_TEN_ROUNDS.md)。七档同源码数据已齐，见[完整表与42张JSON图](results/csa_scheduling_20260927/final_07365e52/README.md)；下列两种长度代表档保持原样。

| 档位 | Native均值 | PTO本体均值 | 本体p50 / p95 | 完整PTO均值 |
| --- | ---: | ---: | ---: | ---: |
| 8K/B16 | 922.02 | 781.52 | 781.05 / 791.80 | 1084.13 |
| 8K/B40 | 1414.41 | 1306.88 | 1303.22 / 1344.64 | 1657.69 |
| 128K/B16 | 1311.63 | 1247.03 | 1201.37 / 1533.18 | 1677.50 |

单位μs；同轮5预热/20次无profiler计时。第5轮8K有收益，128K没有稳定收益，长尾未解决；不声称整模型验收完成。
8K/B16四窗口Worker首尾776.06–793.26 μs，历史上游727.98 μs仅作参考，不是同输入同实现对照。
Score消费者预派发关闭后merge等待消失，但无profiler长尾仍在，已撤回；不能把setup降低直接等同本体收益。
下面c7a52af5表是本阶段之前的调度基底，保留原始数字，不能与当前同一版本混读。

## 此前调度基底：725 μs历史图＋最新版源码

用户要求调度优化持续对照725 μs上游泳道，源码参考pypto-lib最新版。
已用depth=1拉取确认官方main为`2164563`，本地一致；旧泳道按Worker首尾实际为**727.98 μs**，
没有源码/完整输入配置或Scheduler View，不能冒认为该版本采集，也不能作为严格同输入A/B。

此前`c7a52af5`性能版8K/B16/S6/TP1、正式第4层权重、mode2/atomic1/deterministic0、第二CSA层metadata复用，
单卡5预热/20无profiler计时，另4个DFX窗口。PTO本体均值832.26 μs、p50 799.37、p95 820.80；
一次1424.66 μs长尾保留。Native均值929.81，PTO含拆分写回1100.77 μs，仍慢。

| 不重叠区间 μs | 历史上游 | 当前4窗口 |
| --- | ---: | ---: |
| 首Worker→norm结束 | 66.62 | 86.82–91.48 |
| norm结束→Sparse首receive | 317.14 | 321.56–358.06 |
| Sparse首receive→merge结束 | 184.90 | 168.10–179.84 |
| merge结束→末Worker | 159.32 | 182.10–185.88 |
| 总Worker窗口 | 727.98 | 774.98–809.56 |

前段增加BF16→FP32准备及前置依赖；Q/Indexer部分核内已更快但启动分散仍大；
末段O_A同时存在核内时间和分批执行差异。Sparse分段已短于这份旧图，不能继续沿用旧热点排序。
当前没有indexer_key_repack任务，新增的是head_coefficients，不能把两版额外任务混为一谈。
最新上游O_A按行块×列块并行；接入NZ大batch原先核内串行行块，已按二维grid改造并保留：
B40尾段305–313→276–286 μs、本体1327.86→1318.05 μs，B24短尾行块通过；
[实测及上游差异](results/csa_scheduling_20260927/o_a_row_parallel/README.md)。
上表仍是c7a52af5的B16，B16只有一行块，不能据本次大batch收益改写该表。
只调整O_A登记顺序的先导未获明确本体收益，已撤回。

[完整差异、源码对应和范围限制](results/csa_scheduling_20260927/upstream_725/README.md)、
[逐任务/四窗口表](results/csa_scheduling_20260927/upstream_725/comparison.md)、
[当前关键路径和等待归因](results/csa_scheduling_20260927/upstream_725/critical_path_summary.md)。
原始泳道链接在上述文档内；后续候选继续据此记录，原生最终验收合同不变。

## 以下为此前阶段记录（按各自版本读取）

当时恢复性能优化，阶段要求是先按上游写法优化连续缓存下的CSA本体，拆分和写回暂缓。
历史表格保留其原采样范围，不改名为当前结果。

**六档结果后的优先级修正**：下文O-A/量化优先的结论只覆盖原B16/H8192，不能作为
128K泛化档位的优化顺序。当前先定位128K/B16 Indexer的计算和数据搬运：单query的key读取、
head规约位置、跨核中间结果及额外repack。pypto-lib默认路径同样逐query处理；其N768/FP16
双缓冲分支只对B≥64且压缩历史≥32768启用，当前六档不会命中。
单卡score整组启动对照已撤回：128K/B16中位数仅−0.33%，8K/B40反而+1.77%。
旧泳道的第二波score排队是真实现象，但不能把DFX中的约700μs延迟直接当作无profiler可获得收益。
核内、实际物理核分配、完整span和Native数据流差异见
[长上下文定位记录](results/csa_baseline_20260926/long_context_dispatch/README.md)。

此前阶段状态：精度版数值中性迁移、必要差异定位和整模型输出看护已完成；新连续缓存候选尚未做整模型验收。
已完成性能版 TP1/DP=EP16、EPLB 关闭的泛化对比：128K 测 B4/8/16，8K 测 B24/32/40。
主结果只看预热后10次纯 `_model_forward` 的设备耗时均值；六档数据及profiling/泳道见
[统一结果目录](results/csa_six_case_profiles_20260926/README.md)。当前PTO尚未明确快于Native，
B4专项已排除event模式差异为主因：同硬件模式trace约75%的增量在FFN/MoE，
C4后跨rank到达不齐，GMM变慢的路由/分组原因尚待直接证据。诊断见
[回退定位](results/csa_baseline_20260926/event_mode_diagnosis/README.md)。
之前不同默认event模式的model trace不能用于外推CSA本体快慢；本页旧单卡泳道对比的
具体范围仍按各自配置记录，不能混用新的整模型数据。
后续全部功能与性能测试均关闭 EPLB。
下文优化方向作为恢复性能调优后的依据；完整新口径见主清单。

2026-09-26 尾块修正：当前性能版的 sparse plan 保留 8 行主体分块，补充 runtime 有效行数。
上游当前样例同样使用 `BIAS_T_TILE=min(T,8)`、仅检查配置 T 整除，并按完整块切片；
按配置构造 B1 时 T=6、tile=6。接入侧固定容量 T=384，却需要同一入口接受实际 B1～40，
B1/S6 的 runtime T=6 不满足 8 行分块，不能照搬完整块读写。
因此在现有 plan 内增加尾块 read/store/归约有效行数；没有新增任务、metadata 搬运或重排。
这修复的是内部 scratch 越界，未单独宣称性能收益。上游若用缩小后的动态入参是否也触发，
本轮未上卡验证；不据静态样例推断动态尾块安全。具体数值见
[尾块修复证据](results/csa_baseline_20260926/precision_review/tail_fix.json)。

## 2026-09-27 连续缓存与本轮上游参考

- 接入侧 v1：8K/B16 CSA 本体 847.06 μs 均值、847.79 μs 中位数；
  128K/B16 1747.29 / 1615.93 μs。均为 5 次预热、20 次无 profiler 图重放，
  包含 HC_pre/norm/CSA/HC_post，不含拆分和写回。完整路径另计，不能直接宣布端到端改善。
- 上游参考：`2164563`，TP1/B16/S6/H8192，图外 NPU Event 同样 5/20，均值
  810.20 μs、中位数 810.41 μs；四个 DFX Worker span 为 771.80、776.40、801.06、773.58 μs。
  上游用自身合成权重、FP32 残差/scale、编译容量 16；接入侧用正式层权重、BF16 残差/FP16 scale、容量 64。
  这是性能参照，不能作为同输入模型验收或严格 A/B。
- 为兼容当前 PyPTO，参考脚本只在隔离副本移除 B≥64 才进入的 buffered-score 分支；
  B16 实际执行的 direct-score 代码保持原样。原容量 64 因上游 scope 活跃内存超过 kernel-mode
  256 MiB heap 而失败，故缩小参考容量为 16，未修改上游 checkout 或运行时。
- 新 128K 泳道中连续 Score 每 AIC 核内约 752～767 μs，正常组 span 774～803 μs，
  对比分页 v0 只有小幅变化；单纯减少读取调用没有解决主要计算/同步成本。
- 后续优先核对上游权重 L2 bypass、O projection 自适应分块、长上下文 Score 双缓冲。
  现有 Native NZ 物理缓冲继续直接绑定；不以转置权重掩盖与上游矩阵方向的区别。
- v3 已采用上游 O projection 的 M32/96/128、O-B N256 及大档位完整 K 常驻策略，
  8K/B16 本体 817.61 μs，对上述上游参考 810.20 μs 仍高约 0.9%；并非同输入/容量的严格 A/B。
  128K/B16 本体均值 1863.97 μs、p50 1611.49 μs，长尾仍在，未获得稳定改善。
  v3 起 PyPTO 为 `2a4e09ff`（官方 FIXPIPE 移植），其余工具链不变；与旧工具链分段记录。
- v4 移植上游 N768/FIXPIPE FP16 双缓冲，并将启用范围从上游 B≥64/压缩历史≥32768
  扩大到压缩历史>8192，用于当前 B16 长上下文。输入仍连续，scale 仍按 Native FP16，
  无需恢复上游逐页 gather；新增 FP16 舍入独立于精度版。128K 本体均值 1576.93 μs，
  p50 1580.39、p95 1817.54 μs，仍慢于 Native 1302.27 μs。
  Score 核内降到 480～493 μs，但四窗口中两个仍只覆盖 17 个 AIC，整组 span 从约 505 拉到约 980 μs。
  当前差距同时包含调度长尾及剩余核内工作；不能把 FIXPIPE 能力通过等同于本体目标达成。

结果： [本轮目录](results/csa_split_optimization_20260927/)，
[上游参考报告](results/csa_split_optimization_20260927/upstream_direct_cap16_h8192_b16/timing/report.json)，
[上游新泳道](results/csa_split_optimization_20260927/upstream_direct_cap16_h8192_b16/swimlane/dfx/merged_swimlane.json)。

## Native Score 片上链路候选（2026-09-27）

PyPTO 调试分支 `3e87a843` 已移植 main #2876。性能版 v7 使用 Native 的
FP16 query scale / weight / 乘积、FIXPIPE ReLU+1/1024 写 L1、第二次 Cube FP32 head 规约。
显式安排 QK(current) / WS(previous)，并保持两组 L0C 结果同时存活；
仅改用通用 stage=2 未获益，不能将二者混称相同流水。

128K/B16 本体均值/p50/p95：v4 为1576.93/1580.39/1817.54 μs，
v7 为1458.16/1342.84/1832.98 μs，同轮Native均值1307.66 μs。
本轮均值改善7.53%，尚未快于Native，也未解决长尾；不代表16卡token/DSpark验收。
与 Native 仍有单query粒度、独立系数任务、半叶Top-K/任务编排的差别；
四query共享key尚未接入。来源、失败候选和精度边界见[验证日志 §177](DSV4_FLASH_CSA_VALIDATION_LOG.md#177-将-native-上游的-fp16--cube-score-策略接入性能版2026-09-27)。

## 历史数据与口径

本轮补充 Native 对照（2026-09-27）：当前 A3 实际调用
`npu_vllm_quant_lightning_indexer` 的 `arch32` 实现。
Native 同样使用 `FIXPIPE ReLU + 1/1024 + FP16`，但 `FixpSToL1` 直接写入片上 L1；
随后 `ComputeWs` 用第二次 FP16 输入、FP32 累加的 Cube MMAD 规约 64 个 head，
只把每 query/候选一个 FP32 分数写入 GM。Vector 乘 key scale 并做 Top-K。
当前上游及 v4 则把每个 head 的 FP16 分数写到 GM，再由 Vector 做 FP32 加权/规约。
这一段 Cube→Vector score 逻辑载荷是 64×2=128 B 对 4 B，差 32 倍；
只比较该中间张量，不是总带宽、DRAM 实际流量或总耗时的倍数。

Native 的 `M_BASE_SIZE=256`、`gSize=64` 对应最多四个 query 共用 key 块；S6 分成4+2，
当前 PTO/上游逐 query 处理。Native query scale、weights 及其乘积均有 FP16 舍入，
v4 性能版 head 系数保持 FP32；Native 不补偿公共的1/1024，上游/v4系数乘1024，
正的公共比例本身不改变理想 Top-K 排序，其他舍入及规约顺序仍可能改变选择。

所以 v4 的 FP16 舍入是相对旧性能路径新增，不能说 Native 没有这层舍入。
FIXPIPE 能力也须区分目的存储：本地 PyPTO `2a4e09ff` 允许 Acc→GM 带缩放，
但明确拒绝 Acc→Mat/L1 的 `pre_quant`（移植的 verifier 标注 PTOAS#1570）。
精度版当前通过 Vector 转换再 `aic_gather` 回 Cube 来表达 Native 算术，仍不是 Native 直接片上数据流。
上述是 `2a4e09ff` 时的本地状态。现已确认官方 PTOAS 0.66 修复配套问题，
并在调试分支移植 PyPTO main #2876，提交 `3e87a843`；23 项定向单测和 A3 片上写回用例通过。
因此这项编译限制已解除，后续应按实际数据流与测量判断，而不是继续视为工具链阻塞。
逐项源码依据见 [验证日志 §176](DSV4_FLASH_CSA_VALIDATION_LOG.md#176-native-a3-qli-与上游fixpipe路径的源码对照2026-09-27)。

- 历史接入侧：当时保留的 `7eba45a3` 性能实现；B16/S6/H8192、mode=2、atomic=1，第二个 CSA 层复用 metadata。
  正式第 2 层权重及合成历史，PyPTO `88297437`、Simpler `a54c05095`、PTOAS 0.66。
  [原始泳道](results/csa_baseline_20260926/perf_qproj_upstream/swimlane/dfx/merged_swimlane.json)。
- 上游：原 `shangyou-merged_swimlane_20260924_005402.json`，从 Git `30795c69^` 读取并保留这一份
  [原始 Worker View](results/csa_baseline_20260926/upstream_gap/upstream_worker_trace.json)。
  文件只声明 `tensormap_and_ringbuffer`，没有源码提交、输入权重、NZ mode、工具链完整配置，
  也没有 Scheduler View；它是历史优化参照，不能当作同配置端到端验收。代码模式另核对本地
  pypto-lib `216456332c2a74d89cca23b7824dab264ce34bff`，不冒称这就是旧泳道的采样提交。
- 核内时间使用 `kernel-duration-us`：含核内搬运、同步与计算，不是纯 Cube/Vector 指令时间。
  setup 是 Worker receive→kernel start，当前 ringbuffer 的 early-dispatch 门控也可能落入此段。
  整组窗口是该类任务最早 receive→最晚 kernel end，含分批启动、竞争与等待。
  各任务窗口相互重叠，**不能相加**；核·μs 也不能当墙钟延迟。
- 两侧 Worker 首任务起点各归零。当前 Worker 区间 **806.14 μs**，上游 **727.98 μs**，
  差 **78.16 μs（10.74%）**。当前另有 dispatch→finish 810.00 μs，但上游缺同口径字段。
  不拿当前 851.07 μs 的整模型层中位数与这份上游单次 eager 泳道直接求差。

## 主要 incore task 与窗口差距

单位 μs，箭头均为上游→当前；核内列为每个 worker 实例的平均值。

| Task | 数量 | 平均核内 | 整组窗口 | 当前判断 |
| --- | --- | --- | --- | --- |
| Q 展开 `qproj_matmul` | 24→24 | 36.44→41.17 | 46.10→88.04 | 核内略慢，启动分散更显著；当前已采用上游 N256/M64/完整 K，不能继续归因于旧 K128 实现 |
| Q 反量化/RMS/RoPE | 48→48 | 20.59→19.70 | 48.84→125.66 | 核内已略快，主要异常在启动/资源可用性；仍是上游 8 tokens、head 循环流水模式 |
| Indexer Q 投影 | 24→24 | 18.89→15.77 | 34.40→45.54 | 核内更快，窗口仍变长 |
| Indexer Q 反量化/RoPE | 48→48 | 13.71→13.18 | 20.26→47.06 | 核内接近，任务铺开变慢 |
| Indexer score/Top-K AIC | 24→24 | 62.98→48.78 | 75.98→64.30 | 当前核本身更快，但前面新增 cache 重排 |
| Indexer score/Top-K AIV | 48→48 | 64.11→45.37 | 77.00→65.42 | 同上；不能忽略重排后只报告 leaf 的收益 |
| 稀疏 QK/PV AIC | 24→24 | 137.90→133.00 | 152.84→149.94 | 已接近且略快，不是当前相对上游的主要退化项 |
| 稀疏 QK/PV AIV | 48→48 | 140.23→135.02 | 155.18→152.18 | 128 列、24 核、预发 2 拍/3 槽与上游一致 |
| `merge_norm` | 48→48 | 16.57→21.65 | 23.32→36.62 | 核内及启动等待均需优化；当前在每个 worker 内生成交换索引，上游另有 rope_swap 任务 |
| O-A 投影 | 64→64 | 20.19→29.31 | 65.56→92.44 | 明确的 Worker 核内差距；同形状模拟器未显示权重方向足以解释它，见日志 120 |
| O-A 后量化 | 24→24 | 7.23→9.80 | 73.90→104.70 | 核内有差距，setup 也从 30.96→43.62；包含等待 O-A，不能全算量化算术 |
| O-B 投影 | 64→64 | 16.27→17.03 | 72.70→77.80 | 接近；当前 Native 二维 NZ、K512/N128，上游分组三维 NZ及完整 K 路径 |
| O-B 反量化汇总 | 24→24 | 14.38→14.48 | 39.22→33.50 | 核内基本相当 |

其余任务没有省略：[全部 54 类任务表](results/csa_baseline_20260926/upstream_gap/tasks.md)，
包括两个 Compressor、HC pre/post、QR/KV、量化、状态提交、Top-K 发布及适配任务。
[原始聚合数值](results/csa_baseline_20260926/upstream_gap/comparison.json) 另含最大核内耗时、setup、
物理核覆盖和当前 dispatch/finish 字段。旧 9 月 24 日模拟器结果没有作为当前 incore 数据套用。

## 调度与关键路径

| 可直接比较的指标 | 上游 | 当前 | 含义 |
| --- | ---: | ---: | --- |
| Worker 实例总数 | 983 | 1131 | 多 148，详见下一表 |
| Q 展开 receive 首末间隔 | 11.76 | 54.18 | 24 个任务从覆盖 24 核变为 20 核，部分核重复执行；有资源竞争/分批启动 |
| Q 反量化 receive 首末间隔 | 30.30 | 104.68 | 单核没有变慢，却更晚铺完；当前覆盖 46/48 个 AIV |
| Indexer Q 反量化 receive 首末间隔 | 4.04 | 37.06 | 同样有明显铺开差距 |
| AIC 核内区间占比 | 61.19% | 54.38% | 按 24 核×Worker 墙钟窗口归一；不代表 Cube 指令利用率 |
| AIV 核内区间占比 | 46.16% | 42.10% | 按 48 核×Worker 墙钟窗口归一；不代表 Vector 指令利用率 |
| AIC 核内合计，核·μs | 10690.92 | 10520.64 | 合计未增加，执行窗口却拉长 |
| AIV 核内合计，核·μs | 16130.84 | 16292.06 | 工作量大致相当，不能说所有核都算慢了 |

当前 48 个 Q 反量化实例中，39 个紧接同核的 `qr_hadamard_quant`，另 7 个紧接
`indexer_score_topk_leaf_aiv`；后 7 个到 380 μs 之后才接到任务。物理同核先后关系支持资源竞争，
不等同于已经证明一条额外数据依赖。上游缺 dispatch/finish，不能定量宣布“调度器自身慢了多少”。

用户进一步指出 repack 的物理分配不均：48 条 Worker 记录覆盖 47 个 AIV，
AIV_25 执行两次、AIV_28 没有，原始记录也有两个独立派发序号。
SPMD 逻辑 block 动态分配到物理核，没有一核一块的绑定；记录没有 `block_idx`，
不能据此还原两个具体块编号。AIV_24 上 Q 反量化 316.18–341.90 先于 repack
342.08–353.74，两者无相互依赖；repack 324.96 已派发，等待接收 17.12 μs，
并成为 repack 组的最后结束项。这是具体的核占用与排队证据。
score 同时等待 QR 量化，其 Scheduler 最后 finish 358.72 晚于 repack 的 356.62，
故 17.12 μs 不是可以直接从整层扣除的收益。这里沿用泳道文件原始时间轴。
详见[提取证据](results/csa_baseline_20260926/upstream_gap/aiv_repack_scheduling.json)
及验证日志第 122 节；仅复算已有文件，未追加设备采样。

当前可测的平均 dispatch→receive / kernel end→scheduler finish 分别为：
Q 展开 **9.54/7.77 μs**，Q 反量化 **4.99/8.67 μs**，Indexer Q 反量化 **0.51/5.59 μs**。
这是单实例传播/门控/完成回收区间，会互相重叠；上游缺测，也不能从层耗时直接扣掉。
`merge_norm` 的平均 dispatch→receive 127.71 μs 包含很早派发后等待前序任务的时间，
不能把它当成额外 127.71 μs 可优化延迟。

用两侧相同的四个时间边界把整条 Worker 窗口分成不重叠区间：

| 顺序区间 | 上游 | 当前 | 差值 |
| --- | ---: | ---: | ---: |
| 首 Worker→mix_x_rms_norm 完成 | 66.62 | 85.74 | +19.12 |
| norm 完成→QK/PV 首任务接收 | 317.14 | 346.48 | +29.34 |
| QK/PV 首任务接收→merge_norm 完成 | 184.90 | 183.64 | −1.26 |
| merge_norm 完成→HC_post 完成 | 159.32 | 190.28 | +30.96 |
| 合计 | 727.98 | 806.14 | +78.16 |

这是时间分段，不是各 kernel 的独立因果贡献。当前 Q 反量化在 441.84 μs 完成，Top-K 发布
在 455.38 μs 完成，QK plan 在 465.48 μs 完成；QK/PV 已从 460.78 μs 开始流水。
因此只缩短 Q 反量化，也不保证 QK/PV 提前同样多；必须同时看 Indexer 分支和资源竞争。

## 多做的事情，以及为什么模式不同

| 差异 | 实例净增 | 当前核内均值 / 窗口 μs | 原因、必要性与可优化处 |
| --- | ---: | --- | --- |
| `hc_widen` | +12 | 7.26 / 10.00 | Native 残差流是 BF16，上游 HC 入口是 FP32。转换语义必要，独立转换任务不一定必须；直接把 cast 下沉曾影响尾块和 Cube 调度，不能无证据删除 |
| `csa_row_offsets` | +1 | 2.24 / 3.36 | Native compact 输出按实际完成压缩组排列，需逐请求前缀偏移。可尝试随共享 metadata 复用/融合；不是 NZ 权重重排 |
| `indexer_boundary_init` | +16 | 2.22 / 4.18 | S=6 每请求可能完成 1 或 2 组，初始化预留但未填的行，防止 Hadamard 读取陈旧值。保护必要，单独发任务可优化 |
| `indexer_key_repack` | +48 | 20.92 / 43.20 | Native 页内是 4096B INT8 key + 64B FP16 scale，上游 key/scale 分开。当前每步按逻辑页序紧凑化，让 score 侧批量读；是主动实现选择，不是接入必然要求每步重排。应比较整条 repack→score→publish 链 |
| QR split-K | +48 | 任务 16→64 | 当前 split=8，上游代码 split=2；调优选择，非 Native 接口约束。之前改 2/4 未测到完整层额外收益，不能因此宣称调度已经最优 |
| KV 投影拆分 | +24 | 任务 8→32 | 当前 split=8/N128，上游 split=4/N256；同属调优选择，非格式要求 |
| 删除独立 `rope_swap` | −1 | 上游 1.52 / 2.06 | 当前 merge worker 内生成整数交换索引；少一个任务，但每个 worker 增加工作，需与 merge 总代价一起算 |
| **净增** | **+148** | **983→1131** | 所有任务数差额均已对应 |

其他模式差异不增加 task 数，但仍影响性能：

- `csa_rope_interleave` 16 个任务改成 `csa_rope_sign` 16 个任务。Native 已提供交错 cos/sin，
  可省上游的重复展开；同时处理 S=6 的非整块尾行。此项窗口 17.46→5.50 μs，属于减少工作。
- O-A 当前直接接受 Native 根 `[G,K,N]` NZ，默认 matmul；pypto-lib 是 `[G,N,K]` NZ，
  用 `b_trans=True`。O-B 当前复用 `[G*K,N]` 二维 NZ，核内计算组偏移；上游是 `[G,N,K]`。
  **NZ 物理打包规则相同，逻辑矩阵方向与分组方式不同**。接入处没有每次调用的 device 权重重排。
  当前保留 M128、O-B K512/N128；上游有 M32/96/128 自适应及完整 K/N256 路径。
  这些是需要继续缩小的代码/搬运差异，不能只用“Native NZ”解释 O-A 的全部 9.12 μs 核内差距。
- Native 的 state/cache 页寻址、FP16 scale 写回、有效请求/负 slot 保护仍在多个已有任务内部。
  例如 `idx_kv_scale_commit` 核内 3.82→7.36 μs；任务名相同不代表存储合同相同。
- 两个 Compressor、cache writeback、HC_pre、HC_post 上游也有，不能把整个这些阶段算成接入额外工作。
  本窗口已经复用 compact metadata，不能再把首层的两项 metadata 生产成本重复加进去。

## 按本次证据调整优先级

核内累计工作量已接近：AIC 比历史上游少 1.6%，AIV 多 1.0%；但 O-A、量化、merge 的
单实例均值仍分别慢 45.2%、35.5%、30.7%。其他任务更快抵消了这些热点，不能由总量宣布
incore 已完成。反过来，806.14 对 727.98 μs 的 Worker 墙钟差也不能全部归于调度器自身。

1. **先处理 O-A→量化的核内搬运、数据复用与流水衔接，再处理 merge**。
   明确记录每个候选与上游权重方向、M/N/K 和任务边界的区别；不增加每次调用的权重重排。
2. Q/Indexer 调度保留为第二条线：只围绕 QR/repack 的完成时刻、ready/dispatch 和物理核
   占用提出有依据的改动。多轮调序、worker 数及提前派发没有完整区间收益，暂停泛化尝试；
   不再把 Q 反量化 125.66 μs 的窗口直接当算术耗时，也不把局部窗口缩短等同于整层变快。
3. QK/PV、Q/Indexer 反量化及 O-B 已接近或快于上游，暂不优先微调其核内算术。
   HC 入口转换等附加工作仍保留，待上述关键路径收敛后处理。
4. 每个候选一次必要单卡计时；有明确收益再补受影响边界和整模型 token 看护。
   后续更新必须同时记录 incore、调度窗口、额外工作及其原因，缺测明确保留，不扩大测试矩阵。

单次真机 PIPE_UTILIZATION 诊断进一步提供方向：O-A 的 MTE2 busy/total 为 **84.4%**，
Cube 为 **31.2%**；这支持先查供数和搬运重叠，不等于证明带宽已饱和或已解释全部上游差距。
各 pipe 可重叠；本轮 program 模式、PMU 强制 single-issue，与普通 kernel 图重放不同，
没有上游同口径 PMU，因此不把计数换算成可节省墙钟。
见[验证日志第 124 节](DSV4_FLASH_CSA_VALIDATION_LOG.md)及
[硬件计数摘要](results/csa_baseline_20260926/upstream_gap/pmu_pipe/summary.json)。

补充实测：0/64B 两个 GM 别名已证明可直接消费 Native 页，避免每步 key 重排；
但单卡完整区间 p50/p95 为 837.36/854.98 μs，没有优于保留版，已撤回。
所以额外重排是性能取舍，不是不能直接读取的格式限制。Q NZ 投影开启提前派发为
858.79/878.30 μs，也已撤回。两项未新增泳道分项，本文的核内/调度表仍对应保留版；
详见[验证日志第 116 节](DSV4_FLASH_CSA_VALIDATION_LOG.md)。

后续调度反例：12 个历史重排 worker 提前、16 个尾部 worker 等写回后，实例数
1131→1111，Q 反量化窗口 125.66→36.72 μs，但 Worker 整层 806.14→829.16 μs。
QR 投影启动与 Top-K 发布更晚，未缩短真正关键路径；候选已撤回。
这支持优先研究任务图和资源竞争，但**未证明任务数差异是唯一主因，也未证明调度器实现自身变慢**。
详见[验证日志第 117 节](DSV4_FLASH_CSA_VALIDATION_LOG.md)。

显式依赖对照也没有带来收益：借鉴上游 **TP 入口** 的投影链，当前 TP1 单卡整层
p50 为 907.18 μs；放宽为仅让两个 Compressor 等 Indexer Q，为 880.88 μs，
均劣于保留版 817.22 μs，已撤回。根入口拆分同时改变 scope，未另采泳道，
不能把全部退化只归因于某一条依赖。详见[验证日志第 118 节](DSV4_FLASH_CSA_VALIDATION_LOG.md)。
该轮结论是：任务图和资源竞争值得研究，但“主要差距都来自任务差异”尚未被隔离实验确认。
结合后续多轮反例，本轮优先级改为上节的关键核内任务及局部流水。

恢复原编排后，Q 反量化 48→24 worker 为 835.89 μs；保留 48 worker、
只要求整组启动为 854.24 μs，也均未测到收益并撤回。
未新增候选泳道，不将两项 Event 计时反推成核内或调度分项。
详见[验证日志第 119 节](DSV4_FLASH_CSA_VALIDATION_LOG.md)。

O-A 单核对照进一步收窄了判断：同 M96/N128/K4096、tile K256、stage=2 下，
当前 Native `[G,K,N]` 与上游 `[G,N,K] + b_trans` 的指令窗口为 **13.459/13.417 μs**，
均执行 32 条 MMAD。该合成输入对照不能直接外推真实多核，却不支持把上表全部 9.12 μs
Worker 均值差距归因于权重方向。CANN 9 的 Ascend910B1 camodel 与真机 Worker 指标分开记录；
未修改 Native 存储或增加重排。详见[核内证据](results/csa_baseline_20260926/upstream_gap/oa_incore/README.md)。
同轮 O-A stage=4 的单卡完整区间为 **849.78/866.94 μs**，没有确认收益，已恢复 stage=2；
未采候选泳道，也未交错重测基线，不由该总时长反推 O-A 或 runtime 各自退化多少。

CPU 复算：`python tests/pypto_test/results/csa_baseline_20260926/upstream_gap/compare.py`。

同轮 merge 对照（日志 121）：保留源码重新采样为 **842.57/860.34 μs**；
共享交换索引为 **854.67/873.96 μs**，分四组发布给 O-A 为 **855.54/871.24 μs**。
两项均撤回，不把较早的 817.22 μs 作为唯一基线。前者维持任务数、增加 4 KiB GM 表，
后者保持 48 个 worker、增加 3 个编排 task；与上游的边界差异均有记录。
未采候选泳道，不能宣称已经缩短 merge 核内或提前 O-A，也未扩展正确性和整模型测试。
