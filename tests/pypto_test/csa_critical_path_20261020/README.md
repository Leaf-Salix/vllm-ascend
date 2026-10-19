# 2026-10-20：依据当前关键路径继续动态单 SPMD CSA 优化

## 范围与当前状态

当前正式已在本地提交13259f92采用Score区间优化，完整快照为source_after_score_split；
source_reference是此前E49历史基线。新关键路径诊断见CURRENT_SCORE_PRESSURE_DESIGN.md。
以下按时间保留此前过程，涉及“正式未改”等表述以各段发生时为准。

目标和验收口径完整保存在 GOAL.md。正式源码基于 5df25698 / E49；本目录
source_reference 为开始本轮时的完整 37 文件快照。未启用 SDMA，不修改共享环境。
上一轮是在用户设置 goal 前提供描述，没有新的性能进展；本轮以正式源和任务终态为准。

## 已有证据与缺失项

B16 复用 csa_sdma_prefetch_20261019 的 reference 全 72 核阶段时戳。
其 source_reference 与本轮正式源逐字一致；它使用 SDMA-enabled context，
reference 算子本身没有 SDMA。时戳含探针，不作正式性能。
Q_B 的实际 20 核阶段中位约 67.38 μs；Score 的 Cube/Vector 非阶段 GM 等待
区间中位约 209.92/196.27 μs；Attention 约 171.52/175.60 μs。
后两项仍包含 helper 内 FIFO、DMA、Scalar 和探针，不称为纯计算。
先补 B28 当前源码同类时戳，结合已取得的完整全核 incore 流水定位可暴露工作。
不重复 B16 已有诊断，不先进行七档全量测试。

## SDMA 工作保留

三种 Q_B 预取在 B16/B28 均无完整收益。按用户新 goal 暂时转向其他手段。
Simpler 独立 sdma worktree 的 4 文件通道编号修复仅是未编译、未验证草稿；
补丁保存在 sdma_channel_draft.patch，不应作为可用修复或被本轮测试装载。
新建 ptoas-csa-sdma-20261020 分支仍停在 e5c1c432，没有源码修改或新构建。
公共问题已反馈原 PR，暂不继续消耗本轮优化时间修复 SDMA。

## 验收约束

每个候选记录关键路径、收益依据与停止条件。功能先过冻结检查，再以同轮
三方完整无插桩 L1 决定采用；同一任务运行期间冻结 runner、源码与工具链。
本轮所有仓库禁止 push；阶段提交仅纳入明确列出的本轮文件。

## 三项诊断的完成结果

三项任务均 completed/exit0，完整CSA单次输出、Top-K、cache/state、保护、
同次72核信号协议与原多task精确对照通过。不是连续轨迹、metadata/padding
重放或正式性能通过；本轮未新增这几类声明。

B28全核阶段报告见 b28_phase_summary.json：Score扣除外层GM等待后的中位
AIC/AIV区间369.66/362.80 μs，Attention为277.25/278.69 μs。它们仍包含内部
FFTS、数学、DMA和探针，不能当独占引擎耗时。最初分析脚本漏更新上一阶段
游标而让interval恒0，已在读取原始端点时发现并修正，没有重跑设备或改原始数据。

供数探针在原8处FFTS等待与唯一物理行MGATHER调用前后累计设备计时器。
不额外插入流水屏障；同样不能将Scalar发射区间当成独占MTE服务时间。
每核仅在Attention结束写独占512B信号槽的第3条缓存行，32..46项；其它
padding与原协议检查保留。依据本次真实mask逐核核对全部事件次数：
B16每核20有效块/每AIV16次MGATHER，B28分别35/28。

| 场景 | AIC等KV中位 μs | AIV包围MGATHER中位 μs |
| --- | ---: | ---: |
| 128K/B16 | 101.36 | 26.15 |
| 128K/B28 | 186.50 | 42.82 |

不能拿两列相减认定剩余部分全是数学，也不能据此删除MGATHER屏障。
这份新证据不支持把全部KV等待归罪于MGATHER，故暂不推进其内部屏障修改。
下一候选转向更大的Indexer Score区间，说明见 KEY_BLOCK_CANDIDATE.md。

三项完整任务日志与原始phase clock保留，所有大编译产物仅登记本地路径。

## 整块 Key 候选已结束

两代表任务完整精确、状态、保护、metadata/padding 同图及信号检查通过；
同轮五窗完整 L1 均回退。B16 当前/候选 814.70/864.82 μs，B28 为
1214.20/1299.96 μs，五窗全部较慢。不采用、不扩档、不扫池大小，正式算子
保持原版。详细三方窗口和边界见 KEY_BLOCK_CANDIDATE.md 与
KEY_BLOCK_PERFORMANCE.json。

扩档门槛已获用户确认并记入 GOAL.md：重点档至少净省20μs，另一档无明确
退步后再扩档。下一步只补 B28 原正式源的 Indexer READY/ACK 等待证据，
以决定是否值得改生产消费流水；task_20261020_022224_28339126381 尚待结果。

## Score背压与下一候选

背压任务已终态exit0，单次完整输出/Top-K/score、cache/state及保护/信号
通过，72角色计数配对一致。B28 AIC等ACK中位38.14μs、AIV等READY中位
151.17μs；只作为Scalar插桩证据，优先检查Cube生产链，暂不扩大FIFO。

等待期间的64KiB独立KV缓冲资源尝试仍因201472B>188416B失败，停止不上卡。
随后QK提前FIX的v1因Acc同址复用而重新串行，也不上卡；v2以已有
matmul_acc(init_cond=True)保留独立WS Acc，CPU地址与事件机制审计通过，
已进入B28完整验证。详细边界和停止条件见SCORE_FIX_EARLY_DESIGN.md。

该候选两档现已全部终态：完整精确、状态、保护、metadata/padding图和
信号检查通过。B16当前814.94→807.80μs，B28当前1213.40→1198.56μs，
B16四窗更快、B28五窗更快，但净省7.14/14.84μs均低于20μs门槛。不扩档、不迁入正式
源码，保留完整过程和资源机制证据。两轮设备候选已收齐，无本轮悬挂任务。

## 后续叶分配候选

基于B28真实33/34/41块的不均衡，新增按24核约数对齐叶数并轮转逻辑叶的
候选，Top-K同步反解真实writer。模型覆盖与实际计数对齐；首轮完整精确/
状态/保护/图通过，但B16慢5.38μs、B28仅快3.86μs，不采用。

区间修订版消除新增的动态除模，并按Host已有group容量上界编译选择路径；
实际长度和group仍由device处理。B16关闭后的C++与已验证FIX候选一致，
复用该轮完整结果；B28 task_20261020_033233_30933866282等待完整验证。
数据按v1/v2分开保存在LEAF_BALANCE_PERFORMANCE.json，说明及全部停止条件
见LEAF_BALANCE_DESIGN.md。未达到扩档门槛前不改正式源。

叶分配v2现已终态：B28同轮正式1202.30→候选1195.48μs，仅省6.82μs，
四窗较快；完整冻结检查通过。B16复用相同生成代码原轮净省7.14μs的数据。
两档未达20μs门槛，结束整组方向，不采用、不扩档。详细见
LEAF_BALANCE_DESIGN.md，原始三方五窗全部保留。

## Q_B首M流式保留权重候选

新的source_qb_stream_reuse直接基于正式E49，只在长历史128<T<=256
按资源区间启用。与旧整列前置驻留不同，K2/K3在首M前两次MMA发出后
才载入，随后M复用四份权重。完整CPU与地址/事件审计通过，B16关闭版
完整C++与正式E49相同，不重复其设备测试。B28完整任务
task_20261020_041024_37973320637等待结果。两次测试入口参数错误均
在候选运行前退出，记录及停止条件见QB_STREAM_REUSE_DESIGN.md。

Q_B流式驻留候选终态：完整精确/状态/保护/metadata/padding/图/代际通过；
同轮五窗正式1198.38→候选1198.84μs，净省-0.46μs，3/5窗较快，未达
20μs门槛。结束、不采用、不扩档。完整数据见QB_STREAM_PERFORMANCE.json。
B16关闭版代码相同，未重复设备测试，正式E49和依赖仓均未改。

## Attention独立64KiB供数与四半段在线状态已结束

旧64KiB方案的201472B资源失败经共址KV/分拆oi状态解决，两档UB最终
181248/182016B，低于188416B；完整动态精确、状态、保护、metadata/
padding图和代际通过。完整L1却无收益：B16正式827.00→候选827.60μs，
B28为1198.26→1203.52μs，各仅1/5窗较快，未达20μs门槛。
结束、不采用、不扩档。完整记录见ATTENTION_SHARED_KV_DESIGN.md，
三方五窗见ATTENTION_STATE_HALVES_PERFORMANCE.json，正式E49不变。

## Score早FIX与单次WS写回取得三档明确收益，正在收口区间

B16/B24/B28同轮完整L1分别净省15.38/40.72/43.80μs，全部冻结检查通过；
满足20μs扩档门槛后，七档已测齐。B8净省7.30μs，B4回退5.56μs，故
区间候选仅在原长G6域且T>=32开启，沿已有M16资源边界保留小档旧代码。
两8K执行C++与参考相同，保留原始测量波动。所有原三方五窗见
SCORE_SPLIT_ALL_PERFORMANCE.json，具体选路与生成等价性见
SCORE_SPLIT_INTERVAL_CODEGEN_COMPARISONS.json。
真实B1/S1、B16/S5图回归完成通过；T30关闭与T36开启边界正在验证。
正式源暂未替换，最终30%目标未达。完整过程见SCORE_SPLIT_STORE_DESIGN.md。

Score区间方案现已正式采用。T30关闭回归、T36开启完整检查均通过，
T36的590.72→589.62μs仅基本持平。正式只改四个算子文件，算法AST
与验证版本一致，详情SCORE_SPLIT_ADOPTION.json。当前数据按所选代码
映射于SCORE_SPLIT_CURRENT_PERFORMANCE.json；原始七档（含回退）仍在
SCORE_SPLIT_ALL_PERFORMANCE.json。128K五档30%仍0/5，未push。

## 新正式关键路径与KV置换候选终态

13259f92之后的B16/B28全72角色阶段及Score READY/ACK诊断完成，见
CURRENT_SCORE_PRESSURE_ANALYSIS.json。识别B28部分核前段KV与较多Q_B叠加，
据此试了一次KV逻辑worker置换，仍24核原数学/发布，区别于旧四核KV方案。
CPU及关闭代码等价通过；B28完整冻结验证通过，但同轮当前1152.44→候选
1156.20μs，慢3.76，仅2/5窗快。未达20μs，结束、不采用、不扩档，不扫
核编号。完整过程见PREFIX_KV_BALANCE_DESIGN.md及PERFORMANCE报告。
正式保持Score区间版，所有任务终态；没有把跨轮1166.04→1152.44计成收益。
