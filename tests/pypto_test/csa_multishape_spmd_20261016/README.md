# CSA 单 SPMD：按形状选择编译与调度策略

**当前目标（2026-10-17 更新）：七档各自降低至少 30%，详见 [目标约束](GOAL_AND_CONSTRAINTS.md)。原 700 μs 目标及下文历史记录保留，不再作为当前完成标准。**

用户补充的目标约束见 [GOAL_AND_CONSTRAINTS.md](GOAL_AND_CONSTRAINTS.md)：按 CSA 自身
工作量划分区间，档内动态处理输入；七个规格是回归点，不是支持白名单。HC_pre 只作为
分档思路参考，不照搬其阈值或专用分支。最新区间实现见 [RANGE_POLICY.md](RANGE_POLICY.md)。

**当前正式动态源码与阶段提交（2026-10-17 晚）：** 入口为
`vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/entry.py::decode_csa_spmd`。
完整七档、同址metadata/padding重放和最新候选见
[动态整链验证入口](dynamic_full_pipeline_20261017/README.md)，
优化主记录为 `vllm_ascend/ops/pypto/DSV4_CSA_SPMD_OPTIMIZATION_LOG.md`。
下文13:22及更早的807μs等数据属于历史形状特化原型，不能作为当前动态源码成绩。

**历史选择（2026-10-17 13:22）：** 七档见
[SELECTED_SEVEN_RESULTS.md](SELECTED_SEVEN_RESULTS.md)，128K/B16完整无插桩L1
现选首轮中位807.16μs，最新同进程现选807.50μs；七档均尚未达到新的30%门槛。
不同batch及长度策略允许分别编译、分别调优。


三处RoPE逐行gather合并的候选完成全门槛PASS，但完整L1中位815.10μs，未优于现选，
未采用、不扩档，见[RoPE完整记录](ROPE_GATHER_EXPERIMENT.md)。固定分核MILP三轮CPU搜索
没有找到更好安排，没有上卡，见[模型记录](FIXED_CORE_MILP_EXPERIMENT.md)。

全阶段noinline与现选100tick等待组合的B16诊断已完整PASS，L1中位807.02μs，
与现选持平，未采用、不扩档；该诊断不计正式700目标。全部任务已终态。
见[组合诊断记录](OUTLINED_CACHED_WAIT_EXPERIMENT.md)。当前队列以PENDING_JOBS.json为准，
以下历史段落中的“全部任务终态”仅指相应记录当时。

本轮将QK分工候选的softmax/PV/发布分块分别设为32/16/8，CPU Vec分配167552B，
完整设备精度、图重放、30代和保护区PASS。L1三窗830.46/827.18/832.04μs，中位830.46；
同轮动态911.08μs。仍未优于现选，未采用、不扩档，见[分工及分块记录](QK_ROLE_SPLIT_EXPERIMENT.md)。
固定分核的局部交换和主Q后处理成组后移均未改善CPU模型，没有上卡，
见[局部排程记录](FIXED_CORE_LOCAL_SCHEDULE_EXPERIMENT.md)。全部任务终态，700μs目标仍未达到。

本轮B16 QK提前KV通知完整精度/图/保护区PASS，L1中位823.38μs，同轮动态905.06μs；
未采用、不扩档。见[提前通知记录](QK_EARLY_KV_EXPERIMENT.md)。
Indexer任意stride/GM基址条件选择的CPU接口检查见[页视图记录](INDEXER_PAGE_BASE_SELECT_CPU.md)，
没有上卡。双AIV按KV搬运与attention数学分工的原型完整设备精度通过，
L1中位885.08μs，未采用、不扩档；见[分工记录](QK_ROLE_SPLIT_EXPERIMENT.md)。
单次阶段诊断已完成，主要退步在QK/PV，AIV1数学/内部等待/维护段中位222.80μs，
历史现选158.12μs；插桩及暖机差异限制跨轮比较。当前全部任务终态，七档选择不变。

本轮修正测试驱动的提前编译问题：三个候选的首次artifact审计在编译扩展外，
实际未应用所请求的100tick等待，之前约1000μs的结果不能归因为数学核回退。
已移入正确上下文，并在执行前核对实际AIC/AIV源码。扫描74项已完成测试仅这三项受影响。
修正后WS加载顺序814.12μs、PV状态分块820.24μs、N64 L1环806.38μs，均完整PASS。
未替换现选，也未扩七档。详细证据和所有历史数据见
[编译上下文更正](COMPILE_CONTEXT_CORRECTION.md)。

PV状态分块解决了196608B的UB编译超限，实际占用163872B，但本轮仍未证明性能收益。
见[PV完整记录](QK_MERGE_STATE_EXPERIMENT.md)。
另有[两组query完整流水CPU评估](QUERY_GROUP_SCHEDULE_FEASIBILITY.md)：六种模型均未优于现选，未上卡。
WS首对保留原序、后14对后移已验证；全15对后移的CPU失败证据仍保留。

新的N64两面板循环让第二块Key先搬到L1，再做第一块QK/WS。完整CPU编译、
Right/Acc/L1地址及真实提交/等待顺序审计已通过，只让score双槽跨循环。
任务`task_20261017_111540_229620229969`完成全门槛PASS，但完整L1中位813.42μs，
同轮动态898.80μs；未采用、不扩档。见[两面板预取记录](INDEXER_KEY_PAIRS_EXPERIMENT.md)。

本轮Indexer N32双Acc完整PASS但中位981.12μs，未采用，见
[N32记录](INDEXER_N32_ACC_EXPERIMENT.md)。整个KV标记删除后远端等待增加490，未上卡；
保留汇总通知的纯信号候选完整PASS，首轮802.96μs，但同进程现选807.50、候选812.08μs，
未测出收益、不纳入选择。见[KV标记记录](KV_TOUCH_SIGNAL_EXPERIMENT.md)。
另有[固定L1循环更新CPU探针](TILE_LOOP_INPLACE_CPU.md)：正确接收assemble返回值的版本
完整编译通过，未上卡，不能外推CSA大Tile流水。全部本轮任务已终态，七档选择不变。

上一轮N64两面板成组与2048分数交接均完整PASS，但L1中位为920.58/834.68μs，
未采用；见[分组记录](INDEXER_PANEL_GROUP_EXPERIMENT.md)、
[交接记录与首轮错误修复](INDEXER_TRANSFER_2048_EXPERIMENT.md)。
Key L1池扩大通过CPU编译但未减少等待，未上卡，见[CPU记录](INDEXER_KEY_POOL_CPU_EXPERIMENT.md)。
主Q双N128列块首轮806.46μs；同进程复核现选808.92、候选806.02μs，
差0.36%且三窗区间重叠，未采用、不扩档；见[主Q完整记录](QPROJ_L1_COLUMNS_EXPERIMENT.md)。
所有本轮任务均已终态。七档选择不变，700μs目标未达到。

此前短尾候选B16/B24与B16递增轮询均完整PASS，L1分别820.12/1127.60/829.94μs，
没有明确收益，均不采用；见[短尾记录](INDEXER_SHORT_TAIL_EXPERIMENT.md)与
[递增轮询记录](RAMP_WAIT_EXPERIMENT.md)。N64展开循环化两版均CPU编译失败，
未上卡，见[循环化失败记录](INDEXER_ROLLED_PANELS_EXPERIMENT.md)。本轮所有任务已终态。

已新增完整CSA同地址动态metadata验证：128K/B16为131072→131071→130048→131072，
8K/B16为8192→8191→8186→8192；两侧输出/TopK/六份状态/保护区逐bit检查通过，
每侧固定一张图，地址不变，无新增编译单元。短历史首版7168跨策略失败记录保留，
修正版在派发前比较原编排策略，跨策略明确要求另选模板。见
[DYNAMIC_METADATA_VALIDATION.md](DYNAMIC_METADATA_VALIDATION.md)。这不是连续decode或整网验证。

本轮O-A K512与O-A八阶段共享循环均完整PASS，但L1为827.94/830.74μs，未替换现选。
同一QK数学体独立编译没有减少循环屏障；Indexer Key提交探针为11.81μs，
整体Indexer AIC约210.23μs，均为诊断证据，不参加700μs验收。
新增QK32行KV准备与HC48 AIV流式供数均完整exact/图/保护区PASS，但L1中位
分别831.32/844.40μs，未优于815.84，均未采用、不扩其他档。HC自动pipeline版本
在CPU审计中发现READY前置顺序被改变，未上卡；显式range版本通过。见
[HC_STREAM_PRE_SYNC_AUDIT.md](HC_STREAM_PRE_SYNC_AUDIT.md)。
新增Indexer每叶16384与M192/N64双Acc两项候选也均完整PASS，但L1中位分别829.22/968.92μs，
均未纳入选择。地址/提交顺序核对、完整结果和限制见
[INDEXER_PARTITION_EXPERIMENTS.md](INDEXER_PARTITION_EXPERIMENTS.md)。
早期PV只缩小incoming Tile的版本因UB需196608B失败，未上卡；后续拆分循环状态后的结果见上方最新记录。
QK直接GM→L1候选完整exact/图/30代/保护区PASS，但L1中位1100.10μs，同轮动态892.66μs，
明确不采用。小搬运探针、同步模型、生成代码审计和失败记录见
[QK_DIRECT_L1_EXPERIMENT.md](QK_DIRECT_L1_EXPERIMENT.md)。
新增按query/head细化依赖：128K/B16首轮807.16μs，同进程复核旧841.94→新817.88μs，
完整精度和128K→64K→128K同图动态metadata检查通过。已纳入B16候选，其余六档不动。
见[行依赖实验](ROW_DEPENDENCY_EXPERIMENT.md)。Indexer N80完整精度通过但L1为879.10μs，不采用，见
[候选记录](INDEXER_N80_EXPERIMENT.md)。此前任务均已终态。最新HC整行驻留与主Q显式L0预取均完整通过，但L1为819.02/830.62μs，均未采用，见
[候选记录](HC_MIX_QPROJ_EXPERIMENTS.md)。
新增HC三阶段融合、内部Tile直传、Indexer输入行依赖均完整PASS，L1中位分别
806.76/810.44/812.62μs，均未替换现选。细节见
[本轮候选和失败记录](HC_FRONT_AND_INPUT_DEPENDENCIES.md)。现选版阶段探针完整PASS，
仅供归因，不能作为正式性能结果。原12成员HC mix的分块UB缓冲也完整PASS，
L1中位814.74μs，未采用；见[分块UB记录](HC_MIX_CHUNK_BUFFER_EXPERIMENT.md)。
新增成员级离线排程三种策略只完成CPU评估，没有明显模型优势，未上卡；见
[成员级排程](MEMBER_SCHEDULE_CPU_EXPERIMENT.md)。恢复权重缓存声明后MLIR确实改变，
但C++与现选全文相同，未重复测量；见[缓存策略核对](WEIGHT_CACHE_POLICY_EXPERIMENT.md)。
QK每query地址缓存完整PASS，中位811.82μs，未采用；见
[QK地址缓存](QK_QUERY_METADATA_EXPERIMENT.md)。N80 WS K128完整PASS，但中位855.10μs，未采用；见[WS分段记录](INDEXER_N80_WS128_EXPERIMENT.md)。
QK全有效块路径完整PASS，中位817.26μs，未采用；见[分支记录](QK_ALL_VALID_EXPERIMENT.md)。
全部尝试及失败见ATTEMPTS.md，接续见NEXT_STEP.md；
PENDING_JOBS.json是观测快照，不是实时队列。

## 目标与边界

2026-10-16 用户明确：**不要求不同 batch 共用一份二进制，目标是各档性能都好**。
参考 `vllm_ascend/ops/pypto/hc_pre.py` 的做法，入口允许动态描述维度，按档位选择
分块、分核和流水参数；JIT 可以按结构形状分别编译并缓存，也允许精确档位特化。

旧静态源码、正确性和性能结果在相邻的
`csa_static_spmd_feasibility_20261015/`，已以 `a919f119` 推送到
`nalinaly/vllm-ascend:dsv4-flash-pto-v0.25.1rc1`。本目录不覆盖旧结果。

- 第一版继续使用冻结的同一套 CSA 数学核体，参数化形状、视图、工作区和核序列。
  这样可以把调度变化与数学变化分开验证，暂不切换另一会话的算子代码。
- 各档选择自己的历史调度参考；重新计算实际分片数量与依赖。不得把 B16 的
  Tile、Indexer/O-proj 分支或完整等待表硬套到其他 batch。
- `position_ids`、`kv_seq_lens`、slot、页表、紧凑行 metadata 和 cache/state
  始终是 device Tensor。初始化只读取 shape/stride/dtype，不增加每步 `.item()`。
- 调度的分支选择允许使用调用方已有的 Host `max_seq_len`。它只选择 Indexer
  策略，实际候选范围、页地址和尾块仍在核内读取 device metadata 后计算。
  若跨越策略分界，应选新变体；不把跨分界错误执行称为“支持动态长度”。
- 本轮沿用 CSA 的 S=6 契约。此前独立 HC_pre 的 B/S 测试表不是 CSA 的七档。

## 验证次序

1. CPU 检查各规格的调用匹配、分片覆盖、MIX 配对、工作区边界和依赖偏序。
2. 用正式权重检查单卡输出、Top-K、cache/state、保护区和图重放。
3. 性能只比较原动态 PTO 与单 SPMD，沿用完整 CSA 的 L1 incore 首尾口径，
   每档 3 个窗口。另保留图重放 Event 时间；不执行 Native forward。

原七档为 128K/B4、B8、B16、B24 和 8K/B16、B24、B32。
另外选一个非历史 batch 验证参数化，避免只包装七份硬编码文件。
CPU 准备通过不代表真机正确或性能通过；新档未经测量不预记性能收益。

## 当前基线

已完成的 128K/B16 静态对照：原动态 PTO L1 中位数 897.70 μs，
单 SPMD 1380.22 μs，耗时增加 53.75%。这是保留的旧结果，不是多档版的新测量。
后续需先定位、减少融合后的控制/同步成本，再按各档独立选择策略。

## 当前目标与进展

活动目标：先完成多档形状支持，再将 **128K/B16 的无插桩完整 L1 incore
首尾耗时降到 700 μs 以下**。不以某个子阶段、核内累计忙时或插桩结果替代。

实现入口是 `shape_specialization.py:build_variant`：根据实际 Tensor 描述符
重绑定原符号编排，选择相同数学分支的参考调度，生成当前规格的 JIT 源码及
初始化工作区合同。PyPTO 自身缓存编译产物。它是实验准备入口，尚未替换生产入口。
不同规格允许编译不同二进制，运行时 metadata 保持 device Tensor。

`prepared_v1/result.json`：原七档及额外 128K/B12 的 CPU 形状、视图、MIX 配对、
分片覆盖、依赖偏序与三代进度模型均通过。
`prepared_v1/static_b16_identity.json`：B16 生成源码 AST 与已真机验证静态版一致，
不重复进行相同 B16 基线测量。

`device_compare.py` 从实际 Native 参数描述符生成变体，沿用原静态对照的
正式权重、合成历史、exact 输出/Top-K/六份 cache/state、保护区、图重放和
交替 AB/BA 计时方法。每侧 5 次预热、20 次 Event 样本、3 次 L1 窗口。

阶段探针在 `stage_clock.py`：仅当前测试进程拦截 PTOAS 返回的 C++，保留
插桩前后文件和边界检查记录。用每核进度区空闲的独占缓存行存四个 SYS_CNT
时间点；不写信号所在缓存行，不改 PyPTO/Simpler 仓库。插桩会影响耗时，
该结果标记 `eligible_for_700us_target=false`，只用于阶段归因。

全部新尝试与任务编号见 `ATTEMPTS.md`。

## 多档最新实测

重分配候选已完成原七档真机比较，全部通过固定 fixture 的精度、状态及图重放检查。
详细表格见 [BALANCED_SEVEN_RESULTS.md](BALANCED_SEVEN_RESULTS.md)。
七档矩阵中 128K/B16 无插桩结果为 **1067.36 μs**，同轮原动态为 **902.28 μs**；
700 μs 目标尚未达到，其他六档也仍慢于原动态。

`balanced_schedule.py` 在保持数学 DAG、分片覆盖和 MIX 配对的前提下重新分配。
完成通知汇聚会增加尾延迟，当前候选关闭；独立阶段片上内存规划仍在实验中。

21:22 补充：任务计数、轮询退避、独立阶段编译均未产生整体收益，具体负结果和定位证据已记入 ATTEMPTS.md。当前仍以 balanced_plain 的七档结果为基线；未切换生产算子默认路径。

21:55 补充：普通 balanced 新复测中位数 1054.70 μs，但三窗含一次 3922.40 μs 长尾，
不把中位数变化当作稳定优化。全阶段 noinline 诊断为 1044.54 μs，未切换生产入口。
外部 PMU 与实际参数裁剪记录见 ATTEMPTS.md、noinline_and_pmu_results.json。

22:44 补充：cached GM信号退避取得收益。B16在0/50/100tick下分别
1047.58/894.10/851.32μs，全部正确；100tick三窗850.94/851.32/854.06，
同轮动态907.02μs。700μs目标尚未达到，其他六档尚未测该退避候选。
此前“轮询退避未获收益”仅指已测ld_dev候选，不能概括此DCCI读侧新结果。
见cached_backoff_results.json与ATTEMPTS.md；未修改生产入口。

23:12 补充：新七档已全部完成并通过，见 [CACHED_SEVEN_RESULTS.md](CACHED_SEVEN_RESULTS.md)。
各档独立成本与分核、2μs信号退避使L1较同轮原动态降低2.89%～8.54%；
128K/B16为832.28μs，仍未到700μs。Event单独列出，部分档仍略慢。
当前仅在B16继续验证N256输出投影列分块，复用原算子已有分支，
原动态对照保持；首次完整CPU代码生成v2通过，真机结果尚待完成。


## 2026-10-16 23:55 增量

最新按档保留的七档结果见[SELECTED_SEVEN_RESULTS.md](SELECTED_SEVEN_RESULTS.md)。
QK压缩行地址批量在UB计算，只在128K/B8、B16保留新候选；其他档继续用原标量读取。
B16完整无插桩L1中位815.84μs，700μs目标尚未达到。全部新旧数据、负结果和探针限制
见[QK_SEVEN_RESULTS.md](QK_SEVEN_RESULTS.md)及ATTEMPTS.md。
独立GM复用实验逐位通过，节省25051136B但833.28μs与原性能基本持平。
主Q投影不同worker数只做CPU准备/模型检查，未把模型值当成实测。


## 2026-10-17 09:58：N64循环流水与query完成粒度

新增[Indexer L1环实验记录](INDEXER_L1_RING_EXPERIMENT.md)。v1/v3编译拒绝，
v2虽编译成功但地址错误被审计拦截，未上卡；v4仅score跨循环，完整精度/图/状态/30代通过，
L1中位1008.86μs（同轮动态905.34），未采用、不扩档。现选七档保持。

[query完成粒度模型](indexer_query_completion_model.json)覆盖1..32叶所有权：
仅按request重排而保留整阶段通知，下游仍不能提前。需连同merge/QK消费者及
O-A/O-B整批行限制评估，不把score块数当实际时间。后续边界见[NEXT_STEP](NEXT_STEP.md)。
