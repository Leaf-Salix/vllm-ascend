# CSA 单 SPMD TND：v122 之后的九档优化过程

## 2026-10-22：启动、永久基线与首轮单变量候选

用户已设置新 active goal：实现 `CSA_SPMD_TND_NEXT_10PCT_GOAL_20261022.md`，
过程中参考 cannbot-knowledge 的 ops。此前暂停状态不再适用。
固定口径为八档 S6 + 128K/B16 固定混长，九档等权 candidate/v122 的 L1
中位耗时比几何平均 <= 0.90。当前尚未有本轮设备结果，不宣称新增收益。

本轮开始于 `a584eded`，worktree 为 `csa-spmd-tnd-20261020`；上游仍是
`origin/dsv4-flash-pto-v0.25.1rc1`。没有执行 push。共享SDK清单仍 validated，
release=`csa-shared-20261021-upstream-fixes`，PyPTO e891351e、PTOAS 914863ed、
Simpler 57e49923。没有修改/重建共享SDK。开始时无本会话未完成设备任务。

### 固化与验收入口

实验目录：`tests/pypto_test/csa_spmd_tnd_opt_20261022`。
已核对 v122 全部41文件；相对 a584eded 的默认TND包仅 `policy.py` 与
`indexer_query_stages.py` 不同。用固定完整提交、两个原始文件与逐文件SHA256
永久保存基线；`materialize_reference.py` 可在不依赖旧 runs 的情况下重建。
未把当前默认包或旧BSH当作新分母。

增量harness新增只接受永久v122清单的引用，记录 `kind=frozen_tnd_v122`。
沿用原完整链路计时、精确比较、metadata/service检查；额外启用已有
`score_arena_unused_untouched` 断言，确认UB路径不写GM Score scratch。
旧BSH八档验收器未修改。新增独立九档验收器，逐档五组交替72核单MIX窗口；
固定混长 `[1,2,3,4,5,6,6,5,4,3,2,1,1,3,4,6]`，T=56，不得换分布或填到96。

### cannbot-knowledge 依据与边界

只读使用本地 `cannbot-knowledge/.agents/skills/knowledge-query/SKILL.md`。
首次 discover 返回退出码2：`artifacts/indexes/knowledge.sqlite3` 缺失。
按技能停止索引检索，不重建索引、不扫描全库兜底，不伪造search/ready receipt。
继续阅读此前已明确定位的具体卡片，以下是工程假设依据，不是已完成知识库审计：

- `knowledge/ops/ascendc/guides/synchronization/resident_softmax_state_slot_recurrence_and_reuse_events.md`：
  draft、未定位平台。借鉴同核状态可驻留、复用必须等最后消费者；不直接套其示意代码。
  本轮应用于短历史半叶UB生命周期审计，实际资源和依赖以A3生成代码及设备验证为准。
- `knowledge/ops/ascendc/guides/synchronization/attention_schedule_slots_and_platform_handoffs.md`：
  draft、未定位平台。保留生产/消费身份、READY/ACK与末尾排空；无效数学可裁剪
  不代表协议可以单边跳过。不照搬950的mode4/flag偏移到A3。
- `knowledge/ops/ascendc/apis/simd_api/basic_api/memory_vector_compute/sort_and_merge_isasi/sort32.md`：
  stable，明确包括A3；32元素块内同分优先原较小位置。半叶UB候选不变更排序树、
  叶地址与归并输入顺序；整树精确性仍由现有TopK/score比较验证。

### 首轮候选与CPU检查

`output_balance_off` 仅令 `OUTPUT_WORK_BALANCE=False`，分别测128K/B1、B16；
不调整Q/L、tile、其他阶段或wait。代表档descriptor断言只有这一个constexpr不同。

`short_half_ub96` 仅把短G2组数上界48扩至96，测8K/B32；每个AIV顺序处理4波。
源码helper在每叶内创建并消费私有22×384 FP32半叶，33,792字节，未按总组数
复制4份buffer。完整生成AIV代码中只有一个该Tile声明，映射固定UB地址；重用前
有MTE3→Vector等待。保留原六个排序分支、FIFO与全部尾部检查。此项只能说明
候选具备编译与初步生命周期证据，不能证明性能或所有边界已通过。

三个descriptor均已通过CPU parse与完整codegen：
`runs/compile_output_balance_off_h131072_b1/result.json`、
`runs/compile_output_balance_off_h131072_b16/result.json`、
`runs/compile_short_half_ub96_h8192_b32/result.json`，均 passed=true、device_execution=false。

已提交三项独立设备任务，使用同一冻结v122引用及共享SDK：

| 候选/场景 | task-submit ID |
|---|---|
| output_balance_off / 128K B1 S6 | task_20261022_020842_313460812168 |
| output_balance_off / 128K B16 S6 | task_20261022_020844_313577826313 |
| short_half_ub96 / 8K B32 S6 | task_20261022_020845_313708213322 |

输出位于实验目录 `runs/measure_<候选>_h<历史>_b<B>`；完整命令与启动器hash在
旧实验目录 `csa_spmd_tnd_20261020/jobs.json`。只用 `--status` 观察，不用带
timeout的 `--wait`，避免观察超时取消pending任务。候选尚未采用，等待实测。

CPU门禁：新九档/重建/真实启动器快照27项通过，旧增量与共享SDK23项通过；
定向ruff与git diff --check通过。format.sh ci因环境缺少pre-commit退出，
未执行完整仓库格式CI，不把定向检查写成全量CI通过。三项设备任务均已观察为running。

### 首轮设备结果与取舍

上述三项均 completed(exit=0)，两侧同源SDK审计、严格数值/TopK/score、状态、
图、保护区与代际检查通过。Native原有precision_exact=false继续保留。
全部五窗及源清单见新实验目录 `control_results.json`，不将三个局部对照拼成九档结果。

| 候选/场景 | v122中位μs | 候选中位μs | 耗时变化 | 配对快窗口 | 取舍 |
|---|---:|---:|---:|---:|---|
| output_balance_off / 128K B1 | 372.26 | 378.94 | +1.7944% | 0/5 | 拒绝，保留原输出分工 |
| output_balance_off / 128K B16 | 803.18 | 801.34 | -0.2291% | 4/5 | 微小正向证据，暂不采用 |
| short_half_ub96 / 8K B32 | 933.52 | 940.12 | +0.7070% | 1/5 | 拒绝，UB驻留不等于整链变快 |

8K/B32候选同时通过score_arena保持37断言，说明设备确实走到了UB路径。
不能将回退解释成开关没有生效。没有修改默认算子，没有重复相同候选等噪声转正。

### 混长：裁剪与轮换分开建候选

已读另一会话 `prepare_score_live_half.py`、`prepare_score_leaf_rotation.py`
及live-half实测负例。仅借机制，未借用其不同混长fixture的性能数字。
本轮 `live_half` 对实际live<=3的长G6使用固定M192，只删除全零后半QK/WS；
保留六行FIFO、两个AIV ACK、全部READY/消费/drain调用、原live>=4数学AST。
动态valid_shape不改变物理Acc行距，避免已知A3紧凑L0C传输问题。

`live_half_rotate` 在此基础上，仅对少于24个G6组执行
`group=(ordinal+leaf)%groups`；G2和>=24组保持原分工。单SPMD的根消费者同时
用`ordinal=(group+groups-leaf%groups)%groups`反推真实owner；等待原完整阶段
完成信号，不把FIFO ACK作为根完成。逻辑叶/根地址与归并顺序不变。
CPU执行实际生成的映射/等待helper，覆盖空组、1..64组代表边界、1..32叶、
G2/G6和双AIV，验证双射和每一个消费者真实写者；3项检查通过。

固定混长的源码成本模型：最忙AIC的M192半段工作单位44→39，两个AIV lane
各自最大有效query×Score步数(66,55)→(66,36)。这不是耗时预测，也表明
lane0最长工作仍未降低，不能凭AIC平均节省就认定整链提速。
两个候选的完整CPU代码生成均通过，源码清单与借鉴来源hash记录在
`mixed_candidates.json`，各自直接与冻结v122配对五窗：

- `task_20261022_021508_34186829200`：live_half，128K/B16固定混长T56。
- `task_20261022_021510_342018110895`：live_half_rotate，同一固定混长。

两者尚在运行，不能宣布采用或扩大到八档S6通过。

### 当前版本的定向阶段证据

已准备独立 `phase_harness`，保留shared SDK原生150tick等待和通知，只在每核
每阶段独占512B槽的保留字16..19记录时戳及累计GM wait。原信号行不改；
正式L1 harness不改。诊断允许这四个保留字，其余padding/数值/图检查保留。
阶段区间包含插桩扰动与FFTS/实际流水等待，不将减去GM wait的余数叫纯计算。
新九档门禁显式拒绝带phase_clock或diagnostic_phase_clock标记的结果。
CPU对真实当前SDK整份C++验证：移除插桩后逐字恢复全部原代码；拒绝重复
插桩和非150等待。3项定向检查通过，九档27项仍通过，新增脚本ruff通过。

`task_20261022_021917_359785823204` 已提交128K/B16/S6的v122对自身阶段诊断；
不计入任何性能收益，待核对编译实际插桩、完整数据与阶段端点后再选供数/排序/
等待方向。未用旧v25不同SDK的插桩时间直接判定当前瓶颈。

### 第二轮结果：保留局部正向证据，九档目标未达成

混长两项均 completed(exit=0)，完整严格回归/图/保护/代际通过：

| 候选/场景 | v122中位μs | 候选中位μs | 耗时变化 | 配对快窗口 |
|---|---:|---:|---:|---:|
| live_half / 128K B16 混长T56 | 765.98 | 745.48 | -2.6763% | 4/5 |
| live_half_rotate / 同固定混长 | 734.24 | 725.30 | -1.2176% | 5/5 |

两项独立任务/卡，只与各自同轮v122相除；不能比较745.48与725.30来断言
轮换更好，也不能叠加这两项收益。全部窗口及源清单见 `mixed_results.json`。
优先继续单独live_half的回归，轮换保留候选，未正式采用。

live_half追加S6与动态检查：

- `task_20261022_022636_391372328412`，128K/B8/S6：611.96→611.62μs，
  -0.0556%，3/5配对快，基本持平。
- `task_20261022_022637_391508821796`，128K/B16/S6：795.34→785.52μs，
  -1.2347%，5/5配对快。S6都走完整M384分支，不能把这个结果解释为省掉了
  无效query；加入动态分支后的代码生成/布局及波动均需与真实裁剪收益区分。
- `task_20261022_022508_384730226445`，T41/R12：同址S1～S6、混261/123456
  共8场景及4步连续状态全部通过，12次同图重放；见 `live_half_metadata_t41_summary.json`。

两个S6任务也均 completed(exit=0)，全五窗及Native已有差异单列于
`incremental_progress_results.json`。它们不覆盖尚未测的其它档位或真实服务。

### 阶段诊断与排序合批候选

阶段诊断 `task_20261022_021917_359785823204` completed(exit=0)，每侧1619条
有效发布端点，均为同一个v122源码。数据见 `phase_v122_b16_s6_summary.json`。
两次非计时流程的热身/缓存状态与分配不同，整体端点835.60/785.50μs不能当成
候选收益。较稳定的局部观察是Score AIV区间中位192.38/188.59μs，AIC扣除
累计GM wait后的区间仍约200μs量级；包含FFTS等待与流水，不能称纯计算。
Attention后O_A按列分工仍有约三波，每波约26μs。原归因候选优先研究Score
重复排序发射，未原样重试N128、改数学归约或Score/Attention跨波交错负例。

`batched_sort3_v2`基于冻结v122独立制作，不叠加live_half。仅G6、关闭短前缀，
且本AIV三个query均有完整2048有效分数时合批Sort32及三级MRGSort。先逐query
检查真实Q/L/positions；尾段、失活/陈旧位置、短前缀与其他组型走原完整逐query
分支。每个query仍有各自索引与根，末级合并顺序保持new-before-old。

依据为已读cannbot Sort32卡及当前固定PTO-ISA
`327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d`的
`include/pto/npu/a2a3/TMrgSort.hpp`：format1每次归并四个等长列表，允许重复。
6144个分数的Sort32重复192次；12288个FP32 pair元素在blockLen64/256/1024
时重复48/12/3次，边界恰好保持三个query的独立2048段，没有跨query排序。
底层接口事实不代替设备同分检查。

首版CPU被拒：INT32的batch_sort_valid重新赋普通字面量1，后者推导为INDEX。
保留 `runs/compile_batched_sort3_b16/result.json` 与旧候选清单；v2显式INT32赋值，
并用query_group_size表达未启用分支的物理形状。v2完整CPU代码生成通过，
AllocateMemoryAddr的Vec声明地址上界180224B，低于当前188416B预算；
证据 `batched_sort3_v2_resources.json`，未修改共享SDK。

`task_20261022_023226_417823515723` 的128K/B16/S6完整五窗已 completed(exit=0)：
v122 802.56→候选786.78μs，-1.9662%，5/5配对快；严格输出/TopK/score、cache/
state、图、保护区及代际均通过。保留为正向候选，不能与live_half收益直接相加。
已追加以下受影响边界，尚未收齐：

- `task_20261022_024003_29860510152`：B16/S6历史Indexer key置零制造大量同分，
  correctness-only，不混入性能。
- `task_20261022_024004_29935615774`：T65/R16，S1～S6/混长、padding及连续状态矩阵。
- `runs/compile_batched_sort3_v2_b28`：B28完整CPU编译，收取后再决定设备代表档。

固定基线的两个覆盖文件改用`.py.source`扩展名，保留原始字节及41文件清单，
避免格式器修复历史源码的未使用变量而悄悄改变分母。重建与九档门禁27项再次
通过，当前实验脚本ruff通过。没有改默认生产算子，没有push。当前goal保持active；
九档最终几何平均尚未形成，10%目标尚未达成，恢复入口为实验目录 `current_state.json`。

B28完整CPU编译已通过，追加成对五窗任务`task_20261022_024155_4403114082`，
输出`runs/measure_batched_sort3_v2_b28`；不重复已完成的B16测量。

Git状态补记：共享origin引用在02:06:49被外部push更新到059dab4a，导致
ahead/behind计数变为943/3；本会话仅新增三次本地阶段提交，没有fetch、push、
rebase或改写其他会话。保留当前固定基线和工作树，不能把943理解为本轮新增提交数。

## 2026-10-22：继续控制优化、否决O_A重分块、组合回归

上一goal轮属于实际进展：新增三次本地阶段提交、已完成代表档成对测量，
未缩小九档目标。本轮开始核对goal仍active、工作树干净、公共SDK仍validated。
最初16卡由CI占用；保留原三个pending handle，没有重启或取消。随后三项均
completed(exit=0)：排序合批的B16大量同分、T65/R16八项metadata加四步连续状态
全部通过；B28中位1163.44→1154.60μs，-0.7598%，五窗均快。
完整证据 `batched_sort_validation_results.json`；没有把同分fixture计入性能。

### B1空发布与O_A行tile：CPU模型和实际结果分开

`b1_idle_publish`只在既有BALANCED/G2域裁掉INPUT/QR/ROPE的空角色发布。
INPUT无业务的metadata planner仍执行fence/cacheinvalid/fence，读取当轮Q/L、
positions和页表；其他空角色保留其后原QA获取。所有72角色begin_invocation、
代际推进、活跃发布、FIFO ACK及数学不变。它不同于旧v22直接删INPUT发布的
准备版，明确保留planner获取。关闭的G6/B16完整C++仅去独立pto调试注释后，
与已测v122相同，见 `b1_idle_closed_g6_audit.json`；此证据不扩大成所有关闭档验收。

`oa_rows48/64`改变实际O_A行tile及Cube/AIV两侧的组任务数/等待owner，
保留N256、原K256顺序与有效行；只是独立原型，未作为通用服务策略。
T144/M48的最忙核M16取整行量由384降至192，T168/192的M64降至256，
但总权重tile读取次数由64增加至96。这里是源码工作量模型，不是HBM字节
或耗时预测。与另一会话E48仅轮换满/尾行块的机制不同。

39项CPU合同执行实际Scalar分工，检查空角色metadata获取、每个真实输出
行/列唯一覆盖、K序列和组等待writer集合；全部通过。五个descriptor完整CPU
代码生成通过。设备成对五窗均completed(exit=0)且严格回归/图/状态通过：

| 候选/场景 | v122中位μs | 候选中位μs | 耗时变化 | 配对快窗口 | 决策 |
|---|---:|---:|---:|---:|---|
| B1空发布 / 128K B1 | 367.54 | 361.26 | -1.7087% | 5/5 | 纳入待验证组合 |
| O_A M48 / 128K B24 | 1018.50 | 1020.84 | +0.2297% | 0/5 | 拒绝 |
| O_A M64 / 128K B28 | 1165.02 | 1169.98 | +0.4257% | 2/5 | 拒绝 |
| O_A M64 / 8K B32 | 929.00 | 939.08 | +1.0850% | 0/5 | 拒绝 |

四项task依次为`task_20261022_025444_123873330198`、
`task_20261022_025446_124036225902`、`task_20261022_025447_124188114719`、
`task_20261022_025448_124304110989`。源/模型/五窗见
`control_tiling_candidates.json`、`control_tiling_results.json`。没有因为纸面
均衡而采用回退，也未原样重测相同候选等待噪声转正。

### 继续参考ops资料和其它会话

沿此前明确定位路径阅读cannbot的
`knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/memory_access/l2_cache_mode_config.md`。
卡片为draft、frontmatter未定位平台，正文列出A3；仅用其解释重复读取与L2
替换之间的权衡，不把更多TLOAD等同于更多HBM事务。本轮不改变L2配置。
当前PTOAS914863ed的TLoad.cpp仍要求L2Bypass且存在offset才调整源地址；
虽有get_l2_cache_offset op，本轮未建立PyPTO到实际runtime offset的有效通路。
不运行仅改标签却没有实际指令变化的设备实验，也不硬编码地址偏移。
知识索引缺失状态保持；没有伪造检索receipt或自动重建知识仓。

只读复核另一会话新提交7d8b40d0/d2f35d5c及query_capacity_gate_results：
MPMD的整DAG分档/前段核数联动尚非当前SPMD可直接照搬的已采用收益。
本会话v122诊断中query Hadamard末端177.84μs早于Main输出198.64和query量化
207.92μs，不能凭增加Hadamard核数就断言关键链更短。未机械移植其核数参数。

### combined_v1：组合实测不支持直接叠加

独立生成`combined_v1`，只组合B1空发布、live_half和batched_sort3_v2。
两个Score变换操作不同函数，交换变换顺序得到逐字相同源码；仍需实际测量。
B1、B16、混长、短B32四个完整CPU descriptor均通过。新成对结果：

| 场景 | v122中位μs | combined_v1中位μs | 耗时变化 | 配对快窗口 |
|---|---:|---:|---:|---:|
| 128K/B1/S6 | 374.44 | 370.10 | -1.1591% | 3/5 |
| 128K/B16/S6 | 789.32 | 781.40 | -1.0034% | 3/5 |
| 128K/B16固定混长T56 | 744.50 | 743.60 | -0.1209% | 5/5 |

任务为`task_20261022_031228_215821124043`、`task_20261022_031229_215896931619`、
`task_20261022_031231_216043424472`，全部completed(exit=0)，五窗完整保留于
`combined_v1_representative_results.json`。B1和混长的CPU完整C++分别与对应单项
相同（仅去独立pto调试注释）；不能把本轮较小收益直接归为组合相互抵消，也不
选回此前更大窗口替代当前组合结果。这个源码审计不是完整跨任务复用证书。

组合功能追加结果均completed(exit=0)：

- `task_20261022_031232_216162928598`：B1同址图S1～S6六场景通过；该工具不含
  四步连续状态，不把它写成12次回放。
- `task_20261022_031233_216343514801`：T65/R16八场景加四步连续状态通过，12次同图重放。
- `task_20261022_031234_216428831819`：真实模型准备钩子、单层Torch custom-op、
  ACLGraphWrapper通过；反转请求/物理页后同址图重放且没有Python重新提交；
  空输入处理、旧图拒绝和短历史fallback无SPMD提交均通过。范围不是整模型端到端。
  详见`combined_v1_service_summary.json`。Native原有差异仍为precision_exact=false。

归档工具首版假设所有metadata矩阵都有same_graph_replays字段，B1六场景报告
没有此字段，因此归档退出；已改为保留未报告值，不伪造次数。只重新读取原始
结果，没有重新执行设备测试。生产算子仍未改动，综合10%仍未达标。

### 下一候选：Q_B实际列生产者等待

源码审计发现Q_B之后每个AIV仍等待全24个角色，包含辅助分支空发布者，以及
本AIV没有读取的列块。新的`query_exact_waits`只改这一条依赖边，不叠加组合。
从原反量化函数原样抽取head分组选择，展开helper后整函数数学AST与原版相同；
按本AIV的固定head组计算所读N256/N512列，再反推20个Q_B逻辑worker对应的
真实物理编号。48可整除所有可能的head组数，因此跨token波次仍读相同列组。
相邻列仅等待不同owner一次，保持全部发布/代际/FIFO及有效消费者的缓存获取。
未工作AIV不读q_proj，后继原QUERY发布仍保留业务缓存维护。

传递现有PREFIX_BRANCH_GROUPS/QB_WIDE_N constexpr，未增加Tensor ABI或Host
回读。物理反向映射显式保持取模输入非负，覆盖0<R<24；非协同路径保持原24核
列分工。14项CPU验证从生产者正向分配枚举实际读列，覆盖N256/N512、不同R、
T边界及空角色；B8、B28、混长完整CPU代码生成通过。该改动不同于旧v38/v80
只裁掉无工作消费者的等待，本轮没有重复旧QA/weights/coeff等待裁剪。

已提交三个独立代表档，尚待收取，不预记收益：

- `task_20261022_033052_292215211155`：128K/B8/S6。
- `task_20261022_033053_292310924414`：128K/B28/S6。
- `task_20261022_033054_29245487447`：128K/B16固定混长。

当前优先按这些结果决定下一次组合；目标仍为完整九档等权G<=0.90。
尚无最终九档组合报告，不将三个代表档的平均或单项历史收益代替验收。

## 2026-10-22：收取精确等待结果，尝试B1排序重叠

再次核对goal为active，验收仍是八档S6加固定128K/B16混长九档等权G<=0.90。
上一轮三个Q_B实际列writer等待任务均completed(exit=0)，完整精确/状态/图通过：

| 场景 | v122中位μs | 候选中位μs | 耗时变化 | 配对快窗口 |
|---|---:|---:|---:|---:|
| 128K/B8/S6 | 595.90 | 593.14 | -0.4632% | 4/5 |
| 128K/B28/S6 | 1153.08 | 1149.98 | -0.2688% | 3/5 |
| 128K/B16固定混长 | 754.34 | 753.00 | -0.1776% | 4/5 |

完整五窗及冻结源/SDK指纹见`query_exact_wait_results.json`。只保留小幅正向证据，
尚未采用或与组合叠加。Native原有差异仍单列，未改精度阈值。

重新检查静态容量历史：旧v28已经在B16/B28完整通过，约0.4%/1.0%改善，且
增加容量签名编译数；当前没有新机制证明值得原样重试，未再次上卡。

### B1的4096半叶排序：保持叶边界与同分树

`b1_sort_stream`只对BALANCED/G2、真实lane、四个1024分数块、完整可见4096
半叶生效。原4096分支的Sort32和block_len=64/256/1024只形成两个独立2048根；
候选在收到前两个1024块后先排序前根，后两个块到达后再排序后根，最后仍按
right→left归并。原六个有效长度分支完整保留作为回退，没有改变逻辑叶划分。
全部READY/ACK、双槽排空、scale和Score数学不变；不是旧B1改G6/24叶方案。

独立`b1_sort_unused_fills`只把G2不使用的single_roots/single_segments改为
create，在G6保留原full初始化。原B1生成物可见这两次TEXPANDS，不能假设DCE
已经删除。两项均从v122独立生成，避免把无用填充收益归给排序重叠。

参考此前定位的cannbot Sort32稳定API卡（支持A3）：32项内部排序、不同repeat
互不归并，同分保留输入先后。CPU模型执行实际候选表达式及原4096分支，覆盖
全同分、随机同分、正反序、负值、无穷、非零全局索引；另外检查激活条件、
原FIFO调用和非目标函数不变。14项通过。首轮仅测试中的目标函数名称写错，
13项已通过；更正为实际indexer_score_vector_lane_stage后全部通过。
模型用于比较操作树，不能替代设备指令的同分精确检查。

两候选完整CPU编译均通过，无SDK修改。设备任务已提交，未预记结果：

- `task_20261022_041606_4679556748`：stream，128K/B1/S6，五窗完整L1。
- `task_20261022_041612_4726514515`：unused_fills，同一业务档五窗。
- `task_20261022_041614_47370630607`：stream，H139264/S5/Tcap6，全同分正确性。

只读参考另一会话42b14d97：MPMD两条前段局部MIX确有并发核集合重叠导致串行
的窗口，但其ready/gated派发修改不适用于当前单SPMD。当前前段已经包含四query
Hadamard，不重复计算迁移收益。未拷贝其不同fixture的绝对性能作本轮基线。
默认生产算子尚未修改，本轮10%目标仍未达成，未push。

### B1结果与fixture失败纠正

两项完整五窗任务均completed(exit=0)，严格数值/状态/图通过，结果为：

| 候选 | v122中位μs | 候选中位μs | 耗时变化 | 配对快窗口 | 决策 |
|---|---:|---:|---:|---:|---|
| 4096半叶提前排序 | 367.20 | 375.36 | +2.2222% | 1/5 | 拒绝 |
| 去掉G2无用填充 | 373.42 | 363.90 | -2.5494% | 4/5 | 保留供组合 |

见`b1_sort_performance_results.json`。提前排序增加了分支、两次排序启动及循环
携带根的UB搬运；生成物可见根的独立地址与回边TMOV。这是解释方向，不把静态
指令数量认定为已测瓶颈。没有以“理论重叠”采用回退版。无用填充版生成物已无
single_roots/single_segments，也不把完整收益全部归成12KiB写量的理论耗时。

原S5同分任务`041614_47370630607`在fixture初始化失败，未执行候选：S5/Tcap6/R1
缺少吸收多余padding token的dummy请求槽。这不是算法数值失败或runtime加载故障；
后续“extension未加载”来自清理审计阶段。保留原完整错误，改为真实T5/R1后提交
`task_20261022_041846_57849415001`，仍是H139264/S5同分正确性，不计入性能。
没有增加R改变B1分档，也没有删除原失败记录。

### 复用既有页布局证明，减少AIC页表读取

`key_page_proof`只在原SCALE_PAGE_COALESCING且split-G6路径中读取已有
META的(first_page,direction)，把每个panel内的页号查表改为first+direction*page。
原逻辑页clamp、逻辑/物理容量检查、两种64B偏移Native key view、搬运次序以及
全部QK/WS数学不变。direction=0仍逐页查原表。没有扩张证明启用域、追加Host
metadata、修改页表或增加证明工作；复用前段原有META获取与业务发布协议。

22项CPU检查执行实际地址表达式，覆盖正/逆序、碎片、共享页、页尾clamp、
1..1025页及关闭域不读取陈旧证明。AIC搬运/数学/同步调用AST全部保持原样。
B28启用和B8关闭完整CPU编译通过。B28五窗任务
`task_20261022_042002_63203019442`正在运行，尚无性能结论；B8只是关闭域编译
检查，不为未生效路径安排设备“提速”实验。

### 前段顺序候选：Main AIV后移至Indexer输入发布之后

当前阶段端点显示Main输出先于Indexer query量化/系数。新的`main_vector_late`
把COOPERATIVE_PREFIX下原main_compressor_vector_stage完整调用移至IDX_SCALE
发布之后、主Q反量化之前，使AIC有机会在主压缩Vector处理期间先做Score。
原pool→等全部读者→state commit→norm/cache整段及参数逐字保留；非协同路径
保持原次序。中间Indexer链不读取main cache/state，Attention仍等MAIN_OUTPUT。
所有调用AST多重集合不变，未减少任何通知、代际或FIFO边。

这是根据本轮阶段证据建立的单SPMD顺序实验，不是直接移植MPMD的ready选核。
潜在代价是AIV较晚消费Score产生双槽反压，必须完整成对测量。B16/S6和固定
混长正在完整CPU编译，未预记设备通过或收益；生产包未采用。

追加完成状态：S5真实T5同分任务`041846_57849415001`已completed(exit=0)，
严格输出/TopK/score/状态/图通过，详见`b1_sort_tie_results.json`。该功能证据
不撤销提前排序版的性能拒绝结论。Main后移的两个CPU完整编译均通过，随后提交
`task_20261022_042503_85147929684`（128K/B16/S6）和
`task_20261022_042512_85871224786`（固定T56混长），均五窗、冻结v122成对。
保留这两个handle继续收取；没有重复已完成的代表档或变更九档权重。
