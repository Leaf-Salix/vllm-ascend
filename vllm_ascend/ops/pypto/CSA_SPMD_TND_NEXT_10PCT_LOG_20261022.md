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

## 2026-10-22：前段分链的单SPMD原型与B1组合

上一goal轮有实际进展：两个本地阶段提交、B1正负设备证据，以及三个明确运行
的后续任务。本轮重新核对goal active、工作树干净、公共SDK仍validated；没有
重启任何在途任务。已收齐三项，均completed(exit=0)，严格输出/状态/图通过：

| 候选/场景 | v122中位μs | 候选中位μs | 耗时变化 | 配对快窗口 | 决策 |
|---|---:|---:|---:|---:|---|
| 页证明复用 / 128K B28 | 1161.00 | 1164.34 | +0.2877% | 2/5 | 拒绝 |
| Main AIV后移 / 128K B16 S6 | 811.74 | 803.00 | -1.0767% | 4/5 | 保留候选 |
| Main AIV后移 / 固定T56混长 | 753.38 | 746.70 | -0.8867% | 4/5 | 保留候选 |

证据为`key_page_proof_results.json`、`main_vector_late_results.json`。没有把少查
页表当作已获收益，也没有把两项Main后移结果算作最终九档达标。

### front_partition_i8_v2：Indexer与主QKV使用不重叠核组

参考另一会话MPMD的两条前段链，但本原型仍是原72核单MIX、一个SPMD提交，
没有加入动态task调度器或第二个incore。共享HC/QA/QR/RoPE之后，在原G6且
COOPERATIVE_PREFIX域内划分：AIC0..7及AIV0..15负责Indexer压缩、query、
Hadamard、量化、系数和scale；AIC8..23及AIV16..47负责主压缩、Q_B和反量化。
KV投影仍由全部24个逻辑worker完整分摊，循环置换到以主分支为先；后继
Score、Top-K、Attention、输出仍使用原24组分工和数学。

只克隆前段调度wrapper以改变WORKERS及真实producer编号，数学helper保持
原文件。主反量化仍沿原48波对齐规则选head tile，32个实际worker只改变条带，
不改RMS归约、尾部recip(sqrt)、量化或K累计顺序。所有Tensor签名、GM工作区、
代际和原阶段发布保留；空角色发布先于本分支业务，真实发布仍在写回后。

关键依赖逐项处理：

- 主分支不再经过Indexer wrapper，显式补回PHASE_META获取，不能假定此前
  的QR/RoPE或主投影已经获取本轮compact offsets/counts。
- 主压缩pool等待AIC8..23的真实投影者；state commit等AIV16..47中全部实际
  pool读者，保留共享prefix读完才能覆盖state的生命周期。
- Attention等待主cache的物理角色为24+16起的真实写者，不能被Indexer角色
  提前发出的空MAIN_OUTPUT骗成就绪。
- scale计划仍有48个逻辑页writer。16个Indexer AIV分别领取owner、owner+16、
  owner+32，整页64B RMW始终归一个物理核；不改变META中的页owner链表。
- 原Score的READY/ACK、槽重用及末尾排空完全保留，不通过跳过空lane通信提速。

首版完整CPU编译在NZ非负性证明失败：`worker-8`虽处于正确分支，当前编译器
不把差值视为非负NZ列偏移。未改共享SDK或权重布局；v2仅把Q_B logical worker
改为physical_worker % 16，后16个连续物理编号形成完整双射，消费者仍等待全部
真实16个写者。原失败源及`compile_front_partition_i8_b16/result.json`保留。

CPU调度检查执行实际两条schedule及克隆wrapper，以数学stub记录事件：22项
覆盖4/8/12组方案、T24/41/56/96/168/192/385、请求4..65，检查全部阶段恰好
发布一次、等待图无环、state读者屏障、主META获取、worker唯一覆盖及48页owner。
首版测试漏识别q_b_stage的NZ alias导致前置断言失败；补齐实际_q_b_nz_stage
签名后全部通过，没有放宽依赖断言。该模型不等价于设备数值或流水性能验证。

v2的B16/S6和固定混长完整CPU代码生成通过。设备五窗对照任务：

- `task_20261022_043809_13687342079`：128K/B16/S6。
- `task_20261022_043810_136987923117`：固定128K/B16/T56混长。

这两项尚未收取最终结果，不预记性能通过。默认算子未修改。

### combined_v2：单独保留B1无用填充收益

在已验证combined_v1上加入B1去掉single_roots/single_segments无用填充，
与live_half/batched_sort变换交换顺序后源码逐字一致。没有加入负性能的提前
排序或页表复用，也没有把Main后移及分链实验混入。B1完整CPU编译通过，提交：

- `task_20261022_044119_150603731813`：B1/S6五窗完整L1。
- `task_20261022_044120_15073038393`：B1同址图S1～S6六场景。

继续参考已明确定位的cannbot `resident_softmax_state_slot_recurrence_and_reuse_events`
卡片；它为draft且平台未定位，只用于检查最后消费者、槽复用和事件方向。
没有把增加槽数或声明常驻资源等同于必然10%收益，也没有因知识索引缺失自行
重建知识仓。九档等权分母和目标保持不变，未push。

补充13项CPU合同已通过：直接执行Attention的MAIN_OUTPUT等待表达式，确认只等
主分支的真实物理写者；关闭分链guard后，完整Cube/Vector schedule AST与原版
完全一致。合计35项CPU合同，不等同于最终设备边界验收。组合生成器新参数的
局部循环变量曾覆盖报告名，已把变量分开并将本轮清单改回combined_v2.json；
候选目录、源码字节、已编译及已提交快照未变化，没有重复提交设备任务。

### 前段分链与B1组合收取；按实际32个AIV修正head分组

四个任务全部completed(exit=0)，严格数值、状态、保护区及图检查通过：

| 候选/场景 | v122中位μs | 候选中位μs | 耗时变化 | 配对快窗口 |
|---|---:|---:|---:|---:|
| front_partition_i8_v2 / B16 S6 | 801.20 | 810.78 | +1.1957% | 0/5 |
| front_partition_i8_v2 / 固定混长 | 747.72 | 739.36 | -1.1181% | 5/5 |
| combined_v2 / B1 S6 | 370.72 | 368.68 | -0.5503% | 2/5 |

原五窗分别归档在`front_partition_i8_v2_results.json`、`combined_v2_b1_results.json`。
分链v2不作为通用S6改进采用，混长正向证据保留；不把此轮混长候选绝对值与
其它轮基线拼接。B1组合收益很小且窗口方向混合，不能相加此前空发布/填充的
单项百分比。B1同址图S1～S6六项全部通过，仍不把该工具写成有四步连续状态。

发现分链引入的具体负载问题：主Query AIV由48个变为32个，但原helper继续按
固定Q_DEQUANT_WORKERS=48选择head分组。T96的12个M8块会选16 head，48份工作
分给32核后，16核做16个head、另16核做32个head（每head含8行）。v3克隆该
helper，仅把选择器的48改为实际WORKERS；T96改选8 head，96份工作分给32核，
每核均24 head。总量768个head块不变，这只是负载模型，不是25%实测提速。

每个head的完整计算循环AST与原版相同，FP32两次scale乘法、512列RMS顺序、
rsqrt及尾行recip(sqrt)、BF16边界均未修改。T56仍选择4 head；含尾行T65也仍为4。
原helper不改，只有分链启用时调用新helper，关闭域继续保留原完整schedule。
新增五项CPU检查执行实际选择器，覆盖24/32/40/48个worker和T0..385边界，验证
每个真实(token,head)唯一覆盖及尾行分组。全部通过，v3 B16完整编译通过。

已提交`task_20261022_045018_19010782266`（front_partition_i8_v3、B16/S6五窗），
沿用冻结v122、原工作区方法与公共SDK，尚未预记结果。当前不存在最终九档
G<=0.90证据；生产算子未改、未push，goal继续active。

## 2026-10-22：停止分链扩扫，建立M64/N256 Score独立试验

上一goal轮完成两次本地阶段提交及多项正负对照，属于实际进展。本轮核对原
v3任务仍运行，继续原handle；随后completed(exit=0)，严格回归通过。B16/S6
799.34→809.12μs，+1.2235%，仅1/5窗口更快，见`front_partition_i8_v3_results.json`。
按32个实际AIV均衡head工作量没有转化为完整收益。停止分链这条路线的参数
扩扫；保留v2混长正向和Main后移的小幅收益，但不作为全S6采用结论。

### 新Score资源组织与历史负例的区别

先重新阅读旧v39/v60/v62记录：N128/M192单96KiB Acc与K128 WS回退7.21%；
N128三M128、两个64KiB Acc复用回退3.45%；仅预取到L1的变体回退8.90%。
不重复这些路径。新的`score_query_pipeline`仍独立基于v122，只作用于原
长G6/SPLIT_SCORE_ROWS域：

- QK按六个独立query分别做M64/N256/K128，使用一个64KiB INT32 Acc。
- 加权归约为六段K64，保持全局head顺序0..383，使用独立16KiB FP32 Acc；
  每个query的64个非零系数仍完整处于一段。不能据此预先声称设备逐bit相同。
- 当前query的QK/FIX与前一query的WS交错，最后补WS5；每256列仅写一次六行。
- Left保留六个8KiB query与六个2KiB系数片，共60KiB。Right为32KiB Key及
  32KiB WS；只把下一Key预取到L1，不同时保留第三块Right。
- 每1024列由原32次QK/32次WS变为24次QK/24次WS；这只是操作数模型，
  不能代替完整耗时。原1024列双槽、逻辑叶、Native页检查、FP16 FIX比例/ReLU、
  AIV Scale/Top-K、同分树和READY/ACK/末尾排空均保留。

### 必须确认实际启用；资源与传递事件审计

首次CPU编译误用了固定v122描述符，仍是QK_PANEL=64/TO_L0=True，所以新wrapper
没有选中N256 helper。虽编译成功，但资源审计未找到query_qk_acc，不能作为新
候选编译证据。已从实际候选policy重建描述符，先确认相同输入的v122 policy
逐项等于原描述符，再只更新QK_PANEL=256、TO_L0=False；所有Tensor buffer
形状/dtype保持相同。文件`descriptors/score_query_pipeline_b16.json`。

实际启用版`compile_score_query_pipeline_active_b16`完整CPU编译通过。
`score_query_pipeline_resources.json`从AllocateMemoryAddr和最终C++确认：
QK Acc 0..65536、WS Acc 65536..81920互不重叠，Left末端61440字节；
四个panel的Key各32KiB，WS0..4 Right不覆盖仍被六次QK消费的Key。
最后WS5可以复用已退役Key地址，下一panel的Key也只能在WS5读完后覆盖。

首版审计要求退役Key必须有直接M→MTE1事件，因此停止。逐段查看生成C++确认
实际存在QK5 M→FIX、其FIX→MTE1两个完整set/wait链：WS5的Right正从QK5的FIX
结果加载，这个传递依赖已保证最后一次Key读取完成。审计改为明确验证这两段
事件，而非忽略检查；其它QK Acc和Right复用仍检查各自set/wait。最终审计通过，
并验证旧的未启用N64编译产物会被拒绝。没有修改生成C++或公共SDK。

四项CPU合同检查通过：原分档范围、全局K/head覆盖、页/FIFO/Vector数学不变、
实际policy与CPU描述符一致。首版枚举包含非法T>R×query上界，测试已改为要求
新旧policy都拒绝该输入，没有放宽生产验证。设备对照仅提交代表B16：
`task_20261022_050510_249159220949`，五窗完整L1和严格同输入比较，尚待最终结果。

### 不重复另一会话的multi-ND负例

只读核对当前固定ISA的`pto/common/arch/memory/tload_common.hpp`，确有三维
ND2NZ的ndNum通路，可表示4160字节页stride，但多矩阵stride必须正向。
当前fixture为倒序物理页，不能只把二维Mat reshape就当成逻辑顺序正确。
继续查另一会话记录，找到2026-10-20的Key多ND完整试验：54项UT、12项probe
通过，B28全链1163.64→1310.50μs、0/5快；Mat父双槽assemble不支持，最终
直接进两个Right槽，改变了供数生命周期。此前负例及其数据路径限制保留，
本轮没有重复上卡、换成正序fixture或编辑共享SDK。

当前生产默认未变，最终九档G仍未形成；完整10%目标继续active，未push。

### M64/N256终态与下一步诊断范围

`task_20261022_050510_249159220949`已completed(exit=0)：严格输出/Top-K/score、
cache/state、保护、图和代际均通过；B16完整五窗798.16→848.84μs，+6.3496%，
0/5快。见`score_query_pipeline_results.json`。拒绝采用，不扩其它档，也不将
更少MMA或合法的Acc分配当作性能收益。当前无待收取设备任务。

开始前已复核历史Scalar READY/ACK探针与phase-clock局限：它们包含异步队列
和探针成本，不能把包围intrinsic的Scalar时钟差直接称为引擎阻塞时间。本轮
没有重复已完成的全核阶段端点测量。下一步优先补可解释的生产路径诊断，
再决定新的代码方向，而不是继续扫宽N或前段核数。

已阅读本地`pypto-lib/.claude/skills/incore-profiling/SKILL.md`及其canonical
`docs/debug-and-tune/incore-simulator-profiling.md`，并按技能执行只读函数发现。
构建路径须传绝对路径（相对路径会由脚本按pypto-lib根解析）；正确发现唯一
`dynamic_csa_full`及其配对.cpp/.pto。尚未运行仿真或生成任何trace。

完整CSA的72角色常驻协作不能直接装进该技能的单核合成case。后续若采用
仿真，应单独建立明确标识的Score生产路径诊断，复用原数学和页访问，手工
核对group/leaf/Q/L/页表、系数和Scalar维度，让CUBE/MTE2/FIX均有真实工作。
可评估独立生产者不受FIFO反压时的成本；这种隔离案例不用于解释完整72角色
调度，更不能进入九档性能汇总。不得将原完整kernel预填就绪标志后冒充真实
全核执行，也不得改变正式源、基线或用合成输入的周期替代实机L1。

只读确认正式CANN 9.2.0-beta.2目录已有`tools/msopprof/bin/msopprof`与
`tools/profiler/bin/msprof`；没有使用--msopprof/--auto-msopprof，没有复制worker
或注入库、修改共享CANN/SDK。具体仿真case、输入绑定和有效trace尚待建立。


## 2026-10-22：Score单生产者仿真已跑通，宽N负例得到独立指令证据

独立诊断只复制原Score helper并移除三个FIFO调用及空guard/drain，数学、
页读取、query尾行处理保持。入口一AIC、一G6、一逻辑叶、六个1024列step，
不伪造72角色协作，也不预填完整CSA就绪标志。12个buffer分别按真实维度
分配；从packed wrapper提取14个动态轴及block_idx/block_num，实际绑定0/1。
否则默认block_idx=1会得到空trace。Native页表为192..1倒序，page字节4160，
INT8输入/Key为1，系数仅本query的64个head为1；输出12288个FP32值全部应为8。
两个只读Key别名为合成独立buffer，逻辑字节一致，不声称真实cache别名性能。

初版pl.cluster(1)不合法，保留失败源后改为直接调用单incore入口。两个v2
完整CPU生成通过。首轮task_20261022_054417_411548524006和
054418_411706318223均exit1：技能旧兼容前导与CANN9.2内置FP8/FP4类型冲突。
随后CPU干跑另发现MrgSortExecutedNumList重复定义；仅生成case前导移除已由
当前CANN/PTO-ISA提供的声明。原生成kernel正文、SDK、CANN及技能文件均未改。
v2/v3中间失败目录保留；v4先完成CCEC编译/链接再采集，没有重复正式L1任务。

最终task_20261022_055755_4107924772（v122）与
055756_4117009850（N256）均completed/exit0；均实际执行、输出全部校验通过，
各导出一个Cube核trace/指令CSV。共用已验证工具链，仿真模型Ascend910_9391、
dav-c220-cube；模型不等于对实机具体SoC身份的证明。两项都经task-submit auto，
未复制worker或安装依赖。五项CPU合同通过，相关六个脚本Ruff通过。

轻量结果及输入/代码/trace索引在score_producer_sim_results.json，复现提取器
summarize_score_simulator.py。原始大产物位于runs/score_sim_{v122,n256}_v4，
clean/同时保留raw二进制、clean trace和instr_metrics.json。cleaner分别跳过
10/8个无法重锚定的同步箭头，因此事件细节以raw及生成代码为准，不认为clean
图包含所有依赖。原始trace的时间数值为微秒，displayTimeUnit=ns只是显示设置；
MMAD时间与指令CSV独立核对为1850 cycles/us。非MMAD的CSV周期与trace队列
生命周期不能直接互换，未把WAIT/SET累计周期当作不重叠阻塞时间。

| 单生产者、同一合成输入 | v122 N64 | M64/N256 |
|---|---:|---:|
| 完整X事件首尾跨度（仿真μs） | 46.7697 | 67.0222 |
| INT8 QK MMA次数 | 192 | 144 |
| FP16 WS MMA次数 | 192 | 144 |
| MMAD可见区间并集（μs） | 28.9065 | 28.1773 |
| MTE2搬入字节 | 847872 | 847872 |
| MTE2指令数 | 194 | 194 |
| MMAD之间最大空隙（μs） | 0.4114 | 1.4741 |

此表是无FIFO反压的单生产者诊断，不能进入正式九档、不能外推+43%的硬件
回退；正式B16已测+6.3496%的结论不变。指令区间并集也不是物理引擎利用率。
但它确实否定“少25% MMA即可大幅加速”的假设：Cube区间总量近似不变，
大面板串行依赖/间隙明显增加，GM搬入总字节未减少。停止宽N参数扩扫。

下一步回到N64供数和FIX写回组织。只读核对另一会话日志19849及20571等段：
保留双48KiB QK早FIX的WS八列池方案，TSTORE16→2，旧轮B16/B28省5.42/12.12μs；
换公共环境后B16仅省5.14μs，B28中位略慢。它是可借鉴机制而非本会话已取得
收益，不能整目录覆盖TND或据别的会话采用直接宣称通过。先局部移植、审计
Acc地址和最后消费者，再诊断并仅以同轮完整L1决定是否保留。明确保留旧负例，
不扫池宽、不复跑已被否定的联合短尾。生产默认未改、未push，goal仍active。


### N64 WS列池：局部迁移、真实资源/复用事件与独立仿真正向

prepare_score_ws_pool.py只变换v122的_indexer_score_cube_split_store，保留
原policy、TND geometry、双M192、页保护和FIFO。两段K192按原顺序累计到
八个16×64列窗，满16×512才写原6×512输出，最后READY仍在完整step后。
没有整包复制另一会话的BSH，也没有叠加live_half/其它候选。三个CPU合同
核对原数学/FIX、page/FIFO和真实源偏移产生的完整1024列覆盖，全部通过。

完整B16与单生产者CPU均通过。score_ws_pool_resources.json审计两份最终
AllocateMemoryAddr/C++：QK0 0..49152、QK1 49152..98304、WS98304..131072，
16列窗按8个4096字节地址两遍复用；无WS Acc复制、每step32次QK/32次FIX、
16次WS init+16次续加，只有2次TSTORE。中间pool覆盖有FIX→M事件；最后
store到下一step的复用通过同PIPE_FIX队列、下一QK0 FIX→M→QK panel1→WS0
传递依赖保护，不能因没有紧贴末store的冗余直接flag就误判缺失同步。
此审计是源码/地址/事件机制证明，不是硬件重叠或性能测量。

单生产者task_20261022_060953_246241820844 completed/exit0，输出12288值
全部等于8。score_ws_pool_sim_results.json复用上轮原始N64仿真作同模型
诊断：跨度46.7697→43.5227μs；QK/WS MMA仍各192次，MTE2仍847872字节，
实际GM FIX由96→12次、L1 FIX192次不变。没有把约6.94%的合成生产端改善
称为完整层收益，也没有重跑已完成的reference采集。

完整代表B16/S6任务task_20261022_061056_25055969684已提交，严格冻结v122
同进程五配对窗口；当前仍在运行，不能使用初始化result里的false当终态
失败，也不能预记通过。公共SDK、计时口径、Native差异与九档目标均未变。

并行只读核对Query访存，发现旧“连续head分工”只改worker区间、不改八token
跨行tile，本轮拟评估真正八个相邻head组成Vec行。增加weight-scale重复读取
是明确代价；原512列规约、两次FP32乘法及rsqrt保留。T不整除8时完整回退
原helper，保留tail recip(sqrt)。前驱等全部Q_B、后继Cube等全部48个QUERY
写者，所以重新分配head不需要更改生产者信号合同。三个源级所有权/数学/
回退检查通过。首版统一pl.reshape在incore被解析为Tile，CPU类型检查拒绝；
v2明确使用pl.tensor.reshape构造三个GM别名，正在CPU编译，尚未上卡。
未改共享依赖、生产默认或push；目标继续active。


### WS列池完整L1不受益，停止扩档；Query GM视图接口修正

上述061056任务已completed/exit0。严格v122输出/Top-K/score、cache/state、
保护、metadata和图检查通过；原Native precision_exact=false及5850项原有
输出差异保持独立，不改成新精度PASS。五窗789.00→789.70μs，+0.08872%，
仅1/5快，score_ws_pool_results.json保存全部配对窗口和工具链。暂不采用、
不扩B28或扫池宽。独立生产者更快而整链未受益，只能说明此前合成改善没有
转为当前系统级增益，不能直接据此量化AIV反压或认定某一个物理引擎瓶颈。

Query v2的pl.tensor.reshape仍在ConvertTensorToTileOps中变成Tile，完整CPU
同处拒绝。只读检查公共编译器发现tensor.view明确保留GM Tensor，而reshape
是转换运算；v3改三个别名为pl.tensor.view，未改共享依赖，正重新编译。
v1/v2失败源码、描述和日志保留；没有把类型失败当设备性能或组件bug。
目标仍未达10%，生产默认不变，所有改动只做本地阶段commit，未push。


### Query连续head v4完整编译通过，提交代表档严格对照

v3已经保留三个GM Tensor view，但显式full([8,1], FP32)的行字节4不符合
Vec的32字节对齐而被PTOAS拒绝。v4去掉这块新声明：激活scale使用同轮GM
的device FP32 Scalar read，再做MULS；随后权重scale仍为独立FP32 MUL。
这是设备读数，无Host回读或新预处理。数学乘法次序未合并，但MULS与旧
row_expand_mul逐bit行为仍必须以设备对照确认，CPU数学模型不替代精度。

三项测试改为检查实际v4源码，枚举T0..384的完整行容量、24/32/48 worker，
每个(token,head)唯一覆盖且GM地址与原[T,64*512]一致。原尾行helper整体AST
保持；RMS平方/512列规约/rsqrt、NOPE、RoPE及BF16 rint运算保持。只允许
两次乘法广播方式改变；前后信号合同未改。三个检查与Ruff均通过。

v4完整CPU成功，query_contiguous_heads_v4_resources.json确认：三个view与
原q_proj/q_flat/wq_b_scale根及偏移相同，stride=[512,1]；compiled_meta与
相同描述符、原Query代码编译结果完全相等，没有新增GM根/参数/返回；
Vec地址最大163872B，与原Query相同。实际生成8×512连续读取、FP32 Scalar
read和MULS后MUL。v1..v4全部CPU结果归档compile_history，失败目录不覆盖。
旧连续DQ worker-range候选没有改变8-token跨行tile，本候选改变Vec行的
含义，因此不是原样重跑旧分工负例；增加scale/rope准备读取的代价仍保留。

任务task_20261022_062222_1899330389已提交，128K/B16/S6、冻结v122同进程
五配对窗口，等待严格精度/状态/图/完整L1终态。尚未预记PASS或收益，未扩
其它档。此前WS池完整负例已收取，当前只有这一项待收设备任务，无CPU悬挂。
本轮生产默认保持不变、未push，九档综合10%仍未达成，goal继续active。


### Query连续head：B16完整对照正向，扩固定混长和短B32

062222任务completed/exit0，严格冻结v122输出/Top-K/score、cache/state、
保护、metadata、padding和图检查通过。B16/S6五窗中位796.44→785.62μs，
降低1.35855%，5/5配对更快；全部窗口在query_contiguous_heads_v4_results.json。
Native旧precision_exact=false仍另列，不将旧差异改成通过。不与main_vector_late
或其它独立候选的百分比相加，生产默认未采用，也不能据一档宣称九档10%。

query_head_read_model.py从原函数实际device head分组条件提取模型：在T96，
权重scale逻辑读取1572864→12582912字节（8倍），scale/RoPE准备296448→592896
字节（2倍），Q投影输入、Q输出、8×512 RMS块数相同。T56/T168的准备数据
反而减半，T192准备变4倍；因此不能把B16结论直接扩到所有档。该模型不含
缓存、对齐、异步重叠、设备指令成本，不能把逻辑读取当实际HBM流量或耗时。

已顺序提交两个同源码、同冻结v122、同公共环境的独立配对任务：
- task_20261022_063040_245239019924：固定128K/B16混长T56，原九档的第9档。
- task_20261022_063041_24714692920：8K/B32/S6，检查较大的scale/准备重读代价。

两任务当前running，无CPU任务；使用原句柄继续收取，不因观察等待重提。
目前新增阶段提交af32fd56、0c52ab13，后续结果继续分阶段提交，未push。
完整九档G尚未形成，goal保持active，后续还需受影响边界、实际服务和组合验收。


### Query扩展两档均已收取：暂只保留B16方向，准备独立分档

两项任务均completed/exit0，严格冻结v122精度、Top-K/score、cache/state、
保护及图验证通过。保持独立轮次中位数，完整五窗见对应结果JSON：

| v4场景 | v122中位μs | 候选中位μs | 耗时变化 | 配对更快 |
|---|---:|---:|---:|---:|
| 128K/B16/S6（上一轮） | 796.44 | 785.62 | -1.35855% | 5/5 |
| 固定128K/B16混长T56 | 752.40 | 754.70 | +0.30569% | 2/5 |
| 8K/B32/S6 | 926.10 | 930.56 | +0.48159% | 0/5 |

结果文件分别为query_contiguous_heads_v4_results.json、
query_contiguous_heads_v4_mixed_results.json、
query_contiguous_heads_v4_short_results.json。混长方向混合且幅度小，不能
把0.31%认定为稳定损失；但它没有足够正向采用证据。短B32五窗全慢，不
全域开启。三档数据不构成九档G，也不与其它优化不同轮次相加。

下一步优先构建独立constexpr分档：沿已有长G6、64<=token容量<=96资源域，
保留实际device T不整除8时完整旧helper回退。64处Q_B wide-N及前置几何
策略改变，96是既有LocalStats/O_A资源边界；不是按B16名称写白名单。
这个区间只是有依据的待验证范围，尚未证明T64/T72/T80等都有性能收益。
应显式加QUERY_CONTIGUOUS_HEADS，不能隐式捆绑现有Attention开关；源码
启用端与已测v4、关闭端与v122的生成代码分别核对，再补T64/T80与T65尾行
等受影响边界。没有改生产默认，没有提前写采用PASS。

另外只读复审Score消费链：当前B16关闭scale_page_coalescing，逐step逐页
读取FP16 scale；B28原本开启。另会话已有B16 scale/WS组合小幅正向，不能
直接迁移其百分比。若进一步并行化页布局证明，需同时处理group_offsets
尾区的单缓存行写者、各轮覆盖、IDX_SCALE发布/获取、动态页与碎片回退；
现有2字/request不能分给多个Scalar写者而忽略64B行共享。本轮没有生成
这个候选、修改工作区或启动新任务，只把它列为需完整设计的后续方向。

全部当前NPU/仿真/CPU任务已终态，current_state没有pending。本轮实际完成
单生产者诊断、WS整链负例、Query三档正负结果与源码/资源审计，有实质进展，
不存在重复外部阻塞；目标继续active，生产默认未变，未push。


## 2026-10-22：Query独立分档已生成，启用/关闭完整代码与实测产物一致

上一goal轮完成四个阶段commit、单生产者证据、WS完整负例及Query三档正负
实测，属于实质进展。本轮再次核对goal active、Git干净、共享manifest仍为
validated的同一版本，并用task-submit确认上轮两个最后任务均completed/exit0。
继续九档等权G<=0.90目标，不缩为B16单点，不重新测已完成的三个v4试验。

prepare_query_head_guard.py基于实测v4增加独立QUERY_CONTIGUOUS_HEADS，显式
经过policy→decode→core_group→vector_schedule→Query helper。区间为既有
long G6且64<=token容量<=96；实际T不整除8时仍执行整个原helper。未用
LOCAL_ATTENTION_STATS等其它开关隐式控制新选择，修改它们不会改变该flag。
没有新增环境变量、GM buffer、host metadata或公共SDK修改。暂为候选，未采用。

四项CPU合同通过：枚举历史阈值、query上界、请求/容量和63/64/65/95/96/97
等边界，旧policy各项不变；flag独立；实际转发完整；打开的数学与v4相同、
关闭调用整个原helper，其它调度AST不变。相关三个脚本Ruff通过，格式修复
后额外核对生成policy字节未变化，没有因代码格式重编已通过的候选。

完整B16启用与短B32关闭CPU编译均成功。query_head_guard_codegen_audit.json
直接绑定上轮实际设备产物，先核对实测spmd/original_spmd全部源与相应清单，
再比较guard的C++、AIC/AIV packed wrappers、orchestration和compiled_meta。
两侧五份产物均一致：C++只删除// pto:位置注释，运算、常数、地址、分支、
同步保持；host/ABI metadata逐字相等。未把它扩大为ELF相同，未以此直接
标记最终九档性能通过，最终组合仍须完整验收。

本轮重读已定位cannbot ops的attention_schedule_slots_and_platform_handoffs
（draft、未定位platform）：采用其中区分运行期未走分支与编译期实际消除的
核查思路，并以本次生成代码作证；不照搬平台事件编号、mode4或PIPE_S示意。
知识索引缺失的原限制保留，没有重建索引或更换共享工具链。

已提交并确认running：
- task_20261022_064744_56493329836：B16实际S4、query上界6，T64，五窗完整L1。
- task_20261022_064745_5670722673：B16实际S5、query上界6，T80，五窗完整L1。
- task_20261022_064818_59201812816：T72/R17 metadata矩阵，覆盖启用路径。
- task_20261022_064819_5935356678：T65/R16 metadata矩阵，覆盖token尾行回退。

最初尝试T80 metadata矩阵在提交前被原harness合同拒绝：该矩阵只支持
41<=Tcap<=72，且request容量须能接收S1时全部dummy尾部。没有产生任务ID、
没有占卡、没有放宽harness。改用T72/R17（6+ceil((72-6)/6)）覆盖开启路径；
T65/R16合法且保留。T80仍由前述独立完整L1实际S5用例验证。此处只登记
真实提交和已完成CPU证据，尚未预记任何新设备PASS或性能收益。

当前无CPU任务，四项设备任务使用原句柄收取。生产默认未变、未push，
完整九档10%尚未达成，goal继续active。


### 边界终态与实际分档纠正：S4/S5是关闭对照，不能归因新算法

064744、064745、064818、064819四项均completed/exit0、严格功能/状态检查
通过。T72/R17与T65/R16两个metadata矩阵各8项（S1..6、261、123456）和4步
连续轨迹全部通过，各12次同址图重放。连续轨迹保留前轮状态，包含请求减少、
重排和新增；历史偏移覆盖C4余数。摘要query_head_guard_metadata_results.json，
原metadata_matrix.json保留逐项输出/Top-K/cache/state、写界及同址证据。
T72实际flag=True且容量整除8，覆盖新Query；T65实际flag=True但容量尾行，
覆盖整个原Query回退。没有将这些描述为新增B1服务或完整九档通过。

收取两个L1边界时，原collect_trials正确拒绝S4/S5不是固定九档。未修改或
放宽evaluate_nine_cases。另建collect_query_guard_boundaries.py，显式仅收
四种已定义边界输入，沿用完整72核/单MIX/native150/五交替配对/隔离工作区
验证，强制记录实际QUERY_GROUP和QUERY_CONTIGUOUS_HEADS，并核对实际生成
PTO是否存在head GM-view路径。八项负向/合同测试通过，包括确认原九档gate
仍拒绝边界用例。所有边界报告formal_gate_eligible=False。

更重要的实际分档发现：测试器在显式query-lengths下将query_tokens更新为
真实最大值。原T64全S4记录query_tokens=4/G2/flag=False，T80全S5为5/G2/False；
之前登记“上界6”只是CLI意图，不是实际执行条件，现明确纠正。两个实际候选
与同轮v122的完整C++（仅去pto位置注释）、AIC/AIV wrappers、orchestration、
compiled_meta全部一致，见query_head_guard_disabled_boundary_results.json。
其818.94→829.10μs（+1.24063%、0/5）和978.06→979.52μs（+0.14928%、1/5）
完整保留，但不能归因新Query，也不能作为真正启用区间的性能证据。

没有重测已完成S4/S5，而是采用真实最大S=6且同总T的新输入，保持原harness：
- task_20261022_065845_127797123832：T64，[6,2]+[4]*14。
- task_20261022_065846_127906019085：T80，[6,4]+[5]*14。

两者history131072、B16/Rcap16，无dummy、同卡同进程冻结v122五窗；收取器
将要求实际G6且flag=True，否则拒绝当作启用证据。两项尚待终态，不预记收益。
正式八档S6与固定第9档混长完全不变。当前只剩这两个设备任务，无CPU任务，
生产默认未改、未push，goal继续active。


### 真正G6启用边界回退；组合v3仅编译，转为同址归因

065845、065846均completed/exit0，严格输出/Top-K/score、状态、保护及图通过。
新边界收取器同时验证实际QUERY_GROUP=6、QUERY_CONTIGUOUS_HEADS=True以及
生成PTO含head view，两个输入没有再落入关闭路径。完整五窗结果：

| 场景 | v122中位μs | guard中位μs | 耗时变化 | 更快窗 |
|---|---:|---:|---:|---:|
| T64，[6,2]+[4]*14 | 723.56 | 733.00 | +1.30466% | 0/5 |
| T80，[6,4]+[5]*14 | 780.70 | 798.96 | +2.33893% | 0/5 |

证据query_head_guard_active_boundary_results.json。拒绝把64..96整个资源区间
作为采用策略；不靠只挑T96或换成B16白名单掩盖两个负例。原T96正向证据保留，
但尚不能证明普遍/稳定收益，尤其代码相同的关闭对照也出现约1%的耗时差。

在边界运行期间已准备combined_v3：基于最充分验证的combined_v1
（B1空发布、live-half、batch-sort3），加入main_vector_late和Query guard。
INPUT/QR/ROPE裁剪与Main后移变换逐字可交换，Score原组合源完整复制，独立
Query flag仍显式传递；未加入无整链收益的WS池或combined_v2方向混合的填充项。
B1/B16完整CPU均通过，源码/描述符/结果已保存。但边界结果否决了包含的
Query区间，因此组合v3不提交设备、不采用、不作任何性能结论。

为区分T96的实际代码收益与工作区地址差异，提交
`task_20261022_070839_165489119202`。使用已有--l1-shared-workspace归因模式，
两版串行交替共享全部GM buffer（含signal），相同初态恢复仍在计时外，
五窗完整72核L1及代际/精确验证保留。候选用v4（与guard启用生成代码已证
一致），因为其policy字典与v122完全一致，满足原工具同资源策略约束；
没有修改或放宽harness。该诊断shared_workspace=True，原九档gate将拒绝，
只解释先前小幅正向，不能替代最终独立工作区验收。当前该任务running。

另做只读排重：已有实现日志1684后的B1四→两Hadamard核/Q_B二十二核及
随后公共工具链复核记录，不能把同一配置重命名为新优化。本轮没有重跑。

Score连续scale后续设计发现一个必要约束：binding允许自定义allocator，
只显式检查signal的512B对齐；group_offsets并无64B首地址合同。因此不能
只按Tensor相对16字分包就保证物理缓存行单写者，也不能直接让每请求写者
并行更新当前2字记录。安全设计需实际物理对齐处理/回退，或把记录间隔扩到
能覆盖任意INT32起始偏移的两个缓存行，并同步调整binding、消费者及独立
harness的布局/保留区验证。原META完成之后的IDX_QUANT/IDX_SCALE获取链和
最后消费者也必须明确核对。此方向本轮只完成约束分析，没有生成候选或
修改任何共享SDK/生产布局。不可将这个设计记录当作已获性能收益。

本轮仍为有证据的实质进展，无外部阻塞。当前只有同址归因一个任务待收，
无CPU任务；生产默认不变、未push，九档G仍未形成，goal继续active。


### 同址Query诊断与第二个完整S6边界；容量细化只保留为候选

070839同址任务completed/exit0，严格输出/Top-K/score、状态、保护、图和代际
检查通过。790.76→782.88μs，-0.99651%，4/5快。收取器明确验证原九档gate
因shared_workspace=True拒绝；此结果只支持归因，不能替代正式独立工作区
验收。原harness把baseline.buffers设为candidate.buffers，binding每次调用
从该字典传参，未另改计时/恢复位置。结果与完整窗口见
query_contiguous_heads_v4_shared_diagnostic_results.json，旧Native差异另列。

又测非B16的完整S6容量边界：task_20261022_072146_218734617423，B12/S6/T72，
实际G6/flag=True，completed/exit0，全部严格检查通过。706.90→706.22μs，
-0.09619%，3/5快，仅接近零的小幅证据，不据此宣称稳定改善。边界专用收取器
增加这一明确非cohort输入，原九档case集合完全不动；八项拒绝错误分档、
时钟/核数/次序/共享区和phase-clock的测试仍通过。

准备过query_head_capacity_guard，仅在原64..96长G6资源域再要求
T容量=request容量×query上界。只改policy，不从该乘积推断实际Q/L或请求归属，
动态计算与尾行回退完整保留。两个CPU策略/源码合同通过，T80关闭完整编译
及实际v122生成C++/ABI/host逐字核对通过。B16/T96和B12/T72的kernel文件及
所有constexpr值与已测guard一致；没有再重复编译这两个相同输入。该细化
没有采用到生产，B12证据很小，转而优先处理新发现的实际重复读取。

### 新机制：同一AIV的8-head权重scale只载入一次

连续head映射的work=worker+k*WORKERS。H=64、每work八head，当WORKERS为8的
倍数时，work%8恒等于worker%8；当前48个AIV每轮都读同一组8-head权重scale。
v4却把16KiB重新载入每次迭代。query_resident_weights独立基于v4，只改Query
helper：一次读入readonly Vec tile，循环内复用；WORKERS不是8倍数时保持原
逐work读取，T不整除8时仍走整个原helper。没有改scale乘法顺序、RMS、RoPE、
BF16边界、同步或Tensor ABI。

两个CPU合同通过：实际源表达式枚举多种worker数与T，确认可驻留/必须回退的
所有权差异；还原这一个load移动后，整个旧数学body的AST完全一致。B16
完整CPU编译成功，packed_meta与v4相同。逻辑权重读取量T96由v4的12MiB变为
768KiB，原v122为1.5MiB；这是逻辑元素量，不是HBM流量或耗时预测。

按cannbot resident_softmax_state_slot_recurrence_and_reuse_events的生命周期
核查思路审计最终生成物（该卡仍draft/未定位platform，不搬用硬件结论）：
- resident_head_scale物理UB[65536,81920)，16KiB，整个Query循环前只有一次TLOAD。
- 重复主循环中该物理区间只有两个只读别名，既无输出覆盖也无scratch/tmp占用。
  加强检查按实际UB偏移判断重叠，不能把不同MemRef名当成独立UB地址空间。
- 奇数最后单work尾轮可以在最后一次MUL读取时原地覆盖该区间；它在主循环后，
  之后没有任何原scale消费者。没有把合法的最后一次复用误判成循环内破坏。
- 原行索引准备的V消费者先通过V→MTE2的配对事件退休，之后才加载权重；
  生成的Query输入TLOAD全部晚于权重TLOAD，原MTE2→V获取覆盖前序MTE2工作。
- 整体Vec地址上界仍163872B，没有新增GM根/参数/返回。

审计一度因PTOAS合并了展开的_1/_2名字提示而停止；改为核对真实C++分支中的
INT32 8×512输入Tile/TLOAD，未删除控制或事件检查。逐步审计报告保留，最终
入口query_resident_weights_resources_v4.json；这仍只是静态机制证据，真机
正确性与完整耗时由下列任务另证。Ruff及两个新合同检查通过。

### 驻留版三档终态：B16正向，混长仍回退，不全域采用

三个任务都completed/exit0，严格冻结v122输出/Top-K/score、所有cache/state、
保护、图及代际检查通过，Native原有差异不改标：

| 任务 / 场景 | v122中位μs | 驻留版中位μs | 耗时变化 | 配对更快 |
|---|---:|---:|---:|---:|
| 075108_326400111708 / 128K B16 S6 | 799.88 | 789.18 | -1.33770% | 5/5 |
| 075109_32650589990 / 固定混长T56 | 735.92 | 747.40 | +1.55995% | 2/5 |
| 075110_326637827 / 8K B32 S6 | 930.10 | 928.36 | -0.18708% | 2/5 |

完整数据query_resident_weights_initial_results.json及
query_resident_weights_mixed_results.json。不能跨轮把驻留版与v4绝对值相减，
也不能将少读权重等同于全域收益。只保留B16方向，混长不采用；短B32幅度很小
且窗口混合，不给强正向结论。此前combined_v3继续隔离，不上卡或采用。

### 下一实际省工点及平台边界复审

继续追踪发现，连续head helper每work还重新读取/展开rope_swap、加行偏移并
转换为q_abs_swap。该swap是内部q_rope_prepare_stage写出的固定列配对，所有行
相同，不同于随位置变化的外部cos/sin。vector_schedule已在609行附近获取全部
PHASE_ROPE写者，之后才进入Indexer/Main Query。下一候选可在证明生产者行
一致后只准备一次readonly q_abs_swap；仍每query读cos/sin，保留每次gather的
可写scratch，不根据位置相等或T/B推断外部sin/cos数值。此处尚未生成或上卡，
不预记收益，也不更改其余Goal约束。

只读排除了一个较大但无新传输机制的改写：固定PyPTO的
`docs/en/dev/passes/24-expand_mixed_kernel.md`与TPUSH/TPOP契约明确a2a3经GM，
固定PTO-ISA `include/pto/npu/a2a3/TPush.hpp`的producer为TSTORE到GM、consumer
为TLOAD。不能套用a5的消费者SRAM路径，也不能声称把当前手写Group换成
pl.spmd/split_aiv/aic_gather就自动省去GM。本轮没有为该假设改写Attention或
更改共享SDK；这条结论只针对当前固定实现，不扩为所有硬件能力断言。

当前所有任务已终态，无CPU/设备pending。生产默认未变，仍未形成最终九档
G<=0.90及完整新服务验收；goal继续active，未push。


### 内部RoPE索引常驻：生产者证明与完整生成物审计通过，三档已提交

目标继续为九档等权G<=0.90：八档S6和固定128K/B16混长，分母仍是冻结v122。
新query_resident_swap基于query_resident_weights，只将q_swap_one、
q_swap_repeated和q_abs_swap移到Query工作循环前；读取内部rope_swap的第0行。
新增tokens>0保护，空输入不读不存在的第0行。逐query的cos/sin、逐work的
可写gather scratch、完整原tail helper及所有同步/ABI均保持。

三个CPU合同通过：执行实际q_rope_prepare_stage的数学，覆盖T0/1/6/7/8/9、
T24/48/56/65/72/96/144/168/192/193和1/7/16/48个worker，随机sin不同，
所有有效行仍为(1,0,3,2,...)；恢复三条索引准备语句后整个旧Query函数AST相同；
核对原PHASE_ROPE生产、发布、获取先于消费者，等待范围覆盖row0生产者。
producer用min(workers,token_tiles)分配，row0始终由worker0写；原获取还包含
业务cacheinvalid/fence和本轮generation，没有增加新通知协议。

B16完整CPU编译通过；参考cannbot已读的resident状态复用卡，按真实物理UB
地址核查，不把MemRef名当独立bank。新索引为[51200,53248)，两路展开和最后
单work均无覆盖，包括scratch和别名分配；C++只有一次索引TLOAD，三个TGATHER
均读同一驻留tile。MTE2→V事件成对且在转换前获取。固定a2a3 TGather.hpp的
FP32分支将byte-offset暂存和gather结果写到dst，不修改src1索引。权重驻留
独立复审也通过；整体Vec上界仍163872B，packed_meta与父版完全相同。

复现入口prepare_query_resident_swap.py，CPU合同test_query_resident_swap.py，
审计audit_query_resident_swap.py及原audit_query_resident_weights.py。报告
query_resident_swap_resources.json和query_resident_swap_weight_resources.json。
这些是源码/生成物证据，不是设备正确性或性能通过。Ruff检查通过。

完整72核L1、同轮冻结v122、五个交替窗口任务：
- B16/S6：task_20261022_083320_6419639195
- 固定混长T56：task_20261022_083333_65038322156
- 8K/B32/S6：task_20261022_083346_6590615021

均由冻结harness的submit_probe.py经task-submit自动分卡，不改公共SDK。
当前待终态；无生产采用，不相加父版或独立候选收益。


### RoPE索引驻留三档收齐：全部严格正确，收益有限，不全域采用

三个任务均completed/exit0，严格冻结v122输出/Top-K/score、cache/state、
保护区、图及代际通过；Native原有差异继续单列。完整L1五个交替窗口：

| 场景 | 耗时变化 | 配对更快 |
|---|---:|---:|
| 128K/B16/S6 | -0.36862% | 3/5 |
| 固定混长T56 | +0.14270% | 2/5 |
| 8K/B32/S6 | -0.57553% | 4/5 |

原始窗口、两侧中位数可由query_resident_swap_b16_results.json及
query_resident_swap_mixed_short_results.json重算。减少逻辑load和索引算术
未给出全域稳定收益，不将跨轮绝对值相减来推断比权重驻留更好或更坏；
短B32保留小幅局部证据，不当成正式采用或九档目标完成。

复核既有O_A生成物：已有两份K256 L1以及双K128 Left/Right，不把简单
新增双槽当作新机制；历史B1显式K512/M16供数已是完整负例。本轮不重试。
新发现输出量化的oc_amax/oc_q是相同GM地址的两次真实TLOAD，源和最终IR
均存在。准备独立基于v122的output_quant_reuse，仅用oc_q=oc_amax延长原值
生存期；所有amax/div/scale乘法/三次cast、valid行、尾零、分工AST保持。
CPU合同和完整B16编译已通过；仍需检查abs/scratch不可覆盖原值到最后的
row_expand_mul，再做完整L1实测，不预记收益。


### O_A量化原值复用：生成物安全，开始完整L1与B1尾行矩阵

output_quant_reuse完整B16编译和audit_output_quant_reuse.py通过。输入物理
UB在两个展开lane分别为[65536,98304)、[98304,131072)，abs与row_max scratch
均另有存储；到最后一次row_expand_mul前，重叠地址只有oc_q的只读别名。
最终乘法后原值再无消费者，允许结果覆盖原tile。尾轮也逐项同样核查。
展开后原输入load静态位点6→3，整体Vec地址上界163872→155872B，packed ABI
不变。C++确认实际TLOAD、TABS与TROWEXPANDMUL，别名TASSIGN同址。
审计首次假设别名会使用同一个C++变量名而失败；修为核验别名的TASSIGN地址，
保留物理区间覆盖及生命周期检查，未放宽任何数值/保护区条件。

一个CPU源码合同和Ruff通过；本轮只替换第二次相同load，不改运算顺序或
尾部语义，不改变custom allocator合同。B16逻辑输入字节6MiB→3MiB，不声称
这是实际HBM流量或完整收益。此源独立基于v122，不叠加Query或其他优化。

三项冻结harness/公共SDK完整L1五窗任务：
- B16/S6 task_20261022_084339_11452478515
- 8K/B32/S6 task_20261022_084352_115624411592
- B1/S6及同址S1～S6 task_20261022_084401_116502025524；Tcap6/Rcap1，
  table-history131200满足原矩阵动态边界合同，固定S6计时先于metadata回归。

任务尚待终态。生产默认不变、没有push，目标继续active。
