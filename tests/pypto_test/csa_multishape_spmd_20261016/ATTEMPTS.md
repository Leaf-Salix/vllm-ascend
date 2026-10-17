# 多档 CSA 单 SPMD 尝试记录

## 2026-10-16：建立参数化准备入口

- 静态基线已提交并 push：`a919f119`，目标分支 `dsv4-flash-pto-v0.25.1rc1`。
- 使用独立 worktree `.worktrees/csa-spmd-dynamic-20261016`，未改共享生产入口、
  PyPTO、Simpler 或 pypto-lib 源码。NPU 测试仍使用此前已授权的隔离组件。
- `prepared_v1` 八个 CPU 规格全部通过；B16 源码 AST 与已验证版本相同。
- 测试脚本初次复制时把相对 `Path('.')` 的 parent 当作仓库父目录，读取失败；
  改为 `Path.cwd()` 后生成完成。未启动错误测试或改动原脚本。

## B12 非历史形状验证

1. `device_b12_v1` / `task_20261016_190445_377635926395`：退出 1。
   实际描述符绑定、分片/视图检查成功，尚未执行融合核。
   新增逻辑分片逐块展开使 AIV 源码段从 41 增至 69，PTOAS 触发固定 60 秒超时。
   保留生成源码、编译报错和 result.json，不记精度或性能通过。
2. `device_b12_v2` / `task_20261016_191204_200117150`：退出 1。
   一个原控制位置承接同一任务的多个分片，核内循环后合并发布进度，AIV 段回到
   41，AIC 38。CPU 检查通过。PTOAS 对外层 INDEX 循环变量直接用作内层起点
   报 `emitc.for op expected induction variable to be same type as bounds and step`，
   证据在 `build_output/entry_d8ja688g/report/codegen_errors.txt` 及 `.pto` 346、7581 行。
   没有修改 PTOAS 或放宽同步条件。
3. `device_b12_v3` / `task_20261016_191520_21364431342`：已提交。
   用固定波数循环计算 `block=参考block+wave*参考块数`，再判断范围，保留相同
   逻辑 block 集合、顺序与 MIX 配对，避免上述嵌套起点 lowering 问题。
   结果以该目录 `result.json` 和队列终态为准。

## 分档与阶段归因任务

- `b16_stage_clock_v1` / `task_20261016_191754_31988114769`：B16 阶段插桩；
  40 个 AIC 段与 44 个 AIV 段的插入边界已在旧 C++ 上核对；每核最多 20 个实际
  控制位置，80 个时间戳落在信号缓存行之后，位于每核既有 512 B 空间内。
- `device_128k_b4_v1` / `task_20261016_191807_32922422901`：长历史小档。
- `device_8k_b24_v1` / `task_20261016_191809_33070718758`：短历史 S6 Indexer 分支。

后续结果仍需查看任务终态，不能据“已提交”或编译完成推断 PASS。

## 设备结果与第一轮性能定位

- B12 v3：设备计算 PASS。eager/graph/最终输出、Top-K、六份 state/cache、保护区及
  30 代信号通过；L1 原动态 796.76 μs，单 SPMD 1130.30 μs。队列显示 exit=2：
  Python 已正常写出 complete，随后 Bash 因脚本末行在作业运行期间被修改而报 EOF。
  队列日志明确是包装脚本解析失败，不把整个 job 标为 exit=0。后续不原地修改正在
  使用的 shell 脚本；当前脚本 bash -n 通过，不为该包装问题重跑已完成的设备测量。
- 128K/B4：`task_20261016_191807_32922422901` exit=0，全部检查 PASS；
  原动态 L1 583.28 μs，单 SPMD 799.34 μs。
- 8K/B24：`task_20261016_191809_33070718758` exit=0，全部检查 PASS；
  原动态 L1 854.02 μs，单 SPMD 1286.18 μs。
- B16 阶段探针：`task_20261016_191754_31988114769` exit=0，全部检查 PASS。
  原动态 L1 892.48 μs，插桩单 SPMD 1334.38 μs；此值不用于目标验收。
  973 个成员的四点时间戳全部有效。Indexer 数学段中位数约 217 μs、QK/PV
  约 161 μs，与动态版约 222/162 μs 接近。明显延迟在 Indexer 启动前：
  原动态约 303 μs 启动，插桩融合约 599 μs。不能把等待累计时间直接当作可消除开销。
- 读侧绕过缓存候选：`b16_bypass_wait_v1` / `task_20261016_192452_64367617729`
  exit=0，全部检查及 30 代信号通过。只把 TWAIT 换成 A3 ld_dev 的 GE 轮询，
  TNOTIFY、业务数据维护与数学均未改。原动态 L1 901.88 μs，候选 1366.36 μs，
  未见明显性能改善，不声称已证明 DCCI 轮询是主要瓶颈。

## 保持数学偏序的完成通知汇聚

`completion_aggregation.py` 将每个成员分为计算与发布两个节点。只选择
“下一计算节点本就等待该任务全部生产者”的核，在其计算后、发布前汇聚。
随后消费者可等待一个完成标记。无合格核的任务保持原逐核依赖。

B16 CPU 结果：66 个任务中 37 个可汇聚，等待从 16556 降为 3641
（计算前 3134，发布前 507），发布点 711。检查双阶段图无环，并逐节点投影
核对原数学偏序完全相同；序号翻译后的偏序也相同，三代协议模型通过。
未增加全核屏障，未删除业务数据同步。`prepared_gather_v1` 保存完整准备结果。
本候选仍需真机与 L1 验证，CPU 减少的等待数量不是性能收益证明。

- 真机任务：`b16_gather_v1` / `task_20261016_193007_91656320040`。
  19:48 仍 pending；全 16 卡由前序整网任务占用，没有重复提交。

## 融合前后局部流水差异

检查同一 HC_pre widen 数学体：独立动态 PTO C++ 中 set/wait 各 34 处，
融合 C++ 中各 56 处，循环中额外出现 PIPE_ALL。数学操作种类和数量一致。
这证明生成流水不同，尚不能仅据事件数量推断性能差额或判定寄存器溢出。
`hc_widen_sync_diff.txt` 保留对比。

CPU 编译/反汇编尝试保留在 assembly_*：

1. v1 使用 -S，CCEC 拒绝 device 侧该选项。
2. v2 改为 ELF object，缺 tensor.h；v3 补 runtime Tensor include 后缺
   tensormap_and_ringbuffer/task_id.h；v4 补 common 后缺 intrinsic.h。
3. v5 使用 Simpler 正式 include 查询方法，AICore object 编译成功。
   llvm-objdump 只给出 not available，指定 dav-c220-vec 也没有指令。
   两份 ELF 的 stack-size record 都是编译选项 0x8000，不能据此证明真实栈溢出。
   暂停这个没有产出有效指令证据的分支，不据猜测下结论。

`preserve_math_pipeline.py` 是归因候选：在测试进程 C++ 生成边界，把
hc_widen_rms、mix_x_rms_norm、qproj_matmul、idx_qr_proj_matmul 数学区
替换为同轮动态 PTO 编译的 device 函数调用。核间等待、发布、业务缓存维护
与原分核次序保留；函数以 noinline 编译，仍然只有一个融合任务。
该候选不是最终 PyPTO 接口实现，明确不作为 700 μs 目标完成依据。

- `preserve_pipeline_cpu_v1`：8 个控制位置的实际 Tensor/Scalar/逻辑 block
  绑定核对完成，AIC/AIV CCEC 编译通过。
- `b16_preserve_pipeline_v1` / `task_20261016_194834_198181913198`：
  已提交设备对照，尚未得到结果；不预记正确性或性能收益。

## 汇聚实测与按规格重分配

- 完成汇聚 `task_20261016_193007_91656320040`：exit=0，输出、Top-K、
  六份状态、保护区、图重放及 30 代进度均 PASS。动态 L1 906.14 μs；
  单 SPMD 三窗为 1487.84、4639.64、4005.94 μs，中位数 4005.94 μs。
  Event 中位数 1355.20 μs，但最大 9254.46 μs，长尾明显；不采用此结果宣称优化。
- 保留局部流水 v1：`task_20261016_194834_198181913198` 首次融合执行失败，
  synchronize 报 507018 / AICPU 0x2a。编译成功不等于可执行；目前没有证据
  将故障归因于 noinline、缓存一致性或某一组件。变更前脚本保存在 test_source。
- 新的 `balanced_schedule.py` 按数学 task DAG 优先级和核空隙分配分片，
  保留所有逻辑 block、真实 metadata 计算和 MIX 三核配对。旧 trace 中 qproj 的
  20 片落在 12 个 AIC 上，最多同核 3 片；该偶然资源次序不再固定。
  CPU 代价来自参考 L1 中位数，仅用于分配，不作为实际性能结果。
- `prepared_balanced_all_v1/result.json`：原七档加 B12 的重分配、汇聚、
  视图边界及三代协议模型全部 PASS。真机覆盖不能由 CPU 结果替代。
- `b16_balanced_gather_v1` / `task_20261016_195218_216602122487`：
  全部设备检查和 30 代进度 PASS。动态 L1 897.42 μs，
  单 SPMD 1227.12 μs，Event 954.23/1334.08 μs。仍未达到 700 μs 目标。
- `b16_balanced_plain_v1` / `task_20261016_195623_23741736566`：
  只改变分配，去掉汇聚，独立测量其作用；已提交。
- `b16_preserve_inline_v2` / `task_20261016_195623_23739921091`：
  将归因 helper 改成 always_inline，仍保留原 PTOAS 局部流水，缩小上次
  失败的定位范围；已提交，不预记通过。

## 完成七档的无插桩重分配对照（20:11）

- `b16_balanced_plain_v1` / `task_20261016_195623_23741736566`：exit=0；
  输出、Top-K、六份状态、保护区和 30 代信号全部 PASS。无插桩 L1：
  动态 902.28 μs，单 SPMD 1067.36 μs。只保留重新分配，比通知汇聚更快。
- `b16_balanced_stage_v1` / `task_20261016_195951_255674726835`：
  exit=0，检查通过；插桩 L1 1166.52 μs，只用于归因。Indexer 数学段约
  214 μs；qproj 在该安排下延至 Indexer 后，另有优化资源次序的空间。
- 其余六档分三组顺序运行：
  `task_20261016_200302_26991286640`（128K/B4、B8）、
  `task_20261016_200302_26991692585`（128K/B24、8K/B16）、
  `task_20261016_200302_269932320183`（8K/B24、B32），全部 exit=0。
  七档完整数值见 `BALANCED_SEVEN_RESULTS.md`，均通过相同检查。
  每次恢复固定输入，不能据此宣称连续 decode 轨迹通过。
- 最初的三个 matrix 包装任务
  `task_20261016_200028_25818234812`、
  `task_20261016_200028_258186424881`、
  `task_20261016_200029_258202311263` 在解析队列自动附加的
  `--device` 参数时退出 2，没有执行 NPU；包装器补接收并核对 TASK_DEVICE 后提交上述任务。

## 局部流水替换及阶段内存规划

- `b16_preserve_inline_v2` / `task_20261016_195623_23739921091`：
  与 noinline 版本一样，首次融合执行报 507018 / AICPU 0x2a。
  这不能证明根因是内联。停止此 C++ 数学替换方向，保留失败现场。
- `balanced_memory_dump_v1`：CPU 全流水编译通过，保留 MemoryReuse 前后的 IR。
  `stage_memory_aiv0_v1` 的单阶段内存规划通过，但该阶段不是 widen。
- `stage_memory_widen_v1` 完成内存规划后因未设 backend 无法 codegen；
  补明确 Ascend910B 后，`stage_memory_widen_v2` CPU/PTOAS 编译通过。
  同一 widen 数学体单阶段规划时 set_flag/wait_flag 各 33 处，整层规划时各 56 处。
  这是编译流水差异的证据，不是设备性能收益证明。
- `phase_memory.py` 只在本测试进程逐阶段调用原 PyPTO MemoryReuse 与
  AllocateMemoryAddr，拒绝跨阶段共享片上值，保持原数学、核间依赖和流水排空。
  不修改共享 PyPTO/Simpler。
- `balanced_phase_memory_cpu_v1` 失败于嵌套 pass 触发外层 dump 回调，
  使外层 pass_index 越界；没有设备执行。
  v2 对嵌套 pass 隔离 instrument，同时继承完整验证级别及其他编译选项；
  CPU 编译已启动，结果待查。

## 20:38 新候选结果

- 阶段内存规划 CPU v2 遇到原打印 IR 中的 FREE_VAR，无法文本重解析。
  v3 直接组合 IR 节点、保留 Var/MemRef 身份，完整 CPU/PTOAS 编译通过。
  完整融合 C++ 的 widen set/wait 各 57 处（原各 56），不能把单阶段独立
  编译时的 33 对事件当作完整融合方案的改善；见 phase_memory_event_comparison.json。
- 编译调度表候选按等差分段生成每核字段，并转置/分组等待关系；68 个源码阶段、
  973 个成员的依赖全部精确重建。GM records/waits 不再在核内读取，业务 metadata
  保持 device Tensor。此变化仍未带来实测收益。
- 业务缓存候选保留每轮入口 DCCI、每阶段本核流水屏障、Scalar GM 写后的
  DCCI/DSB、Scalar GM 读前的 DCCI/DSB，以及全部信号维护。
  Tile DMA 路径不依赖 Scalar DCache；实际也未取得明显性能收益。
- earliest_ready 改为先安排可最早执行的已就绪任务，仍保持全部数学 DAG、
  逻辑分片及 MIX 配对。Q 投影在模型中能提前，但实测没有改善。
- b16_phase_memory_v1 / task_20261016_202018_36492382421：exit=0，全部精度/状态/保护区/图重放及 30 代信号 PASS；动态 889.78 μs，单 SPMD 1089.86 μs。
- b16_compiled_control_v1 / task_20261016_202404_384514429980：exit=0，全部精度/状态/保护区/图重放及 30 代信号 PASS；动态 901.92 μs，单 SPMD 1148.16 μs。
- b16_payload_cache_v1 / task_20261016_202739_40037602798：exit=0，全部精度/状态/保护区/图重放及 30 代信号 PASS；动态 903.88 μs，单 SPMD 1077.06 μs。
- b16_earliest_ready_v1 / task_20261016_203314_1112088253：exit=0，全部精度/状态/保护区/图重放及 30 代信号 PASS；动态 892.32 μs，单 SPMD 1115.12 μs。

以上候选均未替代 balanced_plain 最佳版本。没有为这些退步候选重跑全七档。

## 局部编译边界继续定位

- 对旧 C++ 数学替换失败日志复核，实际是 Simpler S1:running-stalled，
  completed=0/1，约 60 秒无进展后退出；不是已定位的硬件故障。
  调度未结束的具体核内位置尚未知。
- stage_codegen.py 从当前融合 IR 提取一个完整阶段（等待、数学、发布），
  使用同一组 GM 参数和明确传入的外层代际独立编译，再作 device 函数调用。
  仍只有一个 SPMD 运行时任务。首次仅选择 hc_widen_rms，CPU 编译进行中。
  此路径是归因实验，不能作为生产完成或 700 μs 达标结果。

## 20:55 编译边界、参数与轮询诊断

- stage_codegen_cpu_v1 因使用 name_hint 当作 SSA 身份，遇到重复 t__tile 而退出；
  v2 按 IRNode.same_as 区分变量，并隔离嵌套 pass instrument，完整 CPU 编译通过。
  widen helper 的 set/wait 各 34 处，GM 参数直接来自同一融合函数，外层代际显式传入。
- stage_codegen_compact_cpu_v1：只保留 helper 实际使用的参数，CPU 编译通过。
  它只缩减内部 device 函数参数，不改变外层运行时任务数量和 ABI。
- b16_independent_widen_v1 / task_20261016_204559_77126426784：已排队。
  使用完整参数的独立 widen，仅作归因，不用于目标验收。
- b16_workspace_base_v1 / task_20261016_204727_83145615808：已排队。
  按真实 arena 基址加偏移重建临时 GM 指针，运行前检查实际 data_ptr 对应关系；
  数学、调度及外层 ABI 不变，仅作跨阶段活跃指针成本诊断。
- balanced_ptoas_cpu_v1：使用已有 PTOAS 规划路径编译失败：
  Mat 需要 4456448 bit（544 KiB），超过 4194304 bit（512 KiB）。
  没有设备执行，没有缩小 Tile 或放宽容量。
- b16_balanced_lddev_v1 / task_20261016_205309_10618254230；
  b16_backoff_25_v1 / task_20261016_205309_106179220657：已排队。
  前者为同一 balanced 分配上的 ld_dev 读侧对照，后者仅在未就绪时等待
  25 个 SYS_CNT tick（当前 50 MHz 下为 0.5 μs）再读，首次读取不延迟。
  写侧 TNOTIFY 的 DCCI/DSB、业务数据维护及代际检查保持不变。
  假设是大量空闲核轮询可能影响前置小核数计算，未预记收益。
- stage_codegen_hot_cpu_v1 正在 CPU 检查四个热点的独立编译及实际参数裁剪；
  尚无该候选设备结果。

- stage_codegen_hot_cpu_v1：四个热点的参数裁剪与独立 PTOAS 编译全部通过，内部函数参数数为 9～13 个；完整数据见 stage_codegen_hot_summary.json。尚未提交四热点设备变体，先等待单 widen 结果。

## 21:10 排队实验完成

- b16_independent_widen_v1：全部精度、状态、保护区、图重放和 30 代信号检查 PASS；动态 894.2 μs，单 SPMD 1120.22 μs。
- b16_workspace_base_v1：全部精度、状态、保护区、图重放和 30 代信号检查 PASS；动态 905.08 μs，单 SPMD 1065.62 μs。
- b16_balanced_lddev_v1：全部精度、状态、保护区、图重放和 30 代信号检查 PASS；动态 897.08 μs，单 SPMD 1327.28 μs。
- b16_backoff_25_v1：全部精度、状态、保护区、图重放和 30 代信号检查 PASS；动态 900.1 μs，单 SPMD 1146.62 μs。

独立 widen、GM 基址重建仅作归因，不作为目标验收；GM 基址重建的 1065.62 μs 与原 1067.36 μs 接近，未证明明显收益。连续 ld_dev 与 0.5 μs 退避都未优于原等待方式，不扩大七档复测。

任务计数候选复用当前 A3 PTO-ISA TNOTIFY(AtomicAdd)：每个原逻辑 task 独立计数，消费者等待原 DAG 前驱全部完成。atomic 防止多写者丢更新，DCCI/DSB 承担发布职责；保留业务数据及本核流水屏障。先做依赖偏序等价核对，再验证设备；不预记收益。

- task_counter_prepare_v1：68 个源码阶段、973 个控制成员，任务计数等待 2149 项，原逐核等待 20435 项；全部祖先偏序完全一致，数学体和业务缓存屏障未改。每个逻辑 task 独占 512 B 行中的第二段，不与代际信号共用缓存行。
- task_counter_cpu_v1：完整 CPU/PTOAS 编译通过。运行时仍须验证 AtomicAdd、图重放及 66 个计数器代际。
- b16_earliest_stage_v1 / task_20261016_211142_18013885165：只补测已知退步调度的阶段时间，定位资源安排与实际数学耗时；插桩结果不作为目标验收。

## 21:16 任务计数与编译同步消除

- b16_task_counter_v1 / task_20261016_211233_181811726212：全部输出、状态、保护区、图重放、30 代单核标记及 66 个任务计数器检查 PASS。动态 L1 905.74 μs，单 SPMD 1235.48 μs；减少等待项没有带来性能收益，不采用此策略。
- earliest_stage 阶段探针通过。Q 投影确实从 632～700 μs 提前到 264～359 μs，但 Indexer 仍到约 659 μs，QK/PV 约 747 μs 才开始，前置任务与末尾投影仍有较长执行段。完整插桩结果为 1126.76 μs，不作为无插桩对照或目标验收。见 scheduling_stage_comparison.json。
- 只读 PTOAS 源码 lib/PTO/Transforms/InsertSync/PTOInsertSync.cpp：hasGatherScatterLikeOps 命中任一 gather/scatter 时，整函数跳过 RemoveRedundantSync。该条件未区分 A3/A5；注释说明是 A5 保守处理。当前融合函数含 gather，独立 widen 不含，这是局部流水差异的待验证原因，不能仅凭源码断言设备退步根因。
- stage_codegen_all_cpu_v1：准备将同一融合 IR 的全部阶段分别编译。完整 outline 后，外层只有 Scalar 分发/代际初始化，检查并去掉外层原数学的成对 auto-sync 初始化/回收，避免污染内部函数独立事件生命周期。每个 helper 原数学、跨核事件与数据维护保持。只在测试进程改写生成代码，不改 PTOAS 源码。

## 21:22 全阶段编译结果与后续范围

- stage_codegen_all_cpu_v1：68 个完整阶段独立规划/编译全部通过，内部函数仅有 8～26 个参数。外层旧数学残留的 set/wait 成对核对后全部移除，内部函数自己维护完整事件生命周期。
- b16_all_stage_codegen_v1 / task_20261016_211655_201090727888：全部输出、Top-K、cache/state、保护区、图重放、30 代信号检查 PASS；动态 901.72 μs，融合 1153.34 μs。仍较 balanced_plain 退步，不采用。
- ptoas_sync_phase_evidence/summary.json：实际 PTOAS 0.66 CPU 日志，整层 AIC 的冗余同步消除有执行（3188→772 对），整层 AIV 没有减少（2584→2584 对）；独立 widen 减少 46→28 对。EventId 分配会再改变数目，不能拿这一步计数当最终 set/wait 数量，更不能直接当性能收益。
- b16_all_stage_noinline_v1 / task_20261016_212120_215646228211：保留全部阶段的设备函数边界，核实 CCEC 重新内联后的影响。外层已没有数学流水事件；与此前旧版局部替换超时实验不同。待设备结果，不预记 PASS。

- b16_pipe_profile_v1 / task_20261016_212705_262039629706：只为当前 balanced 基线追加外部 PipeUtilization，计时完成后两侧各采三步，正确性和代际检查覆盖追加重放。完整持久核 PMU 与 L1 核内计时严格区分；不绕过 Simpler kernel 模式对 enable_pmu 的限制。
- 更完整的编译证据及限制见 PTOAS_GATHER_SYNC_SCOPE_DIAGNOSIS.md。


## 21:55：noinline 完成，外部硬件计数和实际 ABI 裁剪诊断

- `task_20261016_212120_215646228211` / `b16_all_stage_noinline_v1` PASS。
  动态 / 单 SPMD L1 中位数 893.54 / 1044.54 μs，三次单 SPMD 为
  1037.44、1044.54、1063.24 μs。全阶段内联版 1153.34 μs，保留设备调用边界
  更快，但尚未证明相对普通 balanced 的稳定收益；这个 C++ 组合诊断不用于正式验收。
- `task_20261016_212705_262039629706` / `b16_pipe_profile_v1` PASS。
  正常计时先完成，L1 动态 / 单 SPMD 中位数 892.56 / 1054.70 μs。
  单 SPMD 三窗 1054.70、3922.40、1040.26 μs，保留第二窗长尾，不能说稳定到 1054 μs。
  后追加每侧三个 PipeUtilization step，33 代进度、输出、Top-K、状态及保护区均通过。
- 外部 PMU 完整持久核中位数：AIC MAC 189.587→189.167 μs，AIC MTE2
  357.960→464.451 μs，AIV Vector 186.749→189.997 μs，AIV Scalar
  470.434→714.641 μs。以上覆盖外围运行时，流水间可重叠，不能直接相加解释 L1 差值。
  指令缓存 miss rate：AIC 0.002→0.006，AIV 0.003→0.013；按工具原字段值记录，
  没有仅凭比率认定 ICache 为瓶颈。原始 CSV 与 `pipe_utilization_summary.json` 保留。
- 新增 `compact_abi.py` 仅诊断实际入参：130 个根 / 132 个叶子参数收敛为
  59 / 61，其中工作区视图合并为一个 arena。Native 权重、state 和缓存不搬运。
  必须核对同规格源 AST、全部工作区及控制表一致，并在设备启动前核对实际指针。
  原先 workspace_base 只是指针重建、没有减少实际传入参数；这次才验证 ABI 部分。
- CPU 准备 v1/v2 发现保存的包装 C++ 已把 AICORE 改写为 static __aicore__；
  修正严格签名匹配，不修改数学体。v3 导入成功但 mixed extern 多输出返回在 Group
  解析中报 TupleGetItemExpr/UnknownType。v4 改用公开 void/in-place extern，仍以
  InOut 标注全部写入，不改 PyPTO。CPU 编译和设备结果待实际完成后追加。


## 21:58：实际参数压缩设备对照完成，无收益

`task_20261016_215358_37103827122` / `b16_compact_abi_v1` PASS。
CPU v4 已通过 kernel artifact 编译，使用 mixed extern 的 void/in-place 形式，
全部写入通过 InOut 声明。设备启动前 71 项工作区地址关系逐项正确；实际根参数
59 项、叶子参数 61 项、72 个 workspace 视图共用一个 arena，无新增 copy。
输出、Top-K、六份状态、图重放、30 代进度、Native/工作区保护区全部通过。
L1 动态中位数 907.10 μs，单 SPMD 1108.48 μs（1108.48/1111.24/1084.90）。
Event 中位数 957.820 / 1186.610 μs。没有整体收益，不切换生产入口。

这只能排除“本次具体参数压缩实现已经有效”，不能证明所有参数/寄存器开销为零。
该实现使用原 PTOAS C++ 和公开 extern，仅为诊断，仍不计入正式 PyPTO 目标验收。

追加同规格实测阶段成本候选：用已有 stage_clock 的数学+维护+通知中位数，
排除等待时间；严格匹配所有根 shape、全部 task/slot；离线重新分配时使用。
不在运行时读取 profile，不改变数学体、依赖偏序或业务 device metadata。

工具记录：本轮两次误用 `task-submit list/status` 被解释成无设备 shell 任务，
编号 task_20261016_214559_336266930669、task_20261016_215431_372084912112，
均 exit=127；没有占 NPU。后续查询已使用正确的 --status/--list 参数。


## 22:03：实测成本重新分核完成，无整体收益

`task_20261016_215905_38576944603` / `b16_measured_schedule_v1` PASS。
CPU 四个成本源/策略组合只用于选择候选，未把估计作为性能：最后选用
同规格 earliest_stage 数学+维护+通知中位数，earliest_ready 策略。
估计 883.40 μs；真正 L1 动态 885.36 μs，单 SPMD 1109.76 μs，三窗
1096.72/1109.76/1357.80 μs。Event 中位数 945.150/1211.380 μs。
相对原 balanced 没有收益，保留负结果；未切换默认策略、未重复七档测试。
等待项 20435→17546，发布 729 次，逻辑分片、MIX 配对和依赖偏序均检查通过。
实际设备输出、Top-K、状态、图重放、信号及保护区均通过。

新增 `core_specialization.py` 归因：结合此前 compiled_control，把物理 group/lane
作为 C++ 模板常量，24 个 AIC 和 48 个 AIV 分别常量折叠各自的调度条件。
仍只有一个运行时 MIX SPMD；长度、位置、页表等业务 Tensor 不特化。
数学/信号 C++ 函数体逐字保持，只替换核编号参数来源并增加分派外壳。
目的在于检验无关阶段判断、控制标量与代码活跃范围的影响。CPU 代码生成与二进制
编译必须先过，再决定设备验证；不是已测收益，也不是最终生产 PyPTO 接口。

另只做了一次 GM 临时区复用可行性分析：按完整 task-DAG 的严格先后关系，
294434944 B 可保守分组到 168272384 B，未修改布局、未执行复用。
空访问 temporary_47 保守独占，没有基于“未找到调用”删除它。其余未完整审查的
生命周期、返回别名和缓存语义必须进一步分析，不能直接据此宣称安全或性能收益。


## 22:23：核身份全展开的编译成本与原数学核完整组合

上一轮属于实际进展：完成两组设备负结果及新的核身份特化源码。当前活动目标没有完成。
CPU `core_identity_cpu_v1` 代码生成通过；`core_identity_binary_v1` 中 AIC 编译/链接完成，
代码段 1532220 B（ELF 含调试信息 24324936 B，不能混成指令体积）。
AIV 的 ccec 进程持续运行，最终由原编译器 900 秒超时终止，exec session 25592 exit=1。
这是已确认的编译超时，不是 NPU 挂死；未占卡、未提交该候选设备任务。没有因日志不动重启。
保留源码和部分 AIC 产物，不反复重编全部数学体；后续优先共享原数学函数，只简化外层控制。

新增 `reuse_identical_core_binaries`：设备候选仅在源 C++ 逐字相同、现有编译上下文相同
时复用已完成 CPU 编译的 ELF；其余回退正常编译。复用不增加 hash，不绕过运行时注册、
编译上下文检查或二进制解析。本次超时候选没有可复用的完整产物。

全部原数学函数组合：`preserve_all_prepare_v2` 已逐项映射 68 个物理阶段的 GM 指针、
scalar、shape/stride、整除恢复维度、block_num 和 MIX lane。和早前四个局部数学替换不同：
全部数学核保留原动态 PTO 编译出的局部流水，外层仅留控制、等待/通知和 payload 维护；
严格成对清理外层原数学 auto-sync 的初始化/回收，避免事件生命周期混用。
还保留原物理包装器每次调用 `set_atomic_none()` 的约定。每个原数学 helper 用 noinline，
共用代码，不为 72 个物理核复制全部数学体。此阶段仍为 C++ 诊断，不是生产接口。

`preserve_all_prepare_v1` 初次映射不支持动态 shape 的 /32 恢复表达式，已补充仅允许
明确 shape/divisor 且精确整除的映射，没有对任意 C++ 字符串执行 eval。
`preserve_all_cpu_v1` 因命令填错历史 build 目录而失败，已保留错误记录；改为实际存在的
`_decode_csa_tp1_layer__lncpugg` 后启动 v2。该输入路径错误不归因于编译器。


## 22:34：完整原数学核设备通过；继续定位等待/计算开销

`task_20261016_222628_28521515432` / `b16_preserve_all_v1` PASS。
动态/单 SPMD L1 中位数 891.18/1086.06 μs，单 SPMD 三窗
1086.06、1098.00、1062.96 μs；Event 中位数 956.650/1188.480 μs。
输出、Top-K、六份状态、保护区、图重放和 30 代信号均通过，仍无整体收益。
本结果不能单独证明旧四阶段替换失败的唯一原因，但完整原数学核的事件生命周期
清理、调用入参和单任务组合已获本规格真机验证。

CPU preserve_all_binary_v1 用 28.54 秒完成两类核和调度二进制；AIC 48728 B、
AIV 122656 B（代码段）。预编译复用未命中，走正常编译，未使用错误二进制。
复查发现缓存 JSON 外层含 context/files，复用检查初版拿外层和 context 字段比较，
造成保守拒绝，已修正解析。源文件也逐字不相同，首批差异为 IR 内联编号注释；
仍不放宽源码一致条件，不把未复用描述成已复用或直接归因于 SDK 不兼容。

已提交 `task_20261016_223130_37234825791`，b16_preserve_all_stage_v1：
相同完整原数学核组合，加阶段 SYS_CNT，区分等待、数学/维护和通知，只用于归因。
另提交 `task_20261016_223226_39023724918`，b16_cached_backoff_50_v1：
保留 PTO-ISA DCCI+volatile GM 的读侧语义，只在信号未就绪时以 SYS_CNT 间隔 50 tick
（1 μs）重新轮询；写侧 DCCI+DSB、PIPE_ALL 和 payload 维护不变。
这是检查轮询争用的独立候选，与此前 ld_dev 绕过缓存的退避不是同一种读法。
二者结果尚待完成，不能预记收益。


## 22:39：DCCI 轮询退避首次取得明确收益

`task_20261016_223226_39023724918` / `b16_cached_backoff_50_v1` PASS。
完整 L1 动态 906.52 μs、单 SPMD 894.10 μs，单 SPMD 三窗
894.10/896.94/892.28 μs；Event 动态956.480/单SPMD999.510 μs。
L1 已接近同轮动态，但 Event 仍更慢，两个口径分别保留。输出、Top-K、
六份状态、图重放、30代信号和保护区全部通过。固定fixture，不扩称连续decode。
只在单元素 GE 信号尚未就绪时退避50个SYS_CNT tick（1 μs）；每次读前DCCI，
写侧和业务数据维护不变。该结果不能单独区分 helper 与退避贡献，故下一步
增加 cached 0 对照、100tick候选和50tick阶段探针，不重复七档负候选。

`task_20261016_223130_37234825791` / `b16_preserve_all_stage_v1` PASS。
插桩 L1 动态902.32/单SPMD1094.30 μs，不计入正式目标。原数学函数下仍有
较长前置阶段：hc_widen_rms数学/维护中位22.44 μs，Indexer约377.56 μs
开始，qproj矩阵乘被本分核安排到Indexer之后。此处先只作为定位证据；
不能把每核等待相加当作整层耗时，也不能认定只改数学编译就能解决。


## 22:44：退避对照与原动态实测成本

同一 cached helper 的0/50/100 tick版本均通过。0 tick
`task_20261016_223952_51986417600`为动态902.46/单SPMD1047.58 μs，
50 tick为906.52/894.10 μs，100 tick
`task_20261016_223952_51980931381`为907.02/851.32 μs（三窗850.94/851.32/854.06）。
这支持退避本身有显著贡献；不能仅从时延推断具体L2/互连争用机制。
100 tick的Event为951.840/951.610 μs，完整正确性、状态和保护区均通过。

50 tick阶段探针 `task_20261016_223952_5198234964` PASS，插桩中位892.60 μs，
只作归因。widen数学/维护中位13.51 μs；Indexer约233 μs开始，
主Q投影约444 μs开始，QK/PV约542 μs开始。与无退避探针相比前置
数学阶段也缩短，仍需要区分实际争用与代码生成影响，不先认定硬件瓶颈。

新增 `export_dynamic_cost.py`：从已完成原动态三窗L1提取同规格70个task/slot
成本。严格核对task ID、物理核类型、核体名称、root shape和完整覆盖；
第三列正确当作dispatch_seq，不误认kernel_id。数学成员内部等待仍包含在成本中。
离线模型：critical_path802.51/earliest_ready773.43 μs，不是设备实测；
仅选择后者配50tick重新对照，任务`task_20261016_224301_58928227892`。
业务长度/位置等仍是device metadata，各batch允许自己的分核和二进制。


## 22:49：退避达到平台，检查静态控制和等待热点

200tick task_20261016_224348_60377711059 PASS，L1动态894.12/单SPMD848.38μs，
单SPMD840.16/848.38/858.60。400tick task_20261016_224348_60385920731 PASS，
909.26/846.56μs，846.56/853.76/846.32。较100tick的851.32μs只差几微秒，
不宣称稳定改善；停止继续扩大统一退避间隔，组合候选沿用100tick（2μs）。

原动态成本+earliest_ready+50tick task_20261016_224301_58928227892 PASS，
892.68/855.36μs（859.20/855.36/854.94）。比旧分核50tick894.10更低，
据此提交同分核100tick task_20261016_224732_68404112819。
另提交原balanced+100tick+静态控制表 task_20261016_224732_68414320903，
检查轮询开销下降后，消除调度GM读取是否有用；尚未预记收益。

新增rotated_waits.py只在初始化时重排每个消费者的等待表：按core号循环旋转，
使首个轮询地址分散，仍遍历完全相同的前驱集合，信号不清除。
20435条依赖逐项多重集合一致，714个成员改变次序；kernel源码、记录表和数学体不变。
CPU原协议验证器的9组随机顺序/代际全部通过。第一次CPU执行因脚本多一个右括号
报SyntaxError，修正后通过，不归因于PyPTO。设备结果待后续记录。


## 22:59：七档退避+独立成本分核完成

见 CACHED_SEVEN_RESULTS.md / cached_seven_results.json。七档全部PASS，
L1均较同轮原动态低2.89%～8.54%，B16当前832.28μs仍未达700μs。
保留Event差异：多数档Event接近或仍略慢，不能宣称整网端到端均加速。

B16静态控制表 task_20261016_224732_68414320903 PASS874.24μs，
等待顺序旋转 task_20261016_224904_7179821170 PASS839.48μs，
未超过实际成本+earliest_ready+100tick的832.28μs，不设为默认策略。
新阶段探针 task_20261016_225258_84749429745 PASS，插桩830.74μs仅用于定位。

CPU另扫描16个就绪时间/剩余关键路径权重，未优于原earliest_ready；
结果b16_schedule_weight_search.json，所有分片、MIX配对和依赖偏序证明通过，
未把这些CPU负候选提交NPU。原动态成本的资源/关键路径下界只是估算，
不能据此承诺700μs或宣布硬件已到极限。

继续两项B16同步候选：只保留有Scalar业务读写的DCCI（已有独立审查与旧版验证），
以及保持数学偏序的完成通知汇聚，在有效退避基础上重新检查其效果。
其他六档不随之重复测试，除非B16有明确收益。汇总脚本初版的命令示例字符串
转义报SyntaxError，改成单行命令后完成；未影响已完成的设备结果。


## 23:01：退避后裁剪缓存维护和完成汇聚没有额外收益

`task_20261016_225624_9520089233` / b16_cached_dynamic_precise_v1 PASS：
动态898.72/单SPMD838.68μs，三窗845.16/838.68/826.80，Event954.870/940.560。
`task_20261016_225727_97074417001` / b16_cached_dynamic_aggregate_v1 PASS：
动态906.82/单SPMD860.30μs，三窗870.28/854.96/860.30，Event959.590/966.030。
未超过已测832.28μs；不把这两个同步变体设为默认、不重测其他六档。

CPU用有效退避下的新阶段数学+维护+通知成本再分核：原earliest_ready估计760.19μs，
加0.05关键路径权重估计754.30μs；这仅选候选，不是实际提速。后续继续用CPU
检查前置主Q投影优先级能否减少资源串行，再选择值得验证的最少设备候选。


## 23:10：新的实测成本仍未更优，进入原有列分块候选

新阶段成本+0.05权重 `task_20261016_230307_106795216098` PASS：
动态909.60/单SPMD863.66μs，三窗871.50/863.66/863.56，未超832.28μs。
CPU关键路径主Q投影偏置扫描最好估计758.78μs，也未优于已测试模型，未提交NPU。

新增oproj_retile.py：从冻结原CSA读取已有proj_a_mm_2（N256）核体，
逐AST节点核对相对N128只改列索引、列分片数及局部Tensor列维。
原有行128、K256、K归约顺序、dtype和cache策略不动；重新绑定原入参名。
B16每个O-A任务8→4逻辑分片，8个group仍覆盖全部1024列。
oproj_n256_body_check_v1和oproj_n256_prepare_v1通过；后者包含分片、MIX配对、
等待偏序、工作区与9组CPU代际模拟。使用相同输出缓冲，不改变原动态参考。

这是首次尝试改变列Tile策略，不再宣称数学核体逐字不变；exact门槛保持，
不能从实数等价或归约顺序一致直接预记浮点逐位通过。新成本尚未测量，
离线初排显式按N列数2倍估计原成员成本，标注unmeasured_duration_scales。
CPU完整代码生成正在oproj_n256_cpu_v1执行，成功之后才提交B16设备验证。


### N256 CPU v1参数重绑定错误及v2修正

oproj_n256_cpu_v1前端拒绝未定义t_dim__ssa_v0。原因是本地AST等价核对
把用于执行的同一份AST也做了去inline编号规范化，后续按原名字重绑定失效。
改为仅对深拷贝做规范化比较，执行AST保留原名；补充函数体自由变量必须
完全由原参数和局部赋值绑定的检查。不是PyPTO bug，不修改依赖仓。
oproj_n256_prepare_v2通过全部CPU协议检查，CPU完整编译v2继续执行。
原v1源码、失败日志和result均保留，不覆盖。


## 23:12：N256 CPU完整代码生成通过，提交B16设备候选

oproj_n256_cpu_v2完整PyPTO/PTOAS代码生成通过；未把CPU通过称为设备通过。
没有编译器或runtime新改动。设备候选沿用原动态成本初排、earliest_ready、
100tick退避，仅新增O-A N256列块；原exact输出与状态检查保持。


## 23:16：N256输出投影真机逐位通过，未形成明确整体收益

`task_20261016_231256_12244625650` / b16_cached_oproj_n256_v1 PASS。
原动态N128中位911.72μs、单SPMD N256为839.00μs，三窗840.92/839.00/838.60；
Event961.780/942.670μs。输出、Top-K、六份cache/state、图重放、30代信号及
全部保护区仍按原门槛通过。不同列块并没有在本规格引入可见逐位差异。
但839.00没有低于此前N128的832.28，且原动态同轮也有波动，不宣称这项列分块
有稳定的额外收益；不切换七档策略、不扩大重测。原N128和N256代码与全部证据均保留。

当前没有未完成的设备任务或CPU编译会话。七档性能汇总以CACHED_SEVEN_RESULTS.md为准，
700μs目标未达到，活动目标保持继续。下一步应依据阶段/生命周期证据继续检查，
不要重新扫描统一退避间隔或重跑本轮已经完成的七档。


## 23:27：临时GM等大小复用的CPU证明通过

新增gm_reuse.py，只用完整task-DAG证明所有旧使用者先于全部新使用者。
不增加依赖，不使用trace时间或单核偶然先后；FFTS temporary_50/62、
未知用途temporary_47和全部控制区独占。PyPTO入口拒绝部分可写重叠，
因此本轮仅允许字节范围完全相同的别名，未改变依赖库的校验。

gm_reuse_prepare_v1：arena从294996992B缩至269945856B，节省25051136B。
生成源码AST及逐核安排与当前832.28μs版本完全相同；没有数学或同步语句变化。
新增最终布局检查，旧workspace_regions_disjoint仅标注baseline，不能冒充复用后互斥。
控制区重叠、部分重叠和同大小但使用者无严格先后均被拒绝。
逻辑scratch的调用后中间值不再保持，正式输出/TopK/Native状态不参与复用。
CPU检查不替代硬件正确性，接下来仅提交128K/B16，保留原exact门槛。


## 23:33：GM复用真机PASS，性能持平

task_20261016_232922_148038020750 / b16_cached_gm_reuse_v1 PASS。
完整L1原动态908.64μs、单SPMD833.28μs，Event955.460/938.520μs。
输出、TopK、六份cache/state、图重放、30代信号及保护区通过。
相对先前832.28μs无明确性能收益；保留其节省25051136B内存的独立候选，
不改七档性能默认策略、不重跑已经通过的七档。

主Q投影原为20worker处理128个列块，在静态图中有独立资源排程，
新增qproj_partition.py允许24/32/64/128个逻辑worker候选，保持24物理AIC上限。
仅改循环次数及列块编号，K256四次INT8点积和数学体逐AST相同；
CPU枚举保证128个列块各一次。首次只准备24worker，不盲目四档全上设备。


## 23:41：主Q投影CPU候选与QK地址计算方向

qproj24_prepare_v1协议和qproj24_cpu_v1完整代码生成通过；并非设备通过。
模型扫描v1误将重分核后的schedule当作原trace anchor，报KeyError end；
v2使用冻结anchor修正调用，不修改调度器或依赖仓。
qproj_partition_model_v2估计20/24/32/64/128worker分别
773.43/782.65/774.52/778.34/777.95μs，均无改善依据，未提交NPU。
新粒度成本按20/workers粗估，不能作为实际性能结论。

QK/PV压缩KV的64行循环每行Scalar读稀疏索引、再读页表。新增
qk_index_cache.py候选：MTE2批量读取索引和当前请求页表，Vector整数
位移+gather计算物理行，Scalar改读本核UB；负索引分支、逐行KV DMA、
QK/PV与softmax数学及FFTS同步不变。只改本实验body_bundle副本。
页表仍是原device Tensor，每块都重读，没有Host取值或GM展开metadata。
页表尾部使用显式valid_shape，未把1026列当成1032列实际数据读取。


## 23:45：QK地址缓存完整CPU代码生成通过，提交单个B16

qk_index_prepare_v1：20016项CPU地址公式检查通过，包含负索引与页边界；
qk_index_cpu_v1完整代码生成PASS。原核体集合只有qk_pv_aiv被改写，
其余函数AST不变，完整调度与工作区分配不变。
提交task_20261016_234523_17348223934 / b16_cached_qk_indices_v1，
采用832.28μs版本的调度与100tick轮询，未叠加GM复用或主Q重分片，
完整exact/图重放/状态/保护区检查及3窗L1保持。尚未获得设备结果。


## 23:49：QK地址缓存B16完整PASS，L1中位815.84μs

task_20261016_234523_17348223934 / b16_cached_qk_indices_v1 PASS。
单SPMD三窗829.04/815.84/813.34，中位815.84μs；原动态中位902.80μs。
Event原动态957.650/单917.870μs。完整输出、TopK、六份状态、图重放、
30代信号和保护区逐位检查通过。相比原832.28中位少16.44μs；
这是跨轮单SPMD比较，同轮动态对照均保留，不把三次测量当大样本统计结论。

因本改动影响所有规格，提交其余六档完整检查，任务清单qk_indices_matrix_jobs.json。
同时提交B16 stage_clock定位本次收益；探针结果明确不计入700μs门槛。
没有重测未改变的旧七档基线；每份新任务内只运行原动态PTO对照和本候选。


## 23:55：七档地址缓存验证完成，按档选择而非统一启用

七个无插桩规格及B16阶段探针全部PASS，任务状态completed(exit=0)。
QK新版本128K B4/B8/B16/B24为541.84/641.56/815.84/1137.90μs；
8K B16/B24/B32为632.58/865.16/943.20μs。
对照旧单SPMD，B4持平，B8/B16改善，其他四档退步；没有删除负结果。
选择只在128K/B8/B16保留新地址缓存，其余五档沿用旧实现。
各档采用既有独立调度，不要求跨batch或策略共用二进制。
完整数据QK_SEVEN_RESULTS.md；当前所选数据SELECTED_SEVEN_RESULTS.md；
机器可读选项selected_shape_policies.json（不是生产入口自动切换）。

B16探针task_20261016_234943_18100881575完整PASS但插桩L1 830.12μs
不进入目标门槛。原/新QK/PV AIC成员数学+通知中位155.82/153.85，
AIV160.58/158.41μs；Indexer AIC209.88/212.10、AIV202.83/202.31μs。
探针变化小于无插桩跨轮差异，不能把完整16.44μs全部解释成地址缓存收益。
探针本身会改变指令、寄存器和计时，当前只作定位；没有用它宣称<700。

当前无未完成设备任务或CPU编译会话。B16最好无插桩815.84μs，
700μs目标仍未完成。原七档、当前七档都已完成，不应无改动再次全量重测。


## 2026-10-17 00:13：Indexer N128首版因L0C容量失败，未上设备

indexer_n128_prepare_v1通过绑定与CPU协议检查，分核、调用、根描述符、
等待表均与现选B16相同。AST审计首版脚本只看module.body，漏掉类内函数
而断言失败；修正为ast.walk后确认仅Indexer AIC变化，其余函数不变。
完整CPU代码生成indexer_n128_cpu_v1失败：Acc204800B超过平台131072B。
理论左/右内存容量足够不代表Acc足够；未改平台限制或依赖仓，未上设备。
原N128/M384的QK本身需要192KiB Acc，不能采用该形状。保留失败生成物。
下一版将每个N128面板分成两个M192 QK，仍让六query共用一次Key加载；
WS分别规约三个query，Right占48KiB，仅保留一个16KiB Key Right，
下一面板只预取到L1。峰值Acc理论104KiB，仍需实际编译验证。


## 00:16：Indexer N128两组三query版CPU通过并提交B16

indexer_n128_prepare_v2 / indexer_n128_cpu_v2完整代码生成通过。
新QK为M192/N128/K128 INT8，WS为M16(有效3)/N128/K192 FP16，
同一个Key Right先后供两个QK使用，下一Key仅预取L1。原分组、
通信双槽和AIV处理不变，没有新增GM缓存或Host数据读取。
任务task_20261017_001600_235577716532，输出b16_cached_indexer_n128_v2，
现选B16调度+QK地址缓存+100tick等待，只额外启用--indexer-n128。
已提交，待设备正确性与三窗L1；未预记PASS或收益。


## 00:21：设备排队期间，准备QK全有效块搬运候选

当前16卡被另一个整机任务占用，N128任务仍pending，没有绕开task-submit。
检查现选B16生成C++可见：64行循环每行仍读一次raw索引、一次物理行，
并执行列边界/负索引判断；没有逐行额外pipe barrier。新增qk_full_block.py：
当前64行raw索引先做INT32 row_min，全部非负时展开8行一组搬运，只读
物理行；含负索引时完整保留原回退。范围判定来自device，不固定历史长度。
不改变数学、FFTS、页表或GM工作区。先做CPU编译，尚未提交设备。
qk_full_prepare_v1通过进程内组合构建，未修改运行中/排队任务的导入模块。


## 00:23：N128设备PASS但性能退步，保留N64

task_20261017_001600_235577716532完成exit=0，所有exact/状态/TopK/
图重放/信号代际和保护区通过。单SPMD L1三窗881.60/883.00/897.02μs，
中位883.00μs；同轮原动态889.42/903.92/884.82μs，中位889.42μs。
对比现选815.84μs没有收益，不扩七档、不设默认。两组三query共享Key
数学正确，但N扩大不自动带来收益；QK/WS调用数不变且失去第二个L0B
预取槽，不能仅按面板宽度宣称更快。未采阶段探针，不将退步精确归因。

QK全有效块CPU首版qk_full_cpu_v1在PTOAS失败：row_min的1x1 INT32
列主序结果只有4B，未满足32B对齐。v2把同一64行索引复制为8个相同行
再规约，结果8x1满足对齐，仍只读第0行；未改编译器，完整CPU编译中。


## 00:24：QK全有效块CPU v2通过并提交B16

qk_full_prepare_v2/qk_full_cpu_v2完整代码生成PASS；使用8x64 INT32
规约输入，8x1结果满足32B对齐。全部input/工作区/等待表保持。原回退
循环在还原结果变量名后AST逐节点一致，8种block/lane组合均覆盖且仅覆盖
当前64行，列索引处于0..511。以上不是设备fallback分支验证。
提交task_20261017_002425_25613488606 / b16_cached_qk_full_v2，
仅在当前815.84μs配置上加--qk-full-blocks，不叠加N128候选。
现有七档性能策略仍保持，未无改动重测。


## 00:28：QK全有效块设备PASS但未获收益

task_20261017_002425_25613488606完成exit=0，完整exact/状态/TopK/
图重放/信号代际与保护区检查均通过。单SPMD三窗840.78/839.56/829.12μs，
中位839.56μs；同轮动态894.94/887.30/901.92μs，中位894.94μs。
未超过现选815.84μs，不扩七档，不设默认。新块级规约和展开改变流水/
代码量，不能仅用减少分支数量推断收益；当前未做精确阶段归因。

现有stage_clock总时长无法区分Indexer和QK/PV的Cube/Vector相互等待。
新增internal_wait_clock.py，仅当前测试进程包装4个对应阶段里的
wait_flag_dev，累计Scalar等待SYS_CNT/次数，并记录该阶段耗时。使用每核
原512B进度区的第2/3条缓存行，不碰信号行，末尾DCCI+DSB发布观测值；
不改变原同步、数学或共享安装。只作诊断，不计入700μs性能目标。
CPU变换审计internal_wait_probe_cpu_v1识别Indexer AIC/AIV的2/1个静态
等待点、QK AIC/AIV的2/2个等待点。其他流水可在Scalar等待时工作，
所以累计等待不是整核空闲，也不能直接当成可消除时间。


## 00:32：提交一次内部FFTS等待归因

任务task_20261017_003153_273890930040 / b16_internal_wait_clock_v1，
基于现选815.84μs的N64+QK地址缓存，不启用两项退步候选，仅增加
--internal-wait-clock。保留全部正确性门槛；插桩结果不参加达标判定。


## 00:37：HC行分工只读审阅与记录归档

已有B16阶段探针中hc_widen_rms与mix_x_rms_norm各12个Vector分片，
数学+维护+发布中位13.48/15.29μs。记录2/4行分块的可行性和约束于
HC_ROW_SPLIT_REVIEW.md：不能复用唯一tail区给多个新满块写；所有输入
和输出有效形状须相应更新，inv_rms不得跨块覆盖，数学K顺序保持。
尚未实现、未编译、未测性能；不把12到48分片当成4倍加速承诺。
本轮文本已镜像到canonical同名目录，大生成物仍在独立worktree，
LOCAL_ARTIFACTS.json只有路径和大小，不做hash。


## 00:39：内部FFTS探针PASS，定位两段的主导侧

task_20261017_003153_273890930040完成exit=0；全部exact/图重放/状态/
TopK/代际/保护区通过，eligible_for_700us_target=false。探针完整L1
837.10μs，只作旁证，不替换无插桩815.84μs。末次重放按成员中位数：

|阶段|侧|阶段耗时μs|Scalar内部等待μs|等待次数中位|等待比例中位|
|---|---|---:|---:|---:|---:|
|Indexer|AIC|214.75|16.34|22|7.70%|
|Indexer|AIV|200.52|74.37|22|37.34%|
|QK/PV|AIC|150.54|109.97|40|73.10%|
|QK/PV|AIV|156.71|4.64|40|2.95%|

证据b16_internal_wait_clock_v1/internal_wait_clock/waits.json，保留每核
原始值和C++插桩审计。比例为逐核比例的中位，不是两列中位相除。
该计时包含读SYS_CNT的开销；Scalar等待期间其他pipe可能运行，且一次
探针样本不足以构成稳定性能统计，不能把等待全部当成可消除或整核空闲。
但它支持优先检查Indexer Cube侧、QK/PV Vector侧的工作和相互供给。
下一步可将QK/PV AIV再分KV搬运、softmax、PV合并/输出几段定位；
无需再扫统一通知退避或原样复测N128、全有效块、七档。

本轮所有CPU会话和三个设备任务均结束；没有待收取任务。当前七档选择
不变，B16最好815.84μs，700μs目标未完成。


## 2026-10-17 00:50：QK AIV分段探针通过，定位编译器插入的交接等待

新增qk_phase_clock.py，在原生成C++的KV搬运、softmax、PV合并、
最终输出分支内累计SYS_CNT，不插入中间pipe屏障。CPU边界审计
qk_phase_probe_cpu_v1通过；task_20261017_004440_298994623828 /
b16_qk_phase_clock_v1完整PASS，eligible_for_700us_target=false。
末次重放48个AIV成员中位：阶段161.82μs，KV58.73/20次，softmax12.36/20次，
PV合并2.88/20次，最终输出14.92/4次。各段允许异步pipe在段外完成，
不能把差额直接等同某项开销，也不能作为未经插桩性能承诺。

qk_phase_probe_cpu_v1/uncovered_sync.txt保存未包入四段的同步指令：
每轮含两处PIPE_ALL及多处跨pipe交接，邻近有效块metadata的GM Scalar读。
新增qk_mask_cache.py，将本核query的有效块标记每次从原device Tensor
预读UB；B16为4x16 INT32共256B，只改AIV内6处读取来源，保留判定、
数学与FFTS。不直接删除任何同步，不新增Host读取或GM展开缓冲。
首个准备脚本误设读取点数量为7而断言失败；逐AST核对确为6处，改正后
准备qk_mask_prepare_v2。实际编译是否减少PIPE_ALL和性能均仍待验证。


## 01:10：mask候选无屏障收益；确认事件槽回退并提交MGATHER

qk_mask_cpu_v1完整CPU代码生成PASS，但原/候选QK tick内PIPE_ALL均4处，
FFTS wait/set均2处。qk_mask_prepare_v2/address_audit.json覆盖187200项
本核query映射；没有设备性能结论，不因CPU通过就采用，不扩大七档。

qk_sync_diagnostic_v1的module-scope打印要求禁用线程，v2尝试的参数
被PTOAS 0.66拒绝；v3去掉module-scope但debug3输出过大，主动终止
(exit143，日志本地保留)。v4用debug2完整通过。实际原最佳B16 IR在
EventId Allocation之前的依赖2708/2709/2712为V→MTE2，2714为MTE3→V，
之后回退为PIPE_ALL；另2687/2688/2689在PV入口回退后合并。
原始IR内无这四处显式屏障。证据qk_dependency_nodes.json/qk_fallbacks.json，
不能直接删这些UB复用依赖，也不能用静态屏障数直接解释所有未测耗时。

新增qk_mgather.py：已有本核UB物理行地址基础上，用MGATHER(row,zero)
替代64次PyPTO逐行控制。负稀疏索引用physical|min(raw,0)保留符号位，
在物理行数<2^31的前提下是越界哨兵；合法行地址不变。ISA仍逐行DMA。
prepare_v1仅qk_pv_aiv变化，其余72个函数不变，60项符号边界CPU检查通过。
CPU v1因保留跨SWA/压缩分支的64KiB初始零Tile，UB200960>188416B失败。
v2将零Tile作用域收至SWA分支，压缩分支由MGATHER完整写出，CPU完整通过；
生成QK区段PIPE_ALL为5处而不是原4处，不宣称减少同步。

任务task_20261017_010828_337615615012 / b16_cached_qk_mgather_v2，
使用当前最佳的earliest_ready/cached100/QK地址缓存，只加MGATHER。
额外qk_mgather_boundary.py在计时外检查64x512 BF16全有效、混合负索引、
全负索引；固定地址图重放时更新device索引并检查输出保护区。
保留整层原exact/TopK/六状态/图/代际/保护区门槛，结果待收取。
没有修改依赖仓或生产入口。


## 01:14：MGATHER完整PASS，无性能收益；准备HC行分工候选

task_20261017_010828_337615615012完成exit0。独立64x512 BF16边界检查
在同一图更新device索引，全有效/混合负索引/全负索引均exact和保护区PASS。
完整CSA输出、TopK、六份状态、图重放、30代信号、Native/工作区保护区PASS。
无插桩L1单SPMD三窗827.18/821.62/820.38μs，中位821.62；同轮原动态
902.20/911.62/911.18μs，中位911.18。没有优于现选815.84，不采用、
不扩七档；证据qk_mgather_result.json和原任务结果。

沿HC_ROW_SPLIT_REVIEW.md实现hc_row_partition.py：只改变hc_widen_rms
与mix_x_rms_norm的逻辑行分片，物理8行Tile及K顺序保留；2/4行可分档。
B16/T96的两个阶段各12→48成员；widen_rows、block_num、member_count
一起更新，再重建分核/完整等待表。显式限制inv_rms有效写行数，pre/mixed
load使用有效行，满块判断改到新行数，避免多核共写唯一tail区。
候选仅允许T%8==0，其他规格仍走原8行实现，不将padding初始化混入本实验。
Scalar循环不随行数缩短，初始调度成本仅估计为原0.5，待实测；没有承诺4倍收益。
hc_rows2_prepare_v1正在准备，未编译/未提交设备，默认hc_rows仍8。


## 01:19：HC行分片CPU通过，提交B16

hc_rows2_prepare_v1脚本把kernel_names误当单元素列表，实际为带None的
三槽位数组，导致自己的断言失败；v2按非空kernel名集合匹配。
hc_rows2_prepare_v2与hc_rows2_cpu_v1完整CPU/PTOAS生成PASS。
address_math_audit.json确认仅两个函数变化，所有for/pipeline表达式相同，
T=8..384步8、每档2/4行共96种行映射逐行无遗漏/重复。

任务task_20261017_011729_35529046475 / b16_cached_hc_rows2_v1已提交，
保留当前最佳earliest_ready/cached100/QK地址缓存，只启用--hc-rows 2。
没有叠加MGATHER等退步候选；全部原正确性门槛保留，设备结果待收取。


## 01:23：HC 2行分片完整PASS，但整层退步

task_20261017_011729_35529046475完成exit0；整层exact/TopK/六状态/
图重放/30代信号/Native及工作区保护区全部PASS。单SPMD无插桩L1
842.48/843.20/850.10μs，中位843.20；同轮原动态893.34/889.28/892.56μs，
中位892.56。未优于现选815.84，不采用、不扩七档；不能把核数增加当收益。
保留--hc-rows 2/4为实验开关，默认8；本轮没有测4行，也不冒称已通过。

新增indexer_page_views.py候选：原N64的32个Key页加载奇偶分支，分别
使用未偏移和偏移64B视图。改为从相同原缓冲按device byte offset切出
4096B再reshape[32,128]，交给原gather_row写L1。页号夹取、QK/WS/FFTS
和所有数学顺序保持；无Host读、无新GM分配。只是准备CPU编译，尚无设备结论。


## 2026-10-17 01:34：页视图CPU失败，核实并消除QK重复清零

indexer_page_views_prepare_v1的地址审计及98个数学调用顺序检查通过，但
indexer_page_views_cpu_v1完整编译失败：InCore中的tensor.slice会lower为
Tile加载，gather_row要求GM Tensor，不能把该写法当零拷贝GM子视图。
没有提交NPU，没有修改PyPTO或越过Tensor窗口边界规避检查。该开关仍默认
关闭，保留失败源码/日志供参考；不宣称算子或性能通过。

重新核对当前最佳B16的真实生成C++：QK KV分支外有一处TEXPANDS，
压缩分支内还有一处TEXPANDS，压缩块确实执行两次64KiB清零。新增
qk_scoped_zero.py，只将前一处移入SWA分支；压缩块保留自己的清零与
原逐行gather/负索引处理，所有数学/FFTS保持。不使用此前MGATHER或
全有效块候选。qk_zero_prepare_v1正在准备，尚无编译或设备结论。


## 01:37：QK初始化作用域CPU通过，提交完整B16

qk_zero_prepare_v1/ast_audit.json确认仅qk_pv_aiv改变，其余72个函数
AST不变；把初始化移回原处后，整个函数AST与原版完全相同。
qk_zero_cpu_v1完整CPU/PTOAS代码生成通过。codegen_comparison.json
确认压缩块由连续两次清零改为分支内一次；整阶段PIPE_ALL静态数未变。

task_20261017_013726_407690029470 / b16_cached_qk_zero_v1提交。
沿用earliest_ready、原动态成本、cached100、QK地址缓存；只增加
--qk-scoped-zero，不叠加其他退步候选。结果未回收前不预记通过/提速。


## 01:42：仅消除重复清零完整PASS，无性能改善

task_20261017_013726_407690029470完成exit0，完整exact/TopK/六状态/
图重放/30代信号/Native及工作区保护区PASS。单SPMD无插桩L1三窗
817.22/822.98/826.50μs，中位822.98；同轮动态897.20/906.32/893.42，
中位897.20。未超过现选815.84，不采用，不扩七档。

继续qk_valid_gather.py：先在device确认64个原始索引全部非负，全有效
块用create未初始化Tile，8行展开覆盖全部64行；否则保留原清零和原
64行条件循环。SWA仍在自己的分支清零。与旧full_block候选的具体区别
是全有效路径由两次清零降到零次，负索引路径为一次；不是原样重跑旧候选。
新增qk_valid_gather_boundary.py，从实际候选AST提取规约与两条搬运分支
复用，拟检查同一图全有效→混合负索引→全负索引→全有效往返。
正在准备CPU与编译，尚无设备或性能结论。


## 01:45：全有效块首版UB超限，改成分支共用目标缓冲

qk_valid_prepare_v1确认只有QK AIV变化，原负索引循环AST完全保留
（仅约定的结果变量改名），8×8展开覆盖64行，无重复或遗漏。
qk_valid_boundary_cpu_v1独立边界代码完整编译通过，但整层
qk_valid_cpu_v1失败：Vec 201472B > 188416B。没有提交NPU。

V2在有效性分支前create共同64KiB目标；全有效分支直接覆盖，负索引
分支先tile.expands(0)再执行原循环。expands广播标量，不读取未初始化
数值；不使用“乘零”规避初始化。SWA仍独立清零。只调整实验源码以表达
共用目标，不修改规划器或容量检查。qk_valid_prepare_v2正在准备。


## 01:48：共用目标V2的字面量类型修正

qk_valid_cpu_v2与qk_valid_boundary_cpu_v2均在类型检查失败：
tile.expands(target, 0.0)的标量0.0被推导为FP32，返回Tile与BF16
注解不符。这发生在内存规划之前，不能声称共用目标已经解决容量。
V3显式将零标量cast为BF16，再广播；保留V1/V2源码与失败日志。
只修正候选写法，不修改PyPTO规则。qk_valid_prepare_v3准备中。


## 01:51：V3仍超UB；不用缺少后端的expands，改现有原地填充

qk_valid_cpu_v3仍报Vec 201472>188416B；显式BF16只解决类型，普通
expands并不保证别名，不能据此宣称两路径共用分配。独立
qk_valid_boundary_cpu_v3另报No codegen registered for operation: tile.expands。
源码/日志全部保留，未修改依赖、未提交设备。

V4使用既有tile.fillpad_inplace(tile.set_validshape(blank, 0, 512))。
PyPTO注册明确set_output_reuses_input(0)，并将结果有效形状恢复为完整
物理形状；当前A3 ISA在srcValidRow=0、srcValidCol=512时填满底部64行，
没有读取有效业务行。这里使用原地填充的已有语义，不降低容量阈值，也
不手改生成C++。实际代码生成与负索引图重放仍须验证，不能只凭源码判PASS。


## 01:53：原地填充的独立编译通过，整层局部变量类型分离

qk_valid_boundary_cpu_v4完整编译通过。qk_valid_cpu_v4在parser失败：
同名stage_30_v255原为无pad标记Tile，原地填充产生pad=zero Tile，
被判断为不同类型的重赋值。尚未到内存规划，不能预记容量已解决。
V5只把压缩分支的初始局部值改名qk_cmp_tile_start，区别于SWA中的
旧同名SSA；保留原地填充、所有搬运和数学。负索引循环审计允许此初值
及原有结果名两处重命名，其余AST必须逐节点相同。


## 01:57：V5的快速/回退初值仍同名，V6分别命名

qk_valid_cpu_v5仍在parser失败，定位到全有效与回退分支共享
stage_30_v302，分别为pad未指定与pad=zero类型。此前只区别于SWA
还不够；V6将三分支初值全部区分，快速分支qk_valid_tile_start、
回退分支qk_fallback_tile_start，内存仍来自同一qk_valid_blank。
没有修改数学或有效性条件，当前qk_valid_prepare_v6在准备。


## 02:00：V6分支结果的pad类型冲突，V7用恒等视图统一

qk_valid_boundary_cpu_v6完整编译通过；qk_valid_cpu_v6仍在parser失败，
这次是融合生成器展开yield后的共同结果stage_30_v303，两个分支的pad
标记不同。初值改名只解决局部重赋值，不能解决共同结果类型。
V7给全有效与SWA分支结果添加64x512、offset[0,0]的恒等tile.slice，
pad_value=zero，统一为与回退相同的类型。三条路径本来都是完整有效Tile，
此处无无效区域，不需要实际填充或拷贝；负索引的真实清零仍由原地填充
完成。实际生成代码需确认未新增搬运/清零，再决定是否上卡。
独立与整层失败/通过范围继续分别记录，不把小例通过算整层通过。


## 02:04：全有效块V7完整CPU通过，提交一次B16

qk_valid_cpu_v7与qk_valid_boundary_cpu_v7完整代码生成PASS。
qk_valid_prepare_v7/codegen_comparison.json：全有效分支8个TLOAD展开点，
没有TEXPANDS/TFILLPAD；SWA保留一处TEXPANDS，负索引保留TFILLPAD。
整个QK AIV阶段静态PIPE_ALL由9增至11，不把清零减少直接说成性能收益。
候选的UB容量已通过原规划器校验，没有手改地址或改依赖。

提交task_20261017_020426_59480118626 / b16_cached_qk_valid_v7。
使用现选earliest_ready/cached100/QK地址缓存，只增--qk-valid-gather。
计时外从实际候选AST提取边界片段做同图4变体往返；整层保持原exact/TopK/
六状态/图/30代信号及保护区闸门，性能为原3个L1窗口。结果待收取。
所有V1～V7的准备和失败产物保留，前面的CPU会话均已收取终态。


## 02:09：全有效块完整PASS，839.84μs，仍不采用

task_20261017_020426_59480118626完成exit0。实际候选片段的同图
全有效→混合负索引→全负索引→全有效四变体，输出exact、只读输入和
保护区PASS。完整CSA输出、TopK、六状态、图重放、30代信号和Native/
工作区保护区全部PASS，未放宽任何准确性门槛。

单SPMD无插桩完整L1三窗839.84/838.18/841.76μs，中位839.84；
同轮原动态899.10/890.14/890.88，中位890.88。没有优于现选815.84，
不采用、不扩七档，不重复同一候选。codegen里额外两个PIPE_ALL是已见
差异，但未做阶段探针，不能断言它单独解释全部回退。

本轮新增qk_scoped_zero.py、qk_valid_gather.py、qk_valid_gather_boundary.py，
只在实验目录接入默认关闭的开关；生产入口、冻结静态、PyPTO和Simpler均未改。
全部CPU与NPU任务结束，没有待收结果。七档当前选择不变，700μs目标仍active。


## 2026-10-17 02:23：同一数学体独立QK编译未减少循环屏障

qk_isolation_cpu_v1 完整CPU编译成功，选中现选版本的同一QK AIV数学体，
局部编译后该阶段仍9处PIPE_ALL，实际QK tick仍4处；循环文本约160205→158220B，
set/wait各69→73。没有从原动态另一数学体的3处PIPE_ALL作因果推断。
证据comparison.json；不提交这个负候选NPU，不删除cached_wait/独立编译组合的安全限制。

## 02:30：Indexer AIC提交区间探针通过

indexer_phase_clock.py 只包围原32个页表读取至Key DMA提交窗口和1024分数块循环，
另计原wait_flag_dev，不给普通异步wait_flag虚构Scalar等待时间。保持原中间流水屏障。
CPU提取审计indexer_clock_prepare_v1通过：32个静态页加载点、1个分数块循环、
2处FFTS等待；每核512B进度区的第二行记录，首行信号不动。

task_20261017_022628_11712641926 / b16_cached_indexer_clock_v1 完整PASS。
24个AIC中位：阶段210.23μs、Key提交累计11.81μs/704次、分数块193.96μs/22次、
FFTS等待15.00μs/22次。普通wait是异步pipe指令，区间会重叠且插桩扰动队列，
不能简单相减成计算/搬运独占耗时。完整插桩L1为845.64μs，不用于700μs目标。
精度/TopK/六份状态/图/30代信号/保护区全部PASS。页地址分支暂不继续优先优化。

## 02:35：完整CSA同地址动态metadata A→B→C→A通过

dynamic_metadata_replay.py 与 --metadata-replay-only：固定B16/S6、页表容量、
全部根参数/工作区地址和每侧捕获的一张图；只在测试准备时改设备Tensor内容。
A=131072，B=131071并交换请求物理页，C=130048并交换页，最后恢复A。
位置、长度、页表、slot、RoPE共19个根输入变化；B/C参考输出确实不同，A恢复完全相同。
每组先运行原动态PTO eager，再重放原动态图和单SPMD图；全部输出、Top-K、六份
cache/state逐bit一致，完整allocation未写区/metadata只读检查通过。

task_20261017_023022_12848062458 / b16_metadata_replay_v1 完整PASS，
无新增编译单元；6代进度准确且变体之间未清零信号，工作区保护区PASS。
这次只做功能，没有Event/L1重复采样。证据dynamic_metadata_result.json及
b16_metadata_replay_v1/metadata_replay/replay.json。尚不代表连续decode、
异构请求长度或生产自动派发，也不要求跨batch共用二进制。

## 02:36：下一候选改为小档O-A的M/K预取分块

oproj_k512.py：T<=96时M按16对齐，保持N128、原逻辑分片、全局视图，
将外层K预取256×16改为512×8，拟保留内层K128及其顺序。较大档仍原策略。
先只准备/编译B16；需核对实际内层代码和内存容量，尚未上卡，不预记性能收益。


## 02:39：O-A K512完整CPU通过，提交性能；另补短历史动态metadata

oproj_k512_prepare_v1与oproj_k512_cpu_v1通过。实际C++左/右矩阵内层仍K128，
同一个Acc覆盖K=0,128,…,3968，外层8个K512块；M96/N128，单阶段PIPE_ALL仍1。
代码审计codegen_audit.json；解析C++常量必须限定AIC函数，不能与AIV同名局部值混合。

task_20261017_023822_152092626476 / b16_cached_oproj_k512_v1已提交，仅B16性能候选。
按既有5次预热/20次Event/3个L1窗口及完整exact检查；没有更改其他档位选择。
短历史的Indexer数学分支不同，另提交B16/8192的同地址A→B→C→A功能检查，
沿用该档原critical_path/cached100策略，不开QK地址缓存、不采性能、不套长历史二进制。


## 02:42：O-A K512完整PASS但没有更快，不扩大测试

task_20261017_023822_152092626476 / b16_cached_oproj_k512_v1 完成exit0。
输出、Top-K、六份状态、图、30代进度、全部保护区继续通过。
无插桩单SPMD L1为827.94/833.74/825.80μs，中位827.94；同轮原动态885.68μs。
没有超过保留815.84μs，因此不扩测其他档，不更改selected_shape_policies。
CPU准备和代码生成证据保留，不能用减少外层加载次数推断整体提速。
短历史动态metadata任务 task_20261017_023946_156511419169 尚在运行，未预记通过。


## 02:45：短历史首版用例跨策略失败，补编排一致性检查

task_20261017_023946_156511419169 已完成exit1。A8192、B8191通过；
C7168在图比较中失败：x_out最大绝对差0.03125、172666个元素不同，
Top-K15576项不同；六份cache/state及所有保护区仍通过。旧探针没有记录
失败的具体graph侧，因此不能据此归因给单SPMD或原动态图。

源码核对：8192+6的压缩长度2049，进入[2048,8192]策略；7168+6压缩长度1793，
已进入另一套score/topk数学分支。当前不要求跨策略共用二进制，原用例错误地
要求复用该捕获图。保留short_metadata_failure_v1.json和原失败日志，不覆盖。

新增metadata_cases用CPU编排模拟器核对原call_signature及selected_control_path，
只使用测试已知Host长度，不回读device。每个待执行变体必须同原策略；
跨策略候选在派发前拒绝并记录。8K档C改为8186（压缩长度恰为2048），
即8192→8191→8186→8192，形状/地址/信号不变；8个更短候选先被CPU拒绝。
short_metadata_strategy_prepare_v2/cases.json记录完整分支证据。
同时失败报告保存active_step和failed_checks，避免再次丢失具体比较侧。
已提交修正版b16_short_metadata_replay_v2，不预记PASS。


## 02:49：短历史策略内动态metadata完整通过

task_20261017_024535_170390925040 / b16_short_metadata_replay_v2完成exit0。
8192→8191→8186→8192四组在完整CSA上输出、Top-K、六份cache/state逐bit一致，
每侧固定一张图、地址固定、无新增编译单元；6代信号及全部保护区PASS。
8个跨策略长度在CPU准备时拒绝。没有重复任何Event/L1性能采样。
证据short_dynamic_metadata_result.json及该目录metadata_replay/replay.json。

继续CPU候选oproj_phase_loop.py：原O-A八个相邻阶段只在group、源行/目标列
基址不同，逐AST正规化后八体完全相同。按原顺序[6,5,7,4,1,3,0,2]循环，
每次仍用原record/wait表，不删等待/通知/缓存维护，不改变任何数学Tile。
原JIT文本587476→569145B，尚不能推断机器码大小或性能。CPU编译已启动，
不改PyPTO/Simpler，不把不同阶段外部事件拼接成独立device函数。


## 02:52：O-A阶段循环完整CPU通过，提交B16

oproj_phase_loop_cpu_v1 完整编译通过。AIC生成C++ 962738→753992B，
静态PIPE_ALL/TWAIT/TNOTIFY位点35/34/35→28/27/28，因为八段共享循环体；
实际动态执行次数与原顺序相同，不把静态位点减少冒充删除同步。
逐AST检查八体除group/源行/目标列外一致；原record基址按432步长保留。
提交b16_cached_oproj_loop_v1，仅在现选B16上检查完整exact/图/信号/保护区及性能。


## 2026-10-17 02:56：O-A阶段循环完整PASS但未改善整层，全部任务已收取

task_20261017_025159_192231313200 / b16_cached_oproj_loop_v1 完成exit0。
全部exact/TopK/六份状态/图/30代进度/保护区PASS。无插桩L1三窗
[835.94, 826.24, 830.74]，中位830.74μs，同轮原动态903.74μs。
AIC bin由66832降至54900B，AIV保持153408B；没有超过已选815.84μs。
因此不采用、不扩其他档，也不把源代码或二进制缩小当成性能收益。
本轮所有CPU session和NPU任务均已收取，没有待等待任务。


## 2026-10-17 03:20：QK按32行分两次准备KV候选，边界包装修正中

用户再次明确：允许batch/长度分档与精确特化，目标是各档均有较好性能；参考hc_pre的编译期分块与入口分派，不要求共用二进制。现选七档不变。

qk_chunked_gather.py保持64行GM交接布局、原通知/等待和QK/PV数学，将AIV本地BF16[64,512]改为两个[32,512]。SWA与压缩分支均分块，压缩metadata仍一次准备64行。CPU地址枚举1696个SWA场景和压缩行序检查通过，qk_chunk_cpu_v1完整代码生成通过。实际tick代码不再有64行KV Tile；PIPE_ALL静态位点4→5，不能据此声称速度提升。

qk_chunked_boundary.py直接提取候选生产片段，拟用同图覆盖SWA短窗口、页边界、负页、长位置，压缩有效/负索引、分支返回及保护区。首轮CPU边界编译qk_chunk_boundary_cpu_v1失败：打印IR中的t__tile_1为保留命名；同时提交的task_20261017_031425_239558810336/b16_cached_qk_chunk_v1因同一问题exit1，完整CSA与性能均未运行。v2对片段局部变量/形参作无碰撞重命名，CPU进一步暴露嵌套incore调用未内联（No codegen registered for operation: _gather）；测试辅助函数改为jit.inline，CPU_v3进行中。以上只修测试包装，未改PyPTO/Simpler。


## 03:23：QK边界CPU通过等待真机；准备HC流式候选

qk_chunk_boundary_cpu_v3完整代码生成通过，jit.inline使实际候选片段在边界入口内联。task_20261017_031908_248319030873/b16_cached_qk_chunk_v2已提交，03:23仍pending，未预记PASS或性能。

hc_stream_pre.py在实验生成源码中改写三个HC阶段，仅接收T96/24组/原M16-K4096四分片。48颗AIV按组内8行×K512加宽，每个K512的row_sum写入原T96闲置尾部暂存的12288字节。AIC用K512 L1供数、两个原K256 matmul依次累加，仍输出4份K4096部分和；原linear_reduce额外按0..31顺序恢复每行RMS。其余HC门控、Sinkhorn、归一化与CSA阶段不变。

原AIC等待全部12个加宽任务的GM轮询被组内FFTS READY8/9与ACK10/11替换。全部48个AIV参与，原12个进度发布记录保留。两槽限制未消费通知，AIV末尾排空两个ACK；不新增Scalar GM信号，也不删后续各阶段数据维护。CPU检查原加宽无前驱、原AIC仅依赖加宽、reduce等24个AIC、两个inv_rms消费者等全部reduce。128个随机调度×3代事件模型无未消费通知覆盖/死锁/残留；这不证明硬件pipe可见性。hc_stream_prepare_v1已生成，hc_stream_cpu_v1编译进行中。尚无真机精度/性能结论。


## 03:29：QK32行候选完整PASS但未提升；HC自动pipeline候选不上卡

task_20261017_031908_248319030873/b16_cached_qk_chunk_v2完整PASS，8项同图边界、输出/TopK/六份状态/图/30代进度/保护区全部通过。无插桩L1为831.32/828.92/838.78μs，中位831.32；同轮原动态886.70。没有优于保留815.84，未采用，不扩其他档。证据qk_chunk_result.json。

HC CPU_v1因把GM store结果重新赋给根参数导致MIX整参别名检查失败，改为原地store表达式，无依赖仓修改。v2的动态事件ID在PTOAS产生作用域错误；v3改为偶/奇静态8/9与10/11，CPU编译通过。但审计两版PTO IR均发现pl.pipeline重排：输入tload先于对应READY wait，生成C++也保留错误顺序。v2/v3都没有上卡，不以编译PASS冒充交接正确。证据hc_stream_pipeline_order_audit.json。v4把两侧手工FFTS循环改为pl.range，RMS纯数学累加仍可pipeline；CPU_v4在编译，待检查READY→输入加载→ACK及双store→READY顺序。


## 03:31：HC显式循环CPU与交接顺序通过，提交B16整层

hc_stream_cpu_v4完整代码生成PASS。逐IR与C++核对READY wait→输入TLOAD→MTE2 ACK，AIV输入/平方和两次TSTORE→MTE3 READY，且末尾等待两个ACK。首个自动审计用固定前后行数误计前一RoPE的store，断言失败；按实际C++循环花括号限定后通过，未改变候选或放松顺序要求。codegen_audit.json保留位点。task_20261017_032959_27383327818/b16_cached_hc_stream_v1已提交，沿用原exact/保护区/图/30代及3窗L1；未预记PASS。


## 2026-10-17 03:35：HC流式版完整PASS但未提速，本轮任务全部收取

task_20261017_032959_27383327818/b16_cached_hc_stream_v1完成exit0。输出、TopK、六份状态、图、30代进度和全部保护区PASS，未放松exact要求。无插桩L1为834.18/844.40/847.60μs，中位844.40，同轮原动态907.64。没有优于保留815.84，未采用、不扩其他档；不能把功能正确、48 AIV参与或开始供数更早当成提速。没有额外采阶段探针，暂不能将整层差值定量归因给ACK、RMS临时区或RoPE前置顺序。

本轮新CPU/NPU任务均已收取。QK分块和HC流式代码、边界、失败、生成代码审计及性能结果均保留。现选七档与生产入口不变；无PyPTO/Simpler修改。目标active，700μs未达到；不存在外部阻塞，不标暂停/完成。

## 2026-10-17 03:47：Indexer叶分片容量实验已提交

新增indexer_leaf_capacity.py/--indexer-leaf16384，仅T96长历史；原8192候选/叶对应6叶，
不是此前NEXT_STEP中误写的9叶。新16384候选/叶为3叶，同步更新Cube、Vector和最终合并。
每worker仍22个1024分数块，query/系数加载4→2次，pair_arena布局/容量/FFTS协议/分数数学不变。
CPU检验4133种动态长度和每种最多5个有效前缀，覆盖/无重叠/active leaf一致性通过；
indexer_leaf_prepare_v1和indexer_leaf_cpu_v1完整lowering通过。仅CPU证据，未证明TopK tie顺序或精度。
真机task_20261017_034636_30078467438，b16_cached_indexer_leaf16k_v1，待结果。
另只读核对payload_cache.py：此前精确Scalar缓存维护候选已覆盖按阶段读写删DCCI，不重复该负候选。
scalar_gm_access_audit_initial.json仅为源级访问清单，不是新的同步正确性证明。

## 03:53：16384候选/叶完整通过，未纳入当前选择

task_20261017_034636_30078467438 exit0；输出、TopK、六份state/cache、图重放、30代信号及保护区均通过。
L1三窗829.22/829.42/813.78μs，中位829.22；同轮原动态897.84μs。
未证明优于现选历史815.84μs，保持七档选择不变，也不按最小值813.78替代中位统计。
现选SPMD与候选没有同进程A/B，不能把历史差值当作确定回退幅度。indexer_leaf_result.json。
后续新候选indexer_score_pingpong.py：M384/N64分成两份M192/N64，使两份QK INT32分数同时活跃，
合计96KiB Acc，原叶数/页访问/1024分数交接/Vector TopK与FFTS保留。与已测N128的区别是
单份Acc由96KiB减到48KiB，两份在源级有重叠生命周期，且Key预取至Right。尚待实际地址/同步与设备验证。

## 03:55：Indexer双Acc的完整CPU编译与地址核对通过

indexer_pingpong_prepare_v1与indexer_pingpong_cpu_v1完成；16组两份INT32 Acc各49152B，
实际地址0与49152不重叠。C++中每组两次TMATMUL均位于第一份TINSERT/FIX之前。
codegen_audit.json记录逐组地址与C++变量；这是地址/提交顺序证明，不是硬件并发或提速证明。
首次审计按TMOV匹配量化写回导致断言失败；实际lowering为TINSERT，改审计匹配后通过，算子代码未为此变化。
未用固定MemRef，也未删同步：通过两份结果同时活跃阻止Acc别名。
task_20261017_035437_31474978052已提交，输出b16_cached_indexer_pingpong_v1，先严格exact后计时。

## 2026-10-17 04:03：双Acc完整PASS但性能回退，本轮收尾

task_20261017_035437_31474978052 exit0，30代信号、输出、TopK、六份cache/state、图重放和保护区通过。
L1三窗968.92/965.56/972.00，中位968.92；同轮原动态904.72。不采用，不扩其他档。
对实际NPU构建再次执行只读codegen地址与提交顺序检查，16对Acc仍不重叠，提交顺序保持。
生成Indexer AIC静态位点TMATMUL32→64、TINSERT16→32、TSTORE16→32、set/wait166→187，PIPE_ALL仍1。
不能把这些静态数直接换算为耗时，无新探针定量解释回退。INDEXER_PARTITION_EXPERIMENTS.md汇总两项。
只读审阅HC前端reduce/split/comb/mix原函数与现有阶段时戳；尚未实现新融合。
全部CPU/NPU任务已终态，依赖/生产/冻结静态不改；七档选择保持，goal active，700目标未达。


## 2026-10-17 04:30：PV分head合并被UB上限拒绝，直接L1供数候选开始验证

qk_merge_heads.py尝试将每AIV的32-head PV合并拆为16-head，再assemble原状态。
qk_merge_heads_prepare_v1及cpu_v1：规划仍产生196608B Vec，超过188416B上限，未上卡。
prepare_v2只生成8-head源码，没有编译或设备执行；看到全状态生命周期证据后没有继续盲扫。
prepare_v3/cpu_v3改为先完成left、再处理right，仍196608B超限。两版CPU失败原样保留，
没有改编译器或扩大硬件容量，没有精度/性能结果，七档选择不变。

新qk_direct_l1.py让Cube直接从Native KV到三槽L1。Vector批量把私有valid-plan索引临时区
改成物理行（外部TopK不改），并准备每组32行零源。一次FFTS4发布后Cube维护Scalar缓存再读。
原事件0从KV_READY改为两AIV完成PV读取的确认；Cube每三槽复用前等待，调用末尾排空。
原score/probability/PV事件1/2/3和全部点积、softmax、归约顺序保持；无新Host读取或根参数。
该候选尚未完整设备验证或计时，不能据此宣称提速。

qk_direct_l1_probe_cpu_v1完整lowering通过；生成C++子视图仍保留父Mat Rows=384，
validRow=1/动态，地址按实际行偏移，因此没有把NZ列跨度错误缩成16。
真实A3小用例首轮task_20261017_042346_360920825849缺少pypto.torch.init，exit1，
没有执行kernel；补测试初始化后task_20261017_042437_362436518114 exit0。
qk_direct_l1_probe_device_v2同一图7项全部PASS：三个槽逐行gather、非对齐起点、动态
有效行数、最后槽最后一行以及返回首槽；全部三槽数据、相邻行/槽、GM保护区/只读输入通过。
只是搬运接口证据，不代表完整CSA正确。未采性能、未运行Native。

qk_direct_l1_protocol.py的CPU交接模型通过3328个调度、322443步、2/3代复用：
枚举两query全部1024种有效块mask并随机两AIV进度，另外检查4query；确认事件按代排空、
score/probability/KV/PV三槽未提前复用，无死锁。1310种SWA长度/负页模式的区间覆盖通过。
模型不证明真实DMA或缓存语义。直接L1完整CPU正在qk_direct_l1_cpu_v1编译，随后核对实际
生成顺序；所有改动只在本实验目录，冻结静态/生产/依赖仓均未改。


## 04:31：直接L1完整CPU与提交顺序通过，上卡验证

qk_direct_l1_cpu_v1完整lowering PASS。IR与实际C++确认：地址/零源store后发布FFTS4；
Cube等待4后fence/cacheinvalid/fence，再开始Scalar地址读取；三槽复用等待0在新QK之前；
两份PV的TLOAD都在MTE2发布0之前；最后一个ACK在退出QK前排空。
qk_direct_l1_prepare_v1/codegen_audit.json保存行号与原文。保留原矩阵乘pipeline，手工交接
使用pl.range，没有出现上轮HC自动pipeline将TLOAD移至READY之前的问题。
任务task_20261017_043021_371809025343/b16_cached_qk_direct_l1_v1已提交；
只比较原动态PTO，固定正式权重fixture，原exact门槛先行，未预记PASS或性能收益。


## 2026-10-17 04:38：直接L1完整通过但明显退步，本轮全部收取

task_20261017_043021_371809025343 exit0。正式权重固定fixture的输出/TopK/六份状态、
图、30代信号与保护区全部PASS。单SPMD L1三窗1096.08/1100.10/1100.20μs，中位1100.10；
同轮动态892.66μs，增加23.24%。明确不采用、不扩其他档，也不追加该负候选的边界/metadata重放。
历史选定815.84μs仍保留；没有选定SPMD同进程A/B，不能把历史差值当成精确回退量。
qk_direct_l1_result.json汇总；实际NPU C++与已核对CPU C++全文相同（文本比较，无hash）。

只读代码归因：AIV整个QK阶段PIPE_ALL静态位点9→5，但Cube逐行搬运变多。
真实compressed128行循环内没有pipe_barrier/set_flag/wait_flag；不能说每行多了屏障。
原两AIV各64行搬运改为单Cube128行ND2NZ，Scalar地址也改从GM读取。这些结构变化
可能增加延迟，但未采内部探针，不能将全部回退定量分配给它们。
第一次按record注释界定C++阶段失败，改用实际花括号边界后生成codegen_comparison，算子未改。
另只读核对当前A3 TPush/Tpop Vec→Mat使用GM FIFO，不实施接口换名式尝试。

下一方向仅做只读初查：query_dependency_inspection_initial.json记录现选控制表中
QK Cube等待17个前驱、valid-plan47个、TopK merge71个、首批部分O-A71个；并不证明可删。
需要先映射数学读写范围，再决定是否细化，不能把当前按核独立推进误说成全卡barrier。
本轮CPU和NPU均终态，无共享依赖、生产或冻结静态改动。goal仍active，700μs未达。


## 2026-10-17 05:06：按行细化成员依赖与同图变长复核

新增row_dependencies.py，严格限定T96长历史，实际Tensor别名/行所有权与FFTS阶段复用有显式契约。
原数学、物理分工、核内顺序和发布/cache维护不变；四类完整task依赖改为成员依赖，
所以不是原task偏序不变。等待16729→12183，发布730→722，细化图约简/翻译后祖先关系一致。
1536个merge worker×leaf_count组合覆盖96query、所有1～32leaf，CPU证明不替代设备。
原生成函数体AST一致；固定成员成本最长链748.55→748.53只是离线分析，不作性能预测。

- task_20261017_044912_40225339550 / b16_cached_row_deps_v1 exit0完整PASS：
  新807.16/804.40/811.26，中位807.16；同轮动态905.08；30代/保护区通过。
- task_20261017_050033_181823706 / b16_row_deps_paired_metadata_v1 exit0完整PASS：
  同进程旧候选841.94/843.14/833.20，新候选806.84/817.88/821.56；中位缩短2.86%。
  同一任务候选图131072→131071→65536→131072，6leaf变3leaf再恢复；
  原动态eager/图与候选图逐bit一致，地址/无新编译/11代/保护区通过。
- paired_row_dependency_perf.py仅为本轮对照辅助；动态metadata增加显式额外history选项，
  先CPU检查原编排call_signature和selected_control_path，无device回读。
- 选用行依赖候选，仅B16长历史。其他档不重测、不套该表。
- 实现、范围和证据见ROW_DEPENDENCY_EXPERIMENT.md；未改生产/F/依赖仓。


## 2026-10-17 05:10：Indexer N80容量与流水检查

indexer_n80.py保持原QK M384/K128、WS M16/K384、FIX量化/ReLU、1024列分数传输、
叶计划及FFTS/TopK AIV。仅N64→N80，13个panel（末64列有效），matmul32→26。
每个80列panel按Native32行页分成32/32/16或16/32/32段，总39段，原来32段。
Native带scale页的0/64字节偏移视图保留；禁止把scale区域连续当Key读。

indexer_n80_prepare_v1与完整CPU生成indexer_n80_cpu_v1通过（session30043已收取）。
和已选行依赖组合，原输出区域/叶所有权/FFTS尾排空不变。
实际C++中13个QK Acc均0地址、122880B；前12个WS Acc在122880、末次WS在0复用，
峰值128000B<131072B。末次复用经过FIX→MTE1→M传递依赖，非无同步覆盖。
WS Right和Key Right均0地址，生成M→MTE1通知/等待保护复用；因60KiB+10KiB超64KiB，
不能沿用N64时同时保留两者的预取。完整设备性能才能判断减少panel是否足以抵消。

审计脚本首版assert失败：跨AIC/AIV合并了重名C++常量，且误要求末次WS也占122880。
限制作用域并识别末次合法复用后通过，未改算子、编译器或容量。证据
indexer_n80_codegen_audit.json/indexer_n80_codegen_fragment.cpp。
任务task_20261017_050914_1722359619/b16_row_deps_indexer_n80_v1已提交，结果待终态。


N80终态：task_20261017_050914_1722359619 exit0，完整exact/图/30代/保护区PASS。
L1879.10/887.62/878.82，中位879.10，同轮动态895.08；Event981.13对956.57。
明显慢于已选行依赖候选，不采用、不扩测。实际NPU与CPU的52个N80 Acc/Right Tile
类型/形状/地址一致；首版阶段文本提取因SSA名称被优化失败，没有把分配一致称为全文一致。
完整限制及代码生成证据见INDEXER_N80_EXPERIMENT.md。

另做两项只读/离线调度核对，没有上卡：Indexer数学父任务为cache scale提交、Hadamard量化、
head coefficients，不包含主Q矩阵。主Q在模型最长链中出现来自资源日历，不可据此直接删wait。
把qproj_matmul与qproj_dequant_rms_nope_rope优先级偏置均设为-500，分别critical_path与
带权earliest_ready，固定旧成本的估计均799.38，对照773.43，不比现选有利，不生成新二进制。
证据main_q_indexer_dependency_inspection.json、deferred_main_q_schedule_analysis.json。
CPU session89931已收取；本轮所有CPU/NPU任务已终态。


## 2026-10-17 05:44：HC整行驻留失败修正及主Q显式L0预取

HC候选见HC_MIX_QPROJ_EXPERIMENTS.md。prepare/cpu v1 runtime tile.full失败，v2 scalar cast失败，
v3/v4 CPU通过。NPU v1 task_20261017_053012_51687711509与v2 task_20261017_053800_64833816891
均exit1：x_out15613项max0.015625、TopK825项不同，六cache/state中四份有差异。未计时。
两版仅Scalar/Vector部分和累加不同，结果相同；不能归因为Scalar加法。
仔细重读冻结body发现原来是16×K256，候选误用8×K512；这是候选实现错误，不是依赖bug。
hc_mix_resident_prepare_v5/cpu_v5恢复16×256和16次原顺序Vector累加，完整CPU通过。
实际IR与V到S屏障已核对；审计脚本首版只找顶层AST导致空集合，改ast.walk后仅mix变化。
任务task_20261017_054339_75460017083/b16_row_deps_hc_mix_resident_v3已提交，未预记PASS。

主Q普通N128的prepare/cpu_v1通过，但Right仍全部地址0，双缓冲未实现，未上卡。
新的qproj_l0_prefetch.py保留N256/20worker/列分工，T96的QR一次加载L1，K128八段按序INT8累加；
先取下一Left/Right再提交当前matmul。prepare/cpu_v1通过，qproj_prefetch_codegen_audit.py
确认两处主Q的Left地址0/12288、Right0/32768交替，下一Right的TMOV先于当前matmul。
这只是实际分配和提交顺序，不是测得的硬件重叠。首版临时检查没有递归解析C++地址cast，
显示None，已改用MLIR分配及C++操作对应关系，不改算子。
任务task_20261017_054303_7422476042/b16_row_deps_qproj_l0_prefetch_v1已提交。

当前七档选择不变，B16仍807.16首轮、817.88同进程复核，700目标未达。
本轮只改N内辅助实现/测试与记录，不改生产、冻结目录或依赖。两个新NPU任务待终态。


## 2026-10-17 05:50：两候选终态，无新性能收益

HC task_20261017_054339_75460017083 exit0，16×256修正后完整exact/TopK/六状态/图/30代/保护区PASS。
L1为822.46/819.02/817.06μs，中位819.02，同轮动态911.56。此前两版15613项输出差异已消除。
主Q task_20261017_054303_7422476042 exit0，同样完整PASS；L1为825.54/831.16/830.62，
中位830.62，同轮动态892.68。实际NPU产物的两处Left/Right双缓冲与CPU审计一致。
两项都不采用、不扩测其他档。七档选择保持，128K/B16首轮807.16/独立同进程复核817.88仍是现选。
没有现选同进程A/B，不将历史差值解释成确定回退幅度。
HC_MIX_QPROJ_EXPERIMENTS.md和hc_mix_qproj_results.json记录全部正负结果与限制。

只读避免重复：Indexer M192/N128、WS K192的两组三query方案已在indexer_n128.py测试过883μs，
没有再次生成或上卡。现有HCpost本身已经整行4096列，不继续基于错误分块前提设计。
本轮所有CPU/NPU任务终态。没有修改生产、F、PyPTO或Simpler仓；动态目录未commit。


## 2026-10-17 06:00：HC reduce→split→mix 合并候选

hc_front_fusion.py保留原数学体，仅把reduce独立行16改8，与split/mix的8行一致；
同一12个AIV每个完成自己的8行三阶段。四份投影部分和仍0..3，RMS仍16×256。
原三份GM写回保留，两处内部fence，显式传递store/loop结果Tensor版本。
原reduce/split两个task移除，Sinkhorn改等融合阶段，仍独立执行；原HCpost等待也重定向。
根参数及全部临时GM分配完全相同；66→64任务，722→704发布，等待12183→12189（重排后略增），
不把阶段减少说成总等待数量必然减少。完整CPU协议/视图/三代模型通过。
HC融合成本仅用三份原动态中位成本之和估计，balanced_schedule新cost kind明确未实测融合核。
prepare/cpu v1完整通过，实际生成TSTORE→PIPE_ALL→TLOAD顺序在两个边界均成立。
task_20261017_055741_99590920590/b16_row_deps_hc_front_fusion_v1已提交，结果待终态。

后续Tile直传版本hc_front_forward_tiles.py已做prepare v1：T96保证split循环恰好一次，
将其显式展开；reduce8×32先转32×8，再按行slice0/4并转回，避免非对齐列4切片。
原GM写回保留供Sinkhorn与诊断，split的两次raw回读和mix的pre回读改为SSA Tile直传，
两处内部PIPE_ALL删除，阶段外完成发布和编译器自动事件保留。
独立CPU准备进程临时绑定helper，不编辑运行任务的导入模块；该版本CPU编译中，尚未上卡。

## 2026-10-17 06:15：HC融合收取与Indexer输入行依赖

HC三阶段融合task_20261017_055741_99590920590完整PASS；L1三窗804.84/810.80/806.76μs，中位806.76μs，对原动态同轮900.14μs。与现选807.16μs是不同运行，不据此升级选型。

HC Tile直传CPU v1因转置临时Tile形状错误失败；v2切片静态valid与临时Tile动态valid不一致；v3改标准三参transpose仍有相同valid类型不一致，均未上卡。v4将对齐行slice改为extract，保持UB数据重排而不新增算术，CPU/PTOAS通过。生成HC数学区域静态TLOAD22→19，pto.fence4→0；保留GM写、阶段外发布/等待与自动事件。task_20261017_061445_12767489743已提交，未预记PASS。

Indexer scale物理页owner候选prepare/cpu v1通过，但历史探针显示204μs完成scale之后仍需等待235μs的query系数，调度估计也不降低总时长。因此只保留CPU候选，未上卡。

新增indexer_input_dependencies.py，扩展原row_dependencies的三个显式合同：quant→系数、quant→Indexer、系数→Indexer。T96每query的64头由w=q%48独占，scale视图平铺逐元素同址；每Indexer组覆盖两个请求，量化写者48→6、系数写者16→2。1..32叶×24组768种覆盖检查通过；保持MIX三成员同一输入要求、原同核顺序及QK的216条FFTS资源边。等待12183→10674，发布722→729。CPU完整编译进行中，尚无真机或性能结论。

## 2026-10-17 06:30：Tile直传、Indexer输入依赖终态与新阶段探针

task_20261017_061445_12767489743 / b16_row_deps_hc_front_tiles_v1 exit0完整PASS，三窗819.74/810.44/810.44μs，中位810.44，同轮动态891.96。
task_20261017_061600_130106629188 / b16_row_deps_indexer_inputs_v1 exit0完整PASS，三窗812.62/815.48/810.34μs，中位812.62，同轮动态882.64。两者均不替换现选。证据HC_FRONT_AND_INPUT_DEPENDENCIES.md及hc_front_and_input_deps_results.json。

为定位后续方向，允许已有row_dependencies组合stage_clock；CPU核对实际PTOAS原始C++的两物理函数、全部阶段边界及信号空闲区，每核88个时戳槽；开始误传wrapper源码导致定位失败，改用正确ptoas/csa_static_group.cpp后通过，未修改算子。task_20261017_062004_137819816687 / b16_row_deps_stage_clock_v1 exit0完整PASS。插桩span796.24μs仅用于归因，不参加700μs目标。

该样本Indexer AIC234.42..458.16、AIV247.90..461.94，QK/PV约481.7..647.36，O-A650.26..735.32，O-B712.20..766.78，HCpost772.68..796.22μs。KV norm虽然被Indexer占核推迟到456us，但本样本已在QK plan完成前完成，不能当作主要瓶颈。按最晚显式前驱回溯的链，数学/核内同步/发布段累计724.36us、就绪至开工间隙71.88us；这些不是硬件理论下限，也不能把后者全称为软同步开销。证据row_deps_stage_summary.json、row_deps_stage_critical_chain.json。

新增hc_mix_chunk_buffer.py候选：保持12成员×8行、原两个16×K256 pipeline和全部算术，以8×4096 BF16 UB Tile存每块结果；norm从UB提取，保留mixed GM写。区别于先前48成员整行resident算法，不增加Scalar归约。prepare_v1通过，CPU编译中；尚未接driver开关或提交NPU，也未声称收益。准备过程的qk_index_cache包装仅在该CPU子进程内生效，未改共享导入或已运行任务。

## 2026-10-17 06:34：HC分块UB缓冲CPU通过，已提真机

hc_mix_chunk_buffer_cpu_v1失败：8×4096宽Tile窄列assemble降低出的pto.tmov不满足A2/A3形状约束。v2把16个8×256块按连续行存入128×256 BF16，norm读取对应8行，未修改数学。prepare/cpu_v2完整通过，65536B驻留Tile只有一份，地址98336..163872，未越188416B可用UB。hc_mix_chunk_buffer_layout_proof.json检查32768元素独占与完整覆盖。
新增正式实验开关--hc-mix-chunk-buffer；prepare_v3生成entry的完整AST与已编译v2相同，没有重复完整CPU编译。task_20261017_063343_160455814507 / b16_row_deps_hc_mix_chunk_buffer_v1已提交，仍按原输出/TopK/六状态/图/30代/保护区和三窗L1检查，不预记通过或收益。
阶段归因增加可复现脚本summarize_stage_path.py；现选探针973个成员与调度一一对应，生成row_deps_stage_analysis/，结果与此前手工汇总相同。

## 2026-10-17 06:38：HC分块UB候选终态

task_20261017_063343_160455814507 / b16_row_deps_hc_mix_chunk_buffer_v1 exit0完整PASS；单SPMD三窗[814.74, 817.52, 812.92]μs，中位814.74μs，同轮动态907.88μs。输出/TopK/六状态、图/30代/保护区均通过；只固定fixture恢复，不是连续decode轨迹。相对现选未显示明确收益，不采用、不扩七档。未做同进程现选A/B，不将跨运行差值作为精确回退幅度。
所有CPU/NPU任务已终态，stage诊断和候选失败保留。无生产/F/依赖仓修改，无新commit。


## 2026-10-17 06:58：成员级CPU排程与权重缓存声明核对

member_schedule.py三种策略完成CPU图检查，973成员/877单元、216条FFTS资源边保留。
同成本原估计771.53us；critical802.49、earliest768.79、finish858.52，最好仅-2.74us且等待更多。
不上卡，不生成新二进制，见MEMBER_SCHEDULE_CPU_EXPERIMENT.md。
O-A物理IR已有K128内层和Left/Right双缓冲地址0/32768，避免重复优化。
进一步确认实验生成器过滤func_attr，导致Tensor数学体QA/主Q/O-A丢失原BYPASS；
KV已有显式Tile cache=1，仍保留两个IR属性，不能混为全部丢失。
weight_cache_policy.py按冻结原声明和本次绑定恢复融合AIC函数属性，检查整个调用图只读合同。
prepare_v1通过，移除新增属性后完整AST与现选相同；cpu_v1编译中。尚未上卡。

## 2026-10-17 07:05：权重缓存候选CPU终态与QK每query地址准备

weight_cache_policy_cpu_v1完整通过，IR缓存属性2→44；但C++完整2784561字节与现选相同。
当前PTOAS只有显式BYPASS offset才调整地址，无offset生成普通TLOAD。原动态也一样，
不是现选突然丢失了已生效的硬件功能。不提交等价NPU任务，保留开关默认关闭。
未修改PTOAS/ISA/Simpler，未猜测硬编码设备地址偏移，详见WEIGHT_CACHE_POLICY_EXPERIMENT.md。

新增qk_query_metadata.py：每query在tick%5==0准备512个压缩KV地址，
两个INT32 Tile共4096字节跨tick携带，四个压缩块复用；尾部tick禁止读超界query。
不假定第一个块有效，保留原负索引判断、逐行DMA、数学、FFTS和三槽流水。
代价是每AIV的地址计算256→512项，收益预期是页表读取最多4→1次。
prepare_v1进行中，未编译/上卡，不声称收益。


## 2026-10-17 07:15：QK每query候选终态、N80 WS分块候选

QK prepare/cpu_v1完整通过，驻留UB两份2KiB末端139264<188416。
任务task_20261017_070649_214406126149 exit0，完整exact/TopK/六状态/图/30代/保护区PASS。
L1三窗811.82/815.08/805.18，中位811.82，同轮动态904.92；没有明确优势，不采用、不扩七档。
详见QK_QUERY_METADATA_EXPERIMENT.md，尚无现选同进程A/B。

新增indexer_n80_ws128.py：针对旧N80的60KiB WS Right无法容纳下一Key，
将WS K384拆为0/128/256三个K128，Right每块20KiB；理论双WS+双Key=60KiB。
提前提交下一Key到Right，保留QK M384/N80、13个panel、原叶/FFTS/AIV TopK。
WS指令分段确实改变，不能只凭零系数或数学等价认定exact；CPU prepare_v1进行中。


## 2026-10-17T07:19:42：N80 WS K128 CPU通过并提交真机

prepare/cpu_v1完整通过。实际13份Key/39份WS Right，WS0/20480、后续Key40960/51200，
最大末端61440；12处下一Key extract先于当前QK且两Key槽不重叠，见物理审计。
task_20261017_071758_23254658630/b16_row_deps_indexer_n80_ws128_v1已提交，未预记PASS。


## 2026-10-17T07:23:07：N80 WS K128终态与QK全有效路径

N80 WS K128任务task_20261017_071758_23254658630 exit0完整PASS，L1三窗844.52/855.10/858.08，
中位855.10，同轮动态906.78。未采用，不扩七档；没有现选/旧N80同进程配对。

新增qk_all_valid.py：每个物理成员先扫描本组query×5个valid-plan项；全部>0才进入
消除重复块判断的快循环，任意无效则保留原循环。只删证明恒真的外层块判断，
稀疏行负索引、尾部、数学、KV搬运和FFTS保持。prepare_v1进行中，未编译/上卡。


## 2026-10-17T07:29:00：QK全有效CPU通过及真机任务

prepare/cpu_v1完整通过；AIV快/回退PIPE_ALL静态点3/4，mask读取0/4，
AIC为0/0、0/2。只说明生成代码变化，不预记性能。
审计FFTS发布初版搜set_flag_dev得到零，改为实际ffts_cross_core_sync统计；未改算子。
task_20261017_072611_24632809131已提交，终态待收取，见QK_ALL_VALID_EXPERIMENT.md。


## 2026-10-17T07:30:43：QK全有效终态

task_20261017_072611_24632809131 exit0，完整exact/TopK/六状态/图/30代/保护区PASS。
单SPMD三窗804.30/817.26/820.92μs，中位817.26，同轮动态905.18。
未显示明确收益，未采用、不扩七档，没有现选同进程A/B。
本轮全部CPU/NPU任务已收取终态。七档选择不变，700目标仍未达。


## 2026-10-17 07:43：Indexer设备短尾快路径

新增indexer_short_tail.py/--indexer-short-tail，最后1024分数块只剩1..64候选时，
仅计算第一个M384/N64 QK与M16/K384 WS。device valid_count选择分支，不写死历史。
原完整路径AST、FP16 FIX缩放/ReLU、FFTS交接/排空、AIV TopK均保持。
未写score_transfer后960列属于无效尾部；核对原九条排序分支的fillpad，
CPU检查叶内1..8192有效数、184314个query/step组合，通过2554个短尾组合。
沿用原S6组内位置有序前提；不是设备证明。见indexer_short_tail_contract.json。
prepare/cpu_v1完整通过，C++短/完整路径matmul2/32、FIX写L1为1/16、GM store1/16，
两条路径内部PIPE_ALL均0。indexer_short_tail_codegen_audit.json。
工作量模型：B4/B8/B16的最大核面板数不变；B24从528到513。不能凭总计算减少推断
B16明显提速；此模型不含TopK、FFTS或调度时间。indexer_short_tail_workload.json。
任务task_20261017_074327_274989723191已提交，只做B16现选基线上一个候选，
先exact/状态/图/30代/保护区，再三窗完整L1；未预记PASS，也未改七档选择。


07:48 B16短尾终态：exit0，全部输出/TopK/六状态、图、30代和保护区PASS。
L1为820.12/820.32/819.08μs，中位820.12，同轮动态902.04，不替换现选。
B24 prepare_v1也通过；工作量模型每核都省15个N64面板，单独提交
task_20261017_074803_283067829725 / b24_indexer_short_tail_v1，不扩大其他档。

另一独立候选--ramp-wait：沿用DCCI后volatile读取，失败后的等待间隔
4/8/16/32/64/100tick，随后保持100。每个新依赖重启短等待，不增加原子或删缓存维护。
wait_ramp_cpu_v1/audit.json核对：默认fixed完整C++与现选逐字相同；递增版
只有helper改变，GE条件、每次读前DCCI、写侧发布、payload及周期PIPE_ALL保持。
不把先前均匀0..400tick扫描重复一遍，仅检验一种由短到长的策略。
task_20261017_074820_28363061302 / b16_row_deps_ramp_wait_v1已提交，未预记PASS。
两项运行时均不再编辑driver/imported helper。


07:53 两项后续任务均exit0、原exact输出/TopK/六状态、图/30代/保护区PASS：
- B24短尾task_20261017_074803_283067829725：1146.14/1127.42/1127.60，中位1127.60μs；同轮动态1175.66。
- B16递增轮询task_20261017_074820_28363061302：833.74/829.94/826.22，中位829.94μs；同轮动态910.94。
两者均无明确性能收益，均未采用，不扩七档或继续扫描退避。未做现选同进程A/B，
不将跨运行历史差值定性为精确收益/回退。全为固定fixture，不代表连续decode。
见INDEXER_SHORT_TAIL_EXPERIMENT.md、RAMP_WAIT_EXPERIMENT.md及相应result。
本轮三个NPU候选均已终态；所有CPU exec session已收取；没有等待任务。


## 2026-10-17 07:58：Indexer N64循环化的CPU尝试

indexer_rolled_panels.py把16个显式展开panel改首块剥离、15次核内循环、末WS排空；
每步仍16次QK与16次WS、原M/N/K、量化/ReLU和FFTS不变，只减少展开代码。
这尚非指令缓存瓶颈的测量证据。prepare_v1通过，cpu_v1失败：编译器插入
Mat→Mat同地址tile.move，源/目标均180224；未提交设备。初版携带Key L1池，
该池每次只供当轮新写入的Key提取，无需跨迭代携带，因此v2改为迭代内创建，
仅循环携带Key Right和上一score L1。prepare_v2通过、cpu_v2编译中。
不改PyPTO/ISA，保留所有失败；循环物理缓冲未审核前不上卡。
indexer_rolled_source_contract.json只记源码操作顺序；其中页地址同表达式求值
不是独立验证，不将其计为正确性测试通过。


08:00 循环化v2终态：cpu_v2失败，实际PTO出现Key Right→Right tmov，
PTOAS拒绝此A3地址空间组合；另有score Mat→Mat移动。session12565已收取exit1。
仅CPU失败，不提交设备、不改依赖、不放宽校验。INDEXER_ROLLED_PANELS_EXPERIMENT.md
记录v1/v2与IR片段。未来可以研究两面板成组、避免Right跨循环携带，当前未实现。
本轮最终三个设备任务全部终态并PASS但未采用；CPU失败均保留，无运行任务。


## 2026-10-17 08:08：N64两面板成组循环

indexer_panel_groups.py / --indexer-panel-group 2：组内两个N64展开，组间8次循环，
组末完成WS再继续，不跨循环携带Key Right或score。QK/WS数学、叶分配、原页地址、
每1024块FFTS及AIV TopK不变。原N64 C++中QK与WS使用不同MTE1事件，
没有把它误当错误的共同等待；证据indexer_selected_qk_ws_order.cpp.txt。
prepare/cpu_v1完整通过。实际Key Right 0/49152，WS Right0，第二Key与前WS不重叠；
QK Acc0/98304B，WS0 Acc98304/4096B也不重叠；无Right→Right move。
生成组循环4个matmul、2个FIX、2个store、0个PIPE_ALL，循环C++片段27706字节。
这些仅证明分配/提交，不是并发或I-cache瓶颈测量。indexer_group2_codegen_audit.json。
生成AST实际store列检查2/4/8组均覆盖0..960的16块；只编译/上卡group2，
没有预记group4/8已通过。task_20261017_080841_317060310914已提交，待设备结论。


08:13 group2终态：task_20261017_080841_317060310914 exit0，完整输出/TopK/六状态、
图/30代/保护区PASS。L1 920.58/924.02/918.76，中位920.58μs，同轮动态909.50。
明显无收益，不采用，不扩其他档/组长。没有测到I-cache miss，不能将代码缩短视为优化成功。
证据indexer_group2_result.json和任务目录。


## 2026-10-17 08:22：Indexer 2048分数交接候选

保留原16个N64展开流水和叶分区，每两个1024块合为一次交接，私有双槽GM
[288,1024]→[288,2048]，Native/root布局不变。奇数尾块不增加数学量；
AIV段span按本叶剩余有效数截断，原三query Top-K/root合并AST保持。
contract核对40962组有效段、全部1..8192叶长与双槽通知计数模型；不是硬件证明。
prepare_v1与cpu_v1完整代码生成通过，实际AIV最大Vec端点131072B，
load为3×2048、stride2048，AIC READY在FIX、AIV ACK在MTE2 load之后，
尾部仍排空min(steps,2)。indexer_transfer2048_codegen_audit.json。
此处full_codegen_passed只指CPU，旧runtime_abi_supported=false来自32参数旧检查，
本会话使用独立256参数runtime；尚未预记设备PASS。

任务task_20261017_082119_33910122896已提交，仅128K/B16；运行期间不改driver/helper。

08:26 首轮2048交接exit1，输出与Top-K不符，六份状态完全一致；未采性能。
本候选遗漏三处root初始化步号：旧1024轮的step<2应改为新2048轮step==0。
保留失败result/log，修复在本目录helper，未修改依赖；v2准备通过，
contract补查初始化/合并选择，source核对仅三处条件改变，其余Top-K AST保持。
任务task_20261017_082629_34925986670已提交，未预记PASS。


## 2026-10-17 08:33：2048交接终态及Key池CPU结论

v2任务task_20261017_082629_34925986670 exit0，exact输出/TopK/六状态、图/30代/保护区全PASS。
L1 836.98/834.68/819.72，中位834.68μs；同轮动态899.98。未显示现选之上的收益，
未采用、不扩档。详见INDEXER_TRANSFER_2048_EXPERIMENT.md，保留v1失败和修正原因。
另indexer_key_pool.py只做CPU：128行Key池扩大1024行，源数学/地址/顺序不变。
完整codegen通过，但完整stage16 MTE1→MTE2 wait 19→20，反向均18，32个matmul不变；
没有目标等待减少，未上卡。抽取采用done_16与C++完整括号，纠正初次v30前缀只覆盖后半段。
见INDEXER_KEY_POOL_CPU_EXPERIMENT.md。CPU session96611已收取exit0，两NPU均终态。

08:35 新CPU候选qproj_l1_columns：保持20worker/原N256列所有权，每组拆两个N128，
QR L1入口驻留，两份K1024权重各一次加载。两个INT32累加链、K256显式交错，
理论L1 360448、Left49152、Right65536、Acc98304B；以实际编译分配为准。
仅准备/CPU编译中，未预记设备或性能PASS，不改变精度门槛。

08:37 主Q双N128列块CPU代码生成通过，实际Left地址0/24576，Right0/32768，
双Acc0/49152且彼此不重叠；两份整列权重位于L1 65536/196608，QR327680，
均在硬件容量内。两次matmul初始化+六次matmul_acc，保持每链K256升序。
证据qproj_l1_columns_codegen_audit.json；CPU session99975已收取exit0。
已提交task_20261017_083715_36883533451，先完整精度再L1，不预记收益。

08:42 主Q双列v1任务exit0，完整exact输出/TopK/六状态、图/30代/保护区PASS。
L1 809.22/806.46/797.56，中位806.46μs；同轮原动态903.86μs。
与现选历史807.16接近，不能直接宣布收益，下一次只作现选与候选同进程三窗L1。
对照版必须开启现选row_dependencies=True、QK地址缓存和100tick轮询；
复用paired_row_dependency_perf，旧调用默认行为保持，不重做动态metadata或Event采样。


## 2026-10-17T08:47:35：主Q同进程终态

task_20261017_084154_377975115763 exit0，双方exact/图/六状态/保护区/7代PASS。
现选L1 805.28/808.92/810.24，中位808.92；候选806.02/804.06/810.32，中位806.02。
中位差-0.36%，均值仅-1.35μs且范围重叠；未采用、不扩档、不重复该比较。
对照生成AST逐项等于b16_cached_row_deps_v1，不是此前未加行依赖的旧版。
测试只2次预热/每侧3窗L1，没有重复Event或动态metadata。
本轮所有CPU exec及NPU任务均已收取终态。七档现选不变，700μs目标未达成。

只读排查也确认：SWA原分支已按Native物理页批量gather，不能再假设它逐行读；
row_dependencies已有成员图传递约简和逐核observed过滤，不需要重复增加相同的seen缓存。
这些没有新增设备任务。下一候选可评估N32的双QK Acc流水（尚未实现），
与已失败/变慢的M192/N64分query方案区别是保留M384/WS K384，只沿候选列拆分。


## 2026-10-17T08:55:50：N32双QK Acc候选启动

只改Indexer AIC面板为M384/N32/K128与M16/N32/K384 WS，保留原归约和1024分数交接。
32面板展开，下一QK在当前FIX前提交。prepare_v1通过；cpu_v1编译中，尚无设备结果。
此方案区别于此前M192/N64分query，容量和源码先后不能当作设备流水收益。

N32 cpu_v1完整codegen通过，31对相邻QK Acc均0/49152交替、生成顺序保持。
Left query0/系数49152；首两Key Right0/4096，后续Key24576与WS Right0分区。
完整阶段64matmul/32FIX/32store。task_20261017_085633_40447024100已提交，尚无性能结论。


## 2026-10-17T09:06:11：N32设备终态

任务task_20261017_085633_40447024100 exit0；完整exact/图/六状态/30代/保护区PASS。
L1三窗985.94/976.20/981.12μs，中位981.12；同轮动态901.76。未采用，不扩档、不重测。
见INDEXER_N32_ACC_EXPERIMENT.md及indexer_n32_acc_result_v1.json。


## 2026-10-17T09:08:10：KV原样回写标记折叠

新增kv_touch_elision.py：将唯一kv_touch的父依赖直接接到QK/PV，保留写回完成和缓存维护。
准备v1被本地合同拒绝：call实参annotation已去掉Out，真实核体形参仍pl.Out。
修正为结合核体形参与allocation检查，属于候选检查器假设错误，不是组件bug。


## 2026-10-17T09:12:13：保留KV汇总者，只删除业务空操作

完全删除kv_touch的prepare_v2/cpu_v1通过，任务66→65、成员973→972，
但远端等待12183→12673，发布数仍722。汇总等待被展开成写回成员等待，未提交设备。
新增kv_touch_signal_only.py保留原阶段、等待表、代际和TNOTIFY/WAIT，只删除1024B自拷贝
以及无业务数据阶段的4个fence/2个全Scalar缓存维护。其他源码逐AST相同。
kv_touch_signal_only_prepare_v1通过，cpu_v1编译中；没有预记性能收益。

纯信号版本cpu_v1完整通过，实际C++ load/store各1→0、PIPE_ALL5→1、全DCCI2→0；
TWAIT/TNOTIFY保留，其他源AST和等待表不变。task_20261017_091307_19138223772已提交。


## 2026-10-17T09:19:37：KV纯信号首轮通过

首轮exit0，完整exact/六状态/图/30代/保护区PASS；L1中位802.96μs（802.96/812.72/798.04）。
同轮动态894.00。与现选差距小，补一次同进程三窗复核task_20261017_091800_27634922996，
只2次预热/3窗L1，不重复Event或动态metadata，候选尚未纳入现选。


## 2026-10-17T09:24:22：KV纯信号同进程终态

任务task_20261017_091800_27634922996 exit0，双方exact/图/六状态/7代/保护区PASS。
现选805.14/808.22/807.50，中位807.50；候选812.08/803.92/812.32，中位812.08μs。
候选未测出收益，不纳入现选、不扩档、不重测；对照生成AST与现选一致。
本轮N32、KV纯信号两轮NPU均终态，所有CPU exec已收取终态。


## 2026-10-17T09:25:07：L1循环原位更新CPU探针

小例子v1因遗漏SPMD index被前端拒绝；v2忽略assemble返回值，触发Mat目标不能复用的
保留区域拷贝错误；v3使用pool=assemble(...)完整CPU/PTOAS通过。
实际循环TINSERT写同一512B地址Mat，循环后TEXTRACT与FIX→MTE1同步保留，无Mat→Mat移动。
只覆盖小FP16矩阵/无分支/无量化，没有上卡，也不证明CSA完整流水。见TILE_LOOP_INPLACE_CPU.md。


## 2026-10-17T09:28:48：N64固定L1环候选启动

上一目标轮完成N32/KV候选终态和固定L1小例子，属于取得新证据的进展；无运行遗留任务。
本轮新增indexer_l1_ring.py：固定三槽Key/双槽score Mat，DPS返回值写回同一变量，
Right留在循环内部，保留M384/N64/K128和WS K384及原1024交接/AIV/叶计划。
先CPU准备和物理代码审计，不把小例子成功当作大核通过。

N64 L1环v1准备通过，但cpu_v1失败：Key池在最后一轮不更新的分支合流处被插入
Mat→Mat同地址188416自拷贝。并非此前Right循环携带，Right已在本迭代局部使用。
v2将最后QK/WS剥离，循环1..14每轮固定预取下一Key，避免可选更新的外层if。
旧源码、passes_dump和错误均保留，不修改编译器或降低检查。


## 2026-10-17T09:41:15：N64环v2被物理地址审计拒绝

v2完整CPU编译成功，但实际Key池中间phi17位于L1地址0，其他池/循环读回位于188416。
第二页TLOAD写入0地址，后续Key提取仍使用188416，且没有有效拷贝。不能上卡。
indexer_l1_ring_codegen_audit.py按物理地址拒绝该版本，证据与rejected_audit均保存。
只读查看gather_row定义确认它是ReadWrite DPS并复用输入0，不是无副作用API。
下一版尝试在同一个分支中完成两页加载，避免两次顺序分支产生中间phi；
继续先CPU审计，不修改依赖仓库，也不绕过地址检查。


## 2026-10-17T09:44:27：L1环v3编译拒绝，v4撤去Key循环携带

v3把两页加载合并在同一分支叶子，prepare通过，但CPU/PTOAS仍因Mat→Mat搬运失败。
错误定位source.py:904:25，原始PTO/passes/report全部保留，未上卡。
v4仅保留score双槽Mat跨迭代，每轮重新创建64×128 Key池并加载本面板。
这会降低Key预取提前量，因此不能预判收益；保留QK(current)/WS(previous)跨面板交错。


v4 prepare/完整CPU/PTOAS通过，物理审计Key池163840、score65536；
QK/WS Acc0/98304、循环内Right49152/0分区，无同层搬运和PIPE_ALL。
实际循环两个matmul、一FIX写L1、一次WS写GM，首块/排空均保留。
新增driver开关--indexer-l1-ring，并在首次设备执行前重新审计实际运行artifact。
task_20261017_094749_7744056502已提交，只128K/B16，不预记精度或性能。


## 2026-10-17T09:50:17：条件gather循环缩减探针

probe_loop_gather_phi.py CPU完整编译通过，初始池、两个if phi和前后transpose视图
物理地址均为0，没有复现完整CSA v2的0/188416不一致。未上卡。
该探针只含循环携带+两次条件DPS写，缺少完整CSA的外层分支和全图L1复用上下文。
保存allocation_audit.json及完整PTO/C++文本，不能把它当作v2已修复的证据。


## 2026-10-17T09:56:58：L1环v4设备终态及下一方向的边界

task_20261017_094749_7744056502 exit0，执行前真实artifact物理审计通过；
完整输出/Top-K/六cache-state、eager/图/30代、未写区域和workspace保护均PASS。
L1 984.56/1034.82/1008.86，中位1008.86μs；同轮动态905.34。
Event候选1133.55、动态951.86μs。未采用，不扩档，不补同进程复测。
现选策略仍807.16μs历史记录，最新现选同进程807.50；这两者不是v4同轮对照。

新增indexer_query_completion_model.py/json：保持各worker叶所有权，枚举1..32叶
共8448个item，证明按request重排不漏不重。实际B16当前6叶，每叶score块6/6/6/5/5/5，
原每核22块；按request重排前8请求的数据可在11块后就绪，后8仍22块。
但原阶段通知不改时所有消费者仍等22块，所以不能只重排循环就宣称收益。
merge每成员跨2个query、QK跨4个query，O-A整批行/O-B M96也会限制下游流动。
此模型没有时间预测、编译或设备验证，不新提交无收益依据的任务。

设备终态后将循环Key/WS Right区间不重叠纳入自动审计，对已有CPU与设备artifact
重新执行纯文本检查均PASS；未重新运行NPU、未改变任何数值结果。


## 2026-10-17T10:15:22：两组query完整成员图评估

上一轮为进展：L1环v4设备终态与两组query所有权模型已保存，任务已复核exit0。
本轮新增query_group_schedule_model.py，将Indexer/merge/plan/QK/O-A/quant/O-B/HCpost按48行拆组，
保持MIX原logical→physical映射，完整检查数学边、资源区间与同worker交接区复用。
1313成员/1121单元/43953数学边。使用原动态同形成本，拆分数学按一半、每成员另留1μs，
这是理想CPU模型，不是设备测量，也未生成设备算子。
前半优先855.325、最早就绪780.470、关键路径813.585μs；同模型现选771.530。
三种排程均未改善，所以不实现/上卡粗粒度两半流水，也不做更悲观投影成本扫描。
前半优先会延迟主Q；其余策略第二半QK仍把第一半O-A挤到后面。
模型与边界保留，不能称拆组方案本身在所有排程下都不可能优化。


## 2026-10-17T10:17:59：PV合并改为8-head，先检查UB

旧16-head的两版均需196608B超过188416B，未上卡。本次将每块16行缩为8行，
保留完整32-head运行状态、相同乘加顺序、先左后右、原FFTS和GM输出。
prepare_v4被旧“row-deps不得与qk_merge_heads组合”保护条件拦截，尚未进入编译。
补入明确合同：所有GM写表达式、scalar metadata读、显式同步与其他函数AST完全不变；
PV读取小Tile的并集仍32×512，且row_dependencies原行分片合同仍执行。
在这个已核对范围内允许组合现选row-deps/QK地址缓存，重新prepare_v5/CPU_v5；
未放宽UB上限或任何数值/保护区门槛，没有改依赖。


## 2026-10-17T10:28:40：两组调度补查及PV状态分块

flexible_mixed earliest允许每半重新分配物理核组，仍严格保护logical worker的GM交接区及FFTS；模型780.470μs，与固定映射相同。
prefix_then_phase0先为未拆分主Q等前缀安排位置，再优先前半链路，模型841.340μs。二者都不优于同模型771.530μs，不生成设备候选。
上述是理想CPU模型，不是实测；完整核对数学边、MIX bijection、物理/GM资源区间。五个模型结果均保留，无NPU。
qk_merge_heads prepare_v5成功，CPU_v5仍因196608B>188416B失败。MemoryReuse转储显示4个32768B状态基址和1个65536B共享arena；缩小incoming为8行不能消除整状态双份分配。
qk_merge_heads_v5_memory_allocation.json保存基址及高端使用点；其中15个动态视图未参与静态末端求值，超限结论来自编译器。
保存旧8-head变换到source_snapshots/qk_merge_heads_8_full_state_v5.py。v6将每侧loop-carried状态本身分成两个16×256值，移除完整状态slice/assemble链，逐元素old*alpha+incoming*beta顺序保持。
显式展开初始化、循环/分支yield、reset及最终16-head输出引用；所有GM写入表达式、scalar metadata读取、显式同步、其他函数AST仍做不变检查。尚待完整编译及设备验证。

### 2026-10-17T10:32:54：v6完整编译、实际地址审计通过并提交B16

qk_merge_heads_prepare_v6/cpu_v6完整codegen PASS；AIV五个分配槽为65536/32768/32768/32768/32B，总163872B。四个16-head状态位于65536/81920/98304/114688，各16384B，初始化/合并/重置原位且互不重叠。
新增qk_merge_heads_codegen_audit.py；CPU已通过，设备driver在第一次kernel执行前也检查实际artifact。禁止仅凭编译通过宣称地址正确或性能改善。
codegen比较显示PV合并局部PIPE_ALL原/新均0，V barrier9→13，set/wait各6→10；最终输出V barrier25→24。候选不保证提速，不把UB缩小外推为耗时下降。
task_20261017_103218_149245329068提交单B16/128K/S6，现选row-deps/QK地址缓存/100tick等待，只启用qk-merge-heads；正式权重，只与原动态PTO比较。待结果；不扩七档。

## 2026-10-17T10:39:13：PV循环状态分块v6设备完成，不采用

task_20261017_103218_149245329068 exit0完整PASS：eager/graph/final输出、Top-K、六cache-state零差异，30代通知和全部保护区通过。
L1候选1033.44/1351.48/1029.72，中位1033.44；同轮动态910.98/914.34/912.20，中位912.20。Event1118.08 vs959.95。未丢慢样本，不采用、不扩档、不重测。
现选此前807.16/807.50不是同轮现选配对。全部阶段静态指令计数只有QK AIV stage30变化：PIPE_ALL9→8、set/wait各80→82、TLOAD22→24、row_expand_mul4→8。不能据此定量归因回退。
QK_MERGE_STATE_EXPERIMENT.md、qk_merge_heads_result_v6.json及full_codegen比较保存。

## 2026-10-17T10:40:56：只调整Indexer的WS Right加载顺序，CPU候选

只读原体确认：现选已经是16个N64展开的QK(current)/WS(previous)，不能将再次展开称为新增流水。
新增indexer_ws_load_order.py，仅交换15对互无值依赖的WS Right extract与当前QK matmul，其他节点AST多重集合完全相同，首块/末WS drain、其他核体不变。
v1/v2准备失败是变换器遍历正在交换的列表，第二次把已移动节点当新模式；修为只遍历原列表快照，同时检查唯一末块排空。cpu_v2因prepare未成功、source不存在而退出，未进入编译，不是编译器问题。日志保留。
prepare_v3只在当前CPU准备进程注入catalog，未改默认catalog、生产入口或驱动参数，尚未NPU。物理Right若因生命周期缩短被复用并串行化，应停止，不直接上卡。

### 2026-10-17T10:44:21：WS顺序候选v4编译通过，但Right复用，不上卡

v3准备仅在末WS名称假设检查失败；核对真实buf_last_pair后改为最后5条AST位置不变检查。prepare_v4及cpu_v4完整通过。
实际地址检查：1/15对当前QK和前块WS Right重叠，现选15对全不重叠；两侧Acc均互斥。加载移到消费后缩短了活跃区间，自动复用破坏预期重叠。
阶段静态matmul32、PIPE_ALL1、M→MTE1wait32、MTE1→Mwait37均不变；set/wait166→165不是性能依据。不上卡、不改默认，无新设备结果。详见INDEXER_WS_LOAD_ORDER_CPU.md。

## 2026-10-17T10:48:26：WS后移仅保留后14对，先核对实际流水

上一goal轮为进展：PV状态分块已完整验证回退，WS全15对后移完成CPU编译并证实仅首对Right复用。本轮复核设备任务已终态、无旧编译进程。
保存all15_v4源码快照。当前只保留buf_previous_pair...ssa_v1原顺序，后14对交换；AST多重集合、首对、末5条排空不变。新增prepare_ws_order.py为可复现CPU准备入口。prepare_v5通过，cpu_v5编译中。
除实际Right分配，还要检查C++ set/wait对应位置，区分源发射顺序与真正等待；如原版本就允许QK与WS加载重叠，则不上卡重复测。

### 2026-10-17T10:52:19：14对CPU及实际C++通过，准备单B16验证

prepare_v5/cpu_v5完整PASS，实际Right与Acc的15对均不重叠，原首对仍load→QK，其余14对QK→load。集成build/device开关后入口文本与已编译v5逐字符相同，无额外CPU编译。
新增C++ set/wait位置审计：原版本在WS TEXTRACT之后到QK之间没有等待WS加载完成的直接串行边，不能声称新版本首次实现计算/搬运重叠。
实际改变是将QK提交放到WS TEXTRACT之前的FIX→MTE1和M→MTE1等待前。例如panel2，旧行1265/1266等待、1269TEXTRACT、1279QK；新1266QK、1276/1277等待、1280TEXTRACT。
新M→MTE1 EVENT_ID3来自更早的set（panel2示例行1001），不等待当前QK。所有14对都检查了该直接边界，无PIPE_ALL介入；不能外推实际重叠时长。
静态总matmul32、PIPE_ALL1、M→MTE1wait32、MTE1→Mwait37、set/wait166均相同。保留首对避免v4的Right复用。设备首次执行前也审计实际地址与14对顺序。


## 2026-10-17T11:04:49：审计提前编译绕过等待扩展，撤回三项退步归因

WS v5任务105218完成exit0、全部精度/状态/通知/保护区PASS；L1 1033.52/1037.68/1032.64，中位1033.52，同轮动态898.64。发现与PV v6异常接近后核查编译流程。
三项新增artifact审计均位于ExitStack外，_resolve_kernel_artifact提前缓存了默认TWAIT版本。实际CCEC输入AIC/AIV均74个默认TWAIT、0个CSA_WAIT_LOCAL_GM、无退避；现选基线则74个替换调用及100tick。不是PyPTO编译bug。
扫描74项已完成cached_wait测试，仅ring v4/PV v6/WS v5三项缺少回调记录，实际默认源码均确认。原始数据保留；精度有效，撤回由此认定数学核回退/不重测的结论。
修复driver三处审计进入所有编译扩展后再解析；新增执行前实际AIC/AIV源检查。已有正确产物通过、3份错误产物均因缺回调及源码不符被拒绝。初次CPU正例因最早manifest无后来ramp_backoff字段失败，改用已有明确false的正例，不篡改旧记录。
修正WS v6任务task_20261017_110148_204947624179，仅128K B16，原精度门槛/3窗完整L1保留。详见COMPILE_CONTEXT_CORRECTION.md、compile_context_*json。
CPU main_q_then_phase0模型862.105μs，不优于同模型771.530；没有设备候选。CPU资源下界当前665.622、两组理想拆分672.955，仅给定成本模型，不是硬件下界或达标证明。

### 2026-10-17T11:08:29：WS等待策略修正后完整通过

task_20261017_110148_204947624179 exit0，实际AIC/AIV均74个替换等待、100tick门禁通过；全部精度/状态/图/保护区/30代PASS。L1 814.12/812.34/816.48，中位814.12；同轮动态902.42/896.00/899.98，中位899.98。未采用、不扩档。
提交PV v7 task_20261017_110518_211373122343及ring v5 task_20261017_110544_21223701219，均仅B16/128K修正配置。期间误提交不完整命令task_20261017_110519_211380121075，发现后cancel提示已运行，立即kill并确认终止；约25秒内取消，没有性能结果，目录保留。

### 2026-10-17T11:13:15：三项修正验证完成，准备N64两面板预取CPU候选

PV v7 exit0，全门槛PASS，L1中位820.24、同轮动态909.56；ring v5 exit0，全门槛PASS，L1中位806.38、同轮动态885.92。两项均实际等待策略门禁PASS。没有七档选择变化。
新增indexer_l1_key_pairs.py/prepare_indexer_key_pairs.py：首块0，循环(1,2)…(13,14)，末块15及排空；每对在第一块QK前提交第二块Key的GM→L1，只让score双槽跨回边，避免原Key分支phi别名。
相同M384/N64/K128、WS K384、FIX 2^-10/ReLU、原1024交接/AIV/叶计划；prepare_v1通过，CPU_v1 session96548运行。实际地址与等待位置未核对，暂不加设备开关。

### 2026-10-17T11:16:31：两面板Key预取完整CPU通过并提交B16

CPU_v1 full_codegen PASS；相同score双槽98304B，Key池分支/循环原位，L1与score不重叠；三个QK/WS并行对的Right/Acc均分离，循环内无PIPE_ALL/非法同层搬运。
实际C++第二块Key四个静态地址分支TLOAD位点269/302/356/389，在首QK415、WS427之前；MTE2完成事件391发出，484才等待，位于首QK/WS后、次QK513前。是提交/等待证据，不是已测重叠收益。
接入独立--indexer-key-pairs/build选项；集成入口文本与已编译CPU入口完全相同，不重复CPU编译。实际设备仍须先进行同样地址/顺序及100tick源码门禁。
task_20261017_111540_229620229969提交，只测128K/B16/S6，正式权重、原动态PTO对照、完整精度和3窗L1，不更改七档选择。

### 2026-10-17T11:20:58：两面板Key预取设备完整PASS，未采用

task_20261017_111540_229620229969完成exit0；设备实际Right/Acc/L1/预取顺序、100tick等待源码门禁均通过。输出、TopK、六cache/state eager/graph/final全部零差异，30代与全部保护区PASS。
完整L1 813.42/814.14/811.40，中位813.42；同轮动态907.18/898.80/892.24，中位898.80。Event926.35 vs950.66，不能替代L1验收。没有优于现选的证据，不采用、不扩档、不重复。
真实C++确有第二Key预取发射提前，但没有转化为整层已测收益，不把静态事件顺序作为速度证明。原N64单面板ring旧产物仍通过扩展后audit默认路径。


## 2026-10-17 11:50：排除重复缓存方案、页基址选择CPU限制，提前KV通知未获收益

交接提出的Scalar读写/Tile DMA分类维护，已由payload_cache.py和b16_cached_dynamic_precise_v1实现验证；旧中位838.68μs，03:47也已复核。本轮不重复实现或上卡。
Indexer任意stride参数当前JIT明确拒绝，读取接口与现有TestUnsupportedLayoutSlot确认。GM双基址先select再gather探针v1因未用block index被拒绝；v2改显式worker后因不同backing SSA的if返回被拒绝。均CPU、无NPU、无依赖修改，见INDEXER_PAGE_BASE_SELECT_CPU.md。
QK early候选只改三个阈值144→96，原AST其余部分不变。新向量索引路径与旧阈值制定时不同，因此只重新评估该T96配置；不把旧阈值当bug。prepare_v1/cpu_v1完整通过，集成入口与编译过的源码文本相同。实际顺序Score wait→KV notify→softmax→Prob notify→PV wait；2wait/2notify/9PIPE_ALL与原相同。所有artifact解析仍在编译扩展入栈后，100tick门禁保留。
任务task_20261017_114240_273482232688、b16_row_deps_qk_early_kv_v1，exit0。eager/graph/final输出、TopK、六cache/state均零差异，30代通知和保护区全PASS。L1 823.38/816.34/833.82，中位823.38；同轮动态905.06/907.42/900.04，中位905.06。没有优于现选证据，不采用、不扩档、不重复。没有本轮现选配对，不精确归因跨轮差额。见QK_EARLY_KV_EXPERIMENT.md。

## 2026-10-17 11:51：双AIV按职责分工CPU原型

qk_role_split.py：AIV0采全128行KV，AIV1负责64head softmax/PV归并/输出，Cube保持原样；两条AIV都保留原Score/PV等待及KV/Prob通知。无任务内新增GM信号、无新增workspace。64head状态128KiB拆成16个8×256 FP32循环值；softmax与最终发布按8head处理，每head运算次序保持设计，尚需设备验证。所有packed输出改由AIV1发布，明确不能沿用旧O-A按head半区选择AIV0/1的依赖。CPU准备暂保留原整父任务依赖，不能作为现选策略的性能比较。
qk_role_split_protocol.py通过3328个调度、471867动作：两query全部1024种valid mask×3个调度，加256种四query随机mask×3代。五份缓冲区槽复用与每代事件排空通过，事件credit抽象峰值3；不证明硬件容量/DMA/缓存语义。
prepare_v1通过；cpu_v1在ConvertToSSA拒绝分支外kvbuf引用，未到内存规划。保留source_v1快照和全部失败产物；v2将原始KV和压缩KV各自store留在各自分支，用独立kv_done接收循环结果，数学与事件不变。prepare_v2通过，cpu_v2正在编译；尚无设备结果或性能。

### 2026-10-17 11:57：双AIV分工完整CPU通过，输出所有权依赖修正后上卡

cpu_v2 full_codegen PASS。实际16个8×256 FP32状态共131072B，起点16384..139264，每份8192B，互不重叠；init/carried/valid地址相同，16个PV add逐个写回。QK显式Vec shape地址范围163328B，非完整allocator报告。C++8个rowmax/rowsum/最终rowdiv，两分支4wait/4notify，5PIPE_ALL，源码事件提交顺序审计PASS。
row_dependencies新增默认关闭的角色所有权分支：从真实源码检查全部packed输出只在AIV1，768行唯一写入且八组O-A消费全覆盖；O-A改等对应AIV1。旧六项源合同和行覆盖证明PASS。prepare_v3使用新证明，角色数学AST与已编译v2相同，无重复数学编译。新增--qk-role-split，生产/依赖不改。
task_20261017_115641_29670009014 / b16_row_deps_qk_roles_v1，只测128K B16 S6、正式权重、原完整精度/图/30代/保护区和三窗L1，设备编译后再检查实际状态地址/事件及100tick等待。提交时无设备结论，见QK_ROLE_SPLIT_EXPERIMENT.md。

### 2026-10-17 12:04：双AIV分工完整设备PASS，但性能无收益

task_20261017_115641_29670009014 exit0。eager/graph/final输出、TopK、六cache/state均零差异，30代进度、全部保护区PASS，实际产物的状态地址/事件顺序/100tick等待门禁PASS。L1 887.72/885.08/870.46，中位885.08；同轮动态897.74/901.30/908.12，中位901.30。未采用、不扩档，不把4个内部PIPE_ALL减少直接当作提速。
观察期40s工具等待超时不是kernel超时，原任务后续正常完成，没有取消重启。只读本地ISA集群映射说明mode2两AIV与4bit计数，未发现设备事件丢失证据。
为了定位角色分工为何没有收益，提交单次阶段时戳诊断task_20261017_120238_306765112993 / b16_row_deps_qk_roles_stage_v1，warmup0/iters1/windows1。含插桩且暖机不同，eligible_for_700us_target=false，不能替代正式结果或精确推导跨轮退步幅度；未重复全量七档。运行中不改其导入代码。

### 2026-10-17 12:08：双AIV分工阶段诊断完成，主要退步定位到QK/PV

task_20261017_120238_306765112993 exit0，eager/graph/final全exact、4代进度、保护区PASS。插桩span867.12，L1单窗878.80均不参加700目标。与历史探针相比，Indexer AIC数学/内部等待/维护中位211.03→212.99；QK AIC150.93→206.39、AIV0 157.90→207.90、AIV1 158.12→222.80μs。本轮QK AIC/AIV0/AIV1最后结束701.30/703.50/718.38，O-A最早720.94，AIV1是组内尾部。
暖机与采样不同，不精确分摊跨轮差额，也不把阶段墙钟当pipe独占忙时。小8-head Tile令softmax变为AIV1八组、发布更细，且单条AIV承担双倍head；这是待CPU核对的开销来源，没有独占阶段探针证明占比。原版不重复上卡，先考虑更大head Tile实际UB/状态别名，不能只凭理论余量扫描。详见QK_ROLE_SPLIT_EXPERIMENT.md和qk_role_split_stage_comparison_v1.json。
只读复核member_schedule.py：原成员三种启发式和query两组六种模型已验证过，不重复；未安装/使用新求解器。当前所有任务终态，生产/F/依赖仓/共享安装保持，七档选择不变，goal active，700目标未达到。


### 2026-10-17T12:22:59：角色分工的head Tile分离，CPU确认发布峰值超UB

在现有8-head分工版本上，将PV状态块扩为16×256、softmax扩为32×128，最初发布也为16head。
qk_role_tiles_h16_s32_prepare_v1准备PASS，cpu_v1在AllocateMemoryAddr拒绝Vec194048B>188416B；无设备执行。
从37_after_MemoryReuse逐项核对，AIV各allocation求和正好194048B。状态仍128KiB，PV的两份16KiB临时区和
最终32KiB FP32发布结果复用；16KiB BF16输出与两份4KiB RoPE临时量、4KiB swap索引同时存在，
因此单独缩softmax并不能消除发布峰值。诊断qk_role_tiles_ub_diagnosis_v1.json保留全部allocation。

将publication_tile从state head_tile解耦：仅发布回到8head，从对应16head状态slice连续行，
仍为32head softmax、16head PV；原运算次序、Cube、GM接口和事件保持。旧8/8及16/32同发布
版本AST回归一致，旧8/8/8编译产物扩展审计PASS。新增快照qk_role_split_source_h16_s32_p16.py.txt。
qk_role_tiles_h16_s32_p8_prepare_v1 PASS，cpu_v1编译中；当前尚无此变体设备结果，不修改七档选择。


### 2026-10-17T12:24:22：16/32/8候选完整CPU通过

qk_role_tiles_h16_s32_p8_cpu_v1 full_codegen_passed=true。MemoryReuse的全部AIV Vec allocation合计
167552B，低于188416B；与h16/s32/p16的194048B相比减少26496B，实际全局复用结果不同，
不能仅按局部Tile字节数估算。8个16×256 FP32状态起点0..114688，互不重叠且原位更新；
incoming不覆盖状态，8次旧值乘法/更新逐槽匹配。softmax2个32head rowmax/rowsum，发布8个8head rowdiv，
仍4wait/4notify/5PIPE_ALL，提交顺序PASS。全部记录qk_role_tiles_h16_s32_p8_codegen_audit_v1.json。

新增默认关闭--qk-role-wide-heads，限定与--qk-role-split组合，明确使用(状态16,softmax32,发布8)；
设备执行前检查实际编译Tile与请求一致。行依赖继续检查AIV1独占全部8组packed输出。
CPU集成准备中，不重复已经通过的数学编译；尚无设备性能，不修改现选。


2026-10-17T12:25:55：集成行依赖后全部数学体AST与CPU通过版本一致；提交task_20261017_122441_342567127888，b16_row_deps_qk_roles_wide_v1。仅B16/128K/S6，全门槛+三窗无插桩L1。任务执行中冻结driver/helper，不改现选。


### 2026-10-17T12:28:27：32head softmax/16head PV/8head发布全门槛PASS，未采用

task_20261017_122441_342567127888 exit0，b16_row_deps_qk_roles_wide_v1。
eager/graph/final的输出、TopK、六cache-state均零差异，30代进度及全部保护区PASS。
实际Tile(16,32,8)、状态/事件及100tick等待源码门禁PASS。无插桩完整L1
830.46/827.18/832.04，中位830.46；同轮动态899.68/911.08/912.48，中位911.08。
比旧8/8/8分工的历史885.08明显降低，但仍未优于现选807.16/807.50；不同轮次无现选配对，
不精确归因差额。未采用、不扩档；暂关闭仅靠分工/增大head Tile的优化方向。
见qk_role_tiles_result_v1.json。所有设备任务终态，700目标未达。

独立新增fixed_core_local_schedule.py，只在CPU从现选顺序进行固定物理分核的相邻交换，
与前轮三种从头贪心分配不同。数学边与216条FFTS组复用边固定，MIX按共同开始建模，
防止误用成员独立开始的时间；原模型771.53，共同开始模型773.41。120秒局部搜索中。
未生成二进制、未接入生产/测试driver，运行中的设备helper没有修改。


### 2026-10-17T12:35:07：固定分核局部搜索及整组后移模型均无收益

fixed_core_local_schedule_cpu_v1：120秒评估7679次、149个环被拒绝、13个等长移动；
整层共同MIX开始模型773.41→773.41，未改善。另将48条AIV上的主Q后处理整组放在Indexer后，
fixed_core_defer_q_dequant_cpu_v1完整图无环，但773.41→788.94，未采用、未生成二进制、未上卡。
Indexer最早开始207.49→191.08，最后结束仍444.24；merge和plan延后，QK最早465.47→481.00。
这些是CPU成本图，不是设备性能归因。源码、全部模型、关键链见FIXED_CORE_LOCAL_SCHEDULE_EXPERIMENT.md。
原始L1不含可直接使用的逻辑block索引，需要匹配派发日志，未尝试从完成顺序猜分片。
目前所有设备与CPU任务都已终态；本轮不改变七档选择，goal仍active、700目标未达。


### 2026-10-17T12:47:41：固定分核全局MILP排程模型开始求解

只读复核Indexer页视图/Key双面板/N128/双Acc旧候选，未重复它们的测试。已有Key提交探针
11.81μs不足以填补整层差额；不把异步提交时间当独占搬运忙时。
尝试在N/cache/schedule_solver_env安装ortools：默认索引Proxy403失败，官方索引也未取到版本。
全部日志保留，未修改共享环境。改用现有SciPy1.13.1/HiGHS1.2.0，无需外部安装。

fixed_core_milp_schedule.py保持物理映射和数学/FFTS依赖，MIX共同开始，每核成员区间互斥，
允许全局换序而非前轮相邻交换。877单元、973成员、2241顺序二进制、34641约束；
已把原773.41μs共同开始安排代入所有约束，最大误差5.68e-14，证明模型保留合法基线。
求解120秒，8线程。若有解还要独立检查数值、同核区间，补足真实核组就绪前驱、重建进度表
并检查偏序，不能把模型共同开始当作已实现的设备同步。目前没有新NPU任务或性能结论。


### 2026-10-17T12:53:57：MILP前两轮结束，约束闭包验证通过，开始定向可行性搜索

v1在120秒内没有找到整数可行解，solver status=1；不能据此称不可行，已有合法基线见证保留。
weighted_precedence.py对带时间差DAG约简24825→20402条边，逐对242701个可达关系的最长
时间差全部保持，最大误差0。另4457条资源顺序已由数学路径蕴含，v2模型约束34641→25762。
v2最多允许96对顺序偏离基线，包含成组换序机会，120秒找到773.41μs合法安排但没有改善。
候选独立约束/整数/同核区间检查PASS；补230条真实MIX就绪前驱后模型仍773.41；
通知/等待翻译的偏序一致，远端等待12176、发布754。
求解gap12.65%，未证明最优，不能把675.54的模型下界当设备下界，也不能当实测达标。
源码v1/v2快照、完整日志、候选图均保留，没有上卡。

v3改为零目标函数的可行性问题，限定模型安排≤750μs，继续相同96对邻域和数学约束；
这是筛选有意义改进的门槛，不是把设备700目标改为750。程序单独读取makespan变量，
不会把零目标值当运行耗时。当前120秒搜索中；不存在新设备任务。


### 2026-10-17T13:02:43：固定分核MILP定向搜索终态，保持现选

v3限定模型≤750μs、最多96个顺序变化，120.074秒内未找到整数可行解；不是不可行证明，也不是700目标达标。全部求解日志与audit保留，无设备任务。当前三个MILP方向均结束，不扩大重复搜索。全部CPU任务已终态，修正PENDING_JOBS旧会话30868。
下一步只做源码级RoPE相邻交换CPU探针，评估是否能合法避开gather触发的整函数冗余同步消除禁用；不删除PTOAS安全检查、不修改依赖、无性能结论。用户明确允许按batch/长度档位分别编译，当前已按此约束实现七档，继续保留。


### 2026-10-17T13:06:38：RoPE交换与批量gather候选

rope_pair_swap_probe_v1在PTOAS因[256,2] FP32行只有8B而失败，A3要求32B对齐。v2用[64,8]转置、4对行extract/assemble再转回，完整CPU编译通过，无gather，但新增两次转置和8行复制；仅保留可行性证据，未上卡、未称性能改善。
另新增rope_flat_gather.py，保留原输入索引，改第r行的c为r*64+c；仅替换kv_rms_norm_rope(32行)、rmsnorm_rope_cache_write(16行)、rmsnorm_rope(16行)的三个逐行gather循环。修改外AST逐项复原一致，枚举262144个合法行/列选择验证索引归属。GM、原数学/尾块/所有跨核同步不改；单行tail保持原样。完整CPU准备进行中，无设备任务。


2026-10-17T13:08:57：rope_flat_gather_prepare_v1完整准备通过；CPU v1命令误用static路径，FileNotFoundError，未进入编译。v2按实际variants目录开始编译。新增默认关闭--rope-flat-gather接入，integrated prepare_v2进行中；数学体与编译候选核对后才提交设备。详见ROPE_GATHER_EXPERIMENT.md。


### 2026-10-17T13:11:33：批量RoPE完整CPU通过并提交B16

rope_flat_gather_cpu_v2 full_codegen PASS。实际三个TGATHER分别为1024/1024/2048 FP32元素，索引均由原列号加整数((linear>>6)<<6)产生；同时活跃的源/输出/INT32索引/scratch分别核对地址区间，无重叠。完整报告rope_flat_gather_codegen_audit_cpu_v2.json。集成prepare_v2全部AST与已编译v1相同，不重复数学编译。新增执行前实际产物门禁，仍在所有编译扩展入栈后解析artifact，100tick检查保持。
task_20261017_131052_534942170 / b16_row_deps_rope_flat_v1，只测128K/B16/S6，正式权重、原动态PTO对照、完整exact/TopK/六cache-state/30代/保护区及三窗L1。提交时无设备结论，不改七档选择；执行中冻结导入driver/helper。


### 2026-10-17T13:15:08：批量RoPE完整设备PASS，未采用

task_20261017_131052_534942170 exit0；eager/graph/final输出、TopK、六cache-state均零差异，30代与保护区全部PASS。实际三处Tile/索引链/UB与100tick等待门禁PASS。完整L1 819.58/815.10/810.50，中位815.10；同轮动态892.30/923.18/920.58，中位920.58。没有优于现选证据，未采用、不扩档；没有本轮现选配对，不精确归因跨轮差额。源码和原始结果保留，不重复该候选。

新增compile_outlined_cached_wait.py，只在CPU组合已有全阶段独立/noinline代码生成与现选100tick等待；数学使用现选原版本，不叠加批量RoPE。原全阶段独立实验发生在旧默认等待时期，不能直接当现选组合的性能结论。未删除driver的组合保护；先核对全部阶段外层事件闭合、helper自身事件保持及最终产物等待替换，尚无新设备结论。该C++组合仍是诊断，不作为生产或700目标完成。


### 2026-10-17T13:19:27：全阶段独立编译加100tick CPU通过，窄范围开放组合诊断

outlined_cached_wait_cpu_v1完整CPU通过，74个noinline helper覆盖全部现选阶段，外层AIC/AIV数学事件只在成对闭合且全部阶段已分离时清理；helper自身事件不改。实际两个CCEC源码各74个CSA_WAIT_LOCAL_GM、无TWAIT残留，DCCI后volatile读取与100tick均在。独立门禁首次仅因后端将AICORE正规化成static __aicore__而拒绝，修正识别语法后完整覆盖/外层无事件/等待链检查PASS，未改候选生成代码。
只开放B16/128K、全部阶段、compact参数、noinline、现选QK索引和rowdeps/earliest_ready/cached100组合，部分阶段仍拒绝。task_20261017_131724_17808923318被另一处旧rowdeps组合保护拒绝，exit1，未执行算子；保留失败日志。该保护只对已受上述严格检查的cached组合开放，CPU直接执行参数解析/校验前缀，确认原现选和允许组合PASS，部分阶段和缺noinline仍拒绝。
重提task_20261017_131834_1981138300 / b16_row_deps_outlined_cached_v2；仍完整exact/30代/保护区及3窗L1，无Native。此C++组合归因eligible_for_700us_target=false，不把诊断数值记作正式实现达标，不改七档选择。


### 2026-10-17T13:22:21：分函数与现选等待组合完整PASS，性能持平

task_20261017_131834_1981138300 exit0。现选数学/行依赖与全部阶段独立noinline+cached100的组合完整通过：eager/graph/final输出、TopK、六状态零差异，30代通知与所有保护区PASS，实际74helper/事件闭合/100tick门禁PASS。完整L1 807.02/805.34/810.66，中位807.02；同轮动态886.36/906.28/902.84，中位902.84。与现选约807持平，不采用、不扩档、不再重复该组合。该归因诊断eligible_for_700us_target=false。不能把静态屏障数减少直接当收益，也不能据此断言同步完全没有成本：函数调用/独立规划可能抵消其变化。
全部任务已终态，七档选择不变，700目标仍未达到，goal active。


### 2026-10-17T13:26:47：下一方向的主Q生产者范围CPU核对

qproj_dequant_ownership.py仅按冻结源码公式枚举B16/T96，不改数学或依赖。现有head_tile16时每个后处理worker读32个N256列块，覆盖全部20个投影worker，因此直接删一部分父核是不成立的。head_tile4重新分配行/head后，每个后处理worker仍执行16个head×8行的数学迭代，但只需8个投影worker；全部12×64个行/head块各有唯一写者。head_tile8需16父核，head_tile1/2也需8父核，但QK关联的后处理worker扩大到48/32；tile4和当前16的QK关联均为16。
这只是读写范围与公式枚举，尚未核对完整GM别名/其他共享产物、绑定常量、全图偏序或实际额外循环开销。没有新等待表，没有CPU时间收益预测，没有设备候选；不能直接删边上卡。下一步先补完整合同并以CPU模型筛选，再决定是否值得编译。该方向与已关闭的仅改投影worker数/主Q后移/Tile预取不同。


### 2026-10-17T13:42:01+08:00：主Q后处理依赖范围模型终态及用户分档澄清

qproj_dequant_dependencies.py已核对投影INT32共享视图、后处理BF16输出与QK reshape地址一致、额外共享只读输入及唯一写者。保留原同核顺序和216条FFTS复用边；通知/等待翻译偏序检查通过。qproj_dequant_dependency_model_v1三种head_tile16/8/4均为771.53μs，未缩短CPU模型关键路径；额外后处理成本1/2/4μs分别使整层等量变长。未修改数学分块、未上卡、未采用，不将模型值作为设备性能。该方向暂结束。

用户进一步明确：参考hc_pre.py按seqlen区间和batch场景选择实现策略，七个历史性能点不能作为算子支持范围。只读核对hc_pre入口实际按T<=32、33..48、49..95、精确96、97..144、>144分支；CSA不能照搬其数值，须结合Indexer历史长度分支、B/T及工作区容量。已有CSA冻结源本身是S6契约，尚不能声称支持其他S。后续先落实范围策略与测试点的区别，128K/B16低于700μs的目标不变。

重新读取七份原始result.json并由每侧3个L1窗口重算中位数，全部与selected_shape_policies一致；未新测、未运行Native。归档selected_seven_readback_20261017.json。

## 2026-10-17T14:00:38.544110+08:00：全档30%新目标及区间入口

用户确认每档单SPMD≤同轮动态分task PTO×0.70，替换原B16<700us单点门槛。原目标与原始粘贴文件均保留。七档历史L1原始起止回读一致，无新测试，差距见target_30pct_historical_gap.json。

新增range_policy.py，按原数学区间、B/T及波次选择策略；七档不是支持白名单。原192点分类、384端点、七点已选策略一致，成本严格全root shape与H匹配。CLI显式默认调优参数被静默覆盖的问题已修，首轮失败及7反例修复证据保留。

新增源码DAG fallback，合并前后期读写效应与显式deps/dummy/manual合同，检查RAW/WAW/WAR，保持MIX配对和通知协议。B12/H65536及缺参考的B17/H65536完整生成/协议/PYPTO lowering PASS；temporary_54 ScalarWriteLineShared告警对应INT32[T,16]逐行信号，行距64B，未修改同步代码。CPU不代表设备。两点分别提交 task_20261017_140019_112364329866 / task_20261017_140020_112371518858；未提前记精度或性能PASS。

### 区间代表点真机完成

上述两任务均exit0。B12/H65536动态714.66、单SPMD690.70μs，降低3.35%；B17/H65536动态893.52、单SPMD863.08μs，降低3.41%。eager/graph/final输出、TopK及六state均通过exact，信号代际/保护区PASS；原始L1起止重新计算一致，单SPMD各窗仅1task/72成员。B17明确使用source_derived_task_dag。均未达到30%目标，不把两个代表点当全区间证明。摘要range_entry_device_results_v1.json。

### 新目标后的独立源码与性能调查

七档阶段读回及O-A双输出tile输入复用见seven_stage_opportunities_v1/README.md。T192单group资源探针v2 CPU通过，Mat327680/Left49152/Right65536/Acc98304B，但Left请求depth2实际退为1；真实跨8组NZ view/stride合同尚未证明。v1 JIT TensorView根参数不支持，v3显式内层展开Right131072B超64KB；保留失败，不上卡、不当性能收益。

query_shape_support_audit_v1给出S5 pair跨请求、双AIV奇数尾漏query、O投影容量320非128整除及mixed[2,4,6]请求定位反例。原动态与冻结IR都需处理，不只是fixture常量；当前S6范围如实标记为限制，未收缩最终目标。

为补新source-DAG的动态metadata证据，提交task_20261017_140653_1348347342，H65536→65535→32766→65536同地址同二进制图重放，C点L32772是长历史下边界。不重采性能。

### B17新源码DAG的动态metadata边界PASS

task_20261017_140653_1348347342 exit0。固定地址/容量/同一二进制，H65536→65535→32766→65536（C1的L32772为长历史策略下边界），页表交换与位置/RoPE等原址更新，输出/TopK/state exact及保护区通过。每侧仅一次图捕获，无新增编译单元，变体间未清信号。此任务不采性能，也不是连续decode轨迹。40秒wait观察超时后继续核对原task，正常完成，未重提。
