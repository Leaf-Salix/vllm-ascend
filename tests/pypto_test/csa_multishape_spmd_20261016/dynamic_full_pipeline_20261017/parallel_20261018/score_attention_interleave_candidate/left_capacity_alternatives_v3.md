# 联合 v3：Left / Mat 实际分配池、必要驻留与复用边界

2026-10-18。只读审查已有 `cpu_long_v3` 与冻结 `source_candidate`；没有修改源码、依赖或安装，没有重新编译或运行设备。本记录中的 KiB 均为 1024 字节。

## 结论与本轮处置

v3 已通过 symbolic parse、Inline、SSA，最终在地址分配时报告 `Left buffer usage (77824 bytes) exceeds platform limit (65536 bytes)`。这个失败不能归结成“Score Query 48 KiB、Attention Left 16 KiB、Coeff 12 KiB 必须同时占 76 KiB”：实际 pass37 已把 Query 与 Attention 的 Left 放入同一个 64 KiB 池，额外的是另一份 12 KiB Coeff 池。

Mat 还有独立问题：pass37 四个 base 分别预留 48、48、256、384 KiB，合计 **736 KiB**。其中 256 KiB 最大成员是一个与父 Tile 地址分裂的 `transpose_view`；这个算子本来不搬数据，应该共享父存储。首次分裂发生在 pass36 `MaterializeSemanticAliases`。另两份 48 KiB 是同一逻辑 Score Query 经分支选择与循环回边形成的不同存储及搬运。应先修复和验证别名传播、循环状态合并及复用，而非立即增加 Coeff 的 Mat 常驻和每步 TMOV。

此前讨论的“Coeff 常驻 Mat，每 step 临时转 Left”只保留为备选补丁方案；本轮没有实现。它可能绕过一部分 Left 分配问题，却不能单独证明当前 Mat 能容纳，也不能替代 view 别名正确性修复。

## 1. 实际产物与地址分配含义

主要输入均在本目录下：

- [v3 结果](cpu_long_v3/result.json)：记录实际失败，不是设备运行结果。
- [pass35](cpu_long_v3/kernel/passes_dump/35_after_InitMemRef.py)：第一次建立 MemRef。
- [pass36](cpu_long_v3/kernel/passes_dump/36_after_MaterializeSemanticAliases.py)：本次 view 分裂首次出现。
- [pass37](cpu_long_v3/kernel/passes_dump/37_after_MemoryReuse.py)：复用后实际 base、offset 与池大小，2,758,088 字节；本报告仅引用，不另复制大文件。
- [AIC 源调度](source_candidate/interleave_cube_wave.py)、[Score step](source_candidate/indexer_score_stages.py)、[Attention tick](source_candidate/attention_tick_stages.py)。

pass37 第 225–233 行的 AIC 声明如下。它们是已声明的池，不是所有 Tile 大小相加：

|空间|base|字节|KiB|
|---|---|---:|---:|
|Right|`mem_right_381`|65536|64|
|Acc|`mem_acc_382`|131072|128|
|Vec|`mem_vec_414`|16384|16|
|Mat|`mem_mat_487`|49152|48|
|Mat|`mem_mat_489`|49152|48|
|Left|`mem_left_497`|12288|12|
|Left|`mem_left_502`|65536|64|
|Mat|`mem_mat_702`|262144|256|
|Mat|`mem_mat_1195`|393216|384|

依赖源码 `src/ir/transforms/allocate_memory_addr_pass.cpp` 第 347–404 行按 memory space 和 base Ptr 身份分组，每个不同 base 单独执行 `footprint.OpenBuffer(buffer_size)`，第 472–473 行核对 `HighWater()`。这个 pass 不再按生命周期把不同 base 叠放。故上述 Mat 736 KiB 是当前地址分配待预留的总量；不能把它说成已经通过硬件容量检查。当前错误先暴露 Left，尚没有完成后的 Mat 地址或可运行二进制。

该依赖文件完整位置：`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016/src/ir/transforms/allocate_memory_addr_pass.cpp`。本轮只读。

## 2. 四个 Mat 池到底是什么

### 2.1 两个 48 KiB 池：同一 Query 的循环版本

`mem_mat_487` 在 pass37:2387 建立 `score_query` 占位 Tile，在 4886 接收 `query_after -> tile.move(..., Mat)` 的回边值。`mem_mat_489` 在 2672–2678 接收本叶 Query load，随后用于 full-valid/padding；2762、2787、4754 又把 `query_iter` 搬到该池，以统一 if 分支返回。最后再搬回 487。

源码只需一份当前叶 Query `[384,128] INT8`。同一个 step 使用当前叶；换叶发生于上一叶所有 step 完成并执行末尾 ACK drain 后。若找到新叶，旧 Query 不再供后续数学使用；未找到或不换叶的分支只是继续原值。不存在源码要求“同时计算旧叶与新叶”的双 Query 流水。因此 487/489 的双份存储和同空间搬运来自控制流实现，不是数学必需双缓冲。

合并的必要条件仍是：分支返回、loop init/IterArg/yield 的整个 alias 集一致；覆盖旧 Query 之前，其上一次 Mat→Left 读取已经完成。不能只把两个数字地址强行设为相同，也不能根据 Scalar 已经过了调用行判断 MTE1 完成。

### 2.2 256 KiB 池：错误继承的 view 加上被复用的临时成员

最有区分力的证据是同一对 producer/view 在三个 pass 的变化：

|产物与行号|gather 输出 `qk_l1_local_inline5359__ssa_v1`|紧接的 `qk_l1_t_inline5348__ssa_v0 = transpose_view(...)`|
|---|---|---|
|pass35:5854 / 5860|`mem_mat_702 @ 0`|`mem_mat_702 @ 0`|
|pass36:5854 / 5860|改为 `mem_mat_486 @ 0`|仍是 `mem_mat_702 @ 0`|
|pass37:4835 / 4841|`mem_mat_1195 @ 0`|仍是 `mem_mat_702 @ 0`|

两者都是 262144 字节。中间没有复制操作；pass34:4489–4493 也只有 gather 与 view。pass37 后续 QK extract 读取这个 view。相同 executor 的尾波展开在 pass37:7165 / 7169 的 producer/view 都使用 `mem_mat_1195 @ 0`，提供同类型的正常对照。

`MaterializeSemanticAliases` 不是本轮 CSA 源级继承/局部别名规避新加的 pass。只读核对依赖 worktree 的 `git show HEAD:src/ir/transforms/memory_reuse_pass.cpp`，既有 HEAD 已有 `TransformMaterializeSemanticAliases`（第 5174 行）及 pass 注册（5469 行），也已有这两个漏 While 的 walker。当前依赖文件有更早的本地修改，故这里不把整个文件等同未修改上游；能确认的是：本次 `attention_tick_stages.py` 的 fresh-local 修正没有引入这个 pass，当前复现只能精确归因到它对该 IR 的处理。

这不是仅凭名称猜测别名：依赖 `src/ir/op/tile_ops/transform.cpp:600–610` 将 `tile.transpose_view` 注册为零复制、`set_output_memory_inherit_input()`、`no_execution_memory_access()`；第 469–475 行也明确保持同一字节存储。因此当前分离至少违反中间 IR 的共享存储合同，应按正确性问题检查；本次编译未完成，不能声称已在设备观察到错值。

同时不能直接删除整个 702 池：MemoryReuse 还把 Score 两份 `[384,64] FP16` panel、无实际消费者的部分 key 占位声明，以及不交错分支的 Attention 临时值放进它。例：pass37:5292 第一 panel 位于 702@0，5543 第二 panel 位于 702@49152。这两份 panel 在原 Score 流水中有相交生命周期。修复 view 后应重新分析这些成员，不能把 256 KiB 全部计作可以无条件省掉的物理空间。

### 2.3 384 KiB 池：跨互斥阶段的复用 arena，不是独立第三份 KV

`mem_mat_1195` 有大量复用成员。交错循环中，Attention KV 两槽位于 `[0,262144)`，Attention Query 位于 `[262144,327680)`；Score 临时值位于更高偏移或另一池。前后 HC、投影及不交错回退分支也复用此池。

主 wave 循环中第一 Score panel 位于 1195@327680、Key pool 位于 1195@376832，第二 panel 在 702；尾 wave 的 Query 已复用到 1195@327680，两份 panel 则在 702@0/@49152，Key pool 仍在 1195@376832。它们对应不同展开位置，不应把两套偏移机械当作同一时刻同时活跃。

同理，不交错 Attention 三槽在 pass37:9209 使用该池的完整 384 KiB；这条分支与两槽交错分支不同时执行。O_B 的 weight 256 KiB 与 activation 128 KiB 在 9668/9673 也复用这个池。池最大值不能直接解释成某一阶段的必需驻留。

## 3. 原数学与生命周期所要求的同时驻留

以下是当前 G6 / 两 Mat 槽方案的源码存储下界及一个容量可行的打包目标，**不是已完成的后端分配证明**。不拆 M、不改变 FP16 FIX/WS 边界、不删双 panel，不减少 KV/概率 GM 三槽。

|Mat 状态|字节 / KiB|何时必须保留|
|---|---:|---|
|Attention Q `[64,512] BF16`|65536 / 64|同 query 后续 QK tick；遇到下一 query 才更新|
|Attention KV `[256,512] BF16`|262144 / 256|两份 128 行槽；各自保留至该块 PV 消费|
|Score 当前叶 Query `[384,128] INT8`|49152 / 48|该叶所有 1024-history step；每步恢复 Left|
|Score 两份 FP16 panel `[384,64]`|98304 / 96|原 QK/FIX 与上一 panel WS 交错期间；不能并成一个槽|
|Score Key Mat pool `[128,128] INT8`|16384 / 16|本步分页 gather、预取与 extract/TMOV 期间|
|Attention PV probability `[64,128] BF16`|16384 / 16|本 tick Mat→Left 读取；与 Score 临时阶段可复用，但要保留硬件依赖|
|Coeff GM→Left 的临时 Mat `[16,384] FP16`|12288 / 12|仅换叶加载；完成 TMOV 后可复用 Score 临时空间|

稳态跨轮的三份 Mat 是 `64 + 256 + 48 = 368 KiB`。Score 服务期间加两 panel 与 Key pool，源码需要 `368 + 96 + 16 = 480 KiB`；Attention 服务期间加 probability，为 `368 + 16 = 384 KiB`。若为便于跨引擎边界保留一份独立 probability 临时，保守按 `480 + 16 = 496 KiB` 也仍低于 512 KiB，但这仍不等于现编译器能完成所需打包和同步。

这个目标不需要把 BF16 KV 与 INT8 Query 的同时活跃字节互相覆盖，也不要求 mixed-dtype reinterpret。它要求的是：同一逻辑 Query 的 phi/backedge 共享存储、KV view 继承父地址、互斥临时成员依生命周期复用；仍保留不同活跃内容的独立窗口。

Left 的源码需求为：Score 当前 Query 48 KiB + 当前 Coeff 12 KiB = 60 KiB；Attention QK 的两个 16 KiB Left 子块共 32 KiB，PV 用 16 KiB，它们可以覆盖已消费的 Query Left，而跨 step Coeff 12 KiB继续保留。因此交错阶段的数学峰值仍可控制在 60 KiB。前段原有 64 KiB Left 峰值发生在交错前。v3 的 64+12 KiB 失败体现的是池边界与复制版本没有合并，不能据此证明硬件上必然放不下。

## 4. 哪些异步边界不能因“源码已死”而省掉

1. Attention 每 tick 先 PV(r−2)，再 QK(r)。内部 Mat 槽 `r % 2` 只在旧块的 PV 最后一次 MTE1 extract 完成后才可由 MTE2 覆盖；GM 槽仍 `r % 3`，原 READY/ACK 条件必须保持。Scalar 经过 PV 末行不等于 MTE1 已完成。
2. QK 的 transpose_view 必须是当前 KV 的原位视图，不能另分配但不复制；也不能为了省池而忽略对应 gather 的 MTE2→MTE1 就绪依赖。
3. Score 两份 panel 分别承接 FIX 写入和 WS 的 MTE1 读取；panel 的重用仍需相应 FIX→MTE1、MTE1→FIX 等真实依赖。Key pool 的 MTE2 写入与 MTE1 读取也必须防覆盖。原两 panel 不能由 Scalar 顺序当成无重叠临时。
4. Score Query Mat→Left 读取完成后，下一叶才能覆盖 Query；Score/Attention 重用 Left 时，要保留 M→MTE1 的再填充依赖。当前 v3 尚未产出最终事件 C++，不能声称这些重用已经得到实际后端验证。
5. 同一个 Tile 的 if/loop 版本可以共享物理地址的前提是语义上没有需要同时观察的旧值和新值；不能全局合并所有同形 Tile。循环 init、分支 false 返回、tail drain、空工作路径都必须保留。

本次只发现中间 IR 与源码可驻留量之间有明确缺口。是否由合理的 PyPTO/PTOAS 修正同时恢复地址、别名与异步事件，需要后续独立编译器验证；不能只降低平台限额检查或固定物理地址。

## 5. Coeff→Mat 的精确备选方案与为何暂不实施

如果别名/复用修正后仍需要源码备选，最小范围仅为 `interleave_cube_wave.py`：

1. `score_cube_leaf_init_g6` 的第二返回 Tile 从 Left 改为 Mat，直接返回原 `buf_coefficients_l1`，删掉该处一次 TMOV。
2. `score_coeff` 的初始句柄和 loop carry 类型相应改成 Mat，仍为完整 `[16,384] FP16`，不改 GM 入参、valid shape、root ABI 或 phase。
3. 每个真实 Score step 在现有 Query TMOV 邻近位置执行 `coeff_left = tile.move(coeff_ready, Left)`，传给原 `indexer_score_cube_step_g6`。step helper 的数学、shape、INT32 QK、FP16 FIX 与 WS 顺序、FFTS 和叶末 drain 均保持。

数学 AST 审计应确认 step helper 完全相同、Coeff GM load 完全相同，新增的仅是纯 FP16 位模式移动的位置/次数；没有 cast、归约或 tile 形状变化。Query 的尾组 full-valid 补齐仍逐项保留。

代价并不为零：原每叶一次 12 KiB TMOV，改成每 step 一次；合法 step 数为 1–8，额外 `(steps−1)×12 KiB`，即每叶 0–84 KiB 的 Mat→Left 搬运。稳态 Mat 还会多驻留 12 KiB。源级 Score Mat 数量变成 492 KiB，若独立保留 probability 临时则 508 KiB，余量很小。当前 736 KiB 池问题未解决前，这些名义数值不能当作可编译证据。

用户随后授权合理修复内存复用，因此当前优先级是保持冻结 v3 数学源码并修复编译器合同；不直接实施上述备选，也不开始新 CPU 或设备测试。

## 6. 历史排重与明确未做事项

- `tests/pypto_test/csa_multishape_spmd_20261016/ATTEMPTS.md:612–655` 已记录 G6 的 M384/N128 Acc 容量失败，以及改成两个 M192/N128 后的完整设备 PASS 但未采用。后者同次 L1 中位 883.00 μs，对比当时现选 815.84 μs没有收益；不能将不同轮比较冒充本候选实测。
- 同树 `INDEXER_PARTITION_EXPERIMENTS.md:38–65` 已记录 M384/N64→两个 M192/N64 的双 Acc 方案，完整通过但增加 MMA/FIX/store 和事件位点，没有提速。
- 因此本报告不建议通过拆 M、扩大 N 或重跑上述负例来避开当前内存池问题。当前 Query/Coeff 版本合并、transpose_view 继承问题与这些数学改法不同。
- 本轮没有 PyPTO/PTOAS 修改，没有源码候选改动，没有新编译、NPU、精度或性能结论，没有 hash 或大文件复制。pass37 的 view 缺口是中间 IR 证据；尚无最终设备错值证据。480/496 KiB 是源级驻留及打包目标，尚不是修复后 allocator 峰值。

## 7. While 遍历遗漏与 Query / Coeff 双份池的独立核对

依赖当前 `memory_reuse_pass.cpp:1286–1298` 的 `VisitForStmts` 递归 Seq、For、If、Scope，未递归 While；`VisitIfPhisForAccumulator:1327–1343` 同样遗漏 While。另一方面 `VisitBranchStorage:995–1001` 已递归 While。即：外层 While 内仍会进行分支存储规范化，但其内部 For 的 loop-carry 规范化和 accumulator phi 合并不会从这两个入口到达。

联合源码的真实结构与此完全吻合：主 wave executor 处于外层 Scalar `pl.while_` 内，内部 `pl.range` 携带 Query/Coeff；尾 wave executor 在该 While 之后。主 wave pass37:2762/2787/4754/4886 出现 Query Mat→Mat 搬运，2763/2788/4755/4883 出现 Coeff Left→Left 搬运。尾 wave 的相同值在 pass37:5228/5231/5259/7089–7096 直接 alias/yield，同样的叶加载与 step 数学不需要这些复制。

这足以给出有针对性的源码缺口与正反对照；还不能在未执行修复后的 pass 时承诺“补两行就一定到 480 KiB”。后续 alias 规范化、分组/活跃区间及对齐仍可能保守。对当前 Mat736/Left76，两处不同层级的问题应分别解决：

- view 继承修复：保证父、transpose_view 以及其读者共享同一物理窗口，不能保留无复制的独立 256 KiB view base。
- 遍历修复：让 While 内的 For/If 经过与外部相同的既有 retarget 与活跃值检查。最小修改是两个 walker 增加 `WhileStmt -> Visit...(body_)` 分支，不需要把 `PropagateFromForStmt` 强行套到 While，不需要放宽旧值可见性或 pinned/offset 限制。
- 后续容量判断必须重新统计整个函数所有不同 base 的实际 extent；不可简单将两个错误的大小从 736 KiB 相减。702 池内的两份 panel 仍有真实用途，1195 的多阶段成员也要重新打包。

对当前源码，一份 Score Query 和一份 Coeff 没有同时读取两个历史版本的语义：换叶真分支用新值，假分支继续旧值；前叶 Score 已完成且 ACK 已排空。当前内容须跨轮保留，但上一轮版本不需要独立留存。Left 的合适目标仍是一个最多 64 KiB 的全函数池，交错区间在其中容纳 Query `[0,48 KiB)` 与 Coeff `[48,60 KiB)`；Attention 使用前 32 KiB，不覆盖 Coeff。此地址只用于说明存在容量安排，不是建议手工固定地址。

### 最小回归设计（只设计，未运行）

|例子|需要证明的事实|不得误作的推论|
|---|---|---|
|根作用域 For 与外层 Scalar While 包住相同 For；For 带一个 Mat Query 和一个 Left Coeff，含“首步新 load / 后续 carry”分支|两种控制流的 loop-carry/phi 均到达同一既有规范化逻辑；若无独立旧值读，最终每个逻辑状态为一个物理窗口，无多余同空间回边 move|不要求所有不同 Tile 一律合并|
|外层 While 零次、内层 For 零次及一次；初始化来自已加载值，最终返回受动态分支选择|零次执行时仍保留正确 init 值；不能从未执行的叶加载或未初始化 create 读数据|不能用固定正 trip count 掩盖 init/yield 错误|
|原始 init Tile 的别名在循环后仍独立 store；循环真分支加载另一个不同输入作为新 carry|旧值与新值必须同时可观察时应保留隔离，不能为了少一个池覆盖旧 init；zero-trip 时返回仍正确|不能全局忽略 old-value observation/liveness gate|
|同一循环中两份独立 Query 或 panel 在写入后都参与后续数学|同时活跃的数据必须保持互不覆盖，维持实际 read/write 依赖|不能把相同 dtype/shape 当作足够的 alias 理由|
|gather 原位更新后接 transpose_view，以及非零 offset 的 slice/view，再经过 if/loop 返回|父地址变化必须传播到 view；保留原 offset/extent，最终读者与写者同址|不能把所有 view 的 offset 清零以通过容量检查|

建议先在现有 pass 级测试设施中添加这些小 IR 对照，复用原检查，随后再核联合 v3 的实际分配和事件。不要求新增设备场景或扩大性能测试。当前只写报告，没有编辑正在由另一代理修改的编译器文件。
