# 完整精度 TND CSA 常驻 MIX 迁移独立审查

## 结论

**可行，适合作为下一项结构实验；不是改装饰器即可完成的迁移。** 先固定数学、逻辑任务编号、tile、累加/舍入及状态读写次序，用一个 24-AIC/48-AIV 常驻 MIX 执行分阶段数学，再逐步缩小依赖集合，是可审计的方向。首版全阶段同步可能损失现有跨任务重叠，不能预先保证更快，也不能由“任务数减少”推出收益。

本报告只读审查当前正式源码与已归档 Nal/兼容 SDK。未改运行源码、未编译或提交设备。源码 SHA 见 `resident-review-source-identities.json`。新 SDK 已完成的 CPU API/ABI 门禁由主线程记录；LLVM/PTOAS 仍构建中，不能视作完整设备迁移已经可用。

## 1. 当前真实结构

`deepseek_v4_flash_csa_tnd_precision_spmd_overlap/decode_csa.py::_full_csa_tnd_prepacked` 有 **64 个 Python Tensor 参数**，不是一个仅含数学叶函数的设备入口：

1. 外层创建 mixed/normed/Q/KV/heads/quant 等 GM 张量及多个 runtime scope。
2. `with pl.spmd(24, sync_start=True)` → `prepacked_prefix_indexer_group` → `pl.cluster`，并排两个 incore 入口，覆盖 HC-pre、输入 Norm、WKV 投影和 Indexer 权重投影。该 cluster 的 union 为 32-Tensor MIX ABI。
3. 之后仍有 `pl.at(CORE_GROUP)`、多组 `pl.spmd`、`task_dummy` 依赖、Scope 生命周期；Q、压缩器、Indexer、QK/PV、O-proj、HC-post 由后续任务执行。

大量函数**本来已经是 `@pl.jit.inline`**；inline 仅说明其 DSL 被展开，不会把内部 `spmd/at/parallel` 自动改成常驻核心中的顺序计算。不能将整个现有 root 包在另一个 `spmd/cluster` 后宣称完成。

## 2. 必须区分逻辑 worker 与常驻物理核

固定常驻物理核编号：AIC `g=0..23`，AIV `v=2*g+sub=0..47`。保留原阶段的逻辑 grid `W` 和原地址公式：

- 原 AIC 逻辑 task `j` 可按 `j=g; j<W; j+=24` 轮转；
- 原 AIV 逻辑 task `j` 可按 `j=v; j<W; j+=48` 轮转；
- **旧 body 内的 `get_block_idx()` 改由显式 logical-worker 参数替代**，内部原 `range(worker, ..., OLD_WORKERS)` 步长也必须保留。不能将其替换为 physical worker 后再把常量改成 24/48，否则任务集合和 buffer owner 会一起改变。

原来的逻辑 grid >24 或 >48 时，“保留 grid”只能指保留逻辑 tile 覆盖及每输出的计算轨迹，不可能保留相同数量的物理并发任务。`pl.parallel(8)` 的外层 group 必须变成显式索引，不能在单 incore 内继续创建 orchestration 并行任务。

对每阶段做 CPU 映射证明：全部合法 T=1..256、真实请求边界、无效 token、每输出唯一 owner、所有原逻辑 worker 恰执行一次、无遗漏/重复。数学内按输出位置决定的 K-order 不得依赖新的物理核编号。

## 3. 各模块迁移难点

| 模块 | 当前实现 | 能复用的最小部分 | 不能机械搬运的部分与门禁 |
| --- | --- | --- | --- |
| HC/Norm/WKV/Indexer 权重前缀 | 已有 24 组、两个 incore + cluster，10 次 matching handoff | `hc_prefix_*_stages`、`norm_wkv_stages` 已接收 worker，是最接近 resident 的数学叶函数 | 不可在新顶层 cluster 中嵌套现有 group/cluster；将两个 incore 的阶段内容并入新 AIC/AIV 入口，先保留每次 handoff。HF32 hi/lo K16/K128 顺序、Norm 精确归约/开方修正不可换成 Nal 快速数学 |
| compact offset / token_request / RoPE sign | 单 CORE_GROUP 串行前缀和 + vector SPMD | offsets 必须仍由一个 owner 顺序生成；sign 独立 worker body | 原标量 GM 写的跨任务可见性不能凭函数调用顺序获得；不要在第一版将其和 checked planner 一起强行驻留 |
| Q_A/QR/Q_B/RoPE | `qkv_precision126`/`qkv_proj_rope` 已 inline，但内部种子 at、多次 spmd、GM scratch、TaskId | 每个 QA 投影、normalize、INT8 matmul/dequant、RoPE tile body 可拆成显式 worker helper | QA 特别处理 **T126**；其余 shape 有 96-column paired forward/reverse K256 路径。必须保留 T126 packet 与 tile_rows 判定。QR_OK=1，不能 split-K；seed→QA→BF16边界→Norm/quant→QB 顺序都要有真实 phase |
| KV Norm/RoPE 与原 cache 写 | 前缀先生成 `projected_kv`，后缀 copy→Norm/RoPE→cache | 后处理 tile body、slot sentinel 写入逻辑 | 名称/注释有“Native projected”，实际值由前缀 PTO WKV 生成，不应误认为还需调用 Native。原 BF16→FP32 暂存和再舍入即使代数多余也先保留。cache 写后到 attention 读之间须显式发布 |
| 主/内压缩器 | project→pool/state读取→commit→Norm/RoPE/write，多处 spmd/at/TaskId | 投影 tile 与每请求 pool/commit 的内部序列 | pool 必须先读完同请求历史再 commit，request 内 token 时间次序保持；不可按 token 重分配而并行改同一状态。主 score projection 对内压缩器有当前 chain_dep，先保留。main/inner compact RoPE、五类 slot 的语义原样 |
| Indexer query/hadamard/系数 | 独立 AIC/AIV SPMD、Hadamard/INT8/FP16 边界 | 每个数学 worker body | 不能将 Native BF16、Hadamard 缩放/量化边界折叠；预留 GM scratch，消除 TaskId 必须补实际 producer phase，不是写空 dummy |
| Indexer score/TopK | **已 24 组 MIX**，2 个 AIV；双通信槽、FFTS event0/1；请求组/双 query/long leaf 动态策略 | 其 AIC/AIV 数学和 owner 映射天然可 1:1 对应常驻24组 | 整个 forest 不是一个可直接搬入的函数；它创建 group表、系数任务、arena、short/long分支、merge tasks。先抽出已选择分支的 score body，保留每 worker leaf 顺序/分段边界/稳定 TopK merge fan-in与比较规则；完整 S6 只是可选组大小，不能将真实 TND 改为 t//6 |
| Sparse checked / plan | Scalar checked48 后 planner32/8；注释明确依赖 task-exit Scalar cache flush | checked 单 token512整数验证、pair swap、planner tile body | **明确风险：当前注释称同 task 的 S/MTE2 event不 flush Scalar cache。** 移除 task 边界后 `bar_all/syncall` 不能自动替代这个合同；先保留独立 planner任务，或单独验证以合法 Vector store 发布 GM/完整写回获取协议。普通 `pl.write/read` 不能当跨核 signal |
| QK/PV + online softmax | **已 24 MIX**、每 AIV32 heads；3×K128 L1 载体；K512数学，QK/PV握手 | AIC/AIV 两侧 body 和 `qk_t=core+24*n` 可不改映射 | 不能缩为 Nal K128 softmax：会改 sum tree/概率BF16/online更新边界。原 SWA K128分支、3-slot复用与最后两片L1保留、MTE1/MTE2/FIX fences/FFTS ACK必须原样；原 `with spmd` 外层与 nested `split_aiv` 必须改成顶层两角色入口，不可再嵌入MIX |
| merge/inverse RoPE/heads发布 | merge48、BF16归一化后逆RoPE、宽copy | 逐 token/head 输出body | 保留 inverseRoPE前 BF16 舍入、实际 key-count sum路径与 signed zero，不能直接用 Nal FP32逆RoPE。旧 task 的结束依赖要换成阶段发布 |
| O-proj | WO-A `parallel(8)×spmd(8)`，行面板；BF16→整行8192量化→WO-B→scale BF16 | 每 `(group,nblock,panel)` Cube body，各 AIV tile/quant body | **不能只换装饰器**。将64逻辑WO-A分配到24AIC，保留原 start[rb,group,block]、rows>=192 seed256路径、较小T旧路径、相邻rb不等start回退、high-start wrap。WO-B有独立逻辑grid，必须重新解码映射。量化仍8192整行，不可换 Nal 8×1024各自scale |
| HC-post | token_tiles SPMD + 每token pipeline，已有 `hc_post_block` inline | `hc_post_block` 最容易先迁成 AIV worker helper | 先FP32 post*x，再按 residual0..3依序mul/add，最后BF16；不能改FMA/MulAddDst或重排到pairwise |

## 4. 同步和 FFTS 是首版阻断点

### 4.1 先全阶段完成，再缩 owner

首版可使用已验收 prefix 的匹配全72参与者 rendezvous，或新 SDK 的 per-core phase generation 协议。全部核无论有无数据工作都必须经过相同阶段，不能某核 return/continue 绕开全核屏障。AIC/AIV 两侧阶段数、顺序必须一致，所有动态 T/短请求分支也必须如此。

新 `signal_notify/wait` 的存在和 CPU API 测试不等于本工作区所有业务 GM 都已可见。Nal 的 `signals.py/phase_sync.py` 明确分开：通知只保护 signal，payload还需本核流水完成、Scalar/业务缓存发布与获取。可参考协议，不照搬其 owner 推导到不同精度版布局。

如果用 generation：每核每 invocation 只推进一次，phase槽单写者，空工作核也发布；graph工作区顺序独占，异常后重建、INT32回绕前重建。完成通知不是 consumer ACK，不可仅见 phase完成就覆盖仍被读的通信/arena缓冲。

### 4.2 原 TaskId 不是设备内同步

`deps=[late_dep, qa_tid, ...]`、`manual_dep` 和 scope退出原先由 runtime任务调度维护。迁成同一任务后，不能保留一个无意义 TaskId 假称依赖仍在。对每条 RAW/WAR/WAW 建立明确阶段边：尤其 compressor pool→state commit、内压缩cache→Indexer、raw/cmp cache→attention，以及出栈前所有 FIX/MTE3 写回。

### 4.3 FFTS / event 复用

现 prefix、Indexer score、QK/PV分别 `set_ffts`，各有原 event编号。分离任务时的初始化/退出习惯不能直接套在同一长任务：

- 保留每阶段独立 workspace 或经证明的 event namespace；
- stage结束前确认所有 set都被对应wait消费、所有跨核写回结束；
- 不在一侧仍等待时改FFTS base，不让上一stage剩余event释放下一stage的wait；
- 两个 AIV参与数/FFTS模式必须与原混合内核一致；
- 原来 task epilogue带来的 fence、Scalar flush、寄存器/本地buffer生命周期要逐项补证据。

这是必须看生成 C++/ISA 和实际最小 handshake gate 的项目，数学 AST 一致不能替代物理同步验收。

## 5. 参数、GM 及片上资源

### 参数

新 Simpler pin 的 `arg_direction.h` 为 core/chip **256 Tensor**、core **16 runtime scalar**、chip128 scalar。当前 root64、prefix32不表示加上全部后缀scratch后自动符合；Nal的130/132 Tensor接口只说明其实现有先例，不能替当前精度版算账。必须统计**实际生成 core ABI 的 union**（含别名和动态维参数），而非只数 Python函数参数/constexpr。可保持服务对外入口而由 orchestration加入workspace参数，但新root/注册schema/manifest需单独身份绑定，不能继续贴旧64标签。

### GM

所有跨阶段 GM 张量及原任务内部 `pl.create_tensor` 所产生的 GM 暂存都必须在调用前准备；不是只搬 root里可见的输出。`pl.create_tile` 的片上私有值仍在局部阶段内创建，不必把 L0C accumulator 全部外化GM。

具体大项：当前 Indexer `T_PAD=432`（静态容量64×6再加一整请求尾巴，48行对齐），最大历史容量 1M/4=262144，32个leaf，因此 `pair_arena` 为 **108MiB**，`score_arena` **13.5MiB**，合计121.5MiB。不能按本次128K或T240猜测静态arena大小。另有 Q/KV、heads、Oproj大 FP32 scratch、三组QK/PV transfer 等。

外部持有可避免 runtime ring heap生命周期叠加，但不减少显存占用；若每个21层各自复制大arena将显著放大显存。优先同一顺序执行服务/图共享workspace，不同并发流必须独占或有明确租约。不要为了控制显存无证明地overlay仍在使用的两个阶段，更不能删旧 TND 末组读尾容量。

### 片上

QK/PV已约 **464KiB L1**，接近A3 512KiB；不能同时保留前一stage L1权重/临时tile。Oproj seed有大L0C，prefixNorm/HC也有多组UB载体。新顶层单函数可能被 allocator认为 live range 跨阶段，若静态配置把局部变量一起活跃可溢出。应采用独立 inline stage/local lexical scopes、仅GM Tensor跨stage；检查实际地址和最后使用，不以Python缩进等同释放。

不要把现有两个独立cluster整体嵌套；最终每个角色一个可生成的顶层 incore函数，数学helper inline展开。若触发原来同空间 Move消除/SSA别名问题，必须修复局部所有权，不可通过放松位精度门禁绕过。

## 6. 精度：保数学还须保真正的执行轨迹

CPU trace至少记录每输出的 M/N tile、valid_shape、Native K-start/permutation、K packet次序、seed方式、累加器类型和每个BF16/FP16/INT8 cast位置：

- QA T126与96/144/192/240的Native特例不能被常驻worker数覆盖；
- WKV与Indexer权重投影的已校准order packet不变；
- Compressor按Native column group进行K shift，不能用物理core号替代列组；
- Oprojcyclic K16（即使packet256）、rows192 dispatch与不等start回退不变；
- QK K512 onlineSoftmax和Native短Key/fullKey求和树不变；
- BF16概率用 round，其他边界 rint，inverseRoPE前BF16，整行Oproj量化及post独立mul/add不变；
- 任何原来多逻辑worker写同输出的 atomic 都先排除或保留确定顺序，不能以数学交换律代替bit证明。本QA精度路径QR_OK=1，正是为了避免split-K差异。

全局同步总序可以改变阶段的时间关系，但只在不涉及状态别名或可见性时才不改变值；不能因此跳过 cache/state逐bit对拍。

## 7. 建议的最小迁移顺序

1. **双基线**：在兼容SDK上先运行未改数学的现有 frozen overlap 与独立 Native，确认新工具链自身不改任何已验收八边界；再建 resident候选。原SDK结果不能直接充当新SDK执行正确性证据。
2. **结构化清单**：冻结root输入/所有GM临时表、每阶段逻辑grid/owner、RAW/WAR/WAW、FFTS/event合同，生成实际ABI清单。先不复用scratch，不改数学。
3. **最小后缀 resident实验**：优先把Oproj BF16/整行量化/WO-B/dequant/HCpost转成一个两角色常驻后缀（或先仅Vector结尾），保持64逻辑WO-A与旧K轨迹；这样直接验证跨阶段GM发布而不引入Indexer/attention的复杂FFTS。该实验仍允许现有前半CSA任务，不谎称已全resident。
4. **前缀并入同角色计划**：展开已有prefixcluster，不嵌套；保留全部hand-off，接Q_A→Norm/quant→Q_B/RoPE/KV后处理。单独gate T126、T144、T240及混合T21。WKV和Indexer权重投影已有worker helper可复用。
5. **压缩器阶段**：依照已冻结DAG，主/内投影→每请求pool历史读取→commit→cache阶段，先全屏障；逐个cache/state/QR/Q边界对拍。补齐不等长1..6请求，不能用S6padding减少问题。
6. **Indexer resident数学**：query/Hadamard/coeff→group plans→score24MIX→leaf/merge。先复用同24物理owner和原slot握手，完整保留短/长、请求组/双query分支。不得把short模式精度验收外推所有long leaf。
7. **QK/PV resident数学**：将已24组混合body并入，两侧循环及event顺序原样；先保留checked/plan独立task，解决Scalar发布合同后再合进去。单stage完成后比较pre-inverse heads/post-inverse heads/8边界，再接后缀。
8. **首次完整入口**：同一24-MIX只留启动/结束runtime任务，所有跨stagebuffer显式外部；CPU无遗漏映射、实际生成物ABI/资源/FFTS审查后，小NPU首差停止。最后才移除全屏障/减少owner等待，不在同一次变化里改tile/搬运/数学。

若阶段拆分/新SDKCPU编译暴露无法表示的控制流或资源上限，应先收敛这个阻断点。不要一次复制全部函数、依靠大补丁后整模型结果猜哪里出错。

## 8. 必须拒绝的捷径

- 外包一个 `spmd(24)` 然后继续调用内含 `spmd/parallel/at` 的旧root。
- 将所有旧worker数量改24/48，改变Logical ID和Native K-order/地址归属。
- 将Nal HC FP32、softmaxK128、FP32 inverseRoPE或分组scale数学迁入精度版以换速度。
- 用图capture/CPU AST通过代替FFTS/Scalar缓存/GM发布硬件证明。
- 用阶段完成信号当consumer ACK，提前复用score/KV/partials缓冲。
- 在graph内分配/清零共享大workspace或让不同流共用generation状态。
- 提前合并runtime scope，重新引入121.5MiB Indexer与Attention/Oproj大临时的重叠生命周期。
- 将TND请求改回每请求6行，或以physicaltile尾部padding冒充request padding。

**建议结论：先做“原数学 worker 化 + 明确同步/外部工作区”的最小常驻后缀，再按上述顺序逐段扩展。完成单入口只说明调度结构成立；性能是否接近Nal、哪些依赖可安全缩小，都需要后续受控测量。**
