# Score / Attention 可暂停一步：源码边界与跨步状态

2026-10-18，基线 formal `66882634`。只读源码与已有报告；没有修改算子、建立调度模拟、编译或上卡，没有性能预测。

结论：最小切点是 **完整 Score step 数学完成** 与 **完整 Attention tick 完成**，不能停在 Score ACK 或某个 matmul 后就认为本步状态已消失。逻辑上可列出有限状态；实现不是两个 helper 调用的简单交替，至少涉及两侧 schedule、两类计算体以及分波 TopK/plan 发布六个源码边界。还须证明实际片上地址与异步流水生命周期。本报告不把源级无环等同于可编译或有收益。

## 来源与范围

本报告行号来自已冻结的 formal 668 全部 33 源快照：

`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-indexer-20261018/tests/pypto_test/score_service_clock_runner_20261018/source_snapshot/`

该快照既有审计 `../score_service_clock_runner_20261018/RUNNER_STATIC_AUDIT.json` 已直接逐字节核对正式源；本轮没有另行复制生产代码。下文文件名均相对该目录。

协调依据：

- 协议审阅：`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_score_attention_interleave_readonly_20261018/README.md`，已读其完成稿。
- 资源审阅：`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/score_attention_interleave_resources_20261018/{README.md,evidence.json,API_GAP.md}`。资源数字来自现成 PTO；未运行新 planner。
- 历史 `PAR/wave_audit/README.md` 只否决固定 owner、完整 helper 排空后的阶段切换，未证明本次细粒度 live 交错无效；不重复旧模型。

## 最小切分位置

| 角色 / 函数 | 初始化、完整一步、收尾位置 | 不宜再缩小的原因 |
| --- | --- | --- |
| AIC `indexer_score_cube_stage`，`indexer_score_stages.py:97` | leaf 描述/Query/coeff 准备 127–172；step 174–375；leaf ACK 排空 376–378 | 一步包含整个 `score_tile` 的 Key 页读取、QK→FP16/ReLU→WS，末 panel 的 WS 362–374 不能漏。375 的 READY 排在 FIX 后，但不代表 Scalar 当场已观察到硬件全完成。 |
| AIV `indexer_score_vector_lane_stage`，同文件 383 | leaf 准备 411–434；balanced step 436–591；leaf root 写回 592–602。非 balanced step 610–673，leaf TopK 674–695 | ACK 在 load 后、scale/排序/根归并前发出。暂停点应在本步全部数学后；否则还要带 score/scale 与排序中间 Tile。leaf root 写回独立于 ACK 完成。 |
| AIC `attention_qk_pv_cube_stage`，`attention_stream_stages.py:117` | 初始化 130–139；完整 tick 141–199；总 tick 数 `block_count+2` | 一个 tick 包含 QK(tick) 与 PV(tick−2)。QK/PV 内累加器可在最终消费者完成后结束，但 Q 与 KV 需跨 tick。 |
| AIV `attention_qk_pv_vector_lane_stage`，同文件 205 | swap 与四 Tile 初始化 234–258；完整 tick 261–484；总 tick 数 `block_count+3` | tick 同时做 gather(tick)、softmax(tick−1)、merge(tick−3)，末 panel 还执行归一化/逆 RoPE/packed 写回与数值状态重置。只在 484 的外层 yield 后暂停，不拆在线数学。 |

一步切分是逻辑边界，不承诺当前 JIT inline 可直接接收全部 live Tile。当前源码已有 `pl.range(init_values=...)` / `pl.yield_` 四 Tile carry（Attention 261–264、484）；Score balanced 的 `single_roots`、`single_segments` 也需成为新外层循环的明确状态。[Tile 接口核对](TILE_HELPER_INTERFACE.md) 已确认：底层 typed Inline 支持 Tile，但当前 JIT 参数特化和外部 IR helper 收集两处桥接不支持直接接法；这比笼统说“inline 不支持 Tile”更准确。

## 必须携带与可重建状态

| 子状态 | 必须跨暂停保留 | 可从不变输入重建 / 不必跨完整一步保留 |
| --- | --- | --- |
| Score 公共控制 | 原 `buf_item` 游标；该 item 的 group/leaf、request、query 首行、live、visible、logical_begin、capacity、cache_len；当前 step、总步数；leaf 初始化/尾排空状态 | group plan 的 groups/max_leaves/leaf_tiles/extra 在本次 kernel 不变，可常驻 scalar。FIFO 行由 worker、step%2、G 重算；页号、panel、mask 等只在一步内。保存 item 不等于可按 wave 重新编号。 |
| Score AIC Tile | 当前 leaf 的 INT8 Query `[G*64,128]` 与 FP16 coeff `[16,G*64]`，或能保持原值的驻留副本/明确重装路径 | 两个 FP16 panel、Key pool、Key Right、QK/WS Acc 仅当前完整 step；释放必须等待原 pipe 消费，不能仅依赖 Python 变量离开作用域。原 live 尾组 L1 补满物理 M 的 148–164 行必须保留，GM 仍只读 live query。 |
| Score AIC 通知 | step 序与最多两个未回收 FIFO 槽；step≥2 的 ACK 等待；leaf 末 `min(steps,2)` 次排空后才允许下一 leaf 从 slot0 起 | 这些既含 scalar cursor，也含硬件事件队列状态；不是把 step 设零即可重置。空 work 不发事件。 |
| Score balanced AIV Tile | `single_roots=[G/2,1024] FP32`；`single_segments=[G,1024] FP32`；step 奇偶/当前 pair 进度 | 偶数 step 留下的半段必须保留到下步。两步凑 2048 后原 ≤512/≤1024/>1024 分支、new-root 在 old-root 前的 tie 顺序不变。scale、当前 FIFO load、单 query 排序/merge 临时在完整 step 数学后结束。 |
| Score 非 balanced AIV | scalar leaf/step/lane_span/lane_begin；已有 GM `score_arena` 的已写前缀 | 不需要 balanced 的两组 Tile；只能在同 leaf 分数写完后运行原 `indexer_topk_half_leaf`。不能把另一策略的 root 数或输入顺序搬来。 |
| Attention 公共控制 | 原 core/lane；独立 `qk_tick` / query ordinal / block_count / 当前 wave 可消费范围；真实全局 T | q/sb、slot、PV/softmax 的滞后索引可从 tick 重建。全局 q、GM 地址与 `T>=144` 提前 KV 分支不能改用 wave 内 T。若 wave 查询不连续，不能用减小 `t_dim` 冒充筛选。 |
| Attention AIC Tile | 当前 query Q BF16 `[64,512]`；仍被滞后 PV 消费的 KV L1 槽 | 原方案三槽 KV 共 384KiB；qk_q 每 query 的 sb0 只载一次，不能切步后遗失。无跨 tick 数值 Acc；PV probability/Left/Right/Acc 可在实际消费者完成后复用。 |
| Attention AIV Tile | `m/l=[32,1] FP32`，`oi_left/right=[32,256] FP32`，逆 RoPE swap `[1,1024] INT32` | 四 Tile 属于正在归并的旧 query，与本 tick gather 的较新 query不同。407–474 行仅在旧 query 末块发布/重置；即使该末块 mask=0 也必须执行。gather/softmax/PV/publish 临时原则上只活一 tick，MTE3/FIX 等消费者仍可能异步在用。 |
| 不变 GM 输入与视图 | 同次 kernel 的 Q/L/positions、页表、cache 与计划引用；两类 FIFO；根/TopK/plan/packed 输出的原全局布局 | 本 kernel 输入 metadata 不在这些计算体中改写；新调用必须从新的 device 内容重新初始化，不能跨 kernel 缓存。当前正式没有新 geometry 缓存，不混入后续候选。 |

长 G6 的资源约束不能只写成 scalar 状态：现成 Score Query Left 48KiB 与 coeff Left 12KiB，Attention Left 临时 16KiB，原地址重叠。资源审阅提出 coeff 保留独立 Left、Query 留 Mat 并每次切回重 TMOV 48KiB的可能路径；Attention Mat 64+384+16+48=512KiB 恰满。下一 Score 的 112KiB 临时还需借真正释放的 KV 槽，原布局不能直接拼接。AIV roots/segments 加 Attention 四 Tile/swap 的常驻逻辑量 104.25KiB，仍需重排 UB 并验证异步生命周期。这是容量条件，不是编译结论或收益。

## 轮次、尾步与发布合同

同一硬件组采用协议审阅的互补顺序：AIC `ScoreStep→AttentionTick`，两 AIV `AttentionTick→ScoreStep`。`r` 是该组的逻辑轮，**不是所有组共用的 step 索引**。两边各有 cursor；某一方先结束则只推进另一方。前波所有所需 roots、merge、plan 必须已经 release/acquire，不能在交错循环里等待尚由未来 Score 才能生成的 plan。

原 Score owner 与 Attention owner 不等价。`indexer_group_stages.py:235–250` 在 groups<24 时采用 leaf-major：`item=leaf*groups+group`；groups≥24 才是 group-major。B16 长 G6 的 4 groups×6 leaves 保留原 owner 时，仅落 core0–3/8–11/16–19 共12核，各两叶；对应 q0–23 的 Attention 却在24组。`ceil(24/max_leaves)` 不能保证铺满或对齐。此处是原公式的直接例子，没有新增负载模拟。

边界必须保留：

- Score `work_valid=0` 两角色均跳；有工作的尾组中空 query lane 仍按原 READY/load/ACK 配对，不写不存在的根。实际 visible/页表/causal 范围来自 device，不能用静态 S6=6 替代。
- Attention 的共同事件谓词仍是同一 `valid_block_mask[q,sb]>0`；地址 guard 不能单边取消事件。首块空、全空、dummy query仍按原 sb/query 推进并完成最终零/合法状态发布。
- C 的 +2 尾拍、V 的 +3 尾拍必须完整；末波无下一 Score 也不能提前退出。空 Attention core 只走原排空控制，不能新增越界 plan 读。
- 停在完整 tick 不等于可覆盖 KV：QK/旧 PV 的 MTE1/M/FIX 完成合同仍需显式保留。若另改成 **AIC L1 两槽且 PV 先于 QK、GM 仍三槽**，它是独立的 tick 内顺序/槽别名修订；本表的数值 carry 仍适用，但不能沿用原三槽的地址/释放证明。

两槽替代的现成只读补审为资源目录 `ALTERNATIVE_TWO_L1_SLOTS.md`：Mat 同 dtype 两槽可沿原 create/gather/slice 表达，不必借异 dtype KV 槽；两槽 KV256+Q64+ScoreQ48+Score 临时112+PV16=496KiB，保守加 leaf 初始化 Coeff Mat12KiB 为508KiB。GM `%3` 与 Mat `%2` 必须分开，旧 PV 最后 MTE1 读取完成后才写同槽的新 KV。协议代理确认源级无环，未验证联合 lowering；PV 先行也会丢失原 QK 提前发射机会，不能称免费减容。

## 从两计算体扩大到真正可调用实现的边界

| 文件 | 最小必要变化 / 保持合同 |
| --- | --- |
| `indexer_score_stages.py` | 拆出 leaf enter、完整 step、leaf finalize/drain 的控制；两个角色共同原 item 顺序，保留 balanced 与半叶两套数学。 |
| `attention_stream_stages.py` | 将全循环入口转为可保留 Tile/cursor 的完整 tick 执行域；保留 init、query publish/reset 和最后排空。不能每次调用都重新创建 Q/KV/m/l/oi。 |
| `cube_schedule.py:350–396` | 原全 Score 后全 Attention 调用换成局部轮次；原 Score 输入获取与 QUERY 获取仍有意义。事件0/1与2..5维持各自消息序。 |
| `vector_schedule.py:828–924` | 同组双 AIV 的互补调用顺序；原 `SCORE_VECTOR→TOPK→PLAN` 的完整发布链调整为波间合同，原 cache/rope/main-output 获取不丢失。 |
| `indexer_publish_stages.py:122–176` | 当前遍历所有 `query=worker+48*k`，需只发布该波已完整根集合的 query；同一 query 的根 identity/merge 顺序完全保留。dummy 行仍需每调用覆盖，避免旧代输出。 |
| `attention_plan_stages.py:136–180` | 当前按 M8 stripe 一次读取8行TopK。wave 截断 M8 时不能提前读未来未发布行；需完整 tile 就绪或精确裁剪 ready 行，保持原 owner/valid tail 和 CACHE_QUERY_METADATA 一致。 |

另需确定波进度发布的表示；现有 phase generation 是整 kernel 代际，重复用同一代同一 phase 值不能证明后波已经完成。若新增专用进度槽，`phase_sync`/runtime ABI/host 分配也在实现范围，不能由 ACK 代替。`indexer_topk_stages.py:78–106` 已有 Tile 返回 helper，故“Tile 返回一律不支持”不成立；参数与 allocation 身份的精确边界需读当前前端证据后判断。

本轮没有发现推翻协议审阅入口假设的新增原理反例。现阶段可以明确实现规模与必须携带的状态，但 **当前 JIT Tile 参数桥接有明确缺口，片上复用、波间发布及 mixed-Q/M8 截断也未形成已验证实现**。应先分别解决这些局部合同，不以自动生成巨大计算体替代接口与生命周期论证。
