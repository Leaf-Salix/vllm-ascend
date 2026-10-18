# Attention tick 接入 wave 前的 query 起点与物理环槽合同

本文是只读接口审阅的结论，不是已经实现的 wave 接线。当前冻结的 `source_candidate/attention_tick_stages.py` 仍采用 `qk_core + (work // 5) * CORE_GROUPS` 的全局 query 公式；本文不修改该源码、已有 AST 证据或初始化行为，未执行 PyPTO 编译或设备验证。

## 为什么必须拆分两个 Scalar

`qk_core` 当前同时表示物理 AIC 核组编号和该核负责的第一个全局 query。对从 query 0 开始的完整调用，这两个值恰好相等；对第二个及后续 wave 不再相等。

未来接线需要独立的 `query_first: pl.Scalar[pl.INDEX]`。`qk_core` 始终是物理核组编号 `0..23`，用于 GM 环槽与本核 FFTS 配对；`query_first` 是当前 wave 中归该物理核所有的第一个全局 query。不能把 `qk_core` 改成全局 query，否则固定 24 核的 GM 环槽会越界；也不能直接让 tick 从全局块序号开始，否则旧 wave 的 softmax/PV 尾拍会被重新消费。

## 六处 query 起点表达式

下列行号指本目录当前冻结的 `source_candidate/attention_tick_stages.py`。未来仅将表达式中的起点 `qk_core` 改成 `query_first`，保留 work、`SPARSE_BLOCKS` 和 `CORE_GROUPS` 的计算。

| helper / 用途 | 当前行 | 未来表达式 |
| --- | ---: | --- |
| AIC PV 的全局 query | 159 | `query_first + (pv_work // SPARSE_BLOCKS) * CORE_GROUPS` |
| AIC QK 的全局 query | 201 | `query_first + (qk_tick // SPARSE_BLOCKS) * CORE_GROUPS` |
| AIV gather 的全局 query | 327 | `query_first + (qk_tick // SPARSE_BLOCKS) * CORE_GROUPS` |
| AIV EARLY_KV 分支中上一块 SCORE 的 mask 查询 | 393 | `query_first + ((qk_tick - 1) // SPARSE_BLOCKS) * CORE_GROUPS` |
| AIV softmax 的全局 query | 410 | `query_first + (softmax_work // SPARSE_BLOCKS) * CORE_GROUPS` |
| AIV PV 归并与输出的全局 query | 449 | `query_first + (pv_work // SPARSE_BLOCKS) * CORE_GROUPS` |

由这些 query 派生的 Q、valid mask、Q/L/position 几何、稀疏索引、bias、RoPE 与输出地址继续使用全局 query，不改成 wave 内下标。EARLY_KV 中的内联上一块表达式也必须修改，不能只替换命名为 `*_t` 的赋值。

## 五处 GM ring 地址保持物理核号

以下五处保持原式，不能使用 `query_first`：

| helper / 用途 | 当前行 | 保留表达式 |
| --- | ---: | --- |
| AIC PV 概率/PV 环槽 | 161 | `qk_core * QK_TRANSFER_SLOTS + pv_work % QK_TRANSFER_SLOTS` |
| AIC QK 的 KV/SCORE 环槽 | 205 | `qk_core * QK_TRANSFER_SLOTS + qk_tick % QK_TRANSFER_SLOTS` |
| AIV gather 的 KV 环槽 | 331 | `qk_core * QK_TRANSFER_SLOTS + qk_tick % QK_TRANSFER_SLOTS` |
| AIV softmax 的 SCORE/PROB/mi/li 环槽 | 412 | `qk_core * QK_TRANSFER_SLOTS + softmax_work % QK_TRANSFER_SLOTS` |
| AIV PV 归并的 mi/li/PV 环槽 | 451 | `qk_core * QK_TRANSFER_SLOTS + pv_work % QK_TRANSFER_SLOTS` |

`QK_TRANSFER_SLOTS=3` 保持不变；AIC 内部 L1 KV 仍用相对 work 的 `% QK_L1_SLOTS`，且 `QK_L1_SLOTS=2`。两个 AIV 的 `lane_head/lane_kv` 与 FFTS 所属物理核组均保持不变。GM 三槽和 L1 两槽是两个独立的生命周期，不能合并模数。

## wave first/count 与初始化合同

合法 wave 满足 `0 <= wave_begin <= wave_end <= T`，`T` 为完整调用的实际 token 数，`G=CORE_GROUPS=24`。各核使用相同的 wave 边界，其本核首项与计数为：

```python
query_first = wave_begin + ((qk_core - wave_begin) % G + G) % G
query_count = 0
if query_first < wave_end:
    query_count = (wave_end - 1 - query_first) // G + 1
block_count = query_count * SPARSE_BLOCKS  # SPARSE_BLOCKS = 5
```

双重取模显式处理后端负数余数语义。该公式保证 `query_first % G == qk_core`，后续每次加 G，保持原全局 `query % 24` 所有权，不在 wave 内重新轮转分核。`query_first >= wave_end` 的核是本 wave 空 owner，计数为零，不访问不存在的 query。

两个 tick helper 最小新增入参是独立的 `query_first` Scalar。当前两个 init 在第 117、248 行按完整 T 计算的 count，未来必须替换为上述 wave 几何合同。可在外层单独计算并传入 `query_first/block_count`，让 Tile 分配和固定 FFTS 声明仍属于调用级初始化；不能为了修改 count 就暗中引入每 wave 全量重建 Tile 的行为。

AIV 的 `t_dim` 必须继续来自完整 `position_ids` 的真实 T，不能替换为 wave 长度或 padding 容量，否则 `EARLY_KV_MIN_TOKENS` 的调度分支会改变。wave 范围与实际 T 是不同变量。

## 相对 tick、空 owner 与首尾排空

每个 wave 的相对 tick 从 0 开始。外层必须保持各角色原来的完整循环界：

- AIC：`0 <= tick < block_count + 2`，首两拍填充、末两拍完成 PV；每拍仍先 PV(tick−2)，再 QK(tick)。
- AIV：`0 <= tick < block_count + 3`，softmax 处理 tick−1，PV 归并处理 tick−3，末三拍完成归并与输出。

当 `block_count=0` 时，上述界使所有真实块、softmax 和 PV 访问条件均不成立，不读未初始化环槽。不能只因空 owner 就跳过未来 wave 层协议要求的通知；wave 层协议尚未接线，本文不替其定义参与者集合。

在非空 owner 中，每个 query 的五个 sparse block 仍完整执行；即使末块 mask 无效，原 `sb==4` 的输出与状态重置分支也必须保留。四个在线归约 Tile 的当前 reset 行为不变，不能按 mask 跳过整个 query 的收尾。

下一 wave 复用 GM/L1 槽前，必须确认上一 wave 的三角色尾拍与原配对事件已完成消费；仅观察 Scalar 已走到 tick 末尾不代表异步 MTE/M/FIX 完成。仍需保留原每块有效 mask 的两端事件谓词、PV 读完后覆盖同一 L1 槽所需的 MTE1→MTE2 依赖，以及最终联合 lowering 产生的正确局部事件。本文没有新增 fence、ACK 或波次屏障，也未证明尚未实现的 wave 总调度。

## 当前验证边界

本次只读核对了冻结源码的全部 `qk_core` 使用位置以及原 tick 分支。六处 query 起点和五处 GM ring 的分类是源码事实；上述 first/count 是待接线的接口合同。当前冻结源码仍未新增 `query_first`，没有 wave 设备执行、联合编译、资源或性能结论。
