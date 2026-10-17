# HC_pre 第一阶段紧凑流式交接设计核对（2026-10-17）

本文是未采用候选的源码设计审计，未编译、未启动 NPU。正式动态 SPMD 实现由主会话放入 `vllm_ascend/ops/pypto/`；本目录只保留本次设计与纯 CPU 协议检查。

## 结论与需要纠正的前提

**在当前 A3 软件路径上，`pl.aic_gather` 不是 AIV 的 UB 直接向 AIC 的 L1 发送数据。**

它确实能形成编译器管理的双槽流式交接，但底层仍为 `AIV UB → GM FIFO → AIC L1`。因此不能把采用 `aic_gather` 解释成消除 GM 往返；收益可能来自及时供数、紧凑地址、重用 BF16→FP32 转换、流水交错和较小驻留集合，必须由完整 CSA L1 incore 对照证明。

本次核对的依赖根目录均为独立 worktree，没有修改共享依赖：

- PyPTO：`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016`
- PTO-ISA：`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/simpler-csa-core-args-256-20261016/build/pto-isa`

可直接接手的原始源码证据：

| 位置（以上述相应依赖根为起点） | 实际含义 |
| --- | --- |
| `python/pypto/language/op/tile_ops.py:795` | `aic_gather` 必须处于 `split_aiv` 区域，由编译器形成 Vec→Mat 边界。 |
| `src/backend/910B/backend_910b_handler.cpp:31` | 明确说明 A2/A3 cross-core transfer goes through GM；GM→Mat 边界转换为 NZ。 |
| `include/pto/npu/a2a3/TPush.hpp:184` | `pushVec2GMFiFo` 将 Tile 用 `TSTORE_IMPL` 写入 GM 环形槽；两个 AIV 通过行偏移写不同半块。 |
| 同文件 `:147` 附近 | V2C READY 用 `ffts_cross_core_sync(PIPE_MTE3, ...)` 发布。 |
| 同文件 `:434` | `popMatTileFromGMFiFo` 将同一 GM 槽用 `TLOAD_IMPL` 读入本地 L1 槽。 |
| 同文件 `:373` 附近 | V2C FREE 在 AIC 的 MTE2 上发布，表示 GM 槽已经读入本地。 |
| `include/pto/npu/a2a3/TPop.hpp:22` | TPOP 执行 wait→load→按周期 free；Tile TFREE 本身在 A3 实现为空。 |
| `include/pto/npu/a2a3/TPush.hpp:505` | TPipe 析构按真实剩余 credit 排空 FREE，防止下一次调用误消费旧事件。 |
| `src/ir/transforms/allocate_memory_addr_pass.cpp:107` | reserve_buffer 不是普通有生命周期的 MemRef；其他 Tile 均放在保留区之后。 |

不能用 A5 的直接写 L1 能力，或源码中存在 `TINSERT` 一词，就推导 A3 的该交接路径绕过 GM。

## 与已有候选的实际差别

原数学核来自仓库内 `tests/pypto_test/csa_static_spmd_feasibility_20261015/generated/core_bodies_ir.py`：

- `hc_widen_rms`（第 66 行）：每 8 行按 K512 计算 32 项平方和，逐项 FP32 相加；同时保存完整 `[T,16384]` FP32 输入。
- `hc_pre_linear`（第 108 行）：M16，每行 4 个 K4096 部分和，每部分严格 16 次 K256 顺序累加。
- `hc_pre_linear_reduce`（第 137 行）：4 个投影部分和按原顺序相加。
- `mix_x_rms_norm`（第 327 行，实际输入读取第 346～349 行）：还会再次读取完整 FP32 输入。它不是投影的独占缓冲。

已有 `../hc_stream_pre.py` 已使用 48 AIV、M16/K512 及时供数、两个 READY/ACK 槽和 32 项 RMS 部分值，但仍向完整 `view_2` 的原行跨度写入和读取。它限定 T96；历史真机 `../hc_stream_result.json` 为输出与保护区通过、完整 L1 844.40 μs，未采用。这只能否定该具体实现，不能把紧凑交接或本次未实测设计标为有效。

最新正式 `vllm_ascend/ops/pypto/hc_pre.py` 中 K 流式路径与双槽、L0 交错值得参考；其 `aic_gather` 在 A3 同样经过 GM。该文件第二阶段有全硬件核 `syncall`，不能原样插进只处理 CSA 部分阶段的局部 DAG；本候选继续用组内 FFTS 和原阶段进度协议。

删除完整 FP32 输入前，必须先把 `mix_x_rms_norm` 的四次 FP32 load 改为读原 BF16 并无损扩为 FP32。保持 `y01 + y23`、BF16 rint、RMS 与 norm 运算顺序和写入位置。主会话正在独立验证此变换。只给投影加 FIFO 而保留完整 FP32 写入，会增加一次中转写入，不能称为缓冲消除。

## 可落地的最小候选

优先采用**普通 GM Tensor 双槽 + 现有 `tile.load/store` 与 FFTS READY/ACK**，避免在整个大 SPMD 中永久预留 L1 环形区。

保留所有原数学划分：

```text
t_linear = ceil(T / 16) * 16
linear_units = ceil(T / 16) * 4
linear_workers = min(linear_units, 24)

对原 AIC hc_pre_linear 的每个物理组：
  logical_block = 原 records 中的 mapped_block
  for task = logical_block; task < linear_units; task += linear_workers:
    row0 = (task // 4) * 16
    split = task % 4
    两 AIV 的 row0 分别加 0、8
    各自处理 8 个 K512 块，经两个紧凑 GM 槽交给本组 AIC
    AIC 每次仍按 K256 顺序做两次 MMA，累计原 K4096 的同一个 Acc
    完成本 task 后两 AIV 排空各自两槽 ACK，再进入下一 task
```

`logical_block` 必须来自 layout/records 的真实映射，不能假定它恒等于物理 `group_id`。若 layout 不是一逻辑成员对应一个物理组，或 AIC HC 阶段不在第一个 AIC 阶段，需要拒绝该候选并重新分析依赖。

### 多波次与尾行

| T | M16/K4096 逻辑任务数 | 参与 AIC 组 | 每组 task 数 |
| --- | ---: | ---: | --- |
| 24 | 8 | 8 | 每组 1 |
| 48 | 12 | 12 | 每组 1 |
| 96 | 24 | 24 | 每组 1 |
| 144 | 36 | 24 | 12 组 2、12 组 1 |
| 192 | 48 | 24 | 每组 2 |

这不是五点白名单。相同公式覆盖 T 区间；分界来自 M16 以及 24 组执行资源。组内 AIC 与两 AIV 必须使用同一个 logical_block 和步长，循环次数严格相同。

尾行需要 `lane_rows = max(0,min(8,T-row0-lane*8))`。AIV 原 BF16 load 必须携带该 valid_shape，转 FP32 后先 fillpad，再将交接 Tile 恢复为完整 `[8,512]`。即使第二 AIV 有零行，也发送全零半块和 READY；否则 Cube 聚合等待会死锁。不能让尾行越界访问原输入，也不能把两 lane 都缩短后改变 GM 槽的行偏移。

Cube 的数学有效 M 仍按原 `min(T-row0,16)` 处理；从完整槽加载后为参与 MMA 的 Left Tile 设置原 valid_shape，避免改变尾部累加器的物理解释和 store 规则。原投影结果区是 padded 行，保持原 store 行形状。该部分必须检查真实生成指令，不能仅凭填零断言等价。

### 地址与容量

单组两个 GM 槽：`2 * 16 * 512 * 4 = 65536` 字节。按物理 group_id 划分，24 组最多 1.5 MiB。每个槽的两 lane 各写连续 16 KiB，不共享写回块。

可以在确证 mix 已不再读取 `view_2` 之后，重用其前部作为 GM FIFO；但完整 allocation 的容量可能仍保留，不能宣称释放了整个分配。若重用 `[T,16384]` 的原 allocation，必须给出新 `[T*32,512]` 连续视图，访问起点为：

```text
fifo_row = physical_group * 32 + slot * 16 + lane * 8
```

必须检查 `(max_active_group+1)*65536 <= T*16384*4`；不能只按参与组数检查而忽略稀疏物理组编号。T=1～3 等小输入可能不满足复用容量，这只是候选拒绝条件：现有合法策略仍需保留，或使用明确定义的新小型工作区，不能将其变成算子不支持这些形状的理由。

注意：在 incore 中重建 GM Tensor 视图是否被当前 parser/lowering 接受，尚未编译验证。不要仅改函数参数 annotation 后把原 Tensor 当成新 stride；必须让真实视图描述符与索引一致。正式实现可在已有 Host shape/stride 绑定阶段提供共享同一 allocation 的 FIFO 视图，无需读任何 device 数值。

RMS 保持 `[32,t_linear]` 的逻辑布局，每项 K512 的平方和独立写入。可复用原 `temporary_5: [8,16384] FP32`，要求 `32*t_linear <= 8*16384`。当前 T≤384 的区间远小于容量上限，但仍需确认它不再被尾行 widen 或其他阶段使用；不要与 FIFO 复用同一位置。

归约要从 0 开始按 K512 的全局索引 0..31 顺序累加。不得先在每个 K4096 内归约成 4 项再相加；后者改变 FP32 累加顺序。

### 同步语义

沿用旧候选的静态事件：READY=8/9，ACK=10/11。FFTS 事件是组内交接，不涉及非一致 Scalar GM 信号读取；原跨组阶段仍使用已验证的发布/等待实现。

1. AIV 使用某槽前，若不是本 task 的最初两个块，先消费对应 ACK。
2. 完成 FIFO 写入和该 K512 的 RMS 部分写入后，在 MTE3 上发布 READY。
3. AIC 消费本组两 AIV 的 READY，再把完整槽加载进独立本地 L1 Tile。
4. AIC 在 MTE2 上发布 ACK，允许覆盖 GM 槽；其后仍可用已经读进本地的 Tile 做 MMA。
5. 退出每 task 前两 AIV 排空两个 ACK；多波次及下一次图重放不能继承旧事件。
6. AIC 发布原 HC 完成进度前保留原 fence。消费者等齐原 HC AIC 成员后读取投影与 RMS 部分；RMS 发布先于 READY，且 AIC 已消费 READY，才能建立这一传递链。

全阶段进度约束仍需重新审计：删除原 AIC 等全部 widen 的 wait 只可在该 wait 的前驱集合完全等于已被双槽协议替代的 widen 成员时进行；inv_rms 的消费者必须经过写入新 inv_rms 的 linear_reduce 阶段。未满足条件时拒绝变换。

### 为什么不直接引入显式 TPipe

当前 PyPTO 支持显式 `reserve_buffer`、`import_peer_buffer`、`aic/aiv_initialize_pipe`、`tpush_to_aic`、`tpop_from_aiv`。独立 AIC/AIV 函数使用这些 API 在语义上可行，源码例子为 `tests/st/runtime/cross_core/test_cross_core.py:574`、`:707`。

但 M16/K512 两槽需要保留 64 KiB L1，该保留在大函数全生命周期内占用。已有整 CSA 候选的 Mat 峰值为 475136 字节；简单相加为 540672，超过 A3 524288 字节容量。这是容量风险估算，不是本候选编译失败记录。K256 两槽保留 32 KiB，估算507904可容纳，但通知数量翻倍，RMS仍须保留K512行归约，而且实际对齐/碎片还需编译验证。

普通 `tile.load` 所得 L1 Tile 由内存规划器按阶段生命周期复用，不需要为整 CSA 永久保留一个 FIFO 区，是本轮更小的候选路径。不能手写一个固定 SRAM 地址绕过分配器，否则可能与后续 QK/PV 的 Mat 区别名。

## 本次已完成与后续证据门槛

本目录 `plan_audit.py` 只检查工作覆盖、RMS 地址互斥、复用容量和抽象 READY/ACK 协议；它不模拟硬件 pipe，也不能证明数据可见性、数值或性能。

已执行纯 CPU 检查，证据为 `plan_audit.json`：T=1..384 的恒等/逆序物理映射覆盖通过；另外确认稀疏物理映射必须按最高组编号检查容量。1..4 波、每种 128 个随机交错、连续 3 代的 READY/ACK 模型全部通过。未生成正式变换、未编译、未提交 NPU，尚无本候选数值或性能结论。

接手后顺序：

1. 独立验证 mix 读取 BF16 的无损变换，保留现有全部精度门槛。
2. 给正式路径接入可维护的 FIFO 视图与按 task 映射的供数候选，CPU lowering 检查资源、地址与静态事件；不能把本目录设计当作已经实现。
3. 少量代表值先验证所有输出/Top-K/state/guards/连续图重放及两槽排空，再看同轮完整 CSA L1 incore。
4. 有明确收益才扩展不同 T、尾行和多波次；不得以 HC 子阶段或某单核跨度替代完整 CSA 的30%目标。
