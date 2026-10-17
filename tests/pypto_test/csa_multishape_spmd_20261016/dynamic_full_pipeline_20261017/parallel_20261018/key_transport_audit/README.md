# A3 双 AIV → 同组 AIC Key 传输：只读复核

当前依赖仍没有可供本候选调用的 A3 UB→L1 直接跨核传输路径。已有
`pl.aic_gather`、`pl.tpush_to_aic` / `pl.tpop_from_aiv` 可以在同一 MIX
任务内交接数据，但 A3 实际执行 **AIV UB → GM FIFO → AIC L1**。
因此换成这些接口不能消除 Key 暂存的 GM 写入和读取；前一份设计估算的
长 B16 约 132MiB 额外 Key 交通风险仍在。GM 地址访问不等于每字节都到 HBM，
这里没有测量 L2/NoC/HBM 命中和带宽。

复核沿用历史 `hc_pre_direct_stream_v1/README.md`（主日志历史原文 091）与
`QK_DIRECT_L1_EXPERIMENT.md`，只检查当前相关实现。没有修改依赖、正式候选或
主仓，没有编译、设备作业、性能复测或摘要校验。本结论限定当前本地软件栈，
不声称穷尽未公开硬件能力。`source_excerpts.txt` 保留关键原文，
`source_inventory.json` 记录路径、大小与行号。

## 依赖与证据链

以下路径均以 `/data/pyptouser/qinchuanyu/pto-eager` 为根：

|简称|实际路径|
|---|---|
|Y|`.worktrees/pypto-csa-static-fixes-20261016`|
|I|`.worktrees/simpler-csa-core-args-256-20261016/build/pto-isa`，本次已使用的独立 Simpler 构建头文件|
|A|`ptoas-src`，仅作本地 lowering 源码旁证，未重编或重新核对安装二进制|
|C|`vllm-ascend-dsv4-pto-0251rc1`|

|层级与位置|当前事实|
|---|---|
|Y `python/pypto/language/op/tile_ops.py:795`|`aic_gather` 处于 `split_aiv` 区域；按 UP_DOWN/LEFT_RIGHT 合并两 lane。它定义交接语义，不承诺物理直传。|
|Y `python/pypto/language/op/system_ops.py:363,456`|已有 `tpush_to_aic(tile, split, id)` 与 `tpop_from_aiv(shape, dtype, split, id)`；对应 `tfree_to_aiv`。|
|Y `src/backend/910B/backend_910b_handler.cpp:31`|明确 A2/A3 跨核经 GM；GM→Mat 使用 NZ 布局。|
|Y `src/backend/common/soc.cpp:146`、`backend.cpp:516`|A3 内存图有 Vec→DDR、DDR→Mat，没有 Vec→Mat；后者仅 A5 图具备。|
|Y `src/ir/transforms/inject_gm_pipe_buffer_pass.cpp:14`|自动注入 `__gm_pipe_buffer`，向调用链传播工作区参数；orchestration 建内部工作区。省掉 Python 手写 GM Tensor 不等于省掉 GM。|
|Y `src/backend/common/pto_ops_crosscore.cpp:450`|初始化 pipe 显式携带 `gm_slot_buffer`，按 pipe id 与方向划分区域。|
|A `lib/PTO/Transforms/Lowering/PTOLowerFrontendPipeOpsPass.cpp:220`|A5 生成 `InitializeL2LPipeOp`；A2/A3 必须有 GM buffer，生成 `InitializeL2G2LPipeOp`。|
|I `include/pto/npu/a2a3/TPush.hpp:184`|`pushVec2GMFiFo` 计算 GM 环形槽与两 lane 偏移，调用 `TSTORE_IMPL`。|
|I 同文件 `:437`|`popMatTileFromGMFiFo` 从同一 GM 槽构造 GlobalTensor，`TASSIGN` 本地 L1 槽后 `TLOAD_IMPL`。|
|I `include/pto/npu/a2a3/TInsert.hpp:165`|普通 TINSERT 分支支持 Vec→Vec，另一分支检查 Acc→Mat；没有 AIV Vec→AIC Mat 分支。|
|I `include/pto/npu/a5/TPush.hpp:245,329`|A5 另有 `pushVec2MatFiFo`，向 peer L1 槽 `TINSERT`；此路径不能移用到 A3。|
|I `include/pto/npu/a2a3/grid_intrinsic.hpp:12`|另一个 GridPipe 名称下的 A2/A3 实现是 HCCL 共享窗口与 GM atomic flags 的 mock；不是可用的片上邻居 SRAM 直传。|

旧测试 `test_split_aiv_gather_row_codegen.py` 的“no GM staging”注释描述前端
无需手写中转；其 A3 lowering 和实际 TPush 实现仍经 GM。不能据该注释反推
硬件路径。`GlobalTensor` 类型的 TPUSH/TPOP 则直接交接 GM 槽视图，同样不绕过 GM。

## 同组绑定、事件与生命周期

同一 MIX 的逻辑 block 由运行时/FFTS 绑定一个 AIC 与两个 buddy AIV；
`get_subblockid()` 决定两份写入偏移。`reserve_buffer` 在 AIC 端保留 L1 地址，
`import_peer_buffer(name, peer_func)` 引用同组 peer 的该地址；这个地址也用于
描述消费者本地槽，不能解释成 AIV 可以直接写 A3 L1。相关原文见 I 的
`docs/reference/pto-cvid-cluster-id-mapping.md`，已附当前本地参考文档摘录。

Key `[1024,128] INT8` 若按行分成两份 `[512,128]`，可用 split=1；每份
65536B，两 lane 写同一 131072B GM 槽的不同半区。split=0 没有 per-lane
偏移，两 lane 内容不同会发生覆盖，不能用它拼 Key。生产、消费必须维持相同
shape、dtype、split、pipe id 与 tile 次序，尾组无有效 query 的 lane 仍需按协议参与。

A3 单向 V2C pipe 消耗两个 FFTS ID：base 为 READY，在 AIV 的 MTE3 完成点
发布；mode2 让同组 AIC 等待两个 lane 的 READY。base+1 为 FREE，在 AIC
的 MTE2 读完 GM 后发布。双向 pipe 占四个 ID。
现有 score 0/1、Attention 2..5 不能与新增 pipe 冲突。**frontend `id=6`
并不表示硬件 event 6**：PTOAS 根据 peer pipe 组件另分配 `flag_base`。
当前 `PTOResolveReservedBuffersPass.cpp:452` 收集 pipe init，组件分配从 0
起；该段没有将普通 `sync.set/wait` 纳入占用表。因此把自动 pipe 混入本 CSA
的手工事件协议，不能假设 allocator 会替手工事件避让；本轮未实施这种混用。

`TPUSH` 在环形槽满后等 FREE；`TPOP` 先等 READY，再 GM→L1，随后按
`SyncPeriod=(SlotNum<=2 ? SlotNum : SlotNum/2)` 批量返还 GM credit。
A3 Tile 版本 `TFREE` 自身为空，FREE 已由 TPOP 的 DMA 完成点发送；
`TPipe` 析构按真实剩余 credit 排空。因而不能把前端 TFREE 的源码位置当作
A3 GM credit 实际释放点，也不能认为它证明所有后续 L1 使用都已完成。
L1 本地槽由 `tileIndex % LocalSlotNum` 复用，程序仍须保证 tile 的最后一次
读取先于该槽下次 TPOP 覆写，并保留编译器流水依赖。跨 leaf/调用的计数和
三方参与数仍需一致，不能靠重建 pipe 隐式清掉未配对 READY。

参考文档描述 FFTS 每 ID 的计数范围 0..15，硬件 ID 共 16 个；可见当前
双槽方案不接近计数上限。这个容量事实不替代实际选择的 ID 冲突检查和排空证明。

## 容量与落地边界

当前 PyPTO 自动 cross-core pipe 默认深度已是 **2**，见
`Y/include/pypto/ir/transforms/utils/cross_core_pipe.h:66`；不能照抄旧测试
中“自动默认 8 槽”的注释。手工 `initialize_pipe` 省略 slot_num 时，PTOAS
仍按单向 8、双向 4 槽推导。`pl.cross_core_slot(slot_num=2)` 是现有的显式
自动深度接口。

对 128KiB Key 整块，单向双槽至少需要每组 256KiB GM FIFO，加每 AIC
256KiB L1 保留区（自动路径省略 local_slot_num 时与 slot_num 相等）；
24 组 GM 是 6MiB，Scale 尚未计入。手工 `local_slot_num` 在 A3 可单独
指定且必须不大于 GM 槽数，但缩成 1 需要新的本地 tile 生命周期证明，不是
本轮推荐。手工漏写 slot_num 的 8 槽会让该 Key L1 保留区达到 1MiB，
超过 A3 512KiB L1。

`reserve_buffer` 是整个函数分配的保留底座，不是能在 Score 结束后自动回收
的普通 tile。其他阶段的所有 Mat tile 都会放在保留区上方。因此即使 Score
局部估算能放入 512KiB，也不能据此证明完整 MIX 能放下。当前 A3 安全 UB
预算是 184KiB；每 AIV 64KiB Key 半块还必须与 Scale、分数排序、TopK
的生命周期共同核算。这里只读，不声称该整块 FIFO 接入已通过容量规划。

自动 GM workspace 也是原“复用闲置 score_arena、无新根 Tensor”约束的额外
集成问题：现有前端 `aic_initialize_pipe` / `aiv_initialize_pipe` 暴露本地
buffer、slot_size、slot_num、local_slot_num、id，没有公开的 GM Tensor
参数来直接指定本候选的 score_arena byteview。不能用不存在的 API 接管它。

下面只是已有接口的两端片段，假设同组 pipe 已按真实接口初始化且完成事件
避让；`key_half` 是原 gather 已得的 Tile。它展示可用的 GM 交接，不是新的
可运行实现或直传替代 API：

```python
# 两 AIV：各持有 key_half: Tile[[512,128], INT8, Vec]
pl.tpush_to_aic(key_half, split=1, id=0)

# 同组 AIC：接收完整 Key，然后保留原 QK/WS 运算
key_l1: pl.Tile[[1024, 128], pl.INT8, pl.Mem.Mat] = pl.tpop_from_aiv(
    split=1, id=0
)
# 原 Key→Right / N64 QK 使用 key_l1；此处省略原数学体。
pl.tfree_to_aiv(key_l1, id=0)
```

对应高层接口是在 `pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN)` 区域中调用
`pl.aic_gather(key_half)`；它走相同 A3 GM FIFO。无需为尝试名称不同的接口
重复编译或上卡。

## 当前决定与证据边界

保留 `../key_prefetch_design/` 的协议与资源模型，不实现 GM 版，不把 A5 或
GridPipe mock 当成 A3 新通路。现有静态旧版 Scalar 探针还显示，Key 页读取至
DMA 提交区间未被证明是主要耗时；细节见 `HISTORICAL_SCALAR_CONSTRAINT.md`。
当前动态 Score AIV 的较长 FFTS 等待说明存在生产/消费不平衡，不能单独证明
AIC 的页解析是其中原因，也不能推出把它迁到 AIV 会获得对应等待时长的收益。

事实：可用 A3 FIFO 仍有 GM 中转，且存在额外工作区、L1 保留和事件集成成本。
推断：在约 132MiB 额外 Key 交通与历史提交探针约束下，当前证据不足以推进
GM 预取完整实现。没有新的当前动态 AIC 分项证据时，保持设计归档更合理。
