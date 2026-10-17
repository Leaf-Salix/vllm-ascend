# QA 小 M16 与已采用 DQ 分组的组合只读审阅

结论：两项修改在源码上可直接组合，未发现必须联动修改的 owner、数学或 GM 合同；它们有真实的前后依赖和编译交互，不能据单项 CPU 通过或性能结果推算组合收益。本次仅读取源码和已有生成审计，未生成组合源、未编译、未上卡。

## 对照与实际改动

2026-10-18 对 33 个 Python 文件逐文件直接比较文本：

| 比较 | 不同文件 |
| --- | --- |
| QA candidate 对 QA reference | `qkv_stages.py`、`vector_schedule.py` |
| 正式 `9d5327e5` 对 QA reference | `query_projection_stages.py` |
| 正式源对冻结 DQ candidate | 无，33 文件相同 |

QA 两份快照位于 [ultra_qa_small_m16_20261018](</data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_qa_small_m16_20261018>)；DQ 快照位于 [query_dequant_group_20261018](../query_dequant_group_20261018)。QA 独立设备参照尚未包含 DQ，所以其结果不能直接代表相对当前正式版本的增量。

QA 在 [qkv_stages.py:147](</data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_qa_small_m16_20261018/source_candidate/qkv_stages.py:147>) 新增小 T 的全局 M16 任务分工，在 188 行以实际 `T<=48` 选择；[vector_schedule.py:537](</data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_qa_small_m16_20261018/source_candidate/vector_schedule.py:537>) 同步扩大实际 QA owner 等待。DQ 仅改变 [query_projection_stages.py:174](../query_dequant_group_20261018/source_candidate/query_projection_stages.py) 的 head 分组选择，未改 Q_B matmul、内层逐 head 数学或 GM 布局。

| 实际 T / 代表 | QA 执行变化 | DQ 执行变化 |
| --- | --- | --- |
| 24 / S6 长 B4 | 两个 M16 块由同一组 8 核改为 16 核；QA 等待 8→16；最后 M16 读 8 行有效输入 | 仍 g4，3 个完整八行块，48 个任务；无 DQ 尾行 |
| 48 / S6 长 B8 | M32+M16 改为三个 M16、24 核；QA 等待 8→24；源码逻辑权重读取为原来的 1.5 倍 | g4→g8，96→48 个任务；每 AIV 总 head 数仍为 8；重复准备次数减半 |

上述 B/S 只是代表输入说明；两分支均取设备真实 T。S1、S5、mixed Q 不能用静态 S6=6 代替实际 T；本次未新增这些输入的设备覆盖。

## 依赖与编译交互

实际数据链是 QA→QR 归一化/量化→Q_B→DQ。QA 写入同一 `qa`，原 K 顺序和有效行合同不变；扩大等待后 QR 获取全部实际 QA 生产者。DQ 继续读取同一 `q_proj`、`qr_scale_pad` 和 RoPE 数据，不需要知道 QA 的 M 块 owner。正式 [cube_schedule.py:258](../query_dequant_group_20261018/source_candidate/cube_schedule.py) 保留 QA 发布；同文件 311–312 行保留 Q_B 发布。正式 [vector_schedule.py:756](../query_dequant_group_20261018/source_candidate/vector_schedule.py) 在 DQ 前等待全部 24 个 Q_B owner，768 行由全部 48 AIV 发布 QUERY；Cube 381 行仍等全部 48 QUERY owner。原 [phase_sync.py:84](../query_dequant_group_20261018/source_candidate/phase_sync.py) 的 fence/cache 发布获取合同不变，源级未形成新增等待环。

性能仍可能交互：新增 QA 工作的 AIC 原本可提前进入紧随其后的 IDX_PROJECT（Cube 260–272 行），并非整层空闲核；它会改变 Indexer 后续投影和 Score 的到达时间。DQ 位于 AIV 的 Indexer cache 提交之后、后续压缩/Score 之前，缩短它也会改变两类核的相遇时间。两个局部改善可能重叠、抵消或将瓶颈转移，不能相加。

资源方面，QA 使用 AIC 的 Acc/Mat/Left/Right，DQ 使用 AIV UB，没有新增二者直接共用的 Tile。既有 [QA codegen_audit.json](</data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_qa_small_m16_20261018/codegen_audit.json>) 显示新增分支 15 条分配，原 2365 条分配的地址/类型/形状保留，地址末端未增；DQ [generated_audit.json](../query_dequant_group_20261018/generated_audit.json) 显示 DQ Tile 全部不变、DQ UB 末端 131072 B，完整 AIC 归一生成文本不变。但 QA 的旧分支局部事件顺序有变化，QA 后三个投影的 M→FIX 事件编号也顺延；这已经说明单 MIX 的编译编号/调度并不按源码文件隔离。

以上两个独立 CPU 审计的 Attention controls 还不同（QA 为 OA256/OB128，DQ 为 OA128/OB96）。它们分别与自己的相同 controls 基线比较有效，不能互相拼成组合资源证明。最终组合仍可能改变 SSA/寄存器、生命周期、局部事件分配；即使 T24 的 DQ 实际取值不变，含新分支的整个编译产物也不能自动视为 QA-only 二进制相同。

## 若 QA 独立结果支持推进，最小组合验证

1. 以当前正式 DQ-only `9d5327e5` 的 33 源为参照，仅合入 QA 两文件，冻结并核对三文件差异范围；先检查组合生成物的动态 T/GM alias/ABI、QA 新分支槽和事件、DQ UB 分配、QA 的 16/24 owner 等待以及固定 QB24/QUERY48 依赖。这里是建议，尚未生成或编译。
2. 用同一个组合版本覆盖长 B4/T24 与长 B8/T48：前者验证 QA 尾 M16 和 DQ 规则未变时的编译交互，后者覆盖两项同时执行变化。沿用完整 CSA 的 exact A→B→A、图/guards，再做相同进程、同轮完整 L1 首尾配对；增量参照必须是正式 DQ-only。已有 QA-only 结果可辅助解释，不要求再加第四臂。
3. 不先铺七档或另加 S1/S5；若组合生成物在保留分支发生实际规划/同步改变，或这两档发现回退，再按具体差异决定是否扩边界。两档通过仅说明这两档，不代替全动态范围的设备证明。

当前没有组合 CPU、设备精度或性能结论；本报告不提出新的算法或调度修改。
