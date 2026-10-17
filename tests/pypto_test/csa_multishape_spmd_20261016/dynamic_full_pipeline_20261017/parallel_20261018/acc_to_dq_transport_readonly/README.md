# A3 Q_B Acc→DQ 传输能力与 FIFO 边界（只读）

基线按主会话指定的正式执行源 66882634；读取当前 [动态包](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd) 与实际依赖。本轮没有修改源码、编译、NPU、依赖或已有产物，也没有做摘要校验。

**结论：当前本地 A3 PTO/PyPTO 没有可用的 AIC Acc/L0C→同组 AIV UB 直接传输路径。** 前端 Acc→Vec 边界可表达，但实际 C2V FIFO 仍经 GM。Q_B→DQ 在 T96 的有效 INT32 载荷是单程 12 MiB、双向 **24 MiB（25,165,824 B，约25.166 MB）**，不是25 MiB。当前每个有效 INT32 只写一次、读一次；FIFO 可改变驻留容量与时序，不能自动减少这24 MiB。

现有阶段顺序还有明确背压环：保留全 QB 等待时，小 FIFO 等消费者腾槽会阻止 QB 完成；即使改逐块等待，DQ 前仍先等 QB 后方的 compact Hadamard/scale。因此仅替换交接存储不可执行。本轮不提出实现候选。

## 1. A3 实际实现，不能从 A5 或前端类型反推直传

| 当前本地位置 | 核对结果 |
| --- | --- |
| [A3 TMov.hpp:164](/data/pyptouser/qinchuanyu/pto-eager/.worktrees/simpler-csa-core-args-256-20261016/build/pto-isa/include/pto/npu/a2a3/TMov.hpp:164) | 合法分支为 Mat→Left/Right/Bias/Scaling、Vec→Vec、Acc→Mat，没有 Acc→Vec。Acc→Mat 是本 AIC 的 L1，不能据此认为 AIV 可读它。 |
| [arch_cce_intrinsic.hpp:396](/data/pyptouser/qinchuanyu/pto-eager/.worktrees/simpler-csa-core-args-256-20261016/build/pto-isa/include/pto/common/arch_cce_intrinsic.hpp:396) | `pto_copy_matrix_cc_to_ub` 外层架构条件仅 A5/Kirin 系列，不含 A2A3。头文件里存在函数名不等于 A3 可调用。 |
| [910B handler:30](/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016/src/backend/910B/backend_910b_handler.cpp:30) | 注释和实现明确 A2/A3 跨核经 GM；Vec 端由 GM-backed C2V pop 物化。 |
| [soc.cpp:147](/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016/src/backend/common/soc.cpp:147) | A3 Acc 边是 Mat/DDR，DDR→Vec；安全 UB184KiB、L0C128KiB。内存图本身不是 TMOV 合法表，需与实际指令共同判读。 |
| [tile_ops.py:760](/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016/python/pypto/language/op/tile_ops.py:760) | `aiv_shard` 是 AIC→AIV 交接语义，必须在 split_aiv 区域；split 继承区域。它不承诺物理直传。 |
| [cross_core.cpp:496](/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016/src/ir/op/tile_ops/cross_core.cpp:496) | aiv_shard 的 Acc→Vec 描述生产/消费两端，消费者值类型为 Vec；实际由 tpush/tpop 跨越。Mat 不是合法 C2V push producer。 |
| [PTOAS frontend pipe lowering:224](/data/pyptouser/qinchuanyu/pto-eager/ptoas-src/lib/PTO/Transforms/Lowering/PTOLowerFrontendPipeOpsPass.cpp:224) | A5 用 InitializeL2LPipe；其它本路径要求 GM buffer，再生成 InitializeL2G2LPipe。 |
| [A3 TPush.hpp:165](/data/pyptouser/qinchuanyu/pto-eager/.worktrees/simpler-csa-core-args-256-20261016/build/pto-isa/include/pto/npu/a2a3/TPush.hpp:165) | pushAcc2GMFiFo 将 Acc 用 TSTORE 写 GM 环槽。 |
| [同文件:397](/data/pyptouser/qinchuanyu/pto-eager/.worktrees/simpler-csa-core-args-256-20261016/build/pto-isa/include/pto/npu/a2a3/TPush.hpp:397) | popVecTileFromGMFiFo 构造同一 GM GlobalTensor，TASSIGN 到本地 UB 后 TLOAD。 |

源码树 `pto-isa-src` 与实际 Simpler build 头文件的 TMov、TPush、TPop、arch_cce_intrinsic 四份文本逐字相同；本结论不只是未安装源码的推测。已知 PyPTO 内部有“Acc→Vec on Ascend910B 可被 PTOAS 接受”的注释，指 IR 可表达，不可覆盖上述架构 lowering/物理路径证据。本报告限定当前软件栈能力，不声称穷尽未公开硬件能力。

## 2. 可用 API、同组 owner、尾行与同步约束

已有 [system_ops.py:336](/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016/python/pypto/language/op/system_ops.py:336) 的 `tpush_to_aiv(tile, split, id)`、[:418](/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016/python/pypto/language/op/system_ops.py:418) 的 `tpop_from_aic(shape,dtype,split,id)` 和对应 free；也可由 aiv_shard 自动生成。shape/dtype/split/顺序/pipe id 必须一致，同一 MIX 组绑定一个 AIC 与两个 buddy AIV，并不提供从任意两个 AIC 聚合到任意 AIV 的交换网络。

A3 C2V READY 在 FIX 上发布（TPush:147），AIV 先 wait READY 再 GM→UB；FREE 在 AIV MTE2 完成点发布（:375–377），保证生产者可复用 GM 槽。它不代表 AIV 后续 DQ 数学已用完其 UB Tile。本地 UB 生命周期仍需编译器事件；TPipe 析构还排空剩余 FREE（:505）。不能删 ACK/drain，也不能将 TFREE 的源码位置误作所有 UB 消费已结束。

当前 QB [query_projection_stages.py:132](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/query_projection_stages.py:132) 的 N256列块 n 属于 AIC n%24；完整512维 head h 来自 n=2h、2h+1，即两个不同 AIC。DQ [:200](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/query_projection_stages.py:200) 按八行/head组轮转到48 AIV；T96 每组16个head，单个 DQ 任务覆盖32个列块、涉及全24个 AIC。直接套同组 C2V pipe 会改变消费者分工，不能只替换一条 load。

数学保持要求：每 head 仍先 INT32→FP32、逐行 scale、逐列 scale，随后对完整512列做原 row_sum/RMS，再做 NOPE/RoPE。可以先无损收集两份256列到完整 [8,512] INT32 Tile（16KiB），再沿用原数学；不能用两次256列规约再相加冒充原512规约的逐位等价。

尾行也有实际限制：[pto_ops_crosscore.cpp:166](/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016/src/backend/common/pto_ops_crosscore.cpp:166) 拒绝 compact-normal 且有效行 pitch 与物理盒 pitch 不同的 split Acc。当前 M128、T96 的 Acc 源 pitch 是96，不能简单扩为128行再按两 AIV 切64行。no-split 分支保持源有效行、完整列宽；它仍经 GM，并不消除容量/路由问题。必须保持原动态 valid_shape，不为通过接口而取消有效行限制。

新增 pipe 还需明确避开现有手工 Score/Attention FFTS ID。前端 id 不等于最终硬件 flag_base，自动分配不能假定会看见现有手工事件。现有 [key_transport_audit](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/key_transport_audit/README.md) 已核对此约束，本轮不重复编译模型。

## 3. 精确字节与工作区容量

当前 QB 是 M128/N256/K256，四段 K 累加，INT32 Acc；末 M 用真实 min(128,T-t0)。DQ 每个有效 head 的 INT32 读取恰好一次，没有按五个 Attention 候选块重复读取 QB。

| T96 项目 | 元素/计算式 | 字节 |
| --- | --- | ---: |
| 有效 QB INT32 输出 | 96×64×512×4 | 12,582,912（12MiB） |
| 有效 DQ INT32 输入 | 同上，每元素一次 | 12,582,912（12MiB） |
| 以上双向有效载荷 | 两项之和 | 25,165,824（24MiB） |
| 现 q_proj 分配 | pad128(96)×32768×4 | 16,777,216（16MiB） |
| 单个 DQ [8,512] INT32 | 8×512×4 | 16,384（16KiB） |
| 单个原 QB 物理盒 | 128×256×4 | 131,072（128KiB） |
| 仅作容量算术：24组、每组两个原盒 GM 槽 | 24×2×128×256×4 | 6,291,456（6MiB） |

[绑定:148–174](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/binding.py:148) 分开 p128 容量与 T；q_proj 的16MiB容量不能当成每次写16MiB。PTO TStoreAccNz2nd 的 `mSize=validRow`，现成动态生成 C++ 的 Acc 声明携带运行时有效行，DQ完整块为 [8,512]，与12MiB有效载荷一致。该现成产物是此前同QB数学的 QA v2 False CPU，只用于核对读取/写回形态，不宣称已为66882634重新生成机器码。

以上均为逻辑有效 GM 请求载荷，不是实测 HBM/NoC/L2 字节。非对齐尾部事务、缓存命中、实际请求合并没有观测，不能按峰值带宽换算收益。

普通 FIFO 若替代完整 q_proj，可在改变调度并正确消费后把容量从随T增长改为固定深度；**仍须把每个 INT32 写 GM一次、读 GM一次**。若保留完整 q_proj 又另写 FIFO，则反而增加交通。仅缩容量可能改变缓存驻留，但没有证据证明本场景有收益。

自动 C2V UB 也不是免费：A3 pop 的本地槽地址步长是完整 SLOT_SIZE（TPush:430），不能按每 lane 半块大小估两个本地槽。原128KiB槽的第二个64KiB半块末端已到192KiB，超过当前184KiB安全预算；按完整槽保留两份更是256KiB。可另设local_slot_num、改为小块GM读取或重新安排UB生命周期，但那需要独立资源证明，不能在本轮视作已有可用双槽方案。现有 DQ 峰值地址包络约128KiB，也不能直接与这类永久保留槽相加后声称可放下。

## 4. 小 FIFO 的明确阻塞反例

正式 [cube_schedule.py:313–332](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/cube_schedule.py:313) 顺序是完整 QB → PHASE_QB → compact Hadamard → PHASE_IDX_HADAMARD。AIV [vector_schedule.py:728–773](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/vector_schedule.py:728) 先等 IDX_HADAMARD、完成量化/scale，再等全 QB，最后才执行 DQ。

因此把 QB 写入两槽并增加消费 ACK、其余业务顺序不变时：

`AIC 第3块等待FREE → AIV DQ尚未开始 → 等全QB完成 → AIC尚未完成QB`。

即使将全QB获取改为逐块 ready，只要仍保留 AIV 前段顺序，在有 compact 工作时还有：

`AIC等DQ FREE → AIV等IDX_HADAMARD → 该AIC须先完成QB才能发布IDX_HADAMARD`。

T96每AIC有5或6个N256盒，深度2不足以在消费者进入前存下全部。若为了维持当前顺序把所有未消费块都存下，又退回完整有效输出容量，不再是小环。相同槽内提前覆盖会直接破坏DQ数据，不能只靠phase generation保护。

解除这些环需要连同 DQ/compact 阶段位置、真实owner和通知一起改；若使用同组 FIFO，还需解决一个head的两个AIC来源。属于新的调度/所有权重构，不能称为现有 Q_B GM往返的无代价替换。先前长B16时钟也显示进入DQ时QB已全部完成；本轮没有新增能证明FIFO上限收益的观测。

## 5. 历史排重

检索 MASTER、动态 ATTEMPTS、原 CSA 验证日志和既有 QB/DQ 报告，未找到同一个“当前动态 QB→DQ 小FIFO”的已执行实验；这只说明该精确实现未找到，不能当成正向依据。

- [key_transport_audit](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/key_transport_audit/README.md)、[HC直接交接审计](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/hc_pre_direct_stream_v1/README.md) 已证 A3 前端跨核流仍经 GM，后者是只读设计，不是 QB 设备对照。
- [QB tile 审计](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/qb_tile_readonly/README.md) 已否定简单扩大N就能减半工作；M64/N512还会扩大T96权重载荷，不重包装该方向。
- [2026-09-27 QB紧凑写回](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_incore_20260927/qproj_compact_writeback/README.md) 确实试过 INT32经FIXPIPE乘2^-10写FP16、Vector还原后scale，GM字节减半，但增加舍入；B40的QB与DQ核内均回退，已撤回。它不满足本轮保持数学合同，不能拿来替代无损INT32交接。
- 旧 Attention direct-L1/MGATHER/独立row Tile是另一侧KV路径，不能当作本QB FIFO已测，也不在本轮重做。

没有找到可在现有接口和阶段顺序内直接消掉24MiB往返的候选；没有把工作区缩小当作带宽减少或性能收益。
