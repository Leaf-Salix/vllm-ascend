# CSA 动态形状单 SPMD：生产包依赖闭包审计

2026-10-17。只读代码设计审计，没有运行测试、重新编译、启动 NPU 或修改共享 helper。
本报告与同级 `hc_front_cost_audit_20261017/` 成本报告配套。工作目录为当前 canonical 仓库；
旧动态 worktree 只作为历史来源，不作写入目标。

**迁移进展补记：** root 已将 73 个数学核体迁入生产 `kernels.py`，并迁入 `policy.py`、
`signals.py`、`signal_codegen.py`；新的验证已读取这些生产实现。下文的“当前来源”表
用于交代原依赖闭包，不能误读为要求再复制这批核体。下一步重点是消除剩余逻辑/late
IR 的重复依赖，以及收敛生成与绑定入口。新增只读 AST 审计见 `logical_contract_audit.json`。

## 结论

正式实现可以放到 `vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/`，而且可以完全不读取
七份历史 anchor、trace、binding JSON 或成本 profile。现有 `build_source_schedule` 已具有
这种算法能力：`anchors={}` 时构造任务图、MIX 分组和逐核调度，未测核使用模型权重 10.0。
这不是各档性能已经得到保留的证明；当前 selected 七档使用的是实测成本，换成常量后
分核和等待表可能变化，不能继承原七档耗时或设备 PASS。

目前 `range_policy.build_range_variant` 仍不独立：有同构 anchor 时先走旧重绑定，只有
缺少 anchor 才走源码 DAG。正式入口应把**所有规格统一走源码 DAG**，保留区间策略，
删除“找不到旧 anchor 才允许新形状”的控制结构。

最大未完成项不是 DAG，而是数学源码和运行入口的正式化。现在生成器读取冻结 IR，
运行测试还依赖测试参数准备器及全局编译回调。单纯搬 `range_policy.py` 并改 import
路径不构成独立生产算子。

已经迁入的生产数学体应继续作为唯一维护点，不能再维护一份 tests 候选核体作为线上
依赖；本报告以下给出进一步压缩剩余接口源码的可行方法。

## 当前 selected 路径的依赖闭包

下表只含当前已选路径，不把 `shape_specialization.py` 的几十个历史实验全部带入。
N 表示本目录上级，F 表示同级 `csa_static_spmd_feasibility_20261015`。

| 环节 | 当前来源 | 性质与迁移要求 |
|---|---|---|
| 区间选择 | N/range_policy.py `select_policy` | 纯 Host 描述符算法。保留 B/T/S、长度容量、分支边界；删除 selected JSON 与 CLI 参数处理 |
| 编排及 Tensor view 绑定 | F/bind_orchestration.py | 算法本身无需测量，但当前解释冻结逻辑 IR；保留 TaskRef、dummy、scope、manual_dep、别名及方向合同 |
| 数学源码和效果 | F/generated/core_bodies_ir.py、F/references/symbolic_ir 的 Outline/ExpandMixed 文件 | 算子代码来源，不是性能 trace；当前是历史文件依赖，必须转为包内受维护的源码/明确接口合同 |
| DAG | N/anchor_gap_design_v1/source_schedule.py、audit_dependencies.py 的核心函数 | 可纯源码生成，RAW/WAW/WAR + 显式依赖；移除审计 CLI、F 常量和绝对路径 |
| 分核 | N/balanced_schedule.py | 核日历、earliest_ready/critical_path、MIX 三成员配对是纯算法；成本 profile 读取不是必需项 |
| 等待偏序 | N/shape_specialization.py `rebuild_waits`，F/build_schedule_manifest.py `ancestor_sets` | 抽取纯函数；不能为了一个拓扑工具 import 整个 trace 读取模块 |
| QK 索引优化 | N/qk_index_cache.py | long 且 32<T<=96 的已选优化；纯源码变换但依赖 SSA 名和源合同，适合写成正式数学分支 |
| B16 行依赖 | N/row_dependencies.py | long、T=96、B=16 的已选优化；按函数名查任务，不靠旧 task_id；保留行映射证明和 FFTS 资源边 |
| 工作区与成员批次 | F/prepare_static_layout.py `prepare` | 纯算法；不复用临时 allocation，布局含控制表和进度区；可迁移 |
| 入口生成 | F/generate_static_entry.py `generate/jit_source` | 纯代码生成 + 数学源码依赖；含必要的 NZ、动态维度和别名处理，不能只保留调度外壳 |
| GM 信号 | N/signal_lowering.py、F/local_signal_adapter.py | 算子必需的发布/获取协议，迁为包内相对 import；保持 payload fence/cacheinvalid、bar_all 和 signal notify/wait |
| 等待退避 | N/signal_polling.py | selected 的 cached+DCCI+100 tick 来自这里；当前 monkey-patch 全局编译函数，不宜原样作为生产入口 |
| Native Tensor 绑定 | F/prepared_arguments.py | 可抽取，当前从 tests JSON 读布局；正式版直接接受生成的合同/布局对象 |
| 实际 JIT 调用 | N/device_compare.py 的少量入口逻辑 | `module.entry(*core_args, config=options)`，直接 native prepare/enqueue，不依赖 torch.library 大参数注册；测试驱动不可导入生产 |

`reuse_gm`、HC front/stream 实验、其它 Indexer/QK 实验、stage clock、协议随机推进、L1 采集、
Native 对照和 fixture 均不属于当前生产最小闭包。

## 无 anchor 生成的具体合同

当前已有可调用 API：

```python
schedule, audit = build_source_schedule(
    bindings, catalog, anchors={}, policy=policy.schedule_policy,
    model_cost_overrides=None,
)
```

它已经做一次 `rebalance`，返回含 `tasks/call_bindings/per_core_sequences/root_arguments/
temporary_allocations/wait_plan` 的完整 schedule，可交 `prepare/generate`。不应再把它当作
历史 anchor。若兼容旧二次 `rebalance`，第二参数应是 `audit['cost_seed_anchor']`，但新入口
直接删除重复分核更清楚。

当前 `rebalance` 为每个模型成员成本再加 1.0，所以无覆盖值时实际调度使用 11.0。
这属于无量纲分核权重；旧结构以 `start/end ×50` 承载它，不应输出成预测的实测微秒。
生产版建议直接传 `cost[(task, role)]`，把 seed anchor 名称及伪时间字段从接口中去掉。
可以先使用统一常量，后续按矩阵块数/历史切片数提供明确的算法权重；两者均不需要
历史 profile。权重只影响映射和顺序，不能用于删除数学依赖或宣称 30% 达标。

源码 DAG 替代七 anchor 必须保留：

1. 逻辑接口与后期物理核的读写效果并集。仅看前期普通 `pl.Tensor` 会漏写效应；例如
   `hc_widen_rms` 的 `inv_rms`。已有 allocation 写保守包含读取，并处理 RAW/WAW/WAR。
2. TaskRef 的本轮 ordinal、嵌套 deps、dummy 展开、manual scope 和 manual_dep 原语义。
   手动作用域不能按整个共享 allocation 强行补成全串行链。
3. 每个逻辑 block 覆盖一次；MIX 对应同物理组的 AIC+AIV0+AIV1。保持原数学体内部 FFTS
   信号的资源顺序，不把任务偏序证明误作全部硬件同步证明。
4. 实际 Tensor view 的 allocation、offset、dtype、shape、stride、NZ 格式及返回别名。
   当前算法不自动发现不同根参数的运行时 storage 别名；入口需验证/约束已有别名合同。
5. 未识别的参数注解、task 选项、作用域或核角色组合明确失败。不能静默套用旧模板。

已有 CPU 证据：七档均为 66 个数学任务，旧图 98 条直接边、源码图 101 条直接边，但
可达偏序均为 1377 对，缺失/新增偏序均为 0；WAR 新识别的边也已被原偏序覆盖。
20 个 CPU 案例覆盖无同构组合和短 Vector 分支。**这些历史检查使用已有的成本种子，
并不等价于本次常量成本版本已跑过设备。** B12/H65536、B17/H65536 的后续真机 PASS
是已生成候选的证据，也不能自动覆盖正式包重构后的所有调度。

## 数学源码如何收敛，不能直接搬整份历史 dump

当前三份源码分别为 7,488、14,751、17,619 行，合计 39,858 行；其中存在重叠。
`SourceContracts` 读取 late IR 主要为了核对 catalog 的参数方向；`generate` 的主体来自
core body bundle，但遇到原 NZ 输入时会改用 Outline 阶段逻辑核体，避免 rank-5 NZ load
被二次 lowering 成错误的 Tile。不能盲目只留 late body，然后声称 NZ 路径无变化。

建议最小、可审查的收敛方式：

- 包内保留各分支实际所需的数学函数，包括未在七点出现的短 Vector、B1～3、中间 B
  组合；不要依据七份 bindings 剪掉合法区间。所有 scalar/device metadata 动态计算保留。
- 保存每个逻辑任务的正式参数、返回别名、效果、物理角色及 worker 语义合同，直接从
  包内源码推导/核对，避免同时维护 late IR 全文件和重复 catalog。
- 原 NZ 核保留适合重新进入前端的逻辑数学表达；其它核保留当前已验证的 AIC/AIV 体。
  逻辑编排可先移植现有解释器，但只解释包内受维护的编排源码。
- 让 catalog 由这些包内定义生成，或使用小型源码声明；不从 tests 读取 JSON。旧 SSA
  名相关 AST 变换可先保留严格守卫，随后改为明确的函数参数/数学分支。

这要求一次明确的数学体/接口提取与对照，不是改几个路径即可完成。已有生成器中的
`RestoreBranchYields`、`NormalizeViewValidShape`、维度绑定、InOut 别名回绑、NZ 处理都
有实际前端用途；只有在正式源码自然表达了同等语义之后才能删除。

### 已核实的逻辑 stub 与 late IR 去重方案

本次仅解析源 AST，并执行原 `Recorder.resolve_returns` 的纯 AST 别名分析函数，未执行
编排、数学体、PyPTO lowering 或设备。完整结构结果与 67 份别名映射保存于
[`logical_contract_audit.json`](logical_contract_audit.json)，映射同时列在本文末尾。

**late IR 可去掉：** 生产 `kernels.py` 的全部 73 个物理函数，剔除
`original_block_idx/original_block_num/original_subblock_idx/original_subblock_num` 后，
参数名到完整注解的映射与原 late IR 全部相同，没有缺失或方向/形状差异。
但是 54 个函数的参数列表次序发生了 scalar-first 重排，例如 `hc_widen_rms`、
`hc_pre_linear`、`split_pre_post`。`SourceContracts` 应按参数名比较注解映射；实际生成
调用继续使用生产函数自己的 `inline_parameter_order`。不能只去掉 worker 参数后继续
比较列表，也不能把生产排列当成逻辑函数原调用顺序。

必须区分三种顺序：原逻辑签名顺序用于 `Recorder.call` 绑定参数；物理函数的完整注解
用于效果与接口核对；生产函数含 worker 的顺序用于最终传参。由包内函数推导这些信息
即可，17,619 行 late IR 不需要作为线上依赖保留。剔除的名字应是精确的四项白名单，
并校验其 INDEX 类型及确实来自 worker 改写，不能泛化为删除所有 `original_*` 参数。

**67 个逻辑函数均可解析返回别名：** `Recorder.call` 对非 root 函数只使用签名/参数
注解和 `resolve_returns`，不执行核体。全部 67 个函数的返回都唯一归于已有参数，
没有失败。因此保留原签名及返回注解，将函数体变为直接 `return param`、
`return param1, param2` 或 `return None`，足以服务 CPU 形状编排。必须保留空返回与
单/多返回的差别，不能按 Out 参数数量猜返回值；别名分析结论比参数方向更强。
完整 `_decode_csa_tp1_layer` 编排仍须保留，共 1,513 行，其中包含所有区间分支、
workspace 创建、Tensor view、TaskRef、manual scope 和动态分支选择。

**10 个 NZ 函数不能同时删数学体：** 当前 `generate` 遇到任一 NZ 实参会从 logical
源码读取原逻辑数学体，以避免二次 rank-5 lowering。下列 10 个函数应保留原体，或者
移到单独的生产 NZ 数学模块，并修改生成器明确从那里取体：

1. `_proj_b_mm_nz_kernel`
2. `_proj_b_mm_nz_kernel__2`
3. `_proj_b_mm_nz_kernel__3`
4. `proj_a_mm`
5. `proj_a_mm_0`
6. `proj_a_mm_1`
7. `proj_a_mm_2`
8. `qproj_matmul`
9. `qr_proj_matmul`
10. `qr_proj_matmul_0`

这 10 项按全部逻辑函数签名中的 NZ 参数得到，覆盖各形状分支，不仅是七档调用记录。
最小改动方案保留 root + 10 个 NZ 原体 + 57 个 stub；按源行数与两行 stub 粗估约
2,053 行（不含文件声明/注释格式），代替 14,751 行 Outline dump。后续也可拆为
`orchestration.py` + `logical_contracts.py` + `kernels_nz.py`，但不需要为拆分复制整份 IR。

风险：stub 的生成不能掩盖数学体之后的返回语义变化；需要在源码变更时更新/验证
别名合同。非 NZ 数学体仍以已迁 `kernels.py` 为准，效果以物理注解与逻辑合同的保守
并集为准。stub 仅是 Host 绑定合同，不得被误选作真实生成的计算体。完整 stub 重构
后的绑定/代码生成/精度尚未执行，本次结构可行性不能写成这些验证 PASS。

## 建议的最小生产组织

```text
deepseek_v4_flash_csa_spmd/
  __init__.py              # 稳定调用入口
  policy.py                # CSA 区间/形状策略；无七点 JSON
  math/                    # CSA 数学体与正式编排，按现有职责拆分
  contracts.py             # 形状、效果、返回别名、MIX 及受支持范围
  graph.py                 # TaskRef/dummy/manual + Tensor view DAG、拓扑/等待
  schedule.py              # 分核、算法成本；无 trace/profile 加载
  layout.py                # 工作区及控制表生成
  codegen.py               # 单 MIX SPMD 入口生成、绑定及数学特化
  signals.py               # 单写者进度 notify/wait、代际及缓存维护
  runtime.py               # 参数/工作区准备、JIT/cache、直接调用
```

名称仅表示职责，不要求机械拆成十个文件。`qk_index_cache` 与 `row_dependencies` 可以
归入 math/graph 的带守卫分支，不能将全部历史候选模块一并复制。

`runtime` 应在 warmup/capture 前生成调度和分配工作区，通过对象传递合同/布局；每个
并发调用者拥有独立 workspace/progress。进度初始化一次，图重放不清零，耗尽 INT32
代际时必须在排空后重置。Native 输入、四份 NZ 权重及 state 保留原存储，视图只做
绑定，不做 FP32 权重转换或 state 搬运。

内核 ABI 当前要求 `leaf_tensor_count <= 256` 的独立 PyPTO/Simpler 能力，入口必须按
实际能力检查，不能继续读测试环境变量或通过 tests 的隔离 bootstrap 获得能力。
应保留直接 JIT/native 调用路径，避免重新引入 torch dispatcher 的 64 项注册约束。

## 当前 selected 策略不能遗漏的两个条件

**行依赖：** B16 长历史优化按逻辑函数名重新找 task ID，故 source ordinal ID 本身
不阻塞迁移。但是它还要求唯一的长 Indexer/merge/plan/QK、8 组 O-A、明确 SSA/view
映射和 T96 行划分；不能扩展到 B17 等形状。它额外保留本物理组 Indexer 三成员在 QK
前完成的 FFTS 资源边，这是正确性合同的一部分。

**等待退避：** `local_signal_adapter` 仅有 bar_all + signal_notify/wait；selected 的
DCCI 后 volatile 读 + 100 SYS_CNT tick 退避来自 `signal_polling` 对 PyPTO 私有全局
`_compile_pto_module` 的回调替换。迁移不能悄悄去掉它并沿用旧性能数字，也不宜在可
并发编译的生产进程中原样 monkey-patch。优先形成算子局部、正式受支持的 wait lowering
扩展；若暂保留适配，须明确隔离/串行编译作用域并审查最终 AIC/AIV 源码，不能只检查
选项布尔值。DCCI/DSB 和 payload fence 不可因退避优化被删掉。

## 边界及实施顺序

当前 selector 只接受 S=6、T=B×6、B1～64、总长度容量至 131200；这些是现有能力边界，
不是对最终动态形状目标的重新定义。长/中/短策略跨界时需换模板；档内地址、有效长度、
尾块和循环继续由 device metadata 计算，不把 synthetic metadata 留作真实输入。

建议先完成无历史数据依赖的生产闭包，统一 source DAG + 常量算法成本，形成可调用
且可审核的功能版本。数学体、同步协议、Native 原存储接口先保持明确合同，再把
已选的 QK 索引/行依赖及局部 wait 退避迁进去。测试留在 tests：对照源码偏序/覆盖、
lowering、输出与状态、图重放及最终七档完整耗时分别验证。历史 selected 数字仍留作
历史参照；正式版七档逐档 ≤ 原动态 PTO ×0.70 需新版本自己的证据。

本次没有为上述迁移运行新测试，没有复制大 trace、成本 profile 或 binding 到生产，
没有修改 shared helper、生产代码或依赖仓。

## 附录：67 个逻辑函数的完整返回别名

顺序与逻辑函数的返回顺序一致；`无返回` 表示返回别名列表为空。此表是 CPU 绑定合同，不能取代真实数学体。

| 逻辑函数 | 返回的原参数别名（按顺序） |
|---|---|
| `_proj_b_mm_nz_kernel` | `partials__ssa_v0` |
| `_proj_b_mm_nz_kernel__2` | `partials__ssa_v0` |
| `_proj_b_mm_nz_kernel__3` | `partials__ssa_v0` |
| `comb_sinkhorn` | `comb_t__ssa_v0`、`comb_tail_store_inline449__ssa_v0` |
| `compress_state_commit` | 无返回 |
| `compress_state_commit_0` | 无返回 |
| `csa_cache_writeback` | 无返回 |
| `csa_rope_sign` | `idx_sin_signed__ssa_v0` |
| `csa_row_offsets` | 无返回 |
| `csa_slots_build_valid_qk_plan` | `cmp_sparse_indices_inline4329__ssa_v0`、`sparse_bias_inline4301__ssa_v0` |
| `hc_pre_linear` | `mixes_partials_inline487__ssa_v0` |
| `hc_pre_linear_reduce` | `mixes_raw_inline448__ssa_v0` |
| `hc_widen_rms` | `inv_rms__ssa_v0` |
| `idx_kv_scale_commit` | `idx_native_kv_cache__rv_v2` |
| `idx_qr_dequant_rope` | 无返回 |
| `idx_qr_proj_matmul` | `qr_acc_pad_inline2007_inline4067__ssa_v0` |
| `indexer_boundary_init` | `normed_kv_inline574__ssa_v0` |
| `indexer_head_coefficients` | `coefficients_inline1281_inline8756__ssa_v0` |
| `indexer_head_coefficients_0` | `coefficients_inline2879_inline9040__ssa_v0` |
| `indexer_head_coefficients_1` | `coefficients_inline1735_inline9271__ssa_v0` |
| `indexer_head_coefficients_2` | `coefficients_inline3035_inline9660__ssa_v0` |
| `indexer_score_topk_leaf` | `score_arena_inline3255_inline4168__ssa_v0`、`pair_arena_inline3226_inline4202__ssa_v0` |
| `indexer_score_topk_native_pair` | `buf_score_transfer_inline8845__ssa_v0`、`pair_arena_inline3226_inline4202__ssa_v0` |
| `indexer_score_topk_native_pair_0` | `buf_score_transfer_inline9167__ssa_v0`、`score_arena_inline3255_inline4168__ssa_v0`、`pair_arena_inline3226_inline4202__ssa_v0` |
| `indexer_score_topk_native_pair_1` | `buf_score_transfer_inline9382__ssa_v0`、`score_arena_inline3255_inline4168__ssa_v0`、`pair_arena_inline3226_inline4202__ssa_v0` |
| `indexer_score_topk_native_pair_2` | `buf_score_transfer_inline9767__ssa_v0`、`score_arena_inline3255_inline4168__ssa_v0`、`pair_arena_inline3226_inline4202__ssa_v0` |
| `indexer_topk_query_merge` | 无返回 |
| `indexer_topk_query_merge__2` | 无返回 |
| `indexer_topk_single_leaf_publish` | 无返回 |
| `kv_and_cache_write` | `idx_kv_scale_values_inline3994__ssa_v0`、`idx_native_kv_cache__ssa_v0` |
| `kv_hadamard` | `kv_final_inline3998__ssa_v0` |
| `kv_proj_matmul` | `kv_fp32_inline3665__ssa_v0` |
| `kv_proj_matmul_0` | `kv_fp32_inline3665__ssa_v0` |
| `kv_rms_norm_rope` | `kv_view_inline3736__ssa_v0` |
| `kv_score_proj` | 无返回 |
| `kv_score_proj_0` | `kv_proj_pad_inline3911__ssa_v0`、`score_proj_pad_inline3938__ssa_v0` |
| `kv_touch` | `ori_kv_flat_inline4326__ssa_v0` |
| `mix_x_rms_norm` | `x_normed_t__ssa_v0` |
| `proj_a_mm` | `o_r_pad_inline4514__iter_v1` |
| `proj_a_mm_0` | `o_r_pad_inline4642__iter_v1` |
| `proj_a_mm_1` | `o_r_pad_inline4770__iter_v1` |
| `proj_a_mm_2` | `o_r_pad_inline4871__iter_v1` |
| `proj_b_act_hc_post` | 无返回 |
| `proj_b_act_hc_post_0` | 无返回 |
| `proj_b_act_hc_post_1` | 无返回 |
| `proj_b_act_hc_post_2` | 无返回 |
| `q_rope_prepare` | 无返回 |
| `qk_pv` | `kv_transfer_inline4342__ssa_v0`、`score_transfer_inline4398__ssa_v0`、`probability_transfer_inline4399__ssa_v0`、`pv_transfer_inline4320__ssa_v0`、`mi_transfer_inline4318__ssa_v0`、`li_transfer_inline4403__ssa_v0`、`o_packed_heads__ssa_v0` |
| `qproj_dequant_rms_nope_rope` | `q_flat_inline8548__ssa_v0` |
| `qproj_matmul` | `q_proj_i32_inline7_inline3585__ssa_v0` |
| `qr_hadamard_matmul` | `qh_acc_gm_inline2065_inline4066__ssa_v0` |
| `qr_hadamard_quant` | `qr_hadamard_i8_inline583__ssa_v0`、`qr_hadamard_scale_dq_inline582__ssa_v0` |
| `qr_proj_matmul` | `qr_fp32_inline1_inline3598__ssa_v0` |
| `qr_proj_matmul_0` | `qr_fp32_inline1_inline3598__ssa_v0` |
| `qr_rms_norm_quant` | `qr_scale_pad_store_inline3589__ssa_v0`、`qr_scale_view_inline8490__ssa_v0`、`qr_i8_matmul_inline3588__ssa_v0`、`qr_view_inline8492__ssa_v0` |
| `quant` | `act_scale_dq_inline4531__iter_v1`、`o_r_i8_pad_inline4534__iter_v1` |
| `quant_0` | `act_scale_dq_inline4655__iter_v1`、`o_r_i8_pad_inline4645__iter_v1` |
| `quant_1` | `act_scale_dq_inline4765__iter_v1`、`o_r_i8_pad_inline4738__iter_v1` |
| `quant_2` | `act_scale_dq_inline4892__iter_v1`、`o_r_i8_pad_inline4874__iter_v1` |
| `rmsnorm_rope` | `normed_kv_inline574__ssa_v1` |
| `rmsnorm_rope_cache_write` | 无返回 |
| `rope_cs` | `rope_sin_signed_inline4321__ssa_v0` |
| `scatter_softmax_pool` | `pooled_kv_inline567__ssa_v0`、`pooled_kv_inline567__ssa_v0` |
| `scatter_softmax_pool_0` | `pooled_kv_inline3908__ssa_v0` |
| `split_pre_post` | `post_t__ssa_v0`、`pre_val_store__ssa_v0` |
| `weights_proj` | `weights_partial_inline3185_inline4185__ssa_v0` |
| `weights_proj_reduce` | `weights_inline3183_inline4184__ssa_v0` |
