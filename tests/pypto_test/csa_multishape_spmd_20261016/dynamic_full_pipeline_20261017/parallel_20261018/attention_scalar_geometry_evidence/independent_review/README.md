# Attention本核Scalar几何缓存：独立只读审阅

**结论：在现有入口的只读metadata、独占工作区与同次plan合同下，未发现源级正确性反例；四Scalar的init/yield表达有当前DSL实现依据。** 这不是CPU lowering、资源或设备正确性通过，也不预记收益。本审阅没有编译、上卡、修改候选或扩展其模型。

审阅对象为 `/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/attention_scalar_geometry_cache_20261018/` 的 `source_reference/` 与冻结 `source_candidate/`。33个Python文件中只有 `attention_stream_stages.py` 改动。下文候选行号均指该目录 `source_candidate/`，其它文件两侧相同。

## 核心不变量与边界

`attention_stream_stages.py:286` 的当前gather query为 `core + (tick//5)*24`，同核每个query占连续五拍，跨query单调增加。缓存的tag是完整token编号，而不是request或slot。`298–308` 仅在真实tick（288）、原mask有效（289）且tag不等时调用原 `attention_query_geometry`；更新的是当前gather的owner/position/active。任何有效块观察到的值都等于原版当拍调用的值，因为同query对应的只读输入与helper逻辑不变。

| 边界 | 实际源码结论 |
| --- | --- |
| 首块无效、后续压缩块有效 | 首个有效块发现tag不等后刷新；没有错误依赖`sb==0`。 |
| 全空query、空核、最终三拍排空 | `361–373`原样携带缓存，不读几何。排空时qk_t可越过T，但被真实tick外层guard挡住。 |
| inactive、dummy、陈旧position、重复owner | 调用原helper，连同active=0一起缓存tag；仍保留mask控制的零KV写入及事件，未令两角色事件对数分歧。合法plan通常会使这些query全空；即使独立测试提供非零mask，两个版本也同样产生零KV。 |
| mixed Q、跨request和多wave | tag每query更新，不复用上一query的position；原`query_request`扫描真实Q/L并要求唯一owner，未引入token//S或固定S6。 |
| PV晚拍发布/reset | 在tick7，gather已是下一query的sb2，而PV正发布上一query的sb4。`519–534`仍只重置原m/l/oi四Tile；缓存从当前gather的`geometry_after_*`进入`536–539`，不会被PV reset误清。 |
| 下一kernel、图重放metadata更新 | `269–272`每次helper入口重新赋(-1,-1,0,0)，没有GM/全局缓存；下一调用重新读取当前输入。仍要求调用之间正确同步，不能在进行中的kernel外部并发改metadata。 |
| 大小档提前KV策略 | `t_dim>=144`与另一分支的事件次序、predicate和原数学体均未改，缓存只包住原geometry调用。 |

## 只读输入与发布链

`entry.py:693–715`将positions、main/idx Q/L声明为普通输入Tensor；`binding.py:80–96`直接保留Native对象，未替换成Host值。所有33源的metadata具名参数都是输入方向；显式`pl.store/pl.write`目的地扫描没有这些metadata名。完整调用链仍只将其传给只读几何/计划/数学参数，没有metadata写回。页表也保持只读。该扫描加源码审阅不是对任意用户制造的重叠指针做形式证明；`entry.py:48–49`原本就禁止未声明缓冲重叠，不能声称本候选新增了自动alias检测。

`vector_schedule.py:870–884`的plan与`896–922`的consumer使用同一 `idx_bounds/idx_lengths/positions/original_table/compressed_table`，相同cache容量及 `ATTENTION_QUERY_METADATA` constexpr。`886`完成plan发布，`890–893`在consumer前等待所有真实plan owner、原cache、压缩cache和RoPE。`phase_sync.py:92–98,113–116`的发布/获取fence和cache维护未变。plan的indices/bias/mask在这些owner完成后不再改写；后续环槽、packed等工作区独立。

`CACHE_QUERY_METADATA=False`调用原 `attention_query_geometry:335–342`，保留Q/L合法性、唯一owner和causal position关系。True调用原`327–333`从同次plan第5/6列读几何；plan `180–190`对失活行写(-1,0)，合法position经INT64→INT32整数保存而不经过FP32。新缓存始终用INDEX保存owner/position，active保留INT32，因此没有新增FP32精度范围、INT32截断或改变原非法页处理。所有逐地址页表列/物理页guard及valid-tail gather都原样保留。

## DSL表达、独立结构核对与局限

候选的循环init、iter_args及最终yield各8项：原4个Tile加INDEX/INDEX/INDEX/INT32四Scalar。三个新增分支层的两侧都yield同样4项；`qk_t`与缓存tag同为INDEX，原geometry返回类型与初始化吻合。当前依赖 `.worktrees/pypto-csa-static-fixes-20261016/python/pypto/language/parser/ast_parser.py:2737–2761`逐项创建iter_arg，`2956–2958`解析任意长度init tuple；`python/pypto/ir/builder.py:1008–1043`从每个init表达式推导类型，没有只许Tile或最多五项的限制。`ast_parser.py:3562`要求赋值形式yield，候选已满足。`tests/ut/language/parser/test_control_flow.py:808`有Scalar range携带，`tests/ut/ir/transforms/test_infer_tile_memory_space.py:4861`有Tile+Scalar混合携带的前端/IR例子；后者为950测试，仅证明通用IR表达，不能当A3后端通过。

本次独立stdlib AST检查结果：Cube完整函数一致；`q_active!=0`下整个gather数学/保护体一致；gather之后直到末尾yield之前的softmax/PV所有语句一致；全部`load/read/store/gather_row/sync_set/sync_wait`调用序列及geometry调用实参一致。差异仅在geometry调用的执行次数、额外Scalar控制/携带和注释。已有作者的可逆AST与7987调用模型另见候选目录 `source_model_audit.json`，本审阅未重跑或扩大该模型。

每query每lane原来对C个有效块调用C次几何，现在C>0时调用一次；最多减少4次O(B)扫描。没有减少业务gather字节、矩阵/排序或FFTS事件。旧 `query_metadata_v1` 的GM plan缓存曾长B16回退1.66%、短B32改善1.33%（MASTER约15987–16031行）；那是写/读plan几何的不同实现，不能当本Scalar候选已测，也不能忽略其“删扫描不保证整层收益”的约束。

新增4个Scalar是源码逻辑状态，不等价于只增加固定28字节物理寄存器或零spill；循环phi、inline geometry和全MIX生命周期可能影响实际代码/分配。唯一CPU应确认A3可lower、scalar未误提升到GM/Tile、UB/事件/全局分配无意外变化；最终收益和exact仍须独立完整同轮设备证据。本审阅只给源级无阻塞结论，不替代这些检查。
