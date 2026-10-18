# GM 返回别名与副作用保留：有限只读审查

结论：本候选将 helper 的 GM 返回值接到未使用局部名、最终返回原 GM 别名，当前关键路径有静态语义和 v3 现成 IR 的保留证据。不能把这个结论推广到任意 InOut/assemble 用法；尚未生成最终 binary，v3 在 AllocateMemoryAddr 因 Left 77,824 > 65,536 字节失败。

## 静态语义判据

- `/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016/src/ir/transforms/inline_functions_pass.cpp:240` 的 CloneInlineBody 用实际参数替换 callee 参数并保留完整返回前语句；同文件 409 行附近 tuple-return splice 只替换返回项，不删除 callee body。unused local alias 可以被删，但 body 的 store/wait 不是这个别名的纯计算生产者。
- `/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016/src/ir/transforms/utils/dead_code_elimination.cpp:58` 明确将 `tile.store`、`tile.assemble` 列为不可删副作用。431 行默认删除候选仅为非已知副作用的 AssignStmt；独立 EvalStmt 的 `tensor.write`、signal wait/notify、FFTS、fence、cacheinvalid、bar_all 均保留，并作为活跃根保留其参数/前驱计算。这里不是仅依赖 InOut 注解。
- `/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016/src/ir/op/tile_ops/memory.cpp:1305` 的 tile.store 声明输出复用第 2 个参数，并向该参数 DMA 写；`tensor.write` 在 tensor_ops/memory.cpp:908 声明 Scalar 写。signal_notify 在 sync_ops/sync.cpp:120 声明写，signal_wait 没有写参数但其等待副作用仍由 EvalStmt 保留，不能由 no_arg_writes 推断可删。
- `/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016/src/ir/transforms/utils/scope_outline_utils.cpp:102` 的 StoreTargetCollector 收集外部 tile.store 目标；1610 行附近将其作为可见输出，即使未在 used_after 中使用。Hierarchy 范围依靠 InOut 参数传递该副作用。方向分析另从操作注册的 arg_effect 收集写目标。
- `tensor.assemble` 不同：它是返回更新 Tensor 的 SSA 形式，输出 lineage 会复用目标，但不能只凭它写参数就认为任意未使用结果都安全。上述 DCE 副作用白名单没有 tensor.assemble。本候选所审关键输出的实际 IR 全是 tile.store 或独立 tensor.write，并未走 tensor.assemble，因此没有用泛化假设跨过这一区别。

读取的 overlay 核心库路径指向 `/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016/python/pypto/pypto_core.cpython-310-aarch64-linux-gnu.so`；本审查不重建、不修改该依赖，不对库与源码做 hash。

## 现成 v2/v3 IR 的窄检查

`inspect_passes.py` 只使用标准库 AST 读取 1–2 MiB 级的指定 pass 文件，不读巨大 result/log，不复制 pass 原文。v2 与 v3 的 Inline 后，以下操作总数逐项一致：

| 操作 | 静态语句数 |
| --- | ---: |
| tile.store | 125 |
| tensor.write | 32 |
| signal wait / notify | 64 / 56 |
| FFTS wait / set | 25 / 22 |
| fence / cacheinvalid / bar_all | 238 / 120 / 56 |

Score transfer、Attention KV/score/prob/PV/mi/li/packed、pair roots、TopK scores/indices、plan indices/bias 的 tile.store 目标计数，以及 plan mask 的 tensor.write 目标计数也逐项一致。signal 的 SSA 别名命名不同，所以同步按操作总数和语句形式核对，不用旧变量名漏计。

v3 的 `05_after_Simplify.py` 到 `29_after_NormalizeReturnOrder.py`，这些关键目标写入计数和全部同步操作计数再次逐项一致。所有同步调用仍为独立 EvalStmt；mask Scalar 写同样为 EvalStmt。关键写目标没有 tensor.assemble。例子行号和各目标计数在 `pass_evidence.json`，未保存大 IR 摘录。

这些静态数包含 runtime 分支/fallback，不是设备执行次数，不能由此宣称全部地址、数据或调度正确。Simplify 前后有展开和 constexpr 折叠，因此未强行要求该边界的计数相同。结论仅为：unused GM 返回局部名没有在已完成的 Inline、Simplify/DCE、scope 展开和返回规范化中删除上述关键写入/同步。

## 交付与剩余边界

- 本目录 `README.md`、`inspect_passes.py`、`pass_evidence.json`、stdout.txt、stderr.txt 为小型审查证据；无编译/NPU/hash。
- 既有 DAG 脚本、简短结果、stdout/stderr、README 已直接文本复制到 `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/score_attention_interleave_candidate/source_candidate/../event_dag_evidence/`，附 COPY_PROVENANCE.json，未重跑 DAG。
- 后续只需在成功的最终生成物继续检查实际 GM store/Scalar mask 写和同步；本结论不能绕过当前 Left 容量失败，也不证明异步 DMA 完成/局部覆盖事件。

