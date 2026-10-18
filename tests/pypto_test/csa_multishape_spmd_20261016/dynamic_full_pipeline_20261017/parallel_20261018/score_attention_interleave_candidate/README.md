# 动态 CSA：当前 Score 与前波 Attention 交错候选

## 当前状态

本目录是在正式动态源码 `66882634` 上联合接入的实验候选。正式 `ops/pypto/deepseek_v4_flash_csa_spmd` 的 33 份源码没有修改；`REFERENCE_SOURCE_CHECK.json` 用直接逐字节比较确认参考和正式代码均与该提交相同，没有计算校验和。

截至 2026-10-18：`cpu_long_v5` 已完成保留58个动态轴的整层编译/PTOAS。
独立 PyPTO 的 view 别名传播及 while 内存复用遍历修复使 Left/Mat 分别降到
64/496 KiB，相关158项 CPU 测试通过。此前 v1～v4 的失败全部保留。

设备 v1/v2 卡在首轮等待；实际生成 C++ 定位到两个独立问题：循环回边并行赋值
被覆盖，以及负数 FloorMod 被当成截断余数。采用源级 carry 顺序规避与独立
PyPTO FloorMod 修正后，v3 已从设备正常返回，随后在测试检查器把布尔 true 的
`-1` 表示误判为非法计划处失败。保存快照的两波信号实际写者离线核对通过，
完整数值与性能随后在修正检查器后的 v4 验证完成。官方 PTOAS 0.67 仍能复现
两项问题，详见 [定点核对及复现](PTOAS_067_RECHECK.md)。

`interleave_floor_mod_h131072_b16_v4/result.json`，任务
`task_20261018_103852_401858324030`：128K/B16/S6 的 A→B→A 输出、Top-K/score、
完整 cache/state、Native/工作区保护区、信号写者与代际、基本图重放全部通过。
同轮5窗完整L1中位数：原多task **877.84 μs**，动态交错候选 **889.96 μs**，
历史静态 **787.30 μs**。候选较原多task慢1.38%，未达到30%目标，不采用为性能
优化结果；其它规格与 metadata/padding 专用场景没有在这一轮重测。

组件问题现已提交为 PyPTO #3026/#3027/#3028 与 PTOAS #1622，四份issue正文均附
最小修改diff、源码版本及验证边界，索引见
[上游问题与建议补丁](upstream_issues_20261018/README.md)。用户已明确改为提供issue
和修改建议，不直接提PR，也不添加标签。PTOAS独立构建已完成，20项Scalar数值
回归全通过；完整CSA尚未使用这个新二进制验证，不能据此改写上面的设备结论。

## 区间策略与执行方式

- 仅已有 G6/N64、Score 1024、平衡叶、prefetch1/toL0、每叶一根策略可进入本候选；device 描述符还必须能切出至少两个 wave。不是 B16/S6 白名单。无内部切点或其它策略执行原路径。
- 每个非末 wave 至少8个真实 group，并在后续 query 的 M8 边界切开；组数、边界和有效工作量仍读本次 device Q/L/位置及 group 描述符。
- 首 wave 只执行 Score 并生成其根、Top-K 和 Attention plan。随后 AIC 每轮先当前 Score step，再前波 Attention tick；AIV 反序。末端独立排空最后一个 Attention wave。
- 保留全局 Score `item % 24`、Top-K `query % 48`、plan `(query // 8) % 48`、Attention `query % 24`。各核循环次数独立，不按波内编号重分核。
- Attention `query_first` 与物理 `core` 分离：六处 query 地址使用前者，五处 GM ring 地址仍使用后者。AIC 内部 KV 为两份 L1 槽，先 PV 后 QK；GM 仍三槽，事件0/1和2..5不合并。
- Score Query 留在 Mat，每次切回时 TMOV 到 Left；Coeff 留 Left。Score roots/segments 与 Attention m/l/output 是显式 Tile 状态。容量估算只是设计依据，实际 planner、局部 MTE 依赖和性能仍待验证。

## 跨核依赖与工作区

每个 wave 使用 ROOT/TOPK/PLAN 三个独立 phase，编号 `60 + wave*3 + {0,1,2}`。原60阶段含义保留，每次调用所有72个参与者的 generation 仍各推进一次。Host 仅按已有 shape T 分配 `60 + 3*ceil(T/8)` 的保守容量，实际 wave 由设备决定；不新增根 Tensor 或动态轴。

每核在整 wave 的所属业务写回完成后才发布一次。根等待使用当前 query 实际可见的叶及其真实 AIV owner，不等待空 lane；Top-K 与 plan 只发布自己的真实行。A3 非一致性 Scalar cache 仍使用原信号发布/轮询及业务 fence/cacheinvalid。ACK 只释放 Score FIFO，不能替代叶根完成通知。

`signal_checks.py` 保留原60阶段检查并检查扩展 padding。`wave_signal_checks.py`
已接入 eager A/B/A、基本图重放及 L1 前后检查点，核对实际 writer/代际递推；
metadata/padding 专用重放和 source-matrix 入口尚未全部接入，不能扩大验收范围。
设计与进度见 `WAVE_PHASE_CHECK_DESIGN.md`。

## 并行分工与源码

| 位置 | 内容 |
| --- | --- |
| `interleave_cube_wave.py` | AIC Score 叶初始化、step与Attention tick交错、末叶ACK排空 |
| `interleave_vector_wave.py` | 双AIV的Attention/Score交错及逐叶根写回 |
| `interleave_control.py` | 按原全局owner筛选有效叶及累计实际step数 |
| `wave_boundaries.py` | device wave与M8 query边界 |
| `wave_dependencies.py` | 根/Top-K/plan的实际生产者依赖 |
| `cube_schedule.py`、`vector_schedule.py` | 接入完整CSA，并保留不适用策略的原路径 |
| `attention_tick_stages.py`、`score_vector_steps_g6.py` | 可暂停的数学步骤，Tile参数局部别名规避 |
| `runtime_abi.py`、`binding.py` | 按T分配独立wave信号槽 |

以上均指 `source_candidate/`，不是正式算子的新验收版本。`source_reference/` 是冻结原动态单SPMD参考。原分task PTO仍是最终性能目标的比较对象。

## 验证证据与失败边界

1. `range_source_audit.json`：两个range入口只裁范围，逐query/M8 Tile数学循环体AST完全一致。
2. `combined_call_audit.json`：联合源40个模块语法通过、202个已知helper调用参数数量匹配。它是最初联合接线审查，后续源码变化另见定点审计。
3. `wave_control_report_v1.json`、`wave_control_report.json`：18个代表整数场景，包括S1/S5/S6、mixed、dummy/gap、zero-visible、陈旧位置和T0。44个wave、1503个有效叶，所有权/步数/依赖覆盖正确。新版解释显式while的init/cond/yield与零迭代结果；不实现Tile数学、FFTS或缓存。
4. `cube_executor_audit.json`、`event_dag_evidence/`：叶初始化/事件保留的AST审计及同组事件有向图模型；不代替后端流水或真机。
5. `mat_left_smoke_evidence/`：最小 Mat/Left full-valid Tile参数、tuple、loop carry完成CPU/PTOAS；第一次fixture的GM Tensor carry元数据错误保留，修正fixture后通过。不是整层验证。
6. `cpu_long_v1/result.json`：首次真实联合编译在 `attention_tick_stages.py` 的形参重绑定失败。`inline_rebind_v1/`与`inline_rebind_source_audit.json`保存源码前后及逆变换AST证据。
7. `cpu_long_v2_summary.json`：Inline已过，ConvertToSSA共228条诊断。自然while标量回边、显式Tile phi混入GM赋值及后续unroll外层IterArg深克隆是分别处理的现象。`ssa_v2_source/`保存修改前源码。原始大诊断仅本地保留路径及大小，不把重复整函数IR打印入Git。

8. `cpu_long_v3/result.json`、`cpu_long_v3_summary.json`：保持58个动态轴，无样例shape实值化。通过此前失败的Inline/SSA，进入AllocateMemoryAddr时Left需要76KiB而平台上限64KiB。实际复用不满足此前容量估算，不能绕过平台检查、冒充已编译或直接上卡。
9. `unroll_iterarg_evidence/`：最小直接IterArg与局部Scalar别名对照。前者Inline后无自由变量、Unroll后4个自由变量、SSA报8条；后者SSA通过。动态T、tuple来源、full-valid Tile carry保持，没有if/while/GM混合phi；支持源码桥接的针对性，但原编译器边界未修复。

## 复现

在本仓库内运行：

```bash
python tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/score_attention_interleave_candidate/check_wave_control.py
bash tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/score_attention_interleave_candidate/reproduce_cpu.sh NEW_OUTPUT --strategy long --attention-query-metadata off
```

上面 `reproduce_cpu.sh` 是早期 Tile 注解 overlay 入口。当前编译器修复版本使用
`reproduce_memory_reuse_cpu.sh` 与 `run_device_memory_reuse.sh`，加载新工作区独立
构建的核心库；旧源码与共享安装保持不变。设备任务必须经过
`task-submit --device auto`。先完成代表档位的精度/同步检查，再采原分task PTO、
历史静态SPMD及候选的同轮完整L1；不重复整套七档来诊断编译问题。

## 本轮独立编译器工作区

- 历史静态SPMD依赖：`.worktrees/pypto-csa-static-fixes-20261016`，保留不动。
- 本轮内存复用依赖：`.worktrees/pypto-csa-memory-reuse-20261018`，分支`csa-spmd-memory-reuse-20261018`，同基线`cb2484471`。
- 已继承31份旧源码修改和已验证的Tile注解补丁；来源清单见`compiler_memory_reuse_worktree/PREPARATION.json`。新核心库必须独立构建，不能用旧.so代替本轮验证。
- 用户已授权合理的PyPTO/PTOAS内存复用改进。首个修复针对pass35中相同MemRef的gather/transpose_view在pass36被拆开的正确性问题，不是强行让本应同时活跃的Tile共用地址。
