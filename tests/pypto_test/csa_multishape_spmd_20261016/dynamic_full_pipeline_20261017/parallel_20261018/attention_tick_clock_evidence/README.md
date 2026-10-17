# d253 Attention 逐 tick 诊断（测试专用）

冻结的 `source_snapshot/` 是正式 d253 动态包的 33 份 Python 源码；直接文本对照一致。源码保持 `pl.dynamic` 和运行时 Q/L/position/page。这里只在正常生成的 C++ 上增加测试时钟，不改变正式源码、数学顺序、协议或业务控制流。入口仅接受 128K/B16、S6（T=96）、baseline-only；没有性能采样窗，不用于认定优化收益或 30% 目标达标。

`run_device.sh`、`run_tick_diagnostic.py`、`attention_tick_clock.py`、`source_snapshot/` 已在提交前冻结。主会话提交唯一诊断 `task_20261018_032708_262386210870`，输出为 [attention_tick_h131072_b16_v1](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/attention_tick_h131072_b16_v1)。本目录没有自行提交额外设备任务。后续说明或统计只能写独立文件，不能修改正在执行的入口。

已有证据只覆盖旧静态代码的四段累计时钟（`csa_multishape_spmd_20261016/qk_phase_clock.py`、`b16_qk_phase_clock_v1/qk_phase_clock/phases.json`），以及 593 的阶段端点、累计 GM 轮询与 FFTS 总量（`parallel_20261018/current_clock_h131072_b16_v1`）。它们没有当前 d253 的每 tick gather、softmax、PV 和等待分解，不能直接确定当前 Attention 推进成本。本诊断只填这条观测缺口；此前有限源码/历史审查见 [只读报告](../ultra_attention_aiv_readonly_20261018/README.md)。

完整符号 CPU lowering 仅执行一次，结果 [cpu_v1/result.json](cpu_v1/result.json) 为 PASS；日志为 [cpu_v1.log](cpu_v1.log)。[cpu_audit.json](cpu_audit.json) 对照既有 d253 CPU 产物确认：130 个 tensor ABI 参数、58 个符号维、long/G6 控制值相同；2365 条按序物理 PTO allocation（角色、地址、完整 tile 类型与源位置）一致。33 份源码与 CPU 保存副本直接文本一致。本步没有编译最终设备机器码，不能据此断言最终指令开销或设备正确性。

测试 hook 的源码/原物对照有两层边界：

- `instrument_cpp` 只增加带 `CSA_ATTN_PROBE_BEGIN/END` 的语句块；剥离所有新增块后逐字恢复原始 C++。原同步调用序列（TNOTIFY/TWAIT、FFTS、局部 event、pipe barrier、DSB/DCCI）保持一致；没有删除或重排任何原业务语句。
- 逐 tick hook 后组合原有 100 tick GM wait-backoff 和旧 `compile_diagnostics.instrument_clock`。**零新增硬件 fence 仅指逐 tick hook**；旧 phase_clock 本身仍在每次 notify 附加单行 DCCI 与 DSB。两者的审计分别保存在 [cpu_v1_codegen/audit.json](cpu_v1_codegen/audit.json)。本 hook 的空 asm memory clobber 是 C++ 编译器重排约束，会影响编译与发射，不能说诊断完全无扰动。

原始 C++ 实际位于 [cpu_v1/kernel/ptoas/dynamic_csa_full.cpp](cpu_v1/kernel/ptoas/dynamic_csa_full.cpp)。旧审计 JSON 的 `raw_cpp_path` 少了 `ptoas/`；不改原记录和冻结入口，只在 [artifact_path_mapping.json](artifact_path_mapping.json) 记录更正。真正返回给编译器的改写文本位于正常生成的 AIC/AIV wrapper；没有额外复制大型 C++/PTO。anchor 的局部行号和上下文在小型 audit 中，便于逐项定位；行号针对插桩前函数。

记录区复用 `signal[tick, physical_core, 32:64]`，每条 32 个 uint32，最多 60 条。起点来自该核原 phase0 notify 的 tensor 指针；它已经指向独占的 512 B core 行。AIC 为物理核 0..23，AIV 为 24+2*group+lane，不能用逻辑 group 代替物理核。新增写入不改 phase0 的第 0 项、其它核行、原 phase_clock 的 16..19 项；原通知 padding 1..15 仍检查为零。独立审查枚举 60×72=4320 条记录互不重叠，最大 word 552895 小于容量 552960。

每 tick 先清本地 32 word 暂存，再写 generation/tick 与已执行路径上的时间戳。无效或未执行分支的时间字段为零，导出不把它们当有效区间。tick≥60 时不写越界记录，只把当前 generation 写本核 phase0 的第 64 项作为溢出标记；既有 padding 检查和导出都会将它判失败。因此容量不足不能静默截断成功。该固定输入每 AIC 核恰有 22 条、每 AIV 核 23 条，导出核对完整 tick 集合，每轮共 1632 条。

| 侧 | 逐 tick 边界 | 解释 |
| --- | --- | --- |
| AIV | tick begin/end、tick 首部 MTE3→V reuse wait | 包含原依赖与插桩，不等于纯数学 |
| AIV | gather begin→GM ring TSTORE 后、ring 本地 join wait | 覆盖 gather/地址与元数据/搬运发射；TSTORE 后不是 DMA 完成 |
| AIV | KV ready before/after、SCORE FFTS wait before/after | 大小 T 的两份 SCORE wait/KV send 路径互斥，同 tick 不重复覆盖 |
| AIV | softmax body begin→PROB ready after | 包含原 softmax、prob 搬运与局部依赖 |
| AIV | merge gate begin、PV FFTS wait、merge end | gate 起点在已有 PIPE_ALL 之前；前导排空 tick 可没有 merge end |
| AIV | final publish begin/end | final block 输出发射及 running-state reset；end 在最后 TEXPANDS 发射之后 |
| AIC | tick begin/end、KV FFTS wait、首个 KV L1 TLOAD 后 | 对应 AIV KV 路径；TLOAD 后不是 DMA 完成 |
| AIC | SCORE ready、PROB FFTS wait、PV ready before/after | 给 QK/PV 发射区间及 AIC 等待的对应边界 |

所有时间来自原位 Scalar `get_sys_cnt`，按既有 50 MHz 换算，并对 uint32 回绕做差。新增 hook 不加中间硬件 fence；不能把发射端点解释为向量或 DMA 完成，也不能把含局部依赖等待的区间叫纯计算。A3 FFTS wait 的官方与本地实现证据见 [独立语义核对](/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-physical-plan-20261018/tests/pypto_test/ultra_physical_plan_20261018/A3_FFTS_WAIT_TIMING_READONLY.md)；不能断言 Scalar 必定在仅发射 wait 后立即继续。本诊断未核对新增包裹后的最终机器码。

**tick_end 在 32 word 记录写入之前，也不包含下一 tick 的初始化、最后一 tick 之后的 helper drain/release。publish_end 在最后 TEXPANDS 发射后，不代表 Vector/MTE3 完成。** 记录写入仍会扰动整条流水和对端等待。区间存在嵌套，不能把各项相加当纯计算或拿它替代完整 L1。

时钟暂存通过 Scalar 写入 GM padding，依靠保留的原 ATTN_CUBE/ATTN_VECTOR 发布路径获得主机可见性：helper 末尾原局部 event drain 不变，原 release 的 PIPE_ALL、DSB、ENTIRE_DCCI、DSB 与 TNOTIFY 不变；主机在设备同步并完成完整 oracle 后读取。新 hook 不提前宣告记录完成，也不更改原 output、state/cache 写回和 release。独立 reviewer `hc_tail_scalar_candidate` 核对了 CPU wrapper 的原 drain/release、槽隔离、互斥分支与逐字恢复，未见阻塞性问题；原物路径问题单独更正。

设备流程复用主 fixture、原分 task oracle 和完整输出/TopK/score/cache/state guard/phase0 检查，输入为 A/B/A 三轮 graph replay；此前保留 eager 与 graph warmup 检查。没有 Native forward/token 端到端验收，也没有七档扩展。信号读回保留真实跨调用数据；仅放行 phase 16..19 和 tick 32..63，1..15、20..31、64..127 仍必须为零，phase0 全 72 核仍检查当前代。

每轮导出旧 `phase_clock.json`，另有：

- `attention_ticks.json`：各物理核/各 tick 的 32 word 原始记录、具名时间戳与有效区间。
- `attention_summary.json`：每项的逐核有效次数与区间总和，以及这些逐核总和的中位数。**它不是单 tick 中位数，也不是全核总时间或端到端 latency**。字段 `interval_pairs` 给出端点，`eligible_for_performance=false`。
- `result.json`：完整正确性、协议与三轮导出路径。若设备执行失败、溢出、epoch/tick 集合缺失或时钟顺序不成立，应先定位，不自动重跑。

复现命令见 [commands.txt](commands.txt)。其中 CPU 命令是本次已经执行过的一次完整编译；设备命令只供主会话复核，任务已提交，不能在同输出目录重复执行。
