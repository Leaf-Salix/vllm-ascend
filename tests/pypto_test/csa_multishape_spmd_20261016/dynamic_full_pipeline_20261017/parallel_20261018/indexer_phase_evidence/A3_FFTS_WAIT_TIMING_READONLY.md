# A3 wait_flag_dev / PyPTO sync_wait 的阻塞语义与计时范围（只读审计，2026-10-18）

## 结论

**当前 A3 路径不是“把一个等待异步发到 MTE2，Scalar 随即继续”。官方文档明确 A2/A3 的 CrossCoreWaitFlag 不支持选择等待流水，会阻塞本核后续全部流水指令；本地 CANN、PTOAS 与已生成的 CSA C++ 都把它降为只有 event ID 的 `wait_flag_dev(id)`。** `pl.system.sync_wait(..., pipe=MTE2)` 的 Python 参数，不能解释成 A3 硬件只阻塞 MTE2。

因此，在 **保持实际指令次序** 的前提下，用 Scalar `get_sys_cnt()` 紧贴既有 `wait_flag_dev` 前后取差，有依据覆盖该等待导致的 Scalar/本核后续发射停顿，而不只是“发射一条 MTE2 指令”的成本。不需要为此加 `pipe_barrier(PIPE_ALL)`，也不应添加它来追求所谓更准的数值。这个差值适合称“该调用位置的 FFTS wait 区间”，**不是远端算子纯执行耗时、MTE2 硬件忙碌时间，或可直接从整层扣除的冗余成本**。

本轮只查代码、已有产物和官方资料。没有写生产代码、编译、上卡、修改依赖或做hash。没有新 wrapper 的最终汇编或真机时间样本，所以**未验证今后插桩的两次 SYS_CNT 读取会否被最终编译器移动，也没有测量插桩扰动**。这是需在实际诊断生成物中核对的边界，不影响下面已有文档/源码对 A3 阻塞语义的明确结论。

## 官方一手资料与产品区别

1. [CANN 9.1 CrossCoreWaitFlag(ISASI)](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/ascendcopapi/docs/api/SIMD-API/%E5%9F%BA%E7%A1%80API/%E5%90%8C%E6%AD%A5%E6%8E%A7%E5%88%B6/%E6%A0%B8%E9%97%B4%E5%90%8C%E6%AD%A5/CrossCoreWaitFlag_ISASI.md)：模板参数说明区区分 A2/A3 与 950。A2/A3 忽略 modeId/pipe，阻塞后续所有流水；950 才可选择具体流水。模式2由一组 AIC/双AIV配对，消费者观察计数后解除阻塞并消费一次通知。此处只适用于 A3，不把 950 的等待流水语义套过来。
2. [CANN 9.1 GetSystemCycle(ISASI)](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/tool_interface/system_resources_and_variables/GetSystemCycle_ISASI.md)：计数读取属于 PIPE_S；A3 系统计数频率为 50 MHz，差值除50得到 μs。文档关于测普通异步 Vector/MTE 运算需要额外同步的提示，不能据此推出 A3 的阻塞等待也只是发射耗时。
3. [Ascend 官方 msopprof simulator 指南](https://github.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_simulator_user_guide.md)：现有工具可显示流水指令与同步关系；ResourceConflictRatio 保留同步细节，只有 PipeUtilization 时会去掉这些细节。这是仿真观察途径。本轮没有确认当前工具对该 CSA 的 `WAIT_FLAG_DEV` 逐点解析，以及24组核完整竞争/依赖建模，不能把仿真或聚合 PMU 当成本任务已有的真机逐事件证据。

访问说明：2026-10-18 的官方搜索索引返回了上述两篇 CANN 9.1 接口正文；再次直接 open 页面超时。本地 CANN 9.2 beta2 实现提供独立交叉证据。没有使用第三方博客作为语义依据。

## 本地证据链：从 PyPTO 到 A3 intrinsic

本机 CANN 路径是 `/data/pyptouser/yejia/vllm-cann92-main/env/Ascend/cann-9.2.0-beta.2`；以下位置按本轮实际文件读取行号记录。

| 层次 | 精确源码位置 | 已证实内容 |
|---|---|---|
| PyPTO Python API | `/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016/python/pypto/language/op/system_ops.py:119` | sync_wait 接 event_id/pipe/core_type，并转交 IR。该层文字仍描述 pipe 匹配；它本身不能证明 A3 硬件只等该 pipe。 |
| PyPTO IR→PTO | `/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016/src/backend/common/pto_ops_crosscore.cpp:489` | 产生 `pto.sync.wait <PIPE_...>, id`，保留编译器侧 pipe 属性。不能据此删除/更换原 pipe；它仍可能参与编译器调度分析。 |
| PTOAS A3/A5 区分 | `/data/pyptouser/qinchuanyu/pto-eager/ptoas-src/lib/PTO/Transforms/PTOToEmitC/Arith/ArithSupport.cpp:403` | `buildInterCoreSyncWaitArgs` 只有 A5 传 pipe，其他目标只传 id；`buildInterCoreSyncWaitCall/Dyn` 调用名为 wait_flag_dev。 |
| CANN 公共包装 | `.../aarch64-linux/asc/impl/basic_api/kernel_operator_block_sync_intf_impl.h:261` | `CrossCoreWaitFlag<modeId,pipe>` 调 `WaitEventImpl<modeId,pipe>`。同文件44行把 arch2201 绑定 dav_c220。 |
| CANN A3 对应实现 | `.../aarch64-linux/asc/impl/basic_api/dav_c220/kernel_operator_sync_impl.h:501` | `WaitEventImpl` 忽略 modeId，实际仅 `wait_flag_dev(flagId)`，没有 pipe 参数。495行 notify 则确实传 pipe 给 `ffts_cross_core_sync`。 |
| PTO-ISA A2/A3 | `/data/pyptouser/qinchuanyu/pto-eager/pto-isa-src/include/pto/npu/a2a3/TSync.hpp:100` | 跨核 Event.WaitImpl 使用 `wait_flag_dev(CrossCoreId)`；同核 wait_flag(src,dst,token) 是另一分支，不能混用语义。 |
| intrinsic 声明 | `.../tools/bisheng_compiler/lib/clang/15.0.5/include/cce_aicore_intrinsics.h:1308` 与 `:2752` | get_sys_cnt、wait_flag_dev 分别是 `__builtin_cce_get_sys_cnt`、`__builtin_cce_wait_flag_dev` 的别名，不是用户态函数轮询。 |
| 计数器实现 | `.../aarch64-linux/asc/impl/basic_api/dav_c220/kernel_operator_sys_var_impl.h:88` | GetSystemCycleImpl 的非CPU分支用 `asm volatile("MOV %0, SYS_CNT\\n" : "+l"(sysCnt))`。这里的汇编是 CANN API 源码；不是本次诊断新插桩的反汇编。 |
| Simpler 时基 | `/data/pyptouser/qinchuanyu/pto-eager/.worktrees/simpler-csa-core-args-256-20261016/src/a2a3/platform/include/common/platform_config.h:234` | `PLATFORM_PROF_SYS_CNT_FREQ = 50000000`。这是系统计数器时基，不是 Cube/Vector 工作主频。 |

CANN A3 包装中的实际短片段如下（仅摘本地源码）：

```cpp
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void WaitEventImpl(uint16_t flagId)
{
    (void)modeId;
    wait_flag_dev(flagId);
}
```

已有完整 CSA C++ 也吻合：

`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/diag_h131072_b16_clock_v2/codegen_probe/original.cpp:13909`

该处是 `wait_flag_dev(v4199)`；同文件8602、14107、14284、42557、48693、48736、49022也只有一个 event ID 入参。附近的 `wait_flag(PIPE_FIX,PIPE_M,...)` 等是核内流水依赖，必须与 FFTS `wait_flag_dev` 分开归因。生产 C++ 与本地 API 证据不需要编译器解释器或模拟器支持假设。

## 如何解释 wait 前后 syscnt

拟议诊断的最低侵入形式在概念上是：读取 t0 → 执行**原有一次**wait_flag_dev(id) → 读取 t1 → 在本核寄存器累加 `t1-t0` 与计数。这里没有新通知、没有新等待、没有新barrier，更不能再执行一遍wait来计时（它会消耗下一次计数）。本轮没有实现该代码。

由上述 A3 语义，可推得：当通知尚未到达，位于 wait 之后的 Scalar 计数读取不能按原硬件顺序先行执行。差值包含该位置剩余的通知等待、硬件指令及两次读时钟的基础开销。通知已积累时，差值主要是执行与计时开销。应同时保存次数，不能把零/小值理解成没有生产者工作。

不能推出的结论：

- 它不是从生产者开始到结束的时长；消费者到达 wait 前已经隐藏掉的生产时间不在区间内。
- 它不是所有执行单元在此期间都空闲。阻止后续发射不等于把已发射指令全部排空，也不等同 `PIPE_ALL` barrier。不要把本核等待时间自动当可删除的气泡。
- READY 的等待包含生产者先前该 pipe 工作、依赖链以及通知传播；ACK 等待则可能是槽位回收背压。两者数学和生命周期意义不同。
- 多核 wait 区间彼此重叠，总和不是整层 wall time；可以按核/角色/阶段/调用点查看分布，不能把72核累计值直接从L1耗时扣掉。
- 50MHz 计数的分辨率为20ns；短等待有离散化和读计数器开销。现在没有测量本脚本的空区间成本，不能擅自减某个常数。
- 最终 bisheng 可能安排标量计算和寄存器使用；只看到 C++ 两行顺序不等于已经验证新二进制布局。将来真正启用诊断时应核对最后机器码中两次 SYS_CNT 与原 WAIT 的顺序。本轮没有为此新编译，也没有给未经检查的新插桩宣称有效。

## 当前 phase_clock 没覆盖什么，以及无需新全局 fence 的诊断选择

当前诊断工具位置：

`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/compile_diagnostics.py:15`

其 `CSA_PROFILE_WAIT` 只包 `pto::comm::TWAIT` 或退避版 `CSA_WAIT_LOCAL_GM`；`instrument_clock` 仅替换这些名字及 TNOTIFY。GM TWAIT 的实现是 Scalar 读 GM 条件循环，见 `pto-isa-src/include/pto/comm/a2a3/TWait.hpp:53`。因此已有 phase_clock 的“wait ticks”不包含 `wait_flag_dev` 组内 READY/ACK；把“phase时间减GM等待”命名成纯计算会误导。

**建议当前只补一个诊断维度：在原 wait_flag_dev 调用处累计 syscnt 差值与次数，不新加任何 fence。** 若同一event ID跨阶段复用，应按 AIC/AIV + 源阶段/调用点分桶，不能只按ID汇总。所有值先在本核寄存器积累，到既有安全发布点写本核独占统计区；若逐wait写GM、DCCI/DSB，就会额外扰动循环。寄存器增加可能造成spill，这也需在实际生成物中检查，不能称为零开销。

Indexer源码可直接确定需要区分的角色（不是凭 vN 数字猜）：

- `.../vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/indexer_score_stages.py:176` 与`:378`：AIC 的 SCORE_CONSUMED / 双槽复用 ACK。
- 同文件`:469` 与`:641`：AIV 的 SCORE_READY，随后 load 分数，原 ACK 在对应 MTE2 位置发送。不能将 ACK 提前，也不能把原消费者等待挪过槽位保护。
- `attention_stream_stages.py:151`、`:168`、`:337`、`:354`、`:391`：Attention 的 KV/SCORE/PROB/PV READY。这些是另一组时序，不应与 Indexer 的等待混成一个数字。

已有官方仿真时序可以辅助看流水/同步，但本轮没有证明它能完整还原本工作负载的所有组内等待。未找到并验证一个现成的、在该实际A3任务上无需代码插桩即可逐调用报告 `wait_flag_dev` 耗时的硬件接口。普通 PipeUtilization/PMU 聚合比例也不能唯一分解到每个READY/ACK位置。

以上只提出诊断解释与最小插桩方向，不授权新任务、不建议为计时添加 MTE2→Scalar 事件、更不建议 PIPE_ALL。后者会改变原并发流水，回答的是另一个执行程序的时间。

## 补查：已上卡长 B16 插桩产物的机器码（同日）

根代理指定的完成任务为 `task_20261018_011813_39503015186`。本次只读其真实目录：

`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/current_clock_h131072_b16_v1/build_output/decode_csa_spmd_jqlk5eey`

**最终机器码次序仍未能证实，不能把以下过程记为反汇编 PASS。** 原任务已真机完整通过这一事实与这里的反汇编能力边界是两件事。

实际发现与保留证据：

1. `kernels/aic/cube_schedule.cpp` 和 `kernels/aiv/vector_schedule.cpp` 都存在；每份文本含 AIC/AIV 的预处理分支，共八处 `get_sys_cnt → wait_flag_dev → get_sys_cnt`，具体行号和原文已保存在 `wait_machine_audit/result.json`。文本行12492～12494、17575～17577、17772～17774、17950～17952、52316～52318、58425～58427、58503～58505、58741～58743符合原位包裹。八个文本位置不等同每个编译角色都保留八条机器指令。
2. 任务仅保留 `prebuilt/kernel_0.bin`、`kernel_1.bin` 的原始 `.text`。没有 `.o/.s/.ll`。这符合 PyPTO `_compile_single_kernel` 提取 `.text`，随后 `_prebuilt.py:_prune_build_outputs` 删除私有编译 sidecar 的流程；不是本轮删除了文件。
3. 本机 CANN9.2 `llvm-objdump -d` 不能把 raw `.bin` 直接识别成对象。本轮用 `llvm-objcopy -I binary -O elf64-hiipu64` **仅添加临时 ELF 容器**，不编译代码；分别逐字节确认容器 `.text` 与原任务 `.bin` 完全相等（没有hash）。给定 A3 target 后，官方工具对 AIC/AIV 前64条位置均只打印 `<not available>`，没有指令助记符。
4. 排除仅是临时容器的问题：对已有真实 A3 `sdma_warmup_kernel.o` 做同样只读反汇编，前32条也全部 `<not available>`；CANN8.3 的工具亦未给出助记符。因此当前工具无法给出所需的 SYS_CNT/WAIT 顺序，不能靠猜机器字节编码补结论。
5. 所有命令、直接失败、控制对象结果、C++原位片段及无源码编译的提取脚本，都保留在 `wait_machine_audit/`。`result.json` 明确 `machine_order_verified=false`。临时 `.o` 只是本地反汇编容器，不属于生产产物，不应提交为原始编译证据。未启动新的NPU任务，也未重编已有算子。

### 这些现成区间数据支持什么判断

根代理给出的长 B16 三轮各核区间中位数如下（μs）；它们属于插桩诊断，不是正式 L1 性能结果：

| 阶段/角色 | 轮0 | 轮1 | 轮2 | 对应等待 |
|---|---:|---:|---:|---|
| Score AIC | 16.05 | 16.60 | 16.15 | 双槽回收 ACK / drain |
| Score AIV | 107.90 | 106.69 | 107.63 | score READY |
| Attention AIC | 119.71 | 108.31 | 112.51 | AIV 产生的 KV / probability READY |
| Attention AIV | 18.35 | 14.51 | 14.99 | AIC 产生的 score / PV READY |

**结合已确认的A3硬件语义与源码包裹位置，这组方向性证据与“长Score当前更多等待AIC供数链，Attention当前更多等待AIV侧推进”一致，足以作为下一步优先排查方向。** 但由于最后机器码次序仍未核实，应称“源级原位计时与A3语义支持的诊断解释”，不能称“经汇编确认的精确阻塞分解”。

即使将来核实机器顺序，也仍不可进一步直接断言：Score 必然是 Cube 运算吞吐受限，或 Attention 必然是 Vector 算术受限。AIC供数链包括其cache读取、矩阵计算、FIX写回及调度依赖；AIV推进包括gather、softmax、probability写出等。现有每阶段累计值也没有拆分Attention的KV与probability两类READY。通知传播、先前依赖与插桩本身开销同样可能进入区间。需要保留“供数/推进链”这一范围，不能换成单个算子或某条算术指令。

三个轮次方向一致可以加强这个排查优先级；它不消除编译顺序未知，也不证明其中相同比例能转化为整层加速。保持 `ffts_issue_us` 字段原名时，应在报告旁解释它在A3上可能包含阻塞等待，不能将“issue”误读成只测发射；也不应为了让数字变成“wait”而加PIPE_ALL重新定义程序。
