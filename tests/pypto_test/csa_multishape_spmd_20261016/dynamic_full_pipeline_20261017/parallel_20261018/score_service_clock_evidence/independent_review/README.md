# Score AIC 服务诊断：独立只读审阅

**Ready，无本范围内的阻塞项。** [STATIC_REVIEW.json](STATIC_REVIEW.json) 的 `ready=true` 可用于唯一设备诊断的提交门槛。审阅只执行标准库文本/作用域/槽检查，不启动 PyPTO、C++ 编译器或 NPU。没有修改冻结 hook、runner、33 份执行源，也没有计算 hash。

审阅对象：

- hook：`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/score_service_clock_20261018/score_service_clock.py`。
- runner：`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-indexer-20261018/tests/pypto_test/score_service_clock_runner_20261018/` 的 `run_score_diagnostic.py`、`run_device.sh`、同名 hook 副本和 `source_snapshot/`。
- 既有 C++：主 `parallel_20261018/oa_pair_manual_h131072_b16_v1/build_output/decode_csa_spmd_jp7o4g2r/` 的 raw `ptoas/dynamic_csa_full.cpp` 及 AIC/AIV 两个 wrapper；仅在内存中做插桩，不复制大 C++。

完整可复现结果在 [independent_audit.json](independent_audit.json)，脚本为 [audit_readonly.py](audit_readonly.py)。

## 实际 C++ 作用域与原程序保持

初稿“一个临时时钟”已由实际实现改为 20 个唯一 `const uint64_t csa_score_start_N`。独立审计对 raw 和两个 wrapper 各运行一次文本插桩，等长遮蔽 C++ 注释、字符串后建立实际大括号祖先栈：每个 start 有且仅有一个减法 end；20 对均在**同一作用域**，且声明先于使用。body 位于 Score 外围作用域；step 和各 sample 位于原 step 循环；step ACK 位于原条件分支；leaf drain ACK 位于原尾循环。重复迭代重新执行本次初始化，内层计时不会覆盖外层 start。此项不只依赖“剥除后原文一致”，也不是 C++ 语义编译证明。

三份原物均通过剥除 Score 标记后逐字恢复、原完整同步调用序列相等、Vector 函数逐字不变。新增代码只在 AIC 活跃 guard 内。组合次序为 Score hook → 既有 100 tick wait backoff → 既有 phase_clock；组合后的 Vector 函数与“只做后两种既有变换”逐字相同。runner 的 guard 检查在两份 wrapper 文本含有两函数的情况下仍正确：AIC guard 有 44 对 Score 标记，AIV guard 无 Score 标记，两侧保留 phase_clock。

固定 sample 是第九个 QK（j=8）及与其交错的前一面板 WS；共 6 个原代码分段和 10 个原局部 wait 站点。原 16 QK＋16 WS、每步 READY 和两处 ACK 均保留。leaf 尾 ACK 在 step 计时外，故检查 `ACK ticks ≤ body ticks`，没有误加 `ACK ticks ≤ step ticks`。计数仍按实际工作累加，不固定成某一次观察的 22 步；每 leaf 的复用 ACK 与尾排空 ACK 总数等于实际 step 数。

## 512B 槽、代际和可见性

写地址从原 BEGIN 的本核 signal view 导出。原物中 AIC physical id 乘 128 word 后构造该 view；hook 再加 `34*72*128+32`。24 个 AIC 只写各自 phase34 的 32..63，共 768 个互不重叠 word，线性范围 313376..316351；不碰 signal[0]、其他核槽或其他 phase。这里 512B 是协议槽大小，不是假定的硬件 cacheline 大小。

每调用先清零本地累计量。结束时恰写 32 个字段各一次：32=schema、33=epoch、34/35=leaf/step、36..49=七组 uint64、50..52=ACK/sample/local wait count、**53=编译时 local wait 站点数 10**、54..63=0。epoch 最后写。无工作 owner 仍写本轮 schema/epoch、零计数与合法的 body 区间，避免读取旧调用残留。53 项为实际实现新增的合同元数据，已区别于初稿“53..63 保留零”。

dump 位于全部 Score 叶结束和原局部 drain 之后、原 Score release 之前。原 MTE/FIX barrier、DSB、ENTIRE DCCI、DSB、PIPE_ALL、阶段通知保持，诊断写入借此发布。body_end 在 dump 和原 release 之前，故不含这次记录保存开销。Score hook 自身只加空 asm 的编译重排约束，不加硬件 fence；**组合的旧 phase_clock 仍在自己的 notify 后写 16..19，并保留单行 DCCI＋DSB**，不能把整个诊断描述为零新增维护。

runner 在 eager/graph replay 同步结束后读回，保留 signal/ffts 跨调用，不人为清零旧槽；随后完整数值、Native 保护区、scratch 保护区、phase0 全 72 核代际检查，再导出。padding 只放行全部 phase 的既有 16..19 与 phase34/AIC0..23 的 32..63；本诊断不混用 Attention tick 或 FFTS 20/21 探针。

独立合成检查覆盖空 owner、尾 ACK 时间大于 step 时间的合法情形；11 个反例均拒绝，包括 schema/epoch、phase34/BEGIN 旧代际、AIV/其他 phase 越槽、FFTS20、54 保留位、槽尾、ACK 计数和站点数错误。合成数据仅验证 host 检查合同，不是设备时钟或精度证据。

## runner 与解释边界

runner 的 33 份执行源与已有正式 668 参考逐文件直接字节相同；hook 副本与作者冻结原件相同。逆向撤销声明的有限替换后，runner 全文与 AST 恢复原 Attention 诊断入口。固定 128K/B16 S6、baseline-only、A/B/A 三轮；没有开放性能窗、额外候选或其他形状。

该 hook 记录 Scalar 原位区间及其中原有依赖、计时和累计开销。qk begin/前 WS end、FIX begin/WS math end 共用原源位置时，实际登记顺序让少量读钟/累计开销交叠；局部 wait 计时也可能包含同边界累计语句。故这些量可作有限诊断，**不可当互斥纯硬件分解，不减 nested wait 来宣称纯计算，不将 j=8 乘 16 外推整段**。

机器码反汇编目前不可读；最终 CCEC 时钟顺序、寄存器 spill 和插桩扰动没有验证，不能从本标准库检查声称通过。主会话已提交唯一任务 `task_20261018_073047_323892912778`，结果应另外记录；本报告不包含设备结论。

最小归档：本目录 `README.md`、`STATIC_REVIEW.json`、`independent_audit.json`、`audit_readonly.py`、`audit_stdout.txt`、`audit_stderr.txt`。复现本审计仅运行 `python audit_readonly.py`；不会编译或上卡。
