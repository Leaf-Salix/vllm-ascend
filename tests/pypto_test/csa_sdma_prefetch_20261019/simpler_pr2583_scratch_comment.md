另发现一个与多发起者通道碰撞独立的融合正确性问题，完整复现及对照已贴在 https://github.com/hw-native-sys/pypto/pull/3001#issuecomment-6269721600 。

只启用一个 AIV 发起 SDMA，预取前加载一个业务 Tile、预取完成后再消费它：event/session/Wait 全部成功，但该 Tile 前 8 个 FP32 被覆盖。当前 PTO-ISA `TPREFETCH_ASYNC_IMPL` 把 context 的隐藏 scratch 固定 `TASSIGN_IMPL(..., 0x0)`，与编译器正常分配在 UB 0 的业务 Tile 冲突。两边都在 context 前加 PIPE_ALL 排除未完成 TLOAD；将诊断 scratch 单独放在 UB 8192 后，初次及 10 次变输入 eager 全通过。

这主要需要 PyPTO/PTOAS 与 PTO-ISA 的 scratch 分配及生命周期契约配套，**不是已确认的本 PR runtime provisioning 回归**。这里同步反馈是为了 kernel SDMA 的整体接入验收：原生示例若先预取、后加载业务 Tile，不能覆盖这个真实融合场景。具体源代码位置、任务结果、运行命令和完整脚本见上述 comment。
