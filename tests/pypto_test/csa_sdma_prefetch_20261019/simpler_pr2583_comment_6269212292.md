配套接入 PyPTO #3001 后，A3 完整 MIX 的 SDMA 多 AIV 通道绑定发现异常，详细复现和事件诊断已贴在：https://github.com/hw-native-sys/pypto/pull/3001#issuecomment-6269202617 。

这里补充 runtime 侧要点，**不将其直接归因为本 PR 新引入的回归**：

1. 迁入本 PR head `0b34c256ff9c027f1afaf83c9c71eadda8ec2daf` 并独立重建后，原 kernel-mode SDMA 测试 TMR/HBG 各 5 passed，原每场景 100 轮；日志为 48/48 channels warmed。已有单发起者路径可用。
2. 完整 `MixedKernels{0,1,1}`、24 AIC + 48 AIV、每条 AIV 一个 context/event 的用例，在 TMR/HBG 均出现 Wait 失败及对应结果整行不写回；纯 eager 重复也可见，不只 ACLGraph。
3. 在生成 C++ 中直接读取事件状态的一次 TMR 失败：48 个 event 和 session 均有效，但 11 个 Wait 返回 false。各 worker 独占 64 字节诊断行并做 DCCI/DSB，AIC 输出正确。
4. **确认的通道别名**：实际 48 个 AIV 的 `session.channelGroupIdx` 只覆盖 24 个值，每个值重复两次。它来自 PTO-ISA `BuildSdmaSession` 的 auto channel，默认取原生 CCE `get_block_idx()`。user kernel 上读取的原生 `get_subblockid()` 均为 0，不能直接拼 `raw_block*2+raw_subblock` 修正；Simpler 注入的逻辑 block/subblock 则能正确区分 48 个 worker。
5. 建议在 runtime 核上下文中暴露**按物理 AIV 唯一**的 SDMA channel ID，并由生成代码显式绑定 session，或者在 runtime 中提供等价的物理发起者到通道映射。不要以逻辑 task ID 分配队列；不同 task 并发时逻辑 ID 不唯一。当前阶段尚未声称仅修通道 ID 就已解决全部问题，仍需唯一通道对照。
6. 建议增加完整 MIX 双 AIV、多发起者、重复 eager/capture/replay 的测试，检查 event/session 有效和 Wait 成功，避免仅检查输出拷贝而漏掉无效事件路径。

测试环境：A3，CANN 9.2.0-beta.2，torch 2.10.0 / torch_npu 2.10.0.post2。本地迁移基线另含 CSA 参数扩容等既有改动；具体测试范围和限制见上面的 PyPTO comment。
