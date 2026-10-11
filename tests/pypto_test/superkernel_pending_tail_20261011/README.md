# 没有 pending 后继时普通返回：候选未通过性能验收

2026-10-11，任务 `task_20261011_113308_139157212552`，A3 设备 0，退出 0，设备已释放。
同一 Simpler SDK `58f6581eb`、TMR、CANN 9.2、NZ=2、atomic_add 关闭。
128K/B16 第二层真实权重，含 HC_pre + norm + CSA + HC_post；每组 30 次单次图重放，
两组交替顺序，reset 在计时外，不开 profiler。独立 O-A 对照不再继续测。

| 版本 | min μs | mean μs | max μs | P95 μs |
| --- | ---: | ---: | ---: | ---: |
| 普通版 | 960.70 | 973.75 | 997.58 | 984.25 |
| 条件尾部扩展候选 | 959.28 | 985.54 | 1107.72 | 1002.63 |

候选均值慢 11.79 μs（1.21%），24/30 配对轮次较慢，配对差值中位数 +5.49 μs。
最小值略低不足以覆盖 mean/P95/max 回退。新候选已撤回，恢复此前的私有原型源码和匹配编译器；
此目录仅归档失败候选的补丁与证据，不能作为当前默认实现或性能优化推荐。

尝试内容：Set 读取 TMR 当前派发寄存器。若仍是当前 task token，保持普通完成类型；
PTOAS 在正常尾部选择 PIPE_ALL，免掉无 pending 后继时的异步状态写入和 executor 事件排空。
已有 pending 时保留逐流水异步完成及真实 FIN 等待；DSL 两个局部语句不变。
诊断区分普通完成上界和异步计算完成时间，不把普通返回计作计算重叠。

3 项局部边界设备用例通过；AIV/AIC 同核重叠及普通尾部均有轻量 trace 证据。
完整 CSA 的 9 项输出/状态与普通版逐元素精确一致（torch.equal），改变输入后图重放也一致，
两组各 31 项保护区通过。**功能通过不代表性能通过**。
当前数据不能分解探测、分支、提前放弃交接机会和全图调度各自的时间贡献，不能认定单一根因。

编译阶段曾因引用未暴露给 incore 编译器的 platform header 失败，无占卡；
改为只判断“当前 token 未变化”，特殊退出状态保留既有异步排空路径，随后编译和设备验证通过。

[全部 60 个计时样本及检查](results.json)、[统计与基础 trace 摘要](analysis.json)、
[PyPTO 失败候选补丁](pypto.patch)、[PTOAS 失败候选补丁](ptoas-src.patch)。
完整原始结果及编译产物位于私有 `artifacts/csa_pending_tail_device_v1/` 和
`artifacts/csa_pending_tail_compile_v2/`，完整 trace 位于 `artifacts/pending_tail_local_device_v1/`。
未修改公共算子，未推送私有实现。
