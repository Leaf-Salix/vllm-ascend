# DSV4 Flash CSA padding 验证边界

当前目标与覆盖顺序以 [任务清单](DSV4_FLASH_CSA_TASK_CHECKLIST.md) 的 A5、E1/E2 为准。
旧的“恢复 padding 支持”实施计划已完成其历史用途，旧预测脚本已删除。

仍须保持的 Native 协议：

- 补位 slot 为负，不能写 cache/state；补位页表可能为 0，不能仅靠 `page >= 0` 判断请求有效。
- 补位请求的 `seq_lens=0`，positions 可能保留陈旧值；读取 state/compact metadata/页表前必须按有效请求限定范围。
- 空 rank 的 dummy 与真实 batch 中的补位请求不同，不能只用 `seq_lens == 0` 代表两者。
- graph replay 使用固定地址的 device metadata；保护不能只在 capture 时检查 host 值。
- 当前 DSpark query 长度为 6，仅覆盖清单声明支持的合法档位，不顺带扩展 ragged 输入。

现行接入与保护逻辑位于两版 `service.py`、`service_config.py`、`native_adapter.py`
及 compressor/indexer/sparse attention 的设备函数。修改这些路径时，先单卡验证
有效行、负 slot、padding 和保护区，再补受影响的真实权重 16 卡场景。

旧 padding 试错记录已删除。当前版本的尾块、补位、空 rank 与图内容更新
证据按 A5/E1/E2 收集，不把旧运行结果当成当前全部 graph 档位通过。
