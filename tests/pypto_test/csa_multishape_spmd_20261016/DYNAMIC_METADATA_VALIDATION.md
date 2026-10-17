# CSA 单 SPMD：分档与同地址动态 metadata 验证

按用户要求，不要求多个 batch 共用一份二进制。编译/初始化使用已有
Tensor shape/stride 及 Host 长度上界选择原有数学分支；每一步的实际长度、
位置、slot、页表继续读取设备 Tensor。不同档位的已测策略保留在
`selected_shape_policies.json`，不能把一个档位的优化直接设成所有档位的默认值。

## 128K/B16/S6 已通过的功能验证

任务 `task_20261017_023022_12848062458`，结果目录 `b16_metadata_replay_v1`。
使用正式 `/data/model/DeepSeek-V4-Flash-0731-w8a8` 第4层权重及合成历史，
对比完整 CSA attention 半层的单 SPMD 与冻结的原动态 PTO，不执行 Native forward。

| 重放 | 历史长度 | 请求物理页归属 | 原动态 eager / 原动态图 / 单 SPMD 图 |
|---|---:|---|---|
| A | 131072 | 原顺序 | PASS |
| B | 131071 | 交换顺序 | PASS |
| C | 130048 | 交换顺序 | PASS |
| A_return | 131072 | 恢复原顺序 | PASS |

每侧只捕获一次图，所有根参数和工作区地址保持不变，变体之间不清零进度。
B/C 的参考输出确实区别于 A，恢复 A 后参考输出完全一致。单 SPMD 累计6代
信号检查通过；重放期间没有新增编译单元。

所有变体的 `x_out`、Top-K 和六份 cache/state 均逐bit匹配原动态 eager。
完整 allocation 的非 slot 写区、只读 metadata 与工作区首尾保护区也都通过。
这是独立恢复历史后的四组用例，不代表连续 decode、同批异构请求长度或整网验证。

B/C 实际改变以下19个设备根输入；读取 CPU 内容仅发生在测试构造和结果比较中，
没有给生产算子增加 Host tiling-update：

- 长度/位置：`cmp_seq_lens`、`kv_seq_lens`、`position_ids`。
- 页表：`cmp_block_table`、`idx_block_table`、`ori_block_table`、
  `state_block_table`、`inner_state_block_table`。
- slot：`cmp_slot_mapping`、`idx_slot_mapping`、`ori_slot_mapping`、
  `state_slot_mapping`、`inner_state_slot_mapping`。
- RoPE：`freqs_cos/sin`、`cmp_freqs_cos/sin`、`inner_freqs_cos/sin`。

## 复现入口与性能口径

实现为 `dynamic_metadata_replay.py`，由 `device_compare.py --metadata-replay-only`
调用。仍须经统一 `task-submit --device auto` 分配设备。基于该档已选调度参数，
在 `run_device_compare.sh` 的正常命令末尾加此标志即可；不得覆盖已有结果目录。

该标志只进行首次编译/eager/图检查和 A→B→C→A，不重复 Event、L1 或 PMU 采样。
结果显式标记 `performance_measured=false`、`eligible_for_700us_target=false`。
真实性能仍以无插桩完整 CSA 的 L1 incore 首尾计时为准，当前选中128K/B16结果
仍为815.84μs，尚未达到700μs目标。

## 长度策略边界

原8K档按 `max_seq_len // 4` 选择 Indexer 的 `[2048,8192]` 压缩候选策略。
`max_seq_len` 含本轮6个query token；均匀fixture对应 history 下界8186。
首轮8K测试的 C=7168 跨入 `<2048` 的另一数学分支，图输出/Top-K比较失败；
A=8192与B=8191已通过，cache/state与保护区没有损坏。失败证据保留在
`short_metadata_failure_v1.json`，不把该结果当作当前档内运行失败。

新增 `metadata_cases` 在派发前用原编排模拟器比较函数调用签名和分支选择；
跨策略候选被拒绝并记录，8K同档用例因此改为8192→8191→8186→8192。
这里只是测试入口的边界检查，生产自动选择尚未接入。跨该边界需选择相应
数学模板/捕获图；本轮没有验证更短历史模板。匹配策略只是必要条件，
还须满足页表/工作区容量和S6契约，不能把少数样例通过说成整个区间已全覆盖。

详细证据：`dynamic_metadata_result.json`、
`b16_metadata_replay_v1/metadata_replay/replay.json`、
`b16_metadata_replay_v1/result.json`。8K/B16 的另一 Indexer 分支另做同类检查，
不能用长历史的通过结果代替。


## 8K/B16/S6 策略内验证结果

修正后的 task_20261017_024535_170390925040 / b16_short_metadata_replay_v2 完整PASS：
8192→8191→8186→8192。每次输出、Top-K、六份状态及保护区均通过，
图和地址固定，6代进度正确，没有新增编译单元，也没有做性能采样。
见 `short_dynamic_metadata_result.json`；首版跨策略失败记录继续保留。
