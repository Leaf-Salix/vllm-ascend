# CSA 单 SPMD 动态范围与模板边界：后续参考

记录日期：2026-10-16。来源：用户转来的另一会话分析；本会话只读核对了下述
代码位置。本文件是后续设计参考，不是新增功能或验证通过记录。

## 当前决策

已完成的静态代码、失败快照和验证结果全部保留，不删除。当前仅验证静态
128K/B16（history=131072、B=16、每请求 6 个 query token、T=96）。动态 batch
和 seqlen 的完整支持后续处理，不因本记录而扩大本轮测试范围。

目标是固化较优执行安排，同时保留每步动态工作量。静态模板并不意味着其内部
所有数值和循环次数都成为常量；核内存在动态读取，也不等于整个入口已支持
任意 batch、shape、长度策略和工作区容量。

## 已有动态计算分布

PTO 有自己的工作划分和动态范围计算，没有直接复用 Native SAS/QLI 的调度
metadata；融合为单 SPMD 时，这些职责仍须保留。

| 职责 | 当前代码与作用 |
| --- | --- |
| 选择 Indexer 长短档策略 | `vllm_ascend/ops/pypto/deepseek_v4_flash_csa/decode_csa.py` 的 `_read_device_max_seq_len`（核对时约第 135 行）：TMR 编排从 device lengths 读取最大长度，进入相应分支。文件另有 `_use_host_max_seq_len` 和 `host_metadata` 工厂选择；不能把 TMR 的 device-read 描述套用于 HBG。 |
| 分配 query 与历史分片 | 同目录 `decode_indexer.py`，`indexer_score_topk_native_pair` 附近（约第 688 行）：读取 `kv_seq_lens`、`position_ids`，计算最大叶片数、各 query 可见候选范围、尾块和循环次数。 |
| 准备 attention 有效块 | 同目录 `decode_sparse_attn_csa.py` 的 `csa_slots_build_valid_qk_plan`（约第 207 行）：由本轮 Top-K、位置及相关 metadata 生成 sparse indices、bias、valid-block mask，供 QK/PV 消费。该阶段融合后仍需实际执行。 |

以上路径以 `vllm-ascend-dsv4-pto-0251rc1` 为根。行号会随修改变化，以函数名和
变量名定位为准；静态生成使用的计算基线仍是已冻结的 baseline_ops，不能假定
主仓后续修改会自动进入现有模板。

## 后续设计区分三层

1. 初始化或编译时确定数学 tile、核数、各核阶段安排、依赖关系及容量边界。
2. 每步从现有输入 Tensor 读取长度、位置、页表、Top-K，计算实际分片、访问地址、
   有效尾块和循环次数。不能把这些读取替换为捕获那一步的数值。
3. 将原跨 task 的必要依赖转为 SPMD 内部同步，保证生产者写入完成并可见后，
   消费者才读取；不能仅删除原依赖，也不应无依据增加全卡屏障。

在同一模板覆盖的形状、策略和容量范围内，每步长度增长原则上只需更新现有
device Tensor 内容，不要求每步重新编译，也不由此推出需要新增 Host
tiling-update 接口。这是设计预期；现有静态整层尚未通过，不能声称已验证。

## 当前 B16 模板已经保留的部分

[`static/h131072_b16/entry_jit.py`](static/h131072_b16/entry_jit.py) 中仍有：

- 约第 1050 行读取 `arg_kv_seq_lens` 并计算 Indexer 历史范围；用户转述版本约为
  第 1047 行，本轮 wrapper 变化造成行号偏移。
- 约第 1100～1101 行读取 `arg_position_ids` 与当前请求长度，决定候选可见范围。
- 后续 attention 有效块准备仍消费本轮位置和 Top-K，未被替换成历史快照。

与此同时，生成器仍特化 Tensor shape、部分循环边界、原逻辑 block 数及捕获到
的编排分支。当前七档模板不能直接代表单个完全动态的 CSA 实现。

## 后续应核对的边界与性能问题

- 跨越 Indexer 长短策略边界时，核对是否需要切换模板，或在单 SPMD 内补齐
  动态分支及对应依赖。边界条件必须来自实际代码，不能只按历史长度标签判断。
- batch/token shape 变化及工作区容量增长是另外的边界；“更新 Tensor 内容”
  不能自动解决描述符 shape 或容量不足。
- 同档内请求长度分布变化可能使固定逐核安排出现长尾。动态长度处理正确与
  分核均衡是两件事，不能把不均衡直接归因为缺少每步 Host tiling 更新。
- 后续以边界场景核对功能，再评估分配策略和性能；本轮不据此启动全量测试。

## 与当前执行停滞的关系

静态 128K/B16 的首次设备启动已报告 running-stalled。现有日志只证明单 MIX
任务发射后没有完成，尚未定位到具体核内阶段。没有证据将其归因于本文件提到
的动态模板边界或负载分布；当前故障定位与后续动态化设计分别记录。
