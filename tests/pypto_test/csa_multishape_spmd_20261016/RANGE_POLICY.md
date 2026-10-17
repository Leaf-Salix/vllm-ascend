# CSA 工作量区间入口（实现中）

当前目标是七个验收点各自相对同轮动态分 task PTO 至少降低 30%，见
[GOAL_AND_CONSTRAINTS.md](GOAL_AND_CONSTRAINTS.md)。以下范围是生成器当前可表达的
数学合同；没有把 CPU 检查写成设备验证，也没有宣称这些阈值已经最优。

## 入口及长度定义

`range_policy.build_range_variant(descriptors, max_seq_len=..., output_root=...)`
按输入的 shape/stride/dtype 和调用方已有 Host 长度上界生成变体。
这个函数负责源码与布局；cached100 是设备编译上下文中的策略，不能把单独生成/编译的源码
当作已启用 cached100。`device_compare.py --range-policy` 会在编译前安装现有
`signal_polling.bypass_signal_wait(..., cached=True, backoff_ticks=100)` 上下文，并检查实际CCEC源码。
测试入口增加 `run_device_compare.sh B HISTORY OUTPUT --range-policy`，不再要求手写
排程、QK 索引缓存或行依赖开关。仍可使用原研究开关单独调优，但不能与范围入口混用。

H 是本轮之前的历史长度，S 是每请求本轮 query 数，L 是包含本轮 query 的长度。
当前 fixture 中 L=H+S；原编排选择使用 C=floor(L/4)，故 H=8192 不是精确分界。
当前核体仍限 S=6、T=B×6≤384、6≤L≤131200；实际页表、cache 和工作区也必须足够。
测试 fixture 另有限制 B≤40。其他 S、padding/dummy 和混合 query 数不能视为已支持。

| 本轮 L 区间 | 原 Indexer 数学分支 | query 分组 |
| --- | --- | --- |
| 6～8191 | Vector leaf | 单 query |
| 8192～32771 | 中等历史 Cube | 比较 pair 与 request 的核波次数 |
| 32772～131200 | 长历史 Cube | B<4 为 pair，否则 request |

中等历史的 request 范围并非简单 B≥24，而是 B=24、B33～48、B57～64。
B1～23、B25～32、B49～56 走 pair。公式保留在 select_policy 中，与原 IR 对照。
原 QR/KV 数学在 T=128 切换；输出投影包含 T≤32、32<T<96、T=96、T>96 四种合同。
T=96 来自原 CSA 本身，不是复制 HC_pre 的专用规则。

## 性能选择与证据边界

初始范围策略保持已有七点选择：长历史且 T≤96 用 earliest_ready，较大 T 用
critical_path；中等历史 request 用 earliest_ready、pair 用 critical_path。
长历史 32<T≤96 启用已核对的 QK Vector 索引变换。现有行依赖只在长历史 T=96
的输出所有权合同启用，扩大范围须重新证明。全区间沿用 cached100 通知等待方式。

这些是可测试的初始范围策略，不代表邻近点已经得到相同收益。七点的成本资料只在
B/H/S 和所有 root shape 完全匹配时使用；没有同规格资料的形状仍可生成，成本明确
标为参考模型，不能作为实测性能。长度、位置、slot、页表以及有效循环范围仍由核体
读取实际 device Tensor，CPU 选档不新增每步 item/cpu/tolist。

## 缺少整图参考的处理

已有同构参考的分支保留原生成路径。没有参考时，范围入口从本轮真实绑定、逻辑接口
和后期 IR 的读写效应、显式 deps、dummy、manual scope 构造任务图，再重新分核。
不套用相邻 B 的等待表，不构造整图串行兜底。生成后仍检查逻辑分片覆盖、MIX 配对、
依赖偏序和信号布局；CPU 通过后才进入 lowering 和设备精度验证。

相关代码在 `anchor_gap_design_v1/source_schedule.py`，审计和验证证据保留在同目录。
当前该路径仍属于实验生成器；生产入口接入、各策略代表点/边界的完整真机验证尚未完成。

## 新性能门槛

`performance_acceptance.py` 使用同轮至少三窗完整 L1 中位数检查 `candidate≤baseline×0.70`。
功能通过与性能达标分开记录：`passed` 是本次功能检查，`performance_target.passed`
才是本规格的性能门槛。单规格性能通过也不能代表七档及区间动态要求全部完成。
历史七档差距保存在 `target_30pct_historical_gap.json`，原始 L1 起止重新计算一致，
没有为此重跑测试；最终达标仍须正式复核。

## 当前真机代表点证据

B12/H65536 与 B17/H65536 均完整 exact/图重放/30代信号/保护区PASS。B17确实走源码DAG路径。
L1同轮三窗中位分别为690.70/714.66μs、863.08/893.52μs（单SPMD/动态），仅降低3.35%/3.41%，
均未达30%。见 `range_entry_device_results_v1.json`。这两点不证明整个区间，当前仍需边界、
动态query及后续数学策略验证。

B17另通过H65536→65535→32766→65536同地址图重放；L32772覆盖长历史数学分支的下边界，
每侧只捕获一次且没有新增编译单元。该证据在`range_b17_h65536_metadata_v1/result.json`，
包含页表交换和原址metadata更新；不代表连续decode状态轨迹。
