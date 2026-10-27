# 完整 Indexer 流水候选验证

基准：11b 精度实现；独立候选不覆盖正式源码及 SDK。

## 本轮重点

前段累计 cut 的约293 μs差距只是定位参考，不等于纯 Indexer 成本或可节省时间。
本轮尝试合并整个 IndexQuery 投影、RoPE、Hadamard、量化、系数、Score、TopK
生产消费链，不再做几微秒的单算子优化。IndexKey 完整写回作为外部依赖，主 Q_B
保持独立，避免重现旧 Query 岛延迟 Score 的问题。

首版仅筛选原生 Cube 的单 leaf 历史区间；其余情况显式回原精度实现。
保留原 query 分组判据、全部数学体和排序；先用保守阶段同步确认完整单 MIX 可生成。
这不是已获得收益或已经验收的实现。

## 清单

- [x] 独立源码提取与 AST 数学体核对；相同64参数入口。
- [x] v6双臂 CPU specialize / 完整编译 / load，NPU 未初始化。
- [x] 生成代码确认单 MIX、真实 QR/scale 和最终 Key commit 前驱；独立审查同步、别名及资源。
- [x] 独立审查通过后提交单卡 FULL graph 性能筛选；217秒完成。
- [x] 本轮性能慢2.538%、0/12胜，不做采用确认或Native精度，不扩大配置。
- [x] 未合入，正式基准保持；未跑真实整模型或128K。
- [x] 原始计时、独立dep、CPU编排和结论已归档。

历史排除：physical-gather-v3 已测失败，未证实新 SDK 有实质性 lowering 变化前不重复提交。

## 已完成的静态资格

- v1：coeff Tensor 注解不能引用同函数 constexpr 参数，导入失败。
- v2：冗余 runtime guard 的空 else 导致返回 TaskId 未定义，SSA失败。
- v3：incore 中 GM reshape 被解释为 Tile；改为编排层创建二维 GM view。
- v4/v5：编排动态 shape 中内嵌 dim Call 不受当前后端支持；v6显式标量再建shape，未修改 SDK。
- v6：两臂完整编译、load通过，根64Tensor，NPU未初始化；manifest为
  `9c659403c5d90a672fb7879a0568c6f73a3c4d2a6ba2f6b6229286263b43933c`。
- 候选 artifact：`33c0b4edb48f7a53eefb899196040fc62186bb15466567e50e10f4a871cb9f10`。
- 选中分支只提交1次完整MIX；24Cube/48Vector各7次顶层同步。
- Indexer depth临时量T192约153.5MiB、T256保守约155.5MiB；只对本次默认每depth256MiB负责。
- 该候选不具备128K小ring或真实整模型资格。

提交尝试时SSH关闭，未返回task ID；必须先按唯一命令查询是否创建任务，避免重复提交。
随后唯一查询确认未创建任务，重提一次；已完成性能及独立dep两任务。计时回退，不采用；无Native精度验收。
