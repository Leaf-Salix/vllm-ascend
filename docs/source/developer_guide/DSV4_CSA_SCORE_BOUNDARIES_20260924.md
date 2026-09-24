# G10：TopK 打分与 inverse RoPE 舍入边界

基于 G9 实验候选，正式文档基线 e18b458cf；本轮算法仅存在于独立实验变体，未合入正式服务。
环境、模型、确定性和权重同 G9：CANN 9.2.0-beta.2、Torch 2.10.0、Torch-NPU 2.10.0.post4、vLLM 0.29。
真实 DeepSeek-V4-Flash-0731-w8a8 第 2 层 C4 权重，TP1/B4/S6/block128，seed62 合成 hidden/history。

## 三个边界

- coeff：query scale 与 head weight 的乘积先舍入 FP16，再参与打分。
- qk：ReLU 后 QK 分数除以 1024、舍入 FP16，再恢复尺度，匹配原生 DEQF16 边界。
- inverse：attention 输出先舍入 BF16，再进入 inverse RoPE。

未改变接口、metadata、cache 所有权。score_both 合并前两项；all 合并三项。
原生 score 的 Cube 归约与 CSA 的向量归约仍可能存在次序差异。

## 8K 模块诊断

固定 native QR 注入 Indexer，以下是混合诊断，不能称完整 CSA 精度。
百分比为 relative L2；集合重合率不能代替排序一致性。

|候选|混合最终输出误差|同输入 TopK 集合重合率|TopK 位置差|
|---|---:|---:|---:|
|coeff|0.666169%|99.975586%|1510|
|qk|0.673006%|99.975586%|1671|
|score_both|0.606598%|100%|2|
|inverse|0.533141%|99.967448%|2141|
|all|0.380709%|100%|2|

inverse 将同输入原生 attention 与 CSA heads 的差异由 0.089956% 降至 0.029267%；all 为 0.029867%。
权重不变、guard、finite、同输入和原生重复输出检查通过。

## 无插桩、无 native QR 注入的完整单层 Graph

baseline 为 G9 QKV+Indexer 候选；aligned 再合并上述三项。12 轮×100 次 replay，轮换顺序，reset 与检查不计时。
单位为 ms/attention forward，误差是对原生输出的 relative L2。

|长度|原生延迟|baseline 延迟|aligned 延迟|baseline 误差|aligned 误差|
|---|---:|---:|---:|---:|---:|
|8K|0.585886|0.564881|0.566792|0.736326%|0.444658%|
|128K|0.795900|0.847596|0.883111|0.775025%|0.378008%|

对同轮 baseline，候选延迟约增加 0.34%/4.19%。8K 差异较小；128K 各轮均变慢。
相对原生，本轮候选约快 3.26%/慢 10.96%。跨 G9 测试基线延迟存在漂移，不能把跨轮变化归为算法收益。

源码 hash 与本地候选一致，全部权重不变、guard 和 finite 检查通过。原生开启 level1/HCCL 确定性。
两种长度的输入及原生输出 hash 均与 G9 一致；本轮 baseline 输出 hash 等于 G9 aligned。
两种长度 allclose(rtol=atol=1e-2) 均通过，但分别仍有 57511/56334 个输出元素不同，不能称精度问题已解决。
cache 的变化与 G9 相同：index key/scale 一致，raw KV 仍有少量差异。

## 后续

精度优先：继续单因素对齐 Softmax 的累计最大值和舍入顺序，并引入 BF16 ULP 统计。
本轮未证明整模型精度、动态 padding、真实请求推进或 DP16 吞吐。
完整指标、源码及运行脚本 hash 见 [JSON](DSV4_CSA_SCORE_BOUNDARIES_20260924.json)。
