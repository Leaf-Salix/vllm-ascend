# G8：五类QKV舍入合并后的Graph精度与性能

基线4a83f66d5，runner提交402f6950c。两种长度均完成；原生确定性level1/HCCL=true、全部参数不变、guard与finite检查通过。
输入权重/hidden/history以及原生输出hash与G7同长度control完全一致；实际kernel源码hash逐项匹配生成manifest。

配置：TP1/B4/S6/block128、真实第2层C4权重、seed62合成输入；CANN9.2.0-beta.2、Torch2.10.0+cpu、Torch-NPU2.10.0.post4、vLLM0.29。
PyPTO `54957491ede07ad5d5015f5e69874f367113cf45`，Simpler `32dff953d07f6bd2aacab8532860f28aca6df931`。
同一卡串行原生/baseline/aligned，12轮×100次固定metadata Graph replay，顺序轮换，重置及CPU检查在计时外。
没有诊断Out或hook；精度来自重置后的单次replay，性能不代表真实请求推进100步或整模型吞吐。

## Graph延迟

单位ms/attention forward，数值为12轮中位数。

|长度|原生|正式CSA|QKV对齐候选|候选相对正式|候选相对原生|
|---|---:|---:|---:|---:|---:|
|8K|0.573020|0.672328|0.655887|-2.45%|+14.46%|
|128K|0.786277|0.873531|0.885010|+1.31%|+12.56%|

CSA轮间波动明显：8K正式0.6453～0.6829ms，候选0.6502～0.6821ms；128K正式0.8580～0.8893ms，候选0.8666～0.8994ms。
因此中位数的-2.45%/+1.31%只是本轮观测，不能认定稳定加速或回退。候选仍慢于共同原生基线。
不将与旧轮次不同的卡、JIT和运行状态造成的耗时变化归因于源码。

## 精度

relative L2单位为百分比。

|长度|正式最终输出|候选最终输出|正式raw KV|候选raw KV|候选输出allclose 1e-2|
|---|---:|---:|---:|---:|---|
|8K|1.435012%|1.203559%|0.286057%|0.007072%|false|
|128K|1.691886%|1.427412%|0.289912%|0.009388%|false|

最终输出误差相对下降约16.1%/15.6%，但仍未通过rtol=atol=1e-2。
raw KV不同元素由3344/3475降到4/3，max abs均降至0.00390625。
compressed、main/inner state及index key/scale与正式CSA指标保持一致，符合仅改QKV的范围。
Indexer key误差仍约0.633%/0.635%，scale约0.261%/0.286%；不能宣称整层数值问题解决。
JSON中接近1的cosine可能有约1e-9浮点统计舍入，验收不依赖该指标。

## 结论与后续

这次是仅合并五类QKV边界的候选，不含早期G6混合候选的Indexer/Hadamard等改动；不能与G6约0.38%～0.44%误差混为同一实现。
结果支持继续对齐Indexer投影、Hadamard和量化/score接口dtype；每项仍须同输入单因素验证后再组合。
未修改正式服务算法，未验证整模型、动态padding、DP16或生产metadata刷新。
完整每轮数据、cache误差和工件hash见[结果JSON](DSV4_CSA_GRAPH_ROUNDING_20260924.json)。
