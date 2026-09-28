# 长B24 Query/系数跨leaf驻留

基线c93ec723；两侧整包冻结为`pkg:dsv4_csa_query_resident_c93ec723`，共同依赖不变。
生产实现未修改。本轮针对Query/系数重复搬运，不叠加dummy或矩阵scale候选。

## Native与pypto-lib参照

最新本地ops-transformer 28f40354的
`attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_cube_arch22.h:254`
在`isFirstS2InnerLoop`时调用`QueryNd2Nz`和`WeightDmaCopy`，后续S2块复用。
当前PTO每个leaf重新加载48KiB Query、12KiB系数并搬入L0A，leaf内N面板已复用。
本候选借鉴跨S2复用，但采用PTO现有S6/L0A布局，不声称与Native的L1/L0流水完全相同。

最新本地pypto-lib 2164563的`models/deepseek_v4_flash_dspark/decode_indexer.py:403`
按单query×leaf分配，Query加载也在leaf循环内；其分页cache、Vector head规约及buffered路径
与本实现不同，不能把旧725μs泳道直接当作同边界性能目标。

## 具体变化及边界

- 仅24个S6 query组、各组最后一个query的可见候选数相同且大于8192时使用新路径。
- 一个worker固定一个query组，AIC和两个AIV按相同顺序处理该组全部leaf。
- Query/系数的GM加载及L0A搬运移到leaf循环外；pair根槽、score scratch归属及双槽通信不变。
- 请求长度不均匀、其他batch、短档仍走原分工；不增加环境变量，不改Native cache或算术规则。

128K/B24有5个leaf，源码层面每worker可少4组Query/系数加载，即240KiB；
这是理论搬运量，不代表实际带宽收益。生成代码已确认两个TLOAD/TMOV位于leaf循环之前；
[生成证据](codegen.json)、[候选补丁](candidate.patch)。

两根依赖图、完整CPU编译/load通过。初次编排C++生成不支持布尔`and`表达式，
已改成等价的嵌套标量判断，未修改PyPTO/PTOAS/ISA。

## 设备验证

任务[task.txt](task.txt)经auto分配单卡；128K/B24与8K/B24各侧5预热/20次无profiler计时，
独立四个DFX窗口，并检查八类跨版本状态、图A→B→A和保护区。
CANN9.2、mode2/atomic0/det0、layer4正式权重、独立合成历史和每物理行不同scale。
Native手工图列只作环境控制，不等同真实模板编译基线或EP16/token/DSpark验收。
按长短8:2分别报告核内、CSA与P95/max。

## 结果：不合入

task_20260929_031212_413206032193完成exit=0，两档八类跨版本状态精确一致，
A→B→A和metadata/保护区通过；P95/max没有异常远离主体分布。

128K/B24 CSA1308.917→1311.718μs（+0.214%），P95 1328.620→1328.860μs。
Score AIC345.779→358.652μs（约+3.72%）、AIV362.587→362.451μs（基本持平），
merge13.989→15.866μs。减少重复搬运没有转化为长档核时收益。
核时包含等待，不能据此断言驻留使Cube算术指令本身变慢。

8K/B24 CSA970.529→963.590μs（−0.715%）；短档未选择新路径，Score核时下降但包络变长。
长短8:2 CSA+0.028%，Score AIC+2.199%/AIV−1.266%，merge+4.765%。
两档Native手工控制也下降，短档变化不能当作跨leaf驻留的核内收益。

综合没有明确收益，不合入生产，不扩测不均匀长度或整机。
这只否定当前固定组/L0A驻留候选，不代表Native跨S2复用本身无效。
[完整结果](RESULTS.md)、[计时、状态和泳道路径](evidence.json)。
