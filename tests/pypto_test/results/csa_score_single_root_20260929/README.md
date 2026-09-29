# Indexer完整query单根候选

基底d93bba14，包括已保留的Sparse末块发布和HC_post残差重用，不包含失败的HC_pre调度依赖。
baseline/candidate由已验证HC调度实验的baseline整包复制，独立私有模块冻结后入队。
Native最新七档已有，不重测；沿用PTO当前图包装和同卡两档A/B。

## 源码依据与范围

最新ops-transformer 28f40354的QLI V2 `ProcessVec1`按query/S1分给两个AIV，
每个query处理完整S2；分数在UB内乘scale、SortAll并累积globalTopkUb，最后每query/分片发布一根。
参考文件：`attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_vector_arch22.h`。
当前Native性能二进制仍是release custom，不把参考源码当作该二进制生成码。

当前PTO长S6每个AIV处理六query的一半候选；Cube传输一轮前后各512候选，
两段逻辑位置并不相邻。每query/leaf写两个Top-512根；后续独立归并读取两倍根数。
旧query-split实验只改AIV分工，保留非连续半区及两根，已失败，本次不原样复测。

本候选仅在现有`balance_leaves=True`的长S6路径生效：

1. Cube按连续1024候选组织传输，保留M384/N64、FP16 FIXPIPE缩放、Key预取和双槽握手。
2. 两个AIV各处理三个完整query，scale每核只加载一次供三个query复用。
   因每核读取全候选，scale流量仍比原半区分工增加一倍，这是明确代价。
3. 连续两次1024分数组成UB内2048分段，直接排序，无缩放分数GM写回/重读。
   用全宽行assemble再reshape规避A3部分列UB搬运限制；根12KiB、分段24KiB，实际峰值以编译分配为准。
4. 每query/leaf仅发布一个Top-512根，consumer同步将有效根数减半；保留原arena容量及query跨度，防止混改ABI。
   例如128K/B16的六个leaf从12根减为6根。任务数、worker数和调度标志保持。
5. 短档与长档B<4的双query路径沿原实现，策略由算子输入分支选择。

这是一项完整组织策略的对照，不能将结果单独归因于某一次GM读写或某个维度分工。

## 数值与筛选

每分数的QK、FP16缩放、head规约和KV scale乘法保持。
排序规则明确为：每连续2048段内部沿原sort32/mrgsort；新段在同分时优先旧段，后leaf优先前leaf。
它可能改变原来按half边界分段的同分选择/顺序；允许按既定精度策略评估，但绝不将结构合法等同于精度通过。
先报告八类完整状态零容差结果；若有差异，须归因到分数/同分规则并按误差、token/DSpark合同处理。
性能无收益则停止扩测，有核内收益仍需区分功能问题与精度取舍后决定采用。

先完成两入口依赖图与完整CPU PTOAS/CCE/load，只提交一次长128K/B16、短8K/B24。
每侧5预热20次正式事件、独立四窗DFX、完整状态及保护区；Score AIC/AIV、merge、CSA/P95分别报告。
长短8:2。无新增Native、全七档或EP16测试，生产算子暂不修改。

[构建候选](prepare.py)、[AIV实现片段](long_query_aiv.py.inc)、[唯一补丁](candidate.patch)、
[来源](source.json)、[运行](run.sh)、[收集](collect.py)。

## 当前状态

两入口依赖图、候选完整CPU PTOAS/CCE/link/load通过，baseline依赖图通过；Ruff/shell通过。
编译期在算子侧解决两处表达限制：当前JIT依赖发现无法保留Tile辅助入参，故排序体直接内联；
前端也会校验短档的未选分支，故UB分段物理宽度明确固定2048，不随短档768传输块推成1536。
未修改PyPTO、Simpler、PTOAS或PTO-ISA。

长S6生成IR的Vec静态地址覆盖90112字节（88KiB），其中常驻根12KiB、分段24KiB；
生成AIV函数只有三个TSTORE站点，分别发布三个query的最终根，无scaled-score GM暂存。
这是静态分配/生成码证据，不是已测性能收益，[记录](codegen_summary.json)。

2026-09-29 11:30正常auto提交`task_20260929_113003_337101716086`，已确认running，最长5400秒。
私有源码已冻结，不向排队任务叠加改动。两档计时与核内/状态结果待产生，不提前采用。
