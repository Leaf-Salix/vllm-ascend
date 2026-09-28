# Indexer query排序共用循环：CANN 9.2独立候选

当前状态：编译和两档状态检查通过，**未取得核内收益，不合入生产，不扩测**。
task_20260928_223556_211263013221由auto分配card0，完成退出0。

## 依据与范围

最新ops-transformer 28f40354的A3 QLI V2
`attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_vector_arch22.h`
在`ProcessVec1`中以`innerS1Idx`循环处理query行，复用同一段SortAll/归并代码。
当前PTO性能版`indexer_score_topk_native_cube`在half-leaf发布阶段使用`pl.unroll(query_group_size)`，
把512/1024/2048/2560/3072/4096六种排序分支各复制2、3或6份。

本候选只将这一个末尾query循环换成`pl.range`，顺序保持；前面的scale处理仍展开。
没有改Score算术、量化、排序算法/tie规则、leaf分配、cache、任务数或调度标志；精度版不变。
这是减少重复指令的候选，尚无I-cache stall计数证明其为瓶颈，不能把代码量降幅当成性能收益。
不是历史§236的Score UB驻留方案，没有重新尝试GM分数搬运或UB gather。

pypto-lib仍主要作为PTO语法/连续cache实现参考；本项按Native QLI的query循环组织表达，
继续保留Native分页cache零复制视图和2/3/6query分组，不因Native按行循环而改变Cube分组。

## 冻结与编译

基线`.cache/csa-cann92-baseline-e33d842a`；候选`.cache/csa-sort-query-loop-d0addbb4`。
d0addbb4与e33d842a的生产算子和本轮单层入口相同，期间只有验证文档/结果变化。
候选从Git提取生产源码和非results测试Python，再施加[唯一补丁](candidate.patch)。
使用公共CANN9.2.0-beta.2、PyPTO88f60598、Simplera54c05095、PTOAS0.66、PTO-ISA327cd586；
没有更新工具链源码。复用上一轮已通过的基线编译，候选全链lowering/PTOAS/CCE/链接/load通过。
初次准备脚本列错了不存在的csa_assertions.py，在创建快照/启动编译前报错；修正为Git实际文件列表后完成。

生成目标`.text`实际字节数如下，不是包含调试信息的.o文件大小。

| 策略 | AIV基线字节 | 候选字节 | 生成TSORT32调用点 |
| --- | ---: | ---: | ---: |
| 长S6 | 41200 | 9596 | 36→6 |
| 长三query | 21400 | 8820 | 18→6 |
| 长双query | 14840 | 8760 | 12→6 |
| 短S6 | 42524 | 9960 | 36→6 |
| 短双query | 15336 | 9112 | 12→6 |

核对生成C++保留query循环，未被重新全部展开；每次执行的排序工作量没有随静态调用点数减少。
[codegen.json](codegen.json)、[编译日志](compile.log)、[CPU入口](compile.sh)。

## 设备验证

先测128K/B16和8K/B16，第二个CSA层layer4正式权重、合成历史、物理页反序及逐行变化scale。
mode2/atomic0/det0、出5验6、seed1024、EPLB关闭；每侧预热5、20次无profiler计时，另四个DFX窗口。
八类PTO状态零容差（不含idx_topk_scores）、保护区/metadata和A→B→A图重放。
分别观察Score核时、最慢核、包络及完整CSA/P95，不把独立DFX与正式计时样本一一对应。
按128K/8K七三权重决策；若明确获益，再补受影响的三query及短S6范围和必要七档收口，暂不追加EP16。

[设备入口](run_layer.sh)、[主机汇总](summarize.sh)。

## 实测与取舍

八类PTO状态零容差、metadata/保护区、A→B→A全部通过，不含Native逐bit或整网验收。
每格为基线→候选，单位μs；独立四DFX窗口不与无profiler样本逐个对应。

| 档位 | CSA均值 | 变化 | P95 | 最大值 | Native控制均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B16 | 1084.333→1073.057 | −1.040% | 1095.06→1081.68 | 1099.98→1082.92 | 1269.801→1298.425 |
| 8K/B16 | 801.090→802.335 | +0.155% | 816.64→824.18 | 821.66→830.38 | 914.172→898.517 |

CSA七三−0.681%，短档P95增加7.54μs；Native控制漂移原样保留，不做归一化或删样本。
长档Score AIC242.498→244.600（+0.867%），AIV263.485→265.182（+0.644%），
最慢AIV269.295→270.800，包络272.550→274.970。长档排序正文缩小，没有转化为可见核内收益。
短档AIC38.132→44.718（+17.271%），AIV42.640→49.182（+15.343%），启动分散虽下降，包络仍变长。
核内AIC/AIV七三分别+5.788%/+5.054%。merge及Sparse没有修改，其波动不记作此候选的确定因果收益。
16个DFX窗口Score/Sparse均每核一份，本轮未捕获同核重复派发，不宣称旧EP16尾部已解决。

不以代码量或长档整层均值代替核内目标，故撤回候选方向，生产继续原展开方式；无新依据不重复测试。
[全部最慢核/包络](INCORE.md)、[原始窗口核时](incore.json)、[七三汇总](weighted.json)、
[长档状态与样本](h131072_b16/evidence.json)、[短档状态与样本](h8192_b16/evidence.json)。
