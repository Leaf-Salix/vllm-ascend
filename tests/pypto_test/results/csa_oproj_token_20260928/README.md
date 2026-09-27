# WO-B按整token量化：评估对后续MoE的影响

基底2a740c1f，原Native cache布局、性能版；[候选](candidate.patch)只修改输出投影算术。
独立工作树`.cache/csa-oproj-token-2a740c1f`；当前没有合入生产。

现有性能版沿pypto-lib：WO-A的FP32结果按8个组分别量化，每组自己的amax/scale；
WO-B每组INT32结果先反量化，再合并FP32。Native则先发布WO-A BF16，拼接8192列，
同一token只使用一个量化scale，WO-B整数结果合并后再反量化。
这不是cache布局问题，也不能断言哪种理论精度更高；但算术差异可能改变下游专家工作量。

候选从已有精度版迁入上述Native量化边界，保留性能版当前的NZ权重读取、M/N分块及按形状选tile：

- 所有WO-A组完成后按token块取全8192列BF16的amax。
- 统一scale量化各组，INT32部分和先合并，再按channel→token顺序反量化并转BF16。
- 增加跨组标度依赖，会牺牲一部分组间流水；能否换来整模型收益必须实测。

先做单卡正式layer4/B16/H8192完整CSA计时，以及B3固定规约A→B→A图/尾块/保护区，
不会因更像Native就直接安排七档或声称通过。
[命令](run_layer.sh)；待设备结果。
