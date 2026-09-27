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

## 后续受控模型对照

已准备[两档模型命令](run_model.sh)及[收集入口](collect_model.py)，尚未提交16卡任务。
只有单卡结果支持继续时才执行。使用独立工作树`.cache/csa-oproj-token-ordered-2a740c1f`，
组合既有`ordered/candidate.patch`与本目录的输出投影补丁。
PTO两次候选之间固定分离/新页排序、atomic1、mode2、det0/HCCL=false及EPLB关闭。
Native复用`csa_source_split_ab_20260927/ordered/model/h*/b*/native`；不把复用结果写成新跑的控制。

统一入场和10步CPU position相同能排除既有的请求入场错位，不能单独证明完整设备输入相同：
异步spec decode在设备上更新采样及草稿token，`input_ids.cpu`不一定是实际输入的镜像。
因此不能为补齐记录而把该CPU缓冲当作设备token，也不往正式计时流里添加复制或同步。
正式forward仍衡量实际请求的部署表现；严格的路由因果分析需使用已有独立路由诊断的实际设备输入。
输出token和DSpark统计仍分别验收，长档旧候选DSpark未通过的事实不变。
