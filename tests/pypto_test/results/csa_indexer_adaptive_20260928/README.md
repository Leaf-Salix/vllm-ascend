# Indexer按工作量选择query分组

基于9a01a276，合并两个已取得明确核内收益的输入策略，隔离工作树`.cache/csa-indexer-adaptive-9a01a276`。
Native cache和生产流程不改，精度版算术不改；性能版保持一套源码内部选择。

- 压缩历史大于8192：query数至少96用S6/M384/N64；48～95用三query/M192/N128；更小用双query。
- 单leaf短档：整请求分组覆盖24worker且最忙核计算增量不超过25%时用S6，否则双query。
  当前矩阵B24/B40命中S6，B16/B32保留双query。
- FP16边界、FIXPIPE、Top-K规则、atomic选项及原有长短调度标志均保持原样。

该规则来自当前筛查，不宣称全输入最优。[合并补丁](candidate.patch)。
128K/B8的[三query实测](../csa_indexer_mid_20260928/README.md)：Score AIC下降20.62%，完整CSA下降3.38%。
8K/B40的[短档S6实测](../csa_indexer_six_short_20260928/README.md)：Score AIC下降51.32%，完整CSA下降1.84%；
Native控制存在波动，已保留原始样本及明确边界，不借异常控制宣称更大加速。

完整CSA CPU lowering、PTOAS、CCE及链接通过，Ruff/shell语法通过；
[编译日志](compile_full.log)，编译入口复用S6目录`compile.py --full`。

## 必要状态对照

task_20260928_040830_332851521278，[命令](run_accuracy.sh)。
只补两条新分支：B8/H32768触发长档三query并覆盖完整leaf+尾leaf；B24/H8192触发短档S6。
对照9a01a276，正式layer4权重、atomic0/det1、随物理行变化的FP16 scale，候选另做A→B→A图重放。
逐元素检查8类输出/状态及Native控制，不用摘要或hash代替。
task退出0，两档各8类PTO输出/状态相对9a01a276逐元素零差异，Native控制也精确一致；
metadata/保护区、固定输入自重放、候选A→B→A图重放均通过。
[长档结果](accuracy/long/comparison.json)、[短档结果](accuracy/short/comparison.json)。
此检查不等于全形状Native浮点对齐。按已验证核内收益，合并策略已应用性能版。

## 真实EP16七档

[模型命令](run_model.sh)、[收集器](collect_model.py)。上述单卡通过后提交16卡，
task_20260928_041310_3365315916，冻结本候选源码；没有叠加其他算术候选。
128K/B4/B8/B16、8K/B16/B24/B32/B40使用同一冻结算子，每个历史只初始化一次模型再扫描batch，
重新采集Native控制。两侧mode2、TP1、DP=EP16、DSpark出5验6、EPLB关、atomic1/det0/HCCL=false。
预热后10步无profiler decode forward为主，另留3步PyTorch profile；报告P95/max、每步慢卡，
逐token和DSpark统计门禁保持不变，CPU位置只作为入场对齐证据。
不将候选单卡分项百分比外推为整模型性能优势。

task已退出0，七档采集齐，但验收未通过。[正式结果和全部样本](model/RESULTS.md)。
128K/B4/B8/B16依次快0.36%、慢4.77%、快1.72%；8K/B24/B32/B40快2.37%/0.86%/0.35%。
所有573440输出token零差异；8K/B16的rank8和rank12各少接受1个第5位置草稿，
总接受数76800→76798，正式第17个满档step的末请求S6位置少1。
该档不提供通过可比性检查的forward百分比，不能把收集齐七档标为验收通过。
[失败记录、完整DSpark计数和位置差异](model/h8192/b16/acceptance_mismatch.json)。

B8独立三步profile仍快：CSA合计21.137→18.565ms，但正式forward55.390→58.035ms。
完整主流图边界核对只在HC首尾之外增加约307/268μs，不能解释两个独立轮次的胜负反转；
图本体、事件计时之外的发射间隙和不同轮请求/cache状态尚未分离。
已准备[同轮事件/trace诊断](../csa_forward_boundary_20260928/README.md)，只补B8，生产算子不改。
[CPU离线导出及七档分解命令](export_profiles.sh)；profile与正式计时严格分开。

七档已有 rank0 profile 均完成 CPU 导出；[CSA/FFN/GMM 差距分解](model/MODEL_GAP.md)。
