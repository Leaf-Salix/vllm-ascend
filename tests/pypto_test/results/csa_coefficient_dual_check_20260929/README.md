# 系数优化：补齐双query路径的状态检查

此前取消空worker、按组加载/FP16计算、UB对角块一次发布已在128K/B16和8K/B24通过零容差对照，
两档都走S6。8K/B32走双query，只有编译证据；本次仅补这一个仍缺的运行形状。

两侧冻结同一套公共代码和runner，独立私有包`pkg:dsv4_csa_coefficient_dual_4ffccb7b`：
基线取已完成七档的c93ec723包，候选取当前保留的4ffccb7b算子。
不含未采用的score UB或正在评估的dummy/准入组合。
两套入口的依赖图已通过CPU解析；两版核模板已有完整编译记录，不重复单独编译。

通过task-submit自动分配一张卡，CANN9.2/mode2/atomic0/det0、layer4正式权重、
独立合成历史、EPLB关闭、ring=[256,128,256,32]MiB/task_window4096。
仅各侧一次现有真实编译runner，保存八类完整状态，要求零容差相同；
复用图/eager、Top-K结构和保护区检查。runner附带5预热/20次计时及单次PyTorch profile保留，
不新增四窗DFX，不把该点拼接进旧七档，也不据此宣称精度版或模型token/DSpark通过。

task_20260929_065501_21426545346已完成exit=0，八类完整状态跨版本零容差通过，
图/eager、Top-K结构及保护区检查通过。CSA均值1128.473→1111.137μs（−1.536%），
P951156.160→1135.600μs，max1170.340→1139.060μs；这些是本轮独立PTO组合对照，
没有测Native，也不和旧七档拼接。双query的已知状态验证缺口已补齐。
下一步以同一套保留源码完成阶段出口七档，更新Native/PTO与核内对比。
[检查结果](RESULTS.md)、[精简状态与逐次计时](summary.json)。

[源码范围](source.json)、[基线解析](parse_baseline.json)、[候选解析](parse_candidate.json)、
[设备入口](run.sh)、[状态收集](collect.py)。
