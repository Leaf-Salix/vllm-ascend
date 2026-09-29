# Sparse末块融合后的七档阶段出口

生产优化55b89ee2已通过长B16/短B24完整状态、性能/P95以及B3/H127/padding边界。
此目录统一重取七档，不把旧Native表与本轮PTO数据拼接。

任务：`task_20260929_093737_25396355997`，整个矩阵正常auto分配同一张卡。
档位：128K×B4/B8/B16/B24＋8K×B16/B24/B32；B40退役。
CANN9.2 / mode2 / det0，PTO atomic0，EPLB关闭；第二个CSA（layer4），固定真实权重与合成独立历史。
Native cache布局保持；长短8:2，各上下文内batch等权。

按用户最新口径分两套Native证据：

- **主性能**：显式torch.compile backend=npugraph_ex，static compile和SuperKernel均开启。
  后端持有唯一图，确认无主机更新节点后直接重放；5次预热20次设备事件，独立PyTorch profile。
- **核内细节**：同入口、同权重/shape/static compile，SuperKernel关闭；只作独立kernel profile。
  仅保留一次重放用于状态/保护区检查，不将该样本当作正式CSA性能，也不重做开关收益消融。
- **PTO**：冻结的同一私有包，生产自定义算子图；5预热20次，独立PyTorch profile及四个DFX窗口。
  不编辑正在编译的算子、适配器或runner。

Native使用从已校准副本提取的实际计时函数`native_measure.py`，不会误用PTO入口的手工外图capture。
依赖图解析、生成Python语法、Ruff和shell语法检查通过。收集器用已有两档Native实际profile核对了融合范围解析。
收集时按实际SuperKernel端点记录完整融合范围；另表比较static-only QLI/Sparse核时，保留测量范围差异说明。
PTO四窗使用官方原始join/行数/block检查，短B16/B32按双query路径核对48个系数worker。

最终保留七档两侧主性能PyTorch JSON、PTO四窗泳道、七个Native核内诊断JSON，
均值/P95/max及所有原样本。20次正常样本不替代EP16稳定性，也不代替逐token/DSpark验收。
当前任务进行中，尚未产生完整七档对比结论。

[冻结准备](prepare.py)、[来源](source.json)、[依赖图](parse.json)、[完整矩阵](run.sh)、
[分侧入口](run_side.sh)、[收集器](collect.py)、[任务](task.txt)。
