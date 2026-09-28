# 当前源码：真实编译新七档

状态：task_20260929_032251_2642121879已完成exit=0，auto单卡串行运行38分2秒。
128K B4/B8/B16/B24、8K B24/B32/B40全部完成，不再新增8K B16。
算子及公共依赖冻结为c93ec723，版本及路径见[source.json](source.json)。
Query驻留、矩阵scale及去dummy候选均未混入。

[完整性能与核内表](RESULTS.md)、[原始计时及状态证据](evidence.json)、
[21份原始PyTorch/PTO JSON集中下载](download/README.md)。

## 计量范围

同一auto单卡，每档交替Native/PTO先后次序；各侧独立OPP静态包和AOT缓存。
Native/PTO真实编译HC_pre+norm+CSA+HC_post；Native实际安装静态包，PTO实际custom-op调用已确认。
两侧CANN9.2、mode2、det0，PTO atomic0；EPLB关闭，ring=[256,128,256,32]MiB/task_window4096。
正式layer4权重、独立合成历史/输入、每物理行不同scale及反向物理页表。
Worker绑核和MoE共享专家多流不在单卡半层范围，不将配置请求值冒称Worker验收。

每侧5次预热、20次无profiler图事件，独立PyTorch profile，PTO另采四个DFX窗口。
原始慢样本全部保留；不将独立profile核时与正式事件相减归因调度。
长档内部batch等权、短档内部batch等权，再按8:2计算变化率。
PTO图/eager八类状态、Top-K结构及metadata/保护区通过；不等于两侧精度或EP16/token/DSpark验收。

## 当前结论

长档平均变化−11.880%、短档−4.039%，长短8:2为−10.312%。
128K/B16为Native1232.308/PTO1043.478μs，PTO P95为1061.640μs。
本轮七档PTO P95/P50为1.0127–1.0341，没有超过各档P50的105%的样本。
8K/B40的PTO max1339.720μs仍高于Native1330.740μs；不关闭历史间歇1.4ms拖尾或EP16问题。

## 依赖时序

七档28个level-4窗口由Simpler官方解析器检查时钟域合并、原始/合并行数和每任务block数。
固定window_3展示Observed路径，不选择最快窗口；Static CPM交叉检查。
dummy缺少物理时戳时将完整ready归因置空，logical compute含SPMD跨度与内部等待。

[长B4](h131072_b4/schedule/README.md)、[长B8](h131072_b8/schedule/README.md)、
[长B16](h131072_b16/schedule/README.md)、[长B24](h131072_b24/schedule/README.md)、
[短B24](h8192_b24/schedule/README.md)、[短B32](h8192_b32/schedule/README.md)、
[短B40](h8192_b40/schedule/README.md)。

长B16的window_0/1为cache scale写回最后完成，window_2/3为系数任务；
不能将Score前所有空档归因于dummy或空worker。分别记录FIN、派发和首次开始。
历史727.98μs上游图仅作结构参照，形状/版本不齐，不作为同输入速度对比。

## 结果字段修正

首次collect.py因PTO报告中的variant=performance而停止。核查确认原入口将
`selected_variant()`的类别覆盖了`pkg:`选择器；它返回performance不代表回退了包。
七档编译日志均明确引用冻结私有包的decode_csa.py/decode_indexer.py，源码根也一致。
收集器要求这两处实际编译来源并写入variant_evidence，不修改原报告或重跑成功测试。
后续入口分别记录variant选择器、variant_kind、implementation_package/source，避免混淆。

本轮入口、收集器和下载脚本：[run.sh](run.sh)、[compiled_case.py](compiled_case.py)、
[collect.py](collect.py)、[analyze_schedule.py](analyze_schedule.py)、[bundle.py](bundle.py)。
