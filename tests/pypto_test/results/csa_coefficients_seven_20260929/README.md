# 系数优化阶段出口：当前保留源码的七档对照

生产算子4ffccb7b，私有整包`pkg:dsv4_csa_coefficients_seven_20260929`。
保留c93ec723之后的系数空worker删除、按组准备与UB一次发布；不含被否定的分数驻留UB、
纯去dummy或真实任务准入组合。S6代表档和双query的跨版本状态检查已完成，本轮收齐完整性能范围。

## 范围与口径

- 128K：B4、B8、B16、B24；8K：B24、B32、B40，不新增8K/B16。
- 同一auto分配单卡，逐档Native/PTO交替先后顺序；各5预热/20次无profiler真实编译图计时。
- CANN9.2.0-beta.2、两侧mode2，PTO atomic0、det0，EPLB关闭；layer4正式权重与独立合成历史。
- HC_pre+norm+CSA+HC_post完整设备区间，按同上下文batch等权，再按128K/8K的8:2加权。
- 每档保留两侧原始PyTorch JSON及四个独立PTO level-4窗口；下载包固定window_3，不挑最快窗口。
- 单列P95/max、尾部比例、Score/Sparse及系数核时与任务分派；不把重叠核时相加当CSA。

Native沿用已对齐decode模板的npugraph_ex/static kernel及融合配置，
CPU binding、共享专家多流和recompute配置均记录，但单层没有MoE/EP16，不能据此验收模型行为。
运行Native仍为release custom二进制；最新版ops-transformer仅作为源码优化参考，未声称重编该仓。
PyPTO/Simpler版本及冻结来源见[source.json](source.json)，提交前两根依赖图已解析通过。
完整生产源码、公共依赖和全部设备runner都已复制，排队后不编辑该副本。

## 当前状态与收集

task_20260929_070256_21859435951已auto单卡提交，**正在执行；最新完整结果仍为c93ec723旧七档**。
本轮不重复已有跨版本完整状态矩阵，复用runner内的图/eager、Top-K结构与保护区检查。
Native det0的编译/eager差异继续如实记录，不冒称两侧逐bit或整模型token/DSpark通过。
正式性能、独立DFX/PMU与模型forward分别报告，不将不同采样相减作因果归因。

完成后依次执行：

1. `collect.py`：检查同卡/同配置/真实PTO包和完整七档，输出CSA/P95、Native PMU及PTO核内分项。
2. `analyze_schedule.py --history H --batch B`：复用官方时钟域、行数/block校验和固定window_3关键路径。
3. `bundle.py`：汇聚21份未改写的原始JSON，保留原路径映射。

收集和分析复用上一轮已验证工具，只增加严格私有包检查和系数任务分项，不再复制整套分析代码。
两侧精度版、新CANN9.2模型token/DSpark和稳态decode forward仍为后续任务。

[解析证据](parse.json)、[设备入口](run.sh)、[单档入口](run_side.sh)、[设备任务](task.txt)、
[结果收集](collect.py)、[调度分析](analyze_schedule.py)、[JSON汇聚](bundle.py)。
