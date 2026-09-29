# O-B激活L1复用，恢复基底K128双缓冲

基底为2ed8ae2e，使用两份独立私有整包。第一版长B16 O-B核时回退15.190%，
发现它同时将L0 K128/Right双缓冲变为K256/共用Right，已否定该组合。
本轮仍借鉴最新ops-nn19614968 QuantBatchMatmulV3Tiling::GetIteratorOrder的
AL1-full/N-first条件策略，只将PTO组内K1024的激活跨两个N256块复用。
不声称Native实测K8192也满足整A可驻留条件。

ROW32/96显式加载完整A到Mat；分段B仍K256双槽。A的Mat切片和B直接传给
tile.matmul_acc，由现有AutoTileMatmulL0生成K128及两份32KiB Right。
最终IR已消除中间Mat切片，直接从常驻A提取Left，不产生额外Mat拷贝。
ROW96的Mat为96+64+64=224KiB，Left48KiB、Right64KiB、Acc96KiB；
生成C++的A TLOAD位于两个N块的循环之前。
这是恢复K粒度和Right交替缓冲，不宣称整个自动同步时序与基底完全相同。

手写两次K128的CPU草案因外层stage2展开占用四份Right（128KiB）被拒绝，未上卡；
改用上述已有自动分块后，两个入口依赖解析、完整PTOAS/CCE/link/load通过，
Ruff及shell语法通过。没有修改PyPTO、Simpler、PTOAS或PTO-ISA。
与pypto-lib2164563的差异仍是A生命周期跨N复用；组量化、INT32累加、舍入不变。
ROW128、ND、Native和精度版不改，64份O-B任务及依赖/调度标志不改。

2026-09-29 12:56正常auto提交task_20260929_125617_186453621969。
测128K/B16与8K/B16，使两档均覆盖实际改变的ROW96。每档5预热20次正式设备事件、
独立四窗DFX、八类完整状态零容差；完整CSA/P95与O-B核时分别按长短8:2评估。
无核内收益则不扩测；有收益再补小档/尾行/padding。不新增Native或EP16测试。
主任务已completed(exit=0)，八类完整状态零容差及16个DFX窗口官方join/worker覆盖通过。

| 档位 | CSA均值μs | P95μs | O-B核时μs |
| --- | ---: | ---: | ---: |
| 128K/B16 | 991.231→977.418（−1.394%） | 1003.540→991.160 | 10.826→10.207（−5.720%） |
| 8K/B16 | 780.880→773.819（−0.904%） | 808.020→787.260 | 12.012→12.715（+5.852%） |

长短8:2，CSA改善1.296%、O-B核时改善3.406%，两档候选均无超过各自P50的105%的样本。
长O-B四窗范围10.231–11.233对9.982–10.346μs，存在少量重叠；不写成四窗完全分离。
短档核内回退保留原值；短档未改的O-A/Q_B/Score也变慢，不能把其幅度单独归因A驻留。
长档未改O-A/Score没有同步加速，不能用统一时钟加速来替代O-B优化的解释。
这些是独立DFX观测，不直接相减解释正式CSA均值；仍不是Native或模型token/DSpark验收。

**结论：已按8:2口径保留到性能版单个decode_o_proj.py。**
13:12正常auto提交task_20260929_131204_257112126238：只用H127/B4和B8，
分别覆盖T24→ROW32、T48→ROW96尾行，以及固定图padding；不额外重复长KV或七档计时。
边界任务已completed(exit=0)，两个batch的跨版本八类状态零容差、保护区、compact metadata，
以及active-B=4/3/1/4、8/7/1/8固定图padding全部通过，见[边界证据](boundary/summary.json)。
采用时只将已测decode_o_proj.py移入生产，原文件与私有baseline一致；未改Native、精度版和工具链。
已交付七档泳道仍标明2ed8ae2e，不能将两档收益推算到其余档位。
[两档完整结果](RESULTS.md)、[状态和原样本](summary.json)、[四窗及未改任务上下文](decision.json)、
[最小边界入口](run_boundary.sh)、[边界收集](collect_boundary.py)。

[变换及冻结](prepare.py)、[候选差异](candidate.patch)、[来源](source.json)、
[CPU静态证据](static_evidence.json)、[编译记录](compile_candidate.json)、
[设备入口](run.sh)、[结果收集](collect.py)。
