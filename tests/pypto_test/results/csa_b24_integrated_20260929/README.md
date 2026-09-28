# 长 B24：近期 Indexer 组合及真实编译 Native 基线

共同依赖冻结为c93ec723，仅性能版decode_indexer不同：基线3b27c7fd、候选c93ec723。
候选包含2048分段排序、UB中间根和当前query/leaf选择；B24仍选既有S6。
整包分别保存为私有pkg，生产/测试根解析及完整CPU编译/load已通过。

同一个auto单卡任务依次运行旧PTO、Native、当前PTO的实际编译半层，
再独立采两侧各四个PTO DFX窗口。两侧PTO均走生产custom-op，禁止静默回退；
Native真实static kernel编译、安装和设备profile必须成功。
每个计时进程独立空static_kernel的私有OPP，CANN9.2/mode2/det0，PTO atomic0。
layer4正式权重/合成历史，每物理行不同scale；PTO第二层复用已有compact metadata。
ring=[256,128,256,32]MiB/task_window4096，B24/S6/H131072；半层容量40，实际24请求/144token，
这不是整模型B24容量24、util0.97的验收。

5预热/20次无profiler图事件计时，独立PyTorch profile；跨版本八类状态精确比较，
当前PTO另做A→B→A和metadata/保护区检查。保留P95/max和全部异常样本。
本次补近期组合的长B24范围，不拼接旧9.0或旧手工Native图数据，不重复16卡容量测试。

[编译入口](compile.py)、[实际编译入口](compiled_case.py)、[设备入口](run.sh)、[任务](task.txt)。

首轮task_20260929_023737_351388018732在候选PTO阶段exit=1：实际已加载AOT编译函数，
但vLLM该路径不设置wrapper.compiled，原测试脚本把缓存命中误判为未编译。
已同时识别fresh compile和was_aot_compile_fn_loaded_from_disk，并为每侧隔离VLLM_CACHE_ROOT。
重新申请auto单卡后，三侧在同一卡重取必要对照；首轮已完成读数保留原始目录，不混入final/结果。
这次修正只在测试入口，没有改生产运行路径、算子或共享AOT缓存。

第二轮三侧计时均完成，但随后因swimlane与graph参数互斥而exit=2；
这不是算子执行失败。保留完整计时，用独立task_20260929_025117_388714826078补泳道与A→B→A，exit=0。
collector分别检查阶段任务状态和所有必要报告，不把exit=2的整项任务说成成功。

最终八类跨版本状态精确一致、图及31项保护区通过。近期组合的CSA1376.883→1312.313μs（−4.690%），
同配置Native1396.274μs，当前PTO低6.013%；P95 1337.940μs，低于旧PTO1393.240和Native1403.260。
长B24 Score AIC383.660→349.409、AIV411.575→366.048μs，独立merge16.852→13.960μs。
Native QLI独立profile Duration382.500、PMU AIC/AIV369.919/369.524μs；
PTO Score已接近此参考，但范围不同且PTO有独立merge，不能据均值直接称整个Indexer已领先。
Native det0自身编译/eager输出及Top-K差异单列；未替代整模型token/DSpark验收。
[完整结果及文件路径](RESULTS.md)、[原始读数与状态摘录](evidence.json)。
