# 当前已采用动态CSA的全核阶段时钟

目的：最近详细阶段诊断早于前段协作、精确Key L1依赖与物理行MGATHER采用，
不能继续把旧阶段时长当成当前815/939微秒版本的分解。本次对当前289cc072
数学源采集128K/B16/S6与8K/B32/S6，每档仅一次诊断调用，覆盖24 AIC和48 AIV。
不增加优化组合或重跑七档，两个待测Score候选仍使用原句柄独立执行。

复用dynamic_full_pipeline_20261017/run_device.py的phase-clock路径：正式
权重、同一fixture的原多task精确对照、输出/Top-K/score/状态/cache/保护检查
仍执行。时戳写每核每phase独占512B槽的第二条64B行；第一条信号行不变，
其余padding仍校验为0。实际CCEC输入须含时钟hook，不能靠命令行标志判断。

所有阶段时长含插桩与依赖开销，只用于定位，禁止填入性能对比或30%验收表。
TWAIT累计只统计已有阶段GM等待，helper内FFTS不在其中；other包含Scalar、
DMA、数学、FFTS和fence，不能叫纯计算。两个Cube运算也共享一个M流水，
不能把重叠QK/WS标签当成独占硬件服务。该图是全角色阶段时钟，并非完整
逐指令incore流水图。原等待、数学、归约和业务缓存维护保持；写诊断槽另执行既有诊断代码的DCCI/DSB，这也是插桩扰动。

当前尚未提交/完成，不预记精度、阶段瓶颈或收益。source_reference保留完整
动态源码；工具链为既有NEWP、Simpler参数扩容和PTOAS v3，无共享安装修改。

两任务均completed/exit0，实际完整数值/状态/保护检查通过，分别采1619与1685个
阶段发布端点，均覆盖0..71全部角色。每档只做一次插桩调用，不声明完成动态
场景矩阵、连续轨迹或新性能测量。analyze_clock.py按runtime_abi给组phase补齐
明确名称，避免原导出器未枚举的组号误标BEGIN；原始JSON不改。

长B16：Score AIC/AIV的other中位211.62/199.66μs，Attention为156.95/165.86μs。
短B32：Score为75.59/62.71μs，Attention为249.65/251.14μs。
这些是相邻发布点扣除阶段GM等待的区间，仍含FFTS/DMA/Scalar/fence和插桩；
两侧相近可能由FIFO节流形成，不能据此判定Cube/Vector数学恰好平衡。
结论仅为：短B32应优先研究Attention整条供数/计算流水，长B16仍需同时关注Score。
完整stage_summary及重命名后的stages_all72.json在各结果phase_clock目录。

最初CPU hook检查误用了最终CCEC wrapper（static __aicore__签名），hook实际
作用于_compile_pto_module原始AICORE定义。切回真实original.cpp后45个等待、
48个发布插桩均匹配；未修改正式等待或组件实现，实际设备源码门槛也通过。

后续取证的具体缺口：Attention事件基址为2，AIC分别等2(KV)/4(probability)，
AIV分别等3(score)/5(PV)；Indexer使用0/1，不能合并统计。生成原始C++中
wait_flag_dev对应这四类，AIV的score等待因大小档分支有两个源码站点。
下一步可在原调用前后采Scalar时钟/次数，并在Attention阶段发布后统一写
本核独占的额外缓存行；不为计时插入流水线完成屏障，诊断发布需明确DCCI/DSB扰动；不修改原等待语义，不将发射/Scalar
区间冒充独占硬件引擎耗时。当前尚未实现或提交该探针，不预记任何等待分解。
