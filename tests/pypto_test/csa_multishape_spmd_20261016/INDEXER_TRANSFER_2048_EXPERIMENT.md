# Indexer 2048分数交接：精度通过，未采用

2026-10-17，仅128K/B16/S6。原动态PTO为对照，没有运行Native forward。

## 实现范围

`indexer_transfer_2048.py`保留原M384/N64/K128 QK、M16/N64/K384 WS、
16面板展开流水、FP16 FIX/ReLU、Native页地址和叶分片；每两块1024分数合成
一次2048交接。私有双槽GM由[288,1024]扩大为[288,2048]，增加1179648字节。
AIC只算实际存在的1024子块；奇数尾叶不会新增数学量。AIV每轮加载3×2048分数，
段span受本叶剩余有效数限制，排除尾槽未写区。各query排序/合并数学保持。

原1024轮每两轮才排序一次，新轮每轮排序一次，所以root初始化由step<2改为step==0。
READY仍在AIC FIX，ACK仍在AIV MTE2完成分数读取之后；叶尾排空min(steps,2)。
没有删除跨核数据发布/等待。正式根Tensor与Native cache/state布局不变。

## CPU证据

`indexer_transfer2048_prepare_v1`和`indexer_transfer2048_cpu_v1`完成代码生成；
新score load为3×2048，stride2048，AIV阶段Vec最大端点131072字节。
`indexer_transfer2048_codegen_audit.json`只描述实际分配和指令，不代替生命周期/硬件证明。
旧result中的runtime_abi_supported=false是32参数旧检查，本会话使用独立256参数runtime。

`indexer_transfer_contract_v2.json`枚举1..8192叶长、40962种有效排序段，
同时比较排序段和root初始化选择；双槽计数模型只证明计数流程，不证明DMA可见性。

## 失败与修正

首轮`task_20261017_082119_33910122896` exit1。六份cache/state相同，
Top-K有48999个不同位置，输出最大差0.1484375；在精度门槛退出，没有采性能。
原因是本候选保留旧step<2，第二个2048段错误覆盖第一个段的root。
只修三处初始化判断；反向恢复这三处后，Top-K数学AST与原逻辑逐项相同。
这是本目录变换遗漏，不归因为PyPTO、Simpler或ISA。
失败原始entry、result和日志保留，摘要`indexer_transfer2048_failure_v1.json`。

## 修正后设备结果

`task_20261017_082629_34925986670` exit0，`b16_row_deps_indexer_transfer2048_v2`：
输出/Top-K/六份状态、eager/graph、30代信号、保护区全部PASS。
完整无插桩L1三窗836.98、834.68、819.72μs，中位834.68μs；
同轮动态PTO为889.94、899.98、909.36μs，中位899.98μs。
Event中位单SPMD940.45μs、动态954.57μs，口径独立于L1。

没有显示出优于现选B16策略的收益，未采用、不扩其他档。
现选B16首轮807.16μs及同进程复核817.88μs是历史数据，
这里没有现选同进程A/B，不把差值解释为精确回退幅度。
每次调用恢复fixture，不代表连续decode轨迹。
证据：`indexer_transfer2048_result_v2.json`及完整任务目录。
