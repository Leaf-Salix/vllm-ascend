# QK前瞻流水验收证据

对应[本轮报告](../../DSV4_FLASH_CSA_TND_QK_LOOKAHEAD_20261026.md)。控制为`0a5baa322`，nalinaly参考c346573ab。以下只描述lookahead；后续direct-publish候选不复用其精度结论。

- `results/`保留所有单层原始结果，包括否决的物理搬运两候选。性能失败没有改写成精度通过。
- `executed_sources.tar.gz`按实际manifest封存physical、lookahead、独立dep及正式whole候选源、编译收据，不含权重/SDK二进制/build；checksum、SDK/3399源码/77runtime/43数学源及13静态包门禁原样保留。
- `PROTOCOL_MODEL.json`是3query/64mask/64000随机交错模型，不是NPU精度证明；实际checker为`check_protocol.py <lookahead-v1解包目录>`。
- `LOOKAHEAD_STAGE_ANALYSIS.json`与`PHYSICAL_STAGE_ANALYSIS.json`保留逐窗模块及非重叠里程碑。`l1_dependency_raw.tar.gz`包含无dep计时及另一次dep数据，不能合并采集解释成同一次测量。
- `FORMAL_SOURCE_PROOF.json`连接正式文件、设备候选与whole候选；归一化config/layout导入路径并排除唯一host防护后，每个DSL函数AST一致。
- `whole8k_raw.tar.gz`及`whole8k_summary.json`为16rank真实模型原始/审计与按步骤取最大rank统计；`whole128k_raw.tar.gz`及`whole128k_summary.json`为128K同类证据，147456位置token/事件通过。
- `summarize_whole.py <解包目录>`从audit/configuration及16rank数据重算token/轨迹和计时，不重新占卡。摘要含原始输出，不能把相同请求的16rank重复当独立样本。
- `TASKS.json`记录本轮任务与终态；whole8K观察空档保留，不声称全程每5分钟监控。

## 227同环境复现

解包后按实际路径更新并重新封存manifest，先无设备编译，不能直接复用旧收据。固定SDK为PyPTO2dfa7f12/Simpler57e49923/PTOAS0.66/CANN9.2，路径及二进制SHA见编译收据；只用自有目录。

单层在`lookahead-v1`分别执行`bash compile.sh baseline`及`candidate`；硬件统一`task-submit --device auto --device-num 1 --max-time 780 "bash <目录>/run.sh <新label> 32 8192 <32个6>"`。反向捕获加`--reverse-capture`；128K用batch16、history131072、16个6；混长用batch36、history8192、1..6重复6次并加`--qualify-boundary`。每轮12×100 event和独立9个L1窗，先性能后精度。所有输出label必须全新，不能复用旧输出。

正式服务用`lookahead-formal-v1`，先`bash compile.sh tuned`，检查64Tensor/selector1..256/postload元数据，再提交`task-submit --device auto --device-num 16 --max-time 780 "bash <目录>/whole/run.sh 8k"`。8K审计通过后才提交同目录`128k`。固定bank/Native gold、192 token、确定性、FULL、NZ2、TP1/DP=EP16、EPLB关闭及容量条件均由preflight/audit检查，不跑Native2。每个任务内部750秒；只用--status/--log观察，不使用--wait或scheduler。

重算DFX需将原始目录恢复到`collected/lookahead-8k-b32-v1`和`collected/lookahead-dep8k-v1`，调整`analyze_lookahead.py`的报告根后执行；硬门禁先核对源manifest/派生SHA、CPU/JIT映射及全部task/core覆盖，再统计跨度。inclusive跨度不是纯计算/搬运/等待三者的单独硬件计数。

SOURCE中的hardware_precision_validated是编译前状态，不覆写封存源；设备结果以complete、comparisons、B_comparisons和ABA收据为准。部分旧harness说明保留继承字段，实际两个PTO臂为控制及候选；baseline/checkpoint继承信息不覆盖actual_control和逐文件SHA。公开摘要仅按既有惯例把两类轨迹摘要字段改名，原始值/字段保留在raw归档；转换记录见PUBLIC_SUMMARY_DERIVATION.json，没有修改指标或关闭扫描。正式文件另加SPARSE_BLOCKS==2的host静态检查，值2通过、1/3/5拒绝，DSL函数AST与设备候选完全相同。

`FILES.json`记录本目录除索引自身外各文件的SHA256。`check_protocol.py`为实际协议模型的可执行副本，改为参数传入受测源码根；其64mask/64000交错/2688000步输出与原记录相同。
