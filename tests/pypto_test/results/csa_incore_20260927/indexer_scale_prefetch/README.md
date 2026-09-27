# 保留：Indexer在等待Cube前读取Native页内scale

基底f76b3ad4，仅性能版`decode_indexer.py`改变AIV核内的操作顺序。
将页内scale的加载及FP16→FP32转换移到`SCORE_READY_EVENT`等待之前；
scale只依赖已有cache_write任务，地址、长度、两个query复用、乘法规约、Top-K与所有任务调度标志不变。
Native布局/分配流程、精度版与工具链不改，无入口复制或新任务。

CPU完整PTOAS+AICPU编译通过，生成代码确认scale的TLOAD先于AIV的wait_flag_dev。
任务`task_20260927_230953_118467913832`退出0。仅测单卡B16的8K/128K代表档，
layer4正式权重＋合成历史、第二CSA metadata复用、mode2/atomic1/确定性0，EPLB关闭。
每档5预热20次无profiler计时，4个独立DFX窗口；没有重测七档或整模型。

| 档位 | 基底完整CSA均值 | 新完整CSA均值 | 变化 | 基底/新P95 | 新最大值 | 同轮Native均值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K/B16 | 797.72 | 776.94 | −2.60% | 818.92 / 784.98 | 787.98 | 934.97 |
| 128K/B16 | 1258.27 | 1242.22 | −1.28% | 1275.16 / 1253.72 | 1257.82 | 1308.92 |

单位μs。Native基底均值925.67/1301.79与本轮有约1.01%/0.55%变化，不能把全差值视为严格同场A/B。
新P95/P50约1.010/1.009，未出现已知单卡长档异常尾部；不外推EP16。

| 核内任务 | 8K基底→新四窗口均值范围 | 128K基底→新四窗口均值范围 |
| --- | --- | --- |
| Score AIC | 38.64～51.78 → 38.13～39.53 | 472.78～475.92 → 463.06～467.46 |
| Score AIV | 43.47～56.28 → 42.74～44.03 | 482.51～485.77 → 472.31～476.54 |
| Top-K merge | 7.85～8.63 → 9.35～9.50 | 15.33～19.02 → 16.48～20.37 |

长档Score AIC/AIV四窗口平均下降1.88%/1.97%，范围不重叠；短档范围重叠，不能把短档12%的窗口均值变化当作稳固同比收益。
Top-K merge小幅变慢，其他任务也有变化；核内计时包含等待，不将它们全部称为算术加速。
根据两档完整路径/P95改善及长档Score核内下降保留。

## 与Native及pypto-lib的差别

Native `quant_lightning_indexer_service_vector.h::ProcessVec1/GetKeyScale`按物理页读取scale，
与Score输入同队列消费；当前同样保持页内FP16 scale及Native key/scale交错布局。
不能声称Native已经用了这里的“跨核等待前预取”。
pypto-lib官方main2164563的DSpark buffered Score是在SCORE_READY等待后，
从独立FP32 scale缓存按页gather，再做head规约与scale乘法。
当前已有Native式FP16 QK＋第二次Cube head规约，AIV可在等待该Cube结果之前读取独立scale；
本轮利用这一独立性隐藏分页延迟，不照搬上游等待顺序，也不改变数学策略。
未对最新版pypto-lib相同输入做新的A/B，不能虚构本轮对上游的加速比。

保护区、metadata、索引结构、有限值检查均通过。Native零容差仍FAIL：
8K/128K输出max_abs 0.03125/0.0390625，Top-K替换366/670，和基底的计数相同。
计数相同不等于逐元素一致；本轮未新增逐元素固定规约或16卡token/DSpark验收，旧七档模型结论仍只属于f76b3ad4。

[报告与每窗口核内数据](report.json)、[实际改动](candidate.patch)、[设备命令](run.sh)。
原始数据在`../../csa_split_optimization_20260927/cache_scale_prefetch/h{8192,131072}_b16/`。
