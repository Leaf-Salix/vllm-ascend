# Sparse Attention 核内等待分解

2026-09-27；源码固定为 `21d99f8a` 的性能版 Sparse Attention。
任务 `task_20260927_155702_180181112209` 退出0。
复用前次 PMU 保存的8K/B40 Native Q/cache/Top-K，不再采集权重或完整CSA。

## 实测及解释

以下为同名核的平均区间，单位μs。24个AIC、48个AIV均有完整记录。
每核处理10个query、每query五个有效候选块，两类wait各50次。

| 核 | 区间 | 均值 | 占该核测量总区间 |
| --- | --- | ---: | ---: |
| AIC | 测量总区间 | 306.75 | 100% |
| AIC | 等待KV-ready | 208.00 | 67.81% |
| AIC | 等待Prob-ready | 15.04 | 4.90% |
| AIC | QK发射及原有排空 | 52.01 | 16.96% |
| AIC | PV发射及原有排空 | 25.86 | 8.43% |
| AIV | 测量总区间 | 301.33 | 100% |
| AIV | gather发射及原有排空 | 124.30 | 41.25% |
| AIV | 等待Score-ready | 78.94 | 26.20% |
| AIV | softmax发射及原有排空 | 24.74 | 8.21% |
| AIV | 等待PV-ready | 63.71 | 21.14% |
| AIV | merge发射 | 6.20 | 2.06% |

这些数据支持先处理KV发布与核内流水衔接。它们不意味着208μs等待全部可消除，
也不能将AIC和AIV时间相加：两侧并行，等待包含对方完成有效工作的时间。
探针只读取同核时间差，没有跨核时间戳相减；余量为循环、条件和未归入上述阶段的操作。

基底的AIV先搬入当前KV，再等上一块Score、运行softmax、发布Prob，最后才发布当前KV。
Native `sparse_attn_sharedkv_scfa_kernel.h` 的 `PreloadPipeline` 则在
`ProcessVec0L` 后发布 `syncV0C1`，然后才执行上一轮 `ProcessVec1L`。
因此下一项候选是在消费上一块Score通知后、softmax之前发布当前KV，
保持事件计数和算术不变，让下一块QK与上一块softmax交叠。

## 测量边界

- 仅给生成的C++加 `get_sys_cnt()` 读取，计数累积到局部变量，函数尾才写入独占缓存行。
  不增加pipeline barrier，所有wait和drain沿用原代码。
- 计时锚点是标量发射、已有wait返回和sync指令发射边界，**不是每段算术或DMA独占耗时**。
  若已有边界未等待异步流水完成，该工作可能延伸到下一段。
- 时间基准取 Simpler 的实际本地配置（目录名 `simpler`）：
  `src/a2a3/platform/include/common/platform_config.h` 中
  `PLATFORM_PROF_SYS_CNT_FREQ=50000000`，即50 ticks/μs。
  这里使用系统计数器，不能用该频率换算前次PMU的核心cycles。
- 测量不含wrapper全部开销、完整CSA上下游或稳态图重放；探针自身也有开销。
  不能将本表306.75μs与完整CSA泳道的319.81–334.24μs比较并宣称加速。
- 复用原FFTS指针入参作为调用方提供的1152个INT64记录区；当前生成代码的
  `set_ffts` 不读写此内存。脚本核对生成ABI和事件拓扑，不修改生产源码或工具链。
- 输出与未插桩的固定输入PTO结果逐bit一致，7,864,320元素、0差异、0非有限值。
  这是探针有效性检查，不是对Native或整模型的精度验收。

## 原始记录与复现

- [探针脚本](run.py)：从Git取固定基底到本目录，编译并只修改本地生成的C++。
- [72×16原始计数与输出比较](report.json)。
- [区间汇总](summary.json)。
- Native输入复用 `../sparse_pmu/native/native_sparse.pt`，不将大张量加入Git。

每行16个INT64；前24行为AIC，后48行为AIV。
列0为总ticks；AIC列1～4为KV等待、Prob等待、QK、PV；AIV列1～5为gather、
Score等待、softmax、PV等待、merge。列8/9为两类wait次数；列10/11为开始/结束时间戳。
其余列保留为0。各列先跨同类核求均值，再除以50得到μs。

先source固定环境，在CPU执行 `python <本目录>/run.py --prepare`，须见
`PHASE_PROBE_COMPILE_OK`；再经 `task-submit --device 0 --max-time 300`
执行同一环境下的 `python <本目录>/run.py`。生成代码在本地 `build/`，不纳入Git。
脚本依赖基底的五块候选、八轮AIV流水和事件顺序，不能不经修改套到后续重排实现。
