# QR归一化：输入和gamma留在UB

来源为ops-nn 7a71d54e的norm/rms_norm_dynamic_quant/op_kernel/rms_norm_dynamic_quant_normal_kernel.h。
`CopyInWeights/ComputeRmsNorm/ComputeDynamicQuant`让输入和权重跨RMS、量化阶段留在UB。
候选从a66255ea独立工作树派生，基线d1f170ff，两者生产算子相同；只改性能版QR归一化。
未叠加PV L0B候选，也未包含主工作树另行进行的wo_a布局调整。

原PTO按256列两遍读取8×1024 FP32输入及BF16 gamma；候选每worker一次读gamma并转FP32，
每8行一次读完整输入，两遍计算均切UB。输入32KiB和FP32 gamma 4KiB是基本驻留容量，
总UB还包括流水与临时量，需以生成代码及编译器分配结果为准。
保留256列平方和/amax的累计次序、RMS再乘gamma、现有高精度rsqrt、量化RINT→I32→FP16→I8 TRUNC。
这不是复制AscendC的全部数值策略；精度版原BF16边界及Native规约没有改。
与pypto-lib的当前两遍分块读取相比，本项只借鉴AscendC的UB复用；8行粒度、worker数、调度和Q投影分组不变。

[候选补丁](candidate.patch)、[CPU编译入口](compile.py)、[单卡入口](run_layer.sh)。
完整CPU lowering/PTOAS/CCE/AICPU链接通过。
首版tile.slice会将同一UB子块在三个使用点重复TEXTRACT；设备任务仍pending时取消，未执行。
最终候选改用显式tile.extract形成可复用的实体块，再次编译通过，核对同块一次提取后复用。
仅这些CPU证据不宣称性能收益；设备任务task_20260928_143910_419318230381已完成、退出0。
单卡8K/B16和128K/B16，mode2/atomic0/det0，layer4真实权重/合成历史。
每侧5预热+20图计时、四个独立DFX窗口，长短交换执行顺序；保留Native控制。
八类PTO状态、保护区、自重放及候选A→B→A分别核对；不是EP16验收。

## 首版长短B16结果

两档八类状态逐元素零容差、metadata/保护区、自重放、图计时状态与A→B→A均通过。

| 档位 | QR归一化核内均值 μs | 完整CSA均值 μs | P95 μs | max μs |
| --- | ---: | ---: | ---: | ---: |
| 8K/B16 | 6.801→6.333（−6.89%） | 796.658→777.229（−2.44%） | 816.44→793.52 | 823.84→797.72 |
| 128K/B16 | 7.138→6.685（−6.36%） | 1110.493→1103.800（−0.60%） | 1122.90→1120.66 | 1127.78→1120.72 |

核内为四窗口各12个worker的均值，完整CSA为20次无profiler图计时；两种采集独立。
短档四窗口7.127/6.388/6.643/7.047→6.783/6.077/6.118/6.353μs，
长档7.125/7.068/6.770/7.590→6.760/6.612/7.065/6.302μs。
窗口分布重叠，不是长档每个窗口都更快；未改merge_norm也变快，不能把整层全部差额归因于本项。
Native控制短档938.19→945.08、长档1325.98→1318.03μs，原样保留。
[8K证据](h8192_b16/evidence.json)、[128K证据](h131072_b16/evidence.json)。

## T60尾块发现功能错误，修复后再决定保留

任务task_20260928_153010_192272130205退出1。原定先QR尾块再Sparse诊断，
前置失败后Sparse没有执行；其后续单独任务另记，不把未执行的部分当失败数值。
B10/S6/T60基线自重放通过；首版候选的第56～59行x_out/idx_topk自重放及图A不一致，
其余cache/state和metadata/保护区通过。这是陈旧数据错误，不是允许的精度差异。

生成代码在量化尾行分支先TLOAD临时qr_i8_matmul，然后才TSTORE本轮结果；
将原Tensor赋值改成显式store后，继续通过GM回读有效尾行未维持所需的数据顺序。
原失败报告与生成代码路径保留在[失败记录](tail_b10/failure.json)，不改写为PASS。

独立修正版`.cache/csa-qr-ub-tail-a66255ea`从原候选派生，
scale和INT8尾行都从当前UB结果set_validshape后直接发布，删除冗余GM回读。
满块算术/输入驻留/归约顺序不变；原候选两档性能不自动冒充修正版的新测量。
完整CPU编译/链接通过，生成代码只余gamma和输入TLOAD；
[修正补丁](tail_fix/candidate.patch)、[编译](tail_fix/compile.py)、[日志](tail_fix/compile.log)。
task_20260928_154007_238329511958完成、退出0，只复测受影响T60候选，复用已通过的固定规约基线；
[命令](run_tail_fix.sh)、[八类PTO/Native状态及图重放比较](tail_fix/tail_b10/comparison.json)均通过。
修正版保留到性能版，精度版不变；满档性能仍标首版实测，修正版未另跑满档计时或EP16。
按用户最新7:3权重，首版两档完整CSA变化率综合为−1.154%，核内约−6.515%；
这只汇总上述独立单卡结果，不代表修正版/组合源码或整模型加速。
