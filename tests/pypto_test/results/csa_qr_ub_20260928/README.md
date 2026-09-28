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
仅这些CPU证据不宣称性能收益；设备任务task_20260928_143910_419318230381已提交，待设备结果。
单卡8K/B16和128K/B16，mode2/atomic0/det0，layer4真实权重/合成历史。
每侧5预热+20图计时、四个独立DFX窗口，长短交换执行顺序；保留Native控制。
八类PTO状态、保护区、自重放及候选A→B→A分别核对；不是EP16验收。
