# 固定K的QR投影：L1 K256→512候选

基底e58ddc94，仅性能版atomic0的QR_K_TILE由256改512。固定K4096单FP32累加链、
N128、按行数M32/M64分组、NZ转置权重接口及归一化/量化规则不变；atomic1及精度版不改。
CPU完整lowering/PTOAS/CCE/链接通过；M32/M64及尾M16仍L0 K128/N128，
每输出32次MMAD，L1分段16→8。不能仅凭分块推断设备收益或数值等价。

与pypto-lib 73078d0差异：上游QR是split-K2/K256、按M64遍历、K×N权重视图、atomic合并；
当前固定完整K且最多3个M组，以Native N×K NZ物理权重直接供给转置矩乘。
保留这一路径是为了兼容原生权重存储并维持已验证的EP16固定归约表现，不进行设备权重重排。
KV K512已经有核内收益，但其权重朝向不同，不能直接推导QR同样受益。

task_20260928_101129_50937415458：长短B16同卡对照，layer4真实权重与合成历史，
NZ mode2、atomic0、det1；各20次无profiler图计时，另采各2个独立DFX窗口。
八类状态、保护区及候选A→B→A重放随性能一起验证。单卡det1用于诊断，部署整网另按det0验收。
当前尚待设备结果，不改生产默认值。

[执行脚本](run_layer.sh)、[源码差异](candidate.patch)、[编译日志](compile.log)、[L0分块](lowering_tiles.json)。
