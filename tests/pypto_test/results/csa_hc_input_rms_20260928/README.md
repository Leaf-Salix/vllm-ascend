# HC输入加宽与RMS统计合并候选

基底为e58ddc94算子（9edb8dfe只增加记录），候选冻结在`.cache/csa-hc-input-rms-9edb8dfe`。
仅性能版把`hc_widen`与`hc_pre_rms`合并为`hc_widen_rms`：BF16读取后转FP32，
一份值写入原FP32中间缓冲，另一份按原512列、32段顺序平方归约，最后用原高精度rsqrt。
省去独立RMS任务对整份FP32缓冲的GM读取，B16/S6逻辑读取量约6 MiB；这不是PMU实测流量或时间收益。

Cube仍读取FP32、保持纯AIC；门控、20次Sinkhorn、混合及BF16/RMSNorm边界共用原实现。
共用模块只抽出“已有RMS统计后的门控”与“已有门控后的混合/Norm”，旧精度入口仍自行运行原RMS任务。
尾行在cast后显式恢复valid_shape并填零，避免历史T60无效行参与归约问题。

Native参考：`csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h`的输入转换和Cube消费流程，
`hc_pre_cube_compute.h::ComputeDecode/MmadA2`复用输入计算平方和与投影。
本候选只吸收输入复用思路，仍用现有Vector归约，不把它表述为原样复制Native Cube算法。
pypto-lib 73078d0的HC入口是FP32，无本接入的BF16桥接；因此仍保留必要FP32缓冲，消除的是本地重复读取。

完整CPU lowering/PTOAS/CCE/链接通过，生成代码有AIV `hc_widen_rms`和纯AIC `hc_pre_linear`，
独立`hc_pre_rms`消失。融合可能延后Cube启动，需同时看核内累计工作、HC关键链及完整CSA。

任务`task_20260928_103131_11716825293`已提交。长短B16各20次无profiler图计时及2个DFX窗口，
layer4真实权重、合成历史、mode2/atomic0/det1。若分配到device8，复用上一轮同卡K256基线DFX；
其他卡重新采基线DFX，两侧无profiler计时始终同卡同轮。设备结果尚待收集，不改变生产默认值。
若代表档有收益，再补T60尾块及受影响的真实EP16；候选无收益则不扩测。

[执行脚本](run_layer.sh)、[源码差异](candidate.patch)、[编译日志](compile.log)。

用户追加：本轮收尾后评估[Simpler #2389](https://github.com/hw-native-sys/simpler/pull/2389)调度改动，
先确认当前kernel-mode适用性，再以长短代表档与P95决定取舍。本轮冻结运行时保持不变。
