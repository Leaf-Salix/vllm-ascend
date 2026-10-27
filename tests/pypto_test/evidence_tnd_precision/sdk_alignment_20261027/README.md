# 新SDK三方对拍与SWA实验原始证据

正式计算基准11b380e3f保留。新SDK仅完成8K/B32单层，不能外推整模型token。

- `new_sdk_threeway_raw.tar.gz`：三方普通计时、A/B八边界、ABA、guard、日志及实际SDK预检。
- `old_sdk_pmu_raw.tar.gz`：旧cafc兼容版与我方/Native的独立PipeUtilization观测，含原始CSV和警告日志；不是最新c346计时。
- `swa_raw.tar.gz`：两项未过性能门槛的原始result与L1。顶层pass不代表性能/精度验收通过。
- `swa_v1_cpu.tar.gz`、`swa_v2_cpu.tar.gz`：各两臂实际CPU收据/生成物；未手改C++。
- `new_sdk_cpu.tar.gz`：最终我方64/Nal debug57无设备资格、源/DSO/编译器SHA与实际Nal编排。
- `sdk_build_logs.tar.gz`：新隔离SDK每文件来源SHA、构建与107项最小UT日志，不包含依赖二进制或模型权重。
- `experiment_sources.tar.gz`：三份真正受测的不可变helper及各自manifest。
- `NEW_SDK_THREEWAY_SUMMARY.json`：由原始JSON重新计算的均值/中位数和精度摘要。
- `TASKS.json`：本轮4项硬件任务及最终状态；各1卡、外780s/内750s。

`DERIVATION.json`等继承元数据保留历史cafc谱系；本轮Nal身份以`NAL_COMPATIBILITY.json`的c346原文件SHA及新SDK收据为准，不以继承文字判断。原工具链和正式基准均未覆盖。

## 复现

1. 在新自有目录解包source archive，对每个manifest及SOURCE候选SHA逐项核对。数据和SDK绑定路径见封存env.sh/sdk_bind.py/source_identity.py；更换路径需重新冻结manifest和CPU资格，不能跳过身份检查。
2. 旧SDK的两个SWA目录各执行compile.sh baseline、compile.sh candidate；通过生成物审查后依次run.sh，全S6的8K/B32/history8191。先性能0.5%/10-of-12/L1门槛，不通过不扩测精度。
3. 新SDK从公开PyPTO d44925c1和PTOAS914863ed构建，保留原SDK。PTOAS使用其已有PTO_ENABLE_PYTHON_BINDING=OFF原生CLI模式，复用LLVM19.1.7/ISA327cd586/Simpler57e499；显式ccache和四并行。完整归档SHA/实际二进制SHA见源码manifest与CPU收据。
4. 新SDK三方helper先执行compile.sh ours、compile.sh nal8k，分别64/57参数，load成功且NPU未初始化。以run.sh 新label 32 8192 timing经task-submit提交，PTOAS队列0.63但脚本绑定自有914，单卡外780s/内750s。
5. Native SK0，确定性1/HCCL=true，NZ2，真实第4层权重/合成hidden/history，12×100普通FULL graph计时，无DFX，不删除样本。我方A/B八边界逐bit；三方自身ABA、guard及只读metadata均检查。Nal采用原始debug TopK观察入口，而非普通55参数入口。
6. PMU与普通计时分开，mode pmu独立任务；不能将完整worker计数或重叠pipe相加为某模块耗时。

本目录不包含随后准备的累计阶段截断诊断，其尚未取得设备结果。
