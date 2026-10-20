# SPMD工具链与基线证据（2026-10-20）

主报告：[完整SPMD工具链与基线](../../DSV4_FLASH_CSA_TND_SPMD_TOOLCHAIN_BASELINE_20261020.md)。
复现入口见[REPRODUCTION.md](REPRODUCTION.md)，逐项结果见`raw-results.tar.gz`中的`RUNS.json`。

- `sk0-full-trace-evidence.tar.gz`：正式单层计时、两侧独立CANN trace和分析。
- `dfx/dep/pmu-*-evidence.tar.gz`：各自独立采集的原始证据。
- 原始包中的`sdk-results`：CPU、PTOAS/CCEC、实际wheel来源及后续设备门禁收据；必须按单份收据范围判断。
- 原始包中的`sk1-offline`与`results`：SK1编译包身份、首差与SK0复现。未包含大型tensor。
- `cpu-lowering-history.tar.gz`：常驻后缀v1～v8失败和修正，不丢弃失败记录。
- `reproduction-tools.tar.gz`：实际诊断/构建脚本和实验候选；实验源码不在正式服务路由。
- `resident-suffix-review`及独立审查：数学顺序、实际生成GM写回/资源与报告范围核验。
- `SHA256.json`：本目录除自身外的文件SHA。

不含权重、SDK二进制或wheel、大cache、历史完整bank。PTOAS wheel的Git提交未知，不能把参考源提交作为其身份。

`raw-results.tar.gz`原样保留全部JSON收据，展开到本目录恢复原相对路径；原字段包括官方算子名及layout，不为拼写检查改写。`RAW_SHA256.json`保存归档前逐文件SHA。

新SDK设备门禁已通过：132参数信号MIX，以及真实第4层权重、8K/T18变长FULL的冻结CSA八边界逐位回归。没有新resident后缀设备精度、新SDK整模型或优化性能结论。`sdk-device-logs.tar.gz`保留实际编译告警和两侧通过日志。
