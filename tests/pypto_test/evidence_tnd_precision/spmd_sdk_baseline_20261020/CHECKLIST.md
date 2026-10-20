# 2026-10-20：对齐最新CSA性能目标

- 基准commit：48c58ce7c467dfce63d5cb5c41afd225488d6012；原71源和独立overlap候选保留。
- 用户新目标：10月20日nalinaly动态SPMD截图，S6、NZ2、真实第4层权重、合成hidden/KV，完整HC-pre到HC-post。
- 图中提升为吞吐比 `(baseline/dynamic)-1`，不是耗时下降百分比；不能与整模型forward毫秒直接比较。

## 门禁

- [x] 固定nalinaly实际提交，找到测量脚本、入口、layout及trace口径。
- [ ] 先复现8K/B16同卡三路：优化Native、nalinaly动态SPMD、当前精度TND；记录实际执行和完整跨度。
- [ ] 拆开阶段计算与调度/同步，依据最大差距选择单因素候选。
- [ ] 候选逐模块Native精度/FULL graph通过，首差停止修正。
- [ ] 有明确完整CSA收益后，8K/B40、128K/B24真实整模型192token和接受事件与Native一致。
- [x] 每项NPU任务≤780秒；不跑Native2、SDK仅在明确批准的新独立目录构建、不使用C++数学、不覆盖原基准；记录证据和复现方法。

## 独立兼容工具链（本轮已获明确批准）

- [x] 固定三仓提交及PTO-ISA、LLVM、CANN cmake提交。
- [x] 远端17899不可用，采用本机获取与rsync上传。
- [x] 新venv原生构建PyPTO与Simpler；不复制旧so。
- [x] LLVM源码构建完成；实际PTOAS独立wheel、SHA和导入路径已封存，PTOAS源码build被复用替代、未完成。
- [x] signal notify/wait、132 Tensor ABI完整编译、31次FULL硬件回放与guard通过。
- [x] 新SDK冻结CSA在8K/T18变长FULL逐bit回归通过；完整resident迁移另行验收。

## 独立管线PMU

- [x] 无设备确认Level0会关闭PMU，使用Level1 + PipeUtilization。
- [x] 复用冻结64参数算子与所有精度检查，不采正式计时、不开swimlane/dep-gen。
- [x] 8K/B16单卡，每arm仅3回放，读取实际CSV计数；首个数值差异即停。

- [x] 参考完整动态入口130 Tensor，CPU lowering/codegen至MLIR通过；只有一个MIX提交。
- [x] PTOAS完整编译、最小设备信号与现有精度算子的8K/T18 SDK回归。

- [x] 最小24MIX信号测试CPU lowering通过；使用不同owner payload，FULL设备门禁也已通过。

- [x] 新SDK冻结64参数CSA、132参数信号MIX和16参数resident后缀完整PTOAS/CCEC编译通过；无NPU初始化。
- [x] 单卡信号FULL Graph门禁及冻结CSA的8K/T18硬件精度回归。
