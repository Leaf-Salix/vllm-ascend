# Attention布局实验原始证据（2026-10-26）

正式控制11b380e3f；固定SDK身份和每组实际helper/SOURCE/SDK/静态权重身份见原始result.json。
本目录七项NPU实验全部性能不通过，没有Native精度升级；各JSON顶层pass不代表性能通过。

- `performance_raw.tar.gz`：每组原始result.json与独立L1轨迹，未删除样本。
- `candidate_sources.tar.gz`：13个不可变候选目录及其manifest，含CPU阶段拒绝的版本；不含权重。
- `cpu_evidence_20261026.tar.gz`：两臂编译收据、SDK预检、失败诊断、关键C++和编排生成物。
  生成C++仅为检查证据，没有手改或替换。
- `PERFORMANCE_SCREEN_SUMMARY.json`：组内配对摘要及原始JSON SHA。
- `TASKS.json`：7项设备任务ID、时限与最终退出状态。
- `PIPELINE_SOURCE_PROOF.json`：3槽事件静态映射/浮点表达式检查结果，不是设备精度证明。
- `REVIEW_NOTES.md`：独立源/生成物审查、已修正的预检问题与资格范围。

## 复现顺序

1. 解包candidate_sources到新的实验根；逐个核验manifest及SOURCE中candidates所有SHA。
2. `native512-pipeline-v2/check_pipeline_protocol.py`执行binary发布/消费模型；
   `native512-pipeline-v3/check_pipeline_protocol.py`逻辑相同；v1旧模型不是最终资格依据。
3. `native-wide-softmax-v3/check_reduction_tree.py`核验符号F8树、256个重排索引及fallback源码合同。
4. 保持现有独立SDK、官方13包静态cache、3399文件SOURCE_ROOT和43/77源绑定不变。
   使用封存env.sh/source_identity.py/sdk_bind.py作严格身份检查；更换目录须重新生成绑定及manifest，
   不能跳过旧收据检查。固定PyPTO2dfa7f120/Simpler57e499238/PTOAS0.66/ISA327cd586。
5. 仅以下7个有效目录可进入设备复现：`pv128-k32-v1`、`pv128-k32-v2`、`pv128-k128-v1`、
   `softmax-hoist-v1`、`bf16-quant-v3`、`native512-pipeline-v3`、`native-wide-softmax-v3`。
   对这7个目录分别用compile.sh baseline、compile.sh candidate完成64参编译/load/noNPU。
   其余6个目录仅保留静态/CPU失败或源审拒绝证据，不提交设备。
6. 依次经task-submit运行上述7个目录的run.sh，参数为全新label、32、8191、32个6；先性能再精度。
   队列入口PTOAS0.63，脚本显式绑定自有实际PTOAS0.66；外780秒、内750秒，不使用--wait。
7. 不并发测两组NPU性能；若任何候选过性能门槛，另作反向捕获重复及Native验收。
   本轮没有过门槛的候选，因此不得推断新候选Native精度或整模型token资格。

13候选的源和CPU失败原因对应报告及REVIEW_NOTES；未提交设备的版本没有NPU结果。
当前正式基准的8K/128K整模型token资格仍见上一份Direct heads报告，不混入本轮候选结果。
