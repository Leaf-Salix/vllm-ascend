# 六档 decode forward 泛化对比

最新主结果和下载文件统一在
[六档结果目录](../../csa_six_case_profiles_20260926/README.md)，压缩包为
[全部 profiling 与泳道](../../csa_six_case_profiles_20260926.tar.gz)。

| 历史长度 | 单卡 B | Native forward ms | PTO forward ms | PTO 耗时变化 |
| ---: | ---: | ---: | ---: | ---: |
| 131072 | 4 | 48.483 | 51.715 | +6.67% |
| 131072 | 8 | 58.681 | 67.259 | +14.62% |
| 131072 | 16 | 73.699 | 94.280 | +27.93% |
| 8192 | 24 | 79.210 | 79.279 | +0.09% |
| 8192 | 32 | 90.437 | 91.371 | +1.03% |
| 8192 | 40 | 102.236 | 106.241 | +3.92% |

仅比较 warmup 后连续10次 `_model_forward` 的设备耗时均值，每rank等权。
metadata准备、logits、采样、DSpark草稿、步间等待、加载与初始化编译均不在此边界。
各档两侧先生成96 token/请求预热，再以128 token/请求生成，跳过前8个满档step，记录接下来10步。
独立Level0 trace另采3步，不把其时间混入主表。六档逐token、DSpark总数及逐位置统计均一致。

正式75分片权重、TP1/DP=EP16、出5验6、EPLB关闭、mode=2；PTO性能版atomic=1，
两侧Native实际level=0、HCCL=false。容量40、捕获24/48/96/144/192/240；
128K预算256、8K预算400，扣除DSpark预留160后分别可调度96/240。
当前CANN缺少Native `TransposeBatchMatMulWeightNz`，Native wo_a保留ND，PTO加载期一次转换NZ。
主矩阵保留两侧各自的进程级event默认行为；PyPTO初始化的全局模式切换另作专项诊断，
见 [B4回退定位](../event_mode_diagnosis/README.md)。专项已排除event模式为主因；
同硬件模式trace约75%的增量在FFN/MoE。旧默认event模式的CSA trace有不同profiling扰动，
不能据此宣称B4 CSA本体更快，完整定位与未确认的路由/分组假设均单列记录。

## 采集与复现

128K源码 `a7dc706e`：Native任务 `task_20260926_223502_384103927457` 完成前三档，
随后旧B24被拒绝；PTO任务 `task_20260926_224652_39544812557` 完成前三档。
8K源码 `5d42db04`：任务 `task_20260926_230348_41308902775`，两侧三档完整完成。
单卡DFX入口 `7769b3c7`，使用正式 `model.layers.4` 权重、合成输入/历史与第二CSA层metadata复用。

在新的产物目录执行，每组每侧只加载一次模型：

```bash
task-submit --device auto --device-num 16 --max-time 7200 \
  'bash tests/pypto_test/results/csa_baseline_20260926/model_128k_performance/run.sh 131072'
task-submit --device auto --device-num 16 --max-time 7200 \
  'bash tests/pypto_test/results/csa_baseline_20260926/model_128k_performance/run.sh 8192'
```

`run.sh` 会拒绝覆盖已有采集，重现时先将脚本的result_root指向新的目录。
CPU解析 `analyze.sh B` 自动选择对应history和组目录；不占NPU。
泳道命令为 `six_case_swimlanes/run.sh H B`，通过单卡 `task-submit` 执行，不重跑完整模型。

各组原始记录位于本目录及 `../model_8k_large_batch_performance/capacity40/b{B}/{native,pto}`。
每档完整离线对照为 `performance_comparison.json`；便于审阅的统一记录在下载目录的
`summary.json`，保留每rank的10步样本、DSpark、显存和各CSA层统计。
旧容量4的 `pilot_b4.json` 仅为完整周期计时方法验证，不能当作纯forward，不填入主表。
本轮不替代H8192/B16的750μs目标或完整逐元素验收，也没有恢复新的CSA性能优化。
