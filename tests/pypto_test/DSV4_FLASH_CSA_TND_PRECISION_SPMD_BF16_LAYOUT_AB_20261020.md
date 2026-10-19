# 精度SPMD：BF16布局搬运单因素对拍（2026-10-20）

## 当前结论

继续在`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`上实验，未新建分支。
已验收的完整精度、父SPMD和预打包实现全部保留。本次候选只归档，尚未接入正式服务路由。

| 前缀真实T | 控制中位数μs | 候选中位数μs | 降低 | 更快窗口 | 逐bit检查 |
| --- | --- | --- | --- | --- | --- |
| 21 | 229.312 | 228.878 | 0.19% | 8/12 | 536/536通过 |
| 144 | 325.523 | 313.143 | 3.80% | 12/12 | 536/536通过 |
| 240 | 361.698 | 351.441 | 2.84% | 12/12 | 536/536通过 |

这是HC-pre→输入Norm→WKV→Indexer权重投影前缀的direct ACLGraph计时，
真实C4权重、合成hidden；不是完整CSA、vLLM FULL或整模型性能。
大T有一致方向的配对收益，短T没有稳定收益。每档仅一次独立任务；还不能证明跨任务稳定性。
因此只保留为下一候选证据，不将现有整模型精度验收转授给这项新改动。

## 只改变什么

当前输入来自BF16 hidden，经无损加宽得到FP32。原路径先用FP32执行转置、8元素组复制和转置，
再cast为BF16；候选把同一次cast移到上述纯布局操作前，后续布局tile改用BF16。
无效尾行仍显式写零，完整packed输入包含尾行逐字节相同。

没有改动HF32模拟权重、Cube乘加、归约次序、舍入位置对应的数值、RMS sqrt/div、
ownership、24AIC/48AIV或每个角色10次MIX交接。生成代码分别确认float和bfloat16布局tile，
两个生成源均含20次SYNCALL（两个角色各10次）。没有手写C++数学或修改SDK。

两臂都在加载期预打包相同2MiB常量，不把之前“移出固定权重打包”的收益重复记入本轮。
两臂保持相同尺寸且独立的工作区、捕获图和poison/guard策略。

## 对拍方法及边界

- Native开启确定性level1。先对拍原控制与Native的HC、Norm、WKV及Indexer投影边界，首差立即失败。
- A/B/A输入，每组3次两臂replay，比较公开边界、只读输入、完整2MiB常量及包含尾行的packed输入。
- 12个交错配对窗口，每窗口25次replay，用NPU Event计时；不删离群样本。
- 每窗口后重新核对输出与Native、两臂完整布局字节、只读常量及前后guard。
- 所有运行源、实际助手、program/kernel ABI、Torch 30参数schema与编译receipt绑定SHA。
- CPU另检全部65,536 BF16编码和尾行1～16的布局等价；这不证明设备NaN传输语义，
  设备对拍只接受有限数值。本次也未覆盖全部模型层、全部T和真实整模型token。

三项任务均exit0，均使用780秒调度上限、内部750秒timeout；任务分开提交并已全部结束。
顺序为T21、T240、T144，任务ID及完整12窗数据保留在`summary.json`及原始结果中。

## 证据与复现

证据目录`evidence_tnd_precision/spmd_bf16_pack_20261020/`保存原始JSON、两臂DSL、
候选构造器、测试/编译/启动助手、实际receipt、生成设备源码和冻结父stage，见`raw.tar.gz`。
`manifest.json`绑定归档及摘要SHA；生成C++是编译产物，未作为手写算子实现。
与[预打包整模型报告](DSV4_FLASH_CSA_TND_PRECISION_SPMD_PREPACK_REAL_MODEL_20261020.md)环境相同。
复现需该报告的冻结43源及父14源、已有SDK和真实权重，并设置助手中的独立实验根；
先执行无设备program/ABI/CCEC及Torch注册预检，再通过task-submit分别运行三档。
禁止直接覆盖原实验目录，首差停止，不重装vLLM/PyPTO/Simpler。

离线复算：

```bash
python -m pytest tests/pypto_test/test_precision_spmd_bf16_pack_evidence_cpu.py -q
```

下一门禁应保持短T旧路径，先把大T候选接入独立完整CSA并做FULL对拍；
通过后按真实8K/B40→128K/B24复测Native token一致性。
只有完成这些门禁才考虑进入正式精度SPMD路径，不能用本表直接声称整模型已加速3%。
