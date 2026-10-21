# Native 128K/B1/S6 性能补测

日期：2026-10-21。仅补这一档；使用原 10/20 Native 五口径测试脚本，没有修改算子。

| 计量方式 | 样本数 | 中位数 μs | 最小值 μs | 最大值 μs |
| --- | ---: | ---: | ---: | ---: |
| PyTorch Profiler 全流整层跨度（主对比值） | 30 | 632.75 | 624.75 | 680.50 |
| 无 Profiler 的图外 NPU Event | 20 | 614.91 | 600.48 | 624.70 |

范围：TP1，正式 W8A8 第4层、S6/T6，HC_pre → input RMSNorm → Native CSA → HC_post。
hidden 与历史 KV 是 seed=1024 的合成输入；不是整模型或服务吞吐测试。

优化生效检查通过：

- 四张核心投影实际 NZ 格式29，保持原 Native DSA 多流。
- npugraph_ex，fullgraph=True、dynamic=False；静态编译实际成功并安装。
- SuperKernel 实际成功返回，trace 存在 sk_*static_kernel 与 MatmulWeightNz。
- 唯一 ACLGraph 直接重放，同初态输出/Top-K/cache/state 对照及保护区通过。
- 30次完整设备跨度保留全部流和图内复制，排除图外 fixture reset/输出毒化。

后续五组表格的 Native B1 列使用 **632.75 μs**。Event 值另列，不能
混入原 Profiler 列；与此前 PTO L1 的百分比仍属于跨轮、外层/核内口径参考。
全部原始样本、优化检查证据及实际 trace 路径见 [PERFORMANCE.json](PERFORMANCE.json)。

任务：`task_20261021_114120_30644583266`。
原始结果：`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_five_way_latest_20261020/results/h131072_b1/native_supplement_20261021_v1`。
原七档历史数值保持不变；权重、OPP及静态二进制仅本地保留。

本次按用户要求归档脚本快照、完整原始报告、全部样本、运行日志、profiling JSON
及其 CSV 文本导出。文件清单见 [COMMIT_FILES.json](COMMIT_FILES.json)，大产物
路径与字节数见 [LOCAL_ARTIFACTS.json](LOCAL_ARTIFACTS.json)，没有新增设备测试。

可直接查看[原始报告](../csa_five_way_latest_20261020/results/h131072_b1/native_supplement_20261021_v1/report.json)、
[PyTorch profiling JSON](../csa_five_way_latest_20261020/results/h131072_b1/native_supplement_20261021_v1/profile/liteserver-hps-365a-00001_3066172_20261021114317094_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json)。

复现命令见 [TASK.json](TASK.json)，必须将 attempt 换成新名字，原脚本拒绝覆盖。
原计量与校验代码已保存到 source_snapshot。只重新汇总已有结果：

```bash
python tests/pypto_test/csa_native_b1_20261021/summarize.py
```
