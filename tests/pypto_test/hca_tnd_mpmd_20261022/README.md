# HCA TND MPMD 单卡实现与基准

正式源码：[deepseek_v4_flash_hca_tnd](../../../vllm_ascend/ops/pypto/deepseek_v4_flash_hca_tnd/)。
目标为完整 HCA attention 半层，Native/PTO−1 超过 30%，不是某个子算子的局部收益。
所有新测试共用 `csa_shared_toolchain/manifest.json`，正式第3层 W8A8、NZ2、seed=1024。

## 当前阶段

2026-10-22 首版及去除 HBG 工厂后的 CPU 后端编译通过，见 `runs/cpu_v5/result.json`
和 `runs/cpu_eager_v1/result.json`。128K/B16 与旧 HCA 的输出/三组 cache/state 逐 bit 一致；
混长跨界 + padding 的输出在容差内、写保护通过，固定 Native RMS 输入的 C128 子链通过。
整体 Native storage 严格门禁仍有原有数值差异，不能报告全面精度 PASS。编译过程中修正了旧 HCA 的 O_B 调用签名、
新代码的 INDEX/INT32 变量复用及 Tensor/Tile 混用；未改公共 SDK。

`experiments/initial_v2` 冻结首次设备候选及直接数学依赖。首个 v1 任务在 pending
阶段取消，用于补齐测试层的 `_pto_hca_layer` 绑定；没有执行设备，未覆盖旧记录。
初始任务 `task_20261022_164106_175565826883` 已完成；其图重放结果只保留作历史。
后续全部使用 eager 直接调用，详见正式目录 OPTIMIZATION_LOG.md。

## 复现

```bash
# 只编译，不占卡；每次使用新的输出目录。
bash tests/pypto_test/hca_tnd_mpmd_20261022/launch.sh \
  --cpu --output /absolute/new_cpu_result

# 提交前冻结候选；不改正在执行的快照。
python tests/pypto_test/hca_tnd_mpmd_20261022/prepare.py attempt_name

# 真机检查；--queries/--histories 支持逐请求混长与 C128 边界。
task-submit --device auto --max-time 1500 \
  'bash /absolute/repo/tests/pypto_test/hca_tnd_mpmd_20261022/launch.sh \
    --experiment /absolute/repo/tests/pypto_test/hca_tnd_mpmd_20261022/experiments/attempt_name \
    --output /absolute/new_device_result --history 131072 --batch 16'
```

`run_case.py` 检查 Native/PTO 输出、三组 cache/state、写保护、只读 metadata、
重复直接调用以及非有限值。输出沿用 atol/rtol=0.01/0.01，cache/state=1e-5/1e-5；
固定初态的重复直接调用精确比较。它不代替整模型 token 精度验收。
指定 `--windows 10` 时，在功能检查通过后另采十次完整 L1 任务首尾，附原始记录和泳道。

`launch_native.sh` 使用直接调用；`profile_eager.py` 以同步结束的 CPU 标记圈定每次调用，
统计全部设备流的首尾，保留真实层内搬运和 AICPU。窗口分析的 CPU 回归已通过。
`run_native.py` 和 `measure_hca.py` 只保存用户新指令前已运行的历史图基准，入口停用；
其 573.25 μs 结果不得当作后续 eager 对照的分母。
八档输入见 `cases.json`，从三个 CSA 会话最新固定性能规范迁移，不把旧 HCA 七档数据混算。

大权重、编译二进制、私有 OPP、张量快照仅本地保存；Git 归档时仅纳入源码和有用的
报告/日志/trace，逐次中文说明实现、失败原因与采用依据。

`--diagnostic-perf` 仅允许在新旧 HCA 输出及全部 cache/state 逐 bit 一致时采集诊断性能。
它保留 Native storage 的失败项及 `passed=false`，不把性能测量改写成精度通过。
