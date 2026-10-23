# HCA TND MPMD 单卡实现与基准

正式源码：[deepseek_v4_flash_hca_tnd](../../../vllm_ascend/ops/pypto/deepseek_v4_flash_hca_tnd/)。
目标为完整 HCA attention 半层，Native/PTO−1 超过 30%，不是某个子算子的局部收益。
所有新测试共用 `csa_shared_toolchain/manifest.json`，正式第3层 W8A8、NZ2、seed=1024。
用户最新明确：候选性能只比较完整HCA的L1 incore首尾。JIT重复分析等问题只提主仓issue，
当前不继续修SDK或以Host提交时间指导核内候选。外部服务/Host脚本仅保留复现和历史证据。

## 当前阶段

最新正式attention分档见[完整记录](ATTENTION_SUMMARY.md)：NZ且Native已有Host
max_seq_lens≥131072、T≥33时启用K512；短历史/小T保留K128，129～192行叠加已有投影策略。
后续累计max/N64候选与Native混长参考输入失配的定位见[本轮工作](ATTENTION_NEXT_STAGE.md)。
任务明细与已结束结果见[后续候选汇总](ATTENTION_NEXT_RESULTS.md)。2026-10-24本轮已收口，用户再次要求结束后停止；归档提交后暂停。
双query、Vector mask核内分档、H32 UB重置均未采用；正式源码保持已验证区间策略，未push。
混长padding的Native参考已修正，尚有一项冷S1输出差异；30%目标和严格storage等遗留未完成。
最后的核外分档与直接KV结果见[直接KV收口记录](ATTENTION_DIRECT_KV.md)，两者均未接入正式算子。

2026-10-23最新投影区间优化见[本轮完整结果](PROJECTION_RESUME_SUMMARY.md)：
Native已有Host历史上界至少128K且129≤T≤192时，独立入口对Q_B/O_B采用M192/N128。
最终B24/B28相对本轮原版分别为717.94→701.02μs、782.08→772.30μs，
控制输出及全状态精确、原Native输出容差通过；最终S5真实服务接管通过。
64K策略和O_A组合的负结果全部保留，没有采用或向所有历史长度推广。
完整过程日志已移动到[HCA_TND_MPMD_OPTIMIZATION_LOG.md](../../../vllm_ascend/ops/pypto/HCA_TND_MPMD_OPTIMIZATION_LOG.md)。

最新进展见[输出差异定位与修正](NATIVE_OUTPUT_PRECISION.md)及
[完整配对结果](NATIVE_OUTPUT_SUMMARY.md)。八档输出均通过原容差；8K/B32原3处超差
通过恢复Native O_A BF16及整token量化语义消除。正式`output.py`已采用该数学路径，
大T的O_B使用N256、K分两段，整数相加后才转FP32。Native严格cache/state差异仍保留。
真实`dsv4_hca_forward`在S1和混长/padding中接管成功，KV钩子顺序和按步compact缓存通过；
这不等于完整PD传输、连续轨迹或整模型token验收。

2026-10-22 首版及去除 HBG 工厂后的 CPU 后端编译通过，见 `runs/cpu_v5/result.json`
和 `runs/cpu_eager_v1/result.json`。128K/B16 与旧 HCA 的输出/三组 cache/state 逐 bit 一致；
混长跨界 + padding 的输出在容差内、写保护通过，固定 Native RMS 输入的 C128 子链通过。
整体 Native storage 严格门禁仍有原有数值差异，不能报告全面精度 PASS。编译过程中修正了旧 HCA 的 O_B 调用签名、
新代码的 INDEX/INT32 变量复用及 Tensor/Tile 混用；未改公共 SDK。

`experiments/initial_v2` 冻结首次设备候选及直接数学依赖。首个 v1 任务在 pending
阶段取消，用于补齐测试层的 `_pto_hca_layer` 绑定；没有执行设备，未覆盖旧记录。
初始任务 `task_20261022_164106_175565826883` 已完成；其图重放结果只保留作历史。
后续全部使用 eager 直接调用，详见上述HCA TND MPMD过程日志。

本日后续已核对 [Native核内路径与大项优先级](NATIVE_KERNEL_REVIEW.md)：
Q_B扩24核在B16/B28均回退，未采用；O_B归约/HC_post合并流水在B16严格保持结果，
但完整L1从511.60增至623.54μs，未采用且不扩测B28。两项候选在`candidates/`保留。
同址metadata检查前四场景通过，最后A再次执行Native时报MTE异常，尚不能算完整A→B→A通过。

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
这八档含256K/B16及128K/B16混长；额外测试的128K/B4、8K/B16不替代这两项目标档。

大权重、编译二进制、私有 OPP、张量快照仅本地保存；Git 归档时仅纳入源码和有用的
报告/日志/trace，逐次中文说明实现、失败原因与采用依据。

`--diagnostic-perf` 仅允许在新旧 HCA 输出及全部 cache/state 逐 bit 一致时采集诊断性能。
它保留 Native storage 的失败项及 `passed=false`，不把性能测量改写成精度通过。
`--control-source`可指定冻结TND控制组，与候选在同进程交错采完整L1；
`--compare-legacy`与它互斥。新旧SDK不跨版本计算收益。

`--native-output-math`仅用于这次已做相同输入拼接诊断的数学修正：保留旧版输出差异，
但必须满足Native原输出容差、cache/state与控制精确相同、重复调用和写保护通过，才采诊断性能。
正式修正版本之间的调度候选仍使用普通`--diagnostic-perf`，输出也必须逐bit相同。

`--profile --profile-service`通过真实custom op计时，每次创建新的forward context，
包含资格判定、compact metadata生产、设备描述符绑定和KV钩子；计时前先验证没有回退Native。
两侧同时记录CPU包络、设备首尾和设备忙碌区间并集，不把Host派发空隙算成incore算力收益。
采样后再次精确比较本侧结果及保护区，不能以只运行成功代替正确性验证。

当前`--cpu`已补齐编排C++与全部incore C++的实际目标文件编译；早期同名阶段
只做PyPTO/PTOAS生成，不代表实际AscendC编译。详见精度修正文档的失败记录。

Host重复分析已经提交[PyPTO #3113](https://github.com/hw-native-sys/pypto/issues/3113)，
说明和建议在`EAGER_HOST_LOOKUP_ISSUE.md`，无模型权重复现为`repro_eager_host_lookup.py`。
测试中访问预热registration的代码没有进入正式适配层，未修改共享SDK。

O_A的Native分块、48份均衡网格及33～96区间组合均保留为候选，未进入正式代码；
首轮B16改善在扩档和区间组合中没有形成稳定收益。任务和报告分别见
`OA_NATIVE_TASKS.json`、`OA_RANGES_TASKS.json`及正式目录的优化日志。
区间组合中发现的scope/constexpr SSA编译问题已提交
[PyPTO #3115](https://github.com/hw-native-sys/pypto/issues/3115)，复现文件
`repro_constexpr_scope_yield.py`不占设备、不需要权重。
