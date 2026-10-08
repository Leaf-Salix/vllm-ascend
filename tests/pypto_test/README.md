# DeepSeek-V4 Flash CSA 验证入口

2026-10-08 的TND单因素效率A/B、独立复测及回退记录见
[效率A/B文档](DSV4_FLASH_CSA_TND_EFFICIENCY_AB_20261008.md)，包含可复现补丁和全部计时样本。

最新的Native SK1、nalinaly BSH、TND九档及变长性能/精度结果见
[三方完整矩阵](DSV4_FLASH_CSA_THREEWAY_20261008.md)。

本分支2026-10-08的变长TND移植、CANN9.2同场性能与精度记录见
[TND验证文档](DSV4_FLASH_CSA_TND_20261008.md)，含固定参考SHA、失败修复及原始轮次样本。

任务、约束和验收状态以 [任务清单](DSV4_FLASH_CSA_TASK_CHECKLIST.md) 为准。
历史与当前执行过程持续记录在 [验证日志](DSV4_FLASH_CSA_VALIDATION_LOG.md)，
本轮从 [第 99 节](DSV4_FLASH_CSA_VALIDATION_LOG.md#log-20260926) 开始。
当前基线为 vLLM 0.25.1 / vLLM-Ascend 0.25.1rc1、A3 / CANN 9.0，
正式权重固定为 `/data/model/DeepSeek-V4-Flash-0731-w8a8`。
环境、PyPTO/Simpler 分支及 PTOAS/ISA 版本见清单第 2 节与 A4。

所有测试先构造单卡 case；需要整模型证据时再使用正式权重 16 卡。
NPU 任务统一通过 `task-submit`，用 `--status` 查询，不用 `--wait`。
任务排队或运行期间固定源码，不修改 JIT 会读取的文件。

## 当前入口

| 入口 | 用途与边界 |
| --- | --- |
| `dsv4_csa_single_layer.py` / `run_csa_single_layer.sh` | 正式第 2 层权重、合成输入/历史的 Native/PTO 整层对照；同初态、metadata/保护区检查，图重放设备计时和可选 profiler，可保存 schema=2 快照 |
| `dsv4_csa_single_card_bench.py` | schema=2 快照回放与性能采样；无参考记为 MEASURED，逐元素验收必须提供全部声明输出/状态的参考和容差 |
| `dsv4_csa_replay.py` / `dsv4_csa_validation.py` | 共用快照、布局/别名/初态恢复与逐元素门禁 |
| `dsv4_csa_weight_layout_probe.py` | 单卡核对 Native 格式 29 原始字节、pypto-lib pack_nz、设备打包与 NZ 快照；不加载整模型 |
| `dsv4_csa_reference_lower.py` | 完整层 CPU lowering；`--build` 另做 PTOAS/CCE 编译，不执行设备或宣称数值验收 |
| `offline_pd/run.py` | 正式权重 P 缓存、D16 生成对照、逐层诊断和 profiling，见 [离线 P/D 说明](DSV4_FLASH_CSA_OFFLINE_PD.md) |
| `offline_pd/compare.py` | CPU 比较两侧 decode 的全部 rank/token 和 DSpark 总数、逐位置接受数；缺项或差异失败，不代替层误差/性能验收 |
| `dsv4_csa_bank_replica_diff.py` | CPU 比较既有 P 缓存的 TP 副本，不做 hash |
| `repro_tdiv_high_precision.py` / `run_tdiv_high_precision_repro.sh` | 独立 TDIV 能力复现，见 [问题与原始证据](PTO_ISA_A3_TDIV_HIGH_PRECISION_REPRO.md) |

`dsv4_csa_native_case.py` 与 `dsv4_csa_formal_weights.py` 是新单层用例的辅助模块，
使用当前 release 的 Native builder、物理页布局和 ModelSlim 加载路径。

## 单卡对照

在本仓库根目录执行，使用新的输出目录：

```bash
source ../env-dsv4-0251rc1.sh
task-submit --device auto --max-time 1800 \
  "bash $PWD/tests/pypto_test/run_csa_single_layer.sh $PWD/tests/pypto_test/results/single_layer_b4_h8192 --batch 4 --history 8192 --variant precision --weight-nz-mode 0 --save-case"
```

结果为 `report.json`，其中 `weight_storage_binding` 记录实际逻辑/物理形状、格式及原地址复用；`--save-case` 另存调用前快照到 `case/`。
`--save-state` 单独保存两侧 8 类逻辑输出/状态到 `states.pt`，供 ND/NZ 逐元素比较。
Native/PTO 使用同一 mode。性能版使用 `--variant performance`，NZ 使用 mode=1/2；
这些是可选配置，不表示每条路径已经验收。
`--atomic-add 0` 选择固定规约，`1` 选择 split-K atomic add；
未显式设置时，性能版默认 0，精度版保持默认 1。
性能版固定规约已通过[七档真实EP16对照](results/csa_atomic_matrix_20260928/model/RESULTS.md)，
正式forward、P95和每步最慢rank均优于Native，token/DSpark一致。
等价环境变量为 `VLLM_ASCEND_PTO_CSA_ATOMIC_ADD`，必须在进程导入/编译算子前设置。
关闭时 QR/KV 改为单 K 分片、单写入者；这也改变累加分组，不能视为 atomic1 路径的逐 bit 参考。
配合 `--graph` 检查相同地址上的 A→B→A 输入更新；每次恢复 cache/state，
图输出与对应 eager 输出逐元素精确比较，并检查 metadata 和保护区。
`--graph` 固定形状与 metadata，只检查输入内容更新。
`--padding-graph` 另测同一个捕获图的满档→补位→满档，要求 batch 至少为 2、atomic=0；
Native builder 在图外更新同址 metadata，Native compact producer 和 PTO 在图内执行。
补位 `seq_lens=0`、slot 为负、页表为 0，保留旧 positions；检查有效请求输出、全部 cache/state、
有效 compact 行及保护区。该检查不包含 Native 整图、空 rank 或跨档位切换。
零容差差异用于诊断，算术差异本身不会让该诊断伪装成 PASS；保护区改写、
metadata 改写、shape/dtype 错误和非有限值会失败。逐 token 与 DSpark 一致仍须整模型验证。

单卡性能候选对照，两侧使用相同 mode，分别运行 mode=1/2：

```bash
task-submit --device auto --max-time 1800 \
  "bash $PWD/tests/pypto_test/run_csa_single_layer.sh $PWD/tests/pypto_test/results/timing_b16_mode2 --batch 16 --history 8192 --variant performance --weight-nz-mode 2 --atomic-add 0 --deterministic-level 0 --timing-iters 20 --profile"
```

`--deterministic-level` 默认 1，用于精度诊断，并同步设置 HCCL 确定性；性能配置显式传 0。
`--timing-iters` 默认 0，不增加原有诊断的采样。启用后每侧先预热 5 次，再采完整图重放区间；
默认 `--timing-metadata reuse` 按同一步第二个 CSA 层计时：PTO 复用前层生成的两组
compact metadata，Native 保留其实际逐层生成路径。固定使用 `model.layers.2` 的正式权重和
合成输入/历史，模拟 metadata 已准备好的状态；不宣称是整模型第二层激活快照。
`--timing-metadata produce` 另测首层在区间内生成 metadata 的成本，不混入主结果。
状态恢复、输出毒化与数值比较均不在区间内。图外 Event 包住一次 replay，
包含可能的图派发间隙，并检查时间戳逐次更新；
当前 torch_npu 的图内 Event 不随 replay 更新，不能用于这一测量。
`--profile` 在计时之后每侧独立采一次，核对设备首末任务和热点，不混入无 profiler 的采样。
结果在 `timing` 中列 p50/p95、全部样本和图相对 eager 的逐元素差异；默认 atomic 路径不要求
跨运行逐 bit 一致，但 metadata/保护区、输出完整性和有限值仍严格检查。
此单卡数据只筛选候选；16 卡完整模型的最终区间、token、DSpark 与层误差须另外验收。

需要回放时先查看 `dsv4_csa_single_card_bench.py --help`；
精度版/性能版、ND/NZ 和输入来源必须明确。格式 29 权重保存原始 NZ 字节，
回放重建基础格式承载张量，不把 Tensor.cpu() 的逻辑解码当作原始快照。
当前四张根几何对齐 Native；旧 schema=2 转置权重按明确形状迁移，未知形状直接拒绝。
只运行受改动影响的测试；清单记录各项已完成的 CPU/设备证据，不为清理文件重复上卡。

## 保留的输入和证据

- `results/csa_baseline_20260926/native_b4h8192_precision_nd_v2/`：当前单卡报告与调用前 schema=2 快照。
- `results/csa_baseline_20260926/native_b4h8192_performance_nd/` 与 `native_b4h8192_performance_fixed/`：性能版默认/固定规约对照、Top-K 集合诊断及 Native QLI 输入；仍为 MEASURED。
- `results/csa_baseline_20260926/native_b4h8192_precision_nz2/`：两侧 mode=2 的真实布局及固定规约图重放证据。
- `results/csa_baseline_20260926/native_b4h8192_performance_nz1/`：性能版 mode=1 的真实布局及固定规约图重放证据。
- `results/csa_baseline_20260926/native_b16h8192_performance_nz1/`：目标 B16 形状的单卡同初态和图内容更新检查；跨实现数值仍为 MEASURED。
- `results/csa_baseline_20260926/model_b16h8192_nz1_fixed/`：同 mode=1、固定规约的正式权重 16 卡基线，24576 token 和 DSpark 统计一致；不包含层误差与部署性能验收。
- `results/csa_baseline_20260926/nz_native_single_card/`：两版 B4/S6/H8192、atomic=0 的 ND/NZ 逐元素相同、图重放/保护区及 Native 权重地址复用证据；跨实现数值仍为 MEASURED。
- `results/csa_baseline_20260926/nz_layout_contract/`：Native/pypto-lib 物理字节与快照合同、同一 Native NZ 存储的消费者对照。
- `results/csa_baseline_20260926/nz_native_b16_timing/`：B16/S6/H8192、默认 atomic、Native 确定性关闭的两侧 mode=1/2 完整图区间；`following_mode1/2` 为第二层主口径，`mode1/2` 单独保留首层成本，另有 mode=2 PTO 泳道。单卡趋势，不是最终验收。
- `results/csa_baseline_20260926/nz_native_edges/`：性能版 B1/H255、精度版 B5/H32767 的 ND/NZ 与图边界检查；另保留 B1 短上下文两版数值差异，性能版较大误差尚待归因。
- `results/csa_baseline_20260926/nz_native_padding/`：两版 mode=2、B4/H4095 固定规约图在 4→3→1→4 个有效请求下的输出、状态及保护区检查。
- `results/csa_baseline_20260926/perf_qproj_upstream/`：性能版 NZ Q 展开优化，第二层完整区间 p50 843.53→817.22 μs；保留单卡尾块检查、泳道及未保留候选的精简记录。
- `results/csa_baseline_20260926/model_b16h8192_nz2_perf_qproj/`：上述候选同 mode=2、atomic=1 的正式 16 卡看护，24576 token 与 DSpark 统计一致，含实际配置和 PTO 图路径证据；不是整模型性能验收。
- `results/csa_baseline_20260926/toolchain/`：当前版本记录、最终编译及 11 项标量 API 回归日志。
- `results/release_offline_pd_20260923/`：7 组正式权重 bank，供后续整模型复用，见离线 P/D 说明。
- `results/cann90_20260921/tdiv_high_precision_repro_v1/`：未关闭的 A3 TDIV 能力问题证据；版本范围见复现说明。

冗余、过时用例、重复快照、旧 profile 与失败重试记录已删除。
清理结果时同步删除失效引用；已提交的历史通过 Git 查看，不再维护旧交接目录或第二份操作说明。
**验证日志长期保留并按阶段追加**，是上述清理规则的例外；旧结论只适用于当时的配置与验证范围。
