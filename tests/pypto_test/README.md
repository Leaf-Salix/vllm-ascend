# DeepSeek-V4 Flash CSA 验证入口

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
| `dsv4_csa_single_layer.py` / `run_csa_single_layer.sh` | 正式第 2 层权重、合成输入/历史的 Native/PTO 整层对照；两次同初态执行、metadata 与完整分配保护区检查，可保存调用前 schema=2 快照 |
| `dsv4_csa_single_card_bench.py` | schema=2 快照回放与性能采样；无参考记为 MEASURED，逐元素验收必须提供全部声明输出/状态的参考和容差 |
| `dsv4_csa_replay.py` / `dsv4_csa_validation.py` | 共用快照、布局/别名/初态恢复与逐元素门禁 |
| `dsv4_csa_reference_lower.py` | 当前所选精度版或性能版完整层的 CPU lowering，不执行设备，也不代表数值验收 |
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

结果为 `report.json`；`--save-case` 另存调用前快照到 `case/`。
Native/PTO 使用同一 mode。性能版使用 `--variant performance`，NZ 使用 mode=1/2；
这些是可选配置，不表示每条路径已经验收。
`--atomic-add 0` 选择固定规约诊断，`1` 保留默认 split-K atomic add；
等价环境变量为 `VLLM_ASCEND_PTO_CSA_ATOMIC_ADD`，必须在进程导入/编译算子前设置。
关闭时 QR/KV 改为单 K 分片、单写入者；这也改变累加分组，不能视为默认路径的逐 bit 参考。
配合 `--graph` 检查相同地址上的 A→B→A 输入更新；每次恢复 cache/state，
图输出与对应 eager 输出逐元素精确比较，并检查 metadata 和保护区。
当前图用例固定形状与 metadata，padding/档位切换仍须按清单继续验证。
零容差差异用于诊断，算术差异本身不会让该诊断伪装成 PASS；保护区改写、
metadata 改写、shape/dtype 错误和非有限值会失败。逐 token 与 DSpark 一致仍须整模型验证。

需要回放时先查看 `dsv4_csa_single_card_bench.py --help`；
精度版/性能版、ND/NZ 和输入来源必须明确，不能混用旧快照与当前 schema。
只运行受改动影响的测试；清单记录各项已完成的 CPU/设备证据，不为清理文件重复上卡。

## 保留的输入和证据

- `results/csa_baseline_20260926/native_b4h8192_precision_nd_v2/`：当前单卡报告与调用前 schema=2 快照。
- `results/csa_baseline_20260926/native_b4h8192_performance_nd/` 与 `native_b4h8192_performance_fixed/`：性能版默认/固定规约对照、Top-K 集合诊断及 Native QLI 输入；仍为 MEASURED。
- `results/csa_baseline_20260926/native_b4h8192_precision_nz2/`：两侧 mode=2 的真实布局及固定规约图重放证据。
- `results/csa_baseline_20260926/native_b4h8192_performance_nz1/`：性能版 mode=1 的真实布局及固定规约图重放证据。
- `results/csa_baseline_20260926/native_b16h8192_performance_nz1/`：目标 B16 形状的单卡同初态和图内容更新检查；跨实现数值仍为 MEASURED。
- `results/csa_baseline_20260926/model_b16h8192_nz1_fixed/`：同 mode=1、固定规约的正式权重 16 卡基线，24576 token 和 DSpark 统计一致；不包含层误差与部署性能验收。
- `results/csa_baseline_20260926/toolchain/`：当前版本记录、最终编译及 11 项标量 API 回归日志。
- `results/release_offline_pd_20260923/`：7 组正式权重 bank，供后续整模型复用，见离线 P/D 说明。
- `results/cann90_20260921/tdiv_high_precision_repro_v1/`：未关闭的 A3 TDIV 能力问题证据；版本范围见复现说明。

冗余、过时用例、重复快照、旧 profile 与失败重试记录已删除。
清理结果时同步删除失效引用；已提交的历史通过 Git 查看，不再维护旧交接目录或第二份操作说明。
**验证日志长期保留并按阶段追加**，是上述清理规则的例外；旧结论只适用于当时的配置与验证范围。
