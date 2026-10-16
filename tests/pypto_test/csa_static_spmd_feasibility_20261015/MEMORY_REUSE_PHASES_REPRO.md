# 两个独立可编译 AIC 阶段融合后的片上内存分配复现

日期：2026-10-16。此问题独立于 while 的 #3009，也独立于 Simpler 的 32 Tensor
容量：下面的融合复现只需要 13 个 Tensor 参数，且全部是 CPU 编译。
未修改依赖仓库、未降低 tile 尺寸或 pipeline stage。

## 输入及对照结果

从当前 B4 静态原型直接抽取两个阶段：

- `indexer_score_topk_native_pair_aic`，原 AIC batch 13；独立入口 10 个 Tensor。
- `proj_a_mm`，原 AIC batch 15；独立入口 4 个 Tensor。
- 按原先后顺序放进一个 AIC 函数，两者共享 records 参数，合计 13 个 Tensor。

抽取脚本：[build_memory_probe.py](build_memory_probe.py)。保留原数学、分片读取、
FFTS 操作、pipeline 和本核完成边界；删除进度发布/等待及无关参数。
三个自包含 PyPTO 文件在 `static/memory_indexer/entry.py`、
`static/memory_oproj/entry.py`、`static/memory_pair_compact/entry.py`。
无需模型权重、Native 服务、真实 Tensor 数据或 NPU 即可编译。

| 编译输入 | PYPTO | DSA_RP |
| --- | --- | --- |
| Indexer 单段 | 代码生成及 PTOAS C++ 生成通过 | 代码生成及 PTOAS C++ 生成通过 |
| O-proj 单段 | 代码生成及 PTOAS C++ 生成通过 | canonical greedy 找不到容量内布局 |
| 两段融合，13 参数 | Right 81920 B > 65536 B | canonical greedy 找不到容量内布局 |

结果目录位于 `../results/csa_static_spmd_feasibility_20261015/`：

- `memory_indexer_PYPTO/result.json`
- `memory_oproj_PYPTO/result.json`
- `memory_pair_compact_PYPTO/result.json`
- `memory_pair_compact_DSA_RP_v2/result.json`
- `memory_indexer_DSA_RP_v1/result.json`
- `memory_oproj_DSA_RP_v1/result.json`

`memory_probe_PYPTO`、`memory_probe_DSA_RP` 是尚未裁掉无关根参数的早期复现，
也失败。`memory_pair_compact_DSA_RP` 是本地编译脚本的 PassContext 参数冲突，
**不算组件失败证据**；该调用方式修正后的结果以 `_v2` 为准。

## 分配审计

`static/memory_probe_summary.json` 汇总 MemoryReuse 后各物理复用槽的最大字节数。
这是 IR 中的分配量，不是运行时测得的瞬时峰值。

| 输入 | Left / KiB | Right / KiB | Acc / KiB | Mat / KiB |
| --- | ---: | ---: | ---: | ---: |
| Indexer 单段 | 60 | 64 | 100 | 112 |
| O-proj 单段 | 64 | 64 | 64 | 256 |
| 两段融合 | 80 | 80 | 100 | 256 |

两个原阶段各自可以放进目标资源，合并后 Left/Right 复用槽总量变成 80 KiB。
应检查不同阶段的 allocation 生命周期、复用分组、pipeline 硬约束与候选布局。
**单段通过和顺序关系提示存在可复用空间，但不是完整硬件可行性证明**；仍须
正确处理原流水和同步，不能直接删除编译器约束或把失败当作硬件必然不支持。

## 脱离 vLLM 的 CPU 编译

在组件维护者正常可用的 PyPTO/PTOAS 环境中，以 `entry.py` 为输入：

```python
from pathlib import Path
import pypto.language as pl
from pypto import ir, passes
from pypto.runtime import RunConfig

source = Path("static/memory_pair_compact/entry.py")
program = pl.parse(source.read_text(), filename=str(source.resolve()))
compiled = ir.compile(program, **RunConfig(
    platform="a2a3",
    memory_planner=passes.MemoryPlanner.PYPTO,  # 对照时改为 DSA_RP
    save_kernels=True,
    save_kernels_dir="/tmp/csa_memory_pair_repro",
    dump_passes=True,
).compile_kwargs())
```

将输入分别替换为两个单段入口即可得到对照。不要调用 compiled 对象或 `.load()`。
这两个段缺少其前序数据生产者和 MIX 的 AIV 半边，独立上卡不属于此复现的范围。

这里没有把两段小例子的 PASS 当成完整 CSA PASS；目标是让分配问题可直接接手。

## 三参数的顺序阶段分区复用小例子

[repro_phase_partition.py](repro_phase_partition.py) 进一步去掉 NZ、pipeline、模型、
进度通信和多核。一个 AIC 函数的第一阶段同时使用 48 + 12 KiB Left，结束后
第二阶段同时使用 32 + 32 KiB Left；阶段之间有 `pl.system.fence()`。
每个 Tile 都参与 matmul 和输出写回，没有为构造生命周期添加虚假使用。

| planner | 实际结果 |
| --- | --- |
| PYPTO | Left 81920 B > 65536 B |
| DSA_RP | 完整代码生成通过；Left 地址为 a=0、b=49152、c=0、d=32768 |
| PTOAS | Left 655360 bit > 524288 bit，分配失败 |

证据分别为 `phase_partition_PYPTO_v1`、`phase_partition_DSA_RP_v1`、
`phase_partition_PTOAS_v1` 的 `result.json` 和 `kernel/passes/`。
DSA_RP 的实际 C++ 为 `phase_partition_DSA_RP_v1/kernel/ptoas/body.cpp`。

本地 `memory_reuse_pass.cpp` 的默认分组按 Tile 大小排序，复用槽按最大成员
固定容量。已有 subrange 尝试，但不能把一个 48 KiB 槽扩大成两段连续的
32 + 32 KiB；于是第二个 32 KiB Tile 另开槽，总容量变为 48 + 32 = 80 KiB。
`docs/en/dev/passes/36-memory_reuse.md` 也说明了该分组方式。
这证明小例子的失败不是 A3 物理容量不足；DSA_RP 已给出可放进 64 KiB 的布局。
**它不证明完整 CSA 已有合法布局**：真实 O-proj 单段在 DSA_RP 下仍失败，且
不同 planner 会影响 AutoTileMatmulL0 的选择，需要继续核对实际分配约束。

## 固定原分块后，真实两段的 DSA_RP 分配可以通过

[probe_fixed_tiles_dsa.py](probe_fixed_tiles_dsa.py) 读取 PYPTO 路径保存的
`36_after_MaterializeSemanticAliases.py`，只执行 `allocate_memory_addr`，
在其 PassContext 选择 DSA_RP。没有修改 IR、Tile 大小或流水阶段。
这是一项隔离分配因素的 CPU 诊断，**不是完整编译，也不是生产入口的解决方案**。

| 固定 Tile 的输入 | Left / KiB | Right / KiB | Acc / KiB | Mat / KiB | 分配 |
| --- | ---: | ---: | ---: | ---: | --- |
| O-proj | 64 | 64 | 64 | 256 | 通过 |
| Indexer + O-proj | 64 | 64 | 100 | 256 | 通过 |

真实代码的两段合并在保持原分块的条件下，已有满足容量的地址布局。
证据为 `oproj_fixed_tiles_dsa_v1/allocated.py`、`pair_fixed_tiles_dsa_v1/allocated.py`
及 `fixed_tiles_dsa_summary.json`；空间上界取实际 MemRef 的 offset + size。
复现命令：

```bash
python tests/pypto_test/hc_pre_layer_integration_20261012/bootstrap.py \
  tests/pypto_test/csa_static_spmd_feasibility_20261015/probe_fixed_tiles_dsa.py \
  --source tests/pypto_test/results/csa_static_spmd_feasibility_20261015/memory_pair_compact_PYPTO/kernel/passes_dump/36_after_MaterializeSemanticAliases.py \
  --output /tmp/csa_pair_fixed_tiles_dsa
```

直接选择 DSA_RP 的原始源码编译与上面的固定 Tile 诊断不同：
O-proj 在前者中生成两个同时存活的 128×256 BF16 Left Tile，每个 64 KiB；
`LowerPipelineLoops` 的 IR 顺序是先产生两组 Left/Right，再执行两个 matmul_acc，
所以需要的 Left 已超过 64 KiB。PYPTO 原路径则使用 128×128、每块 32 KiB。
不能要求分配器把同时存活的两块 64 KiB 强行放到一块 64 KiB 中。

建议 PyPTO 组件分开处理：

1. 默认 MemoryReuse 的固定槽大小导致阶段之间无法重新分区，3 参数例子可直接验收。
2. AutoTileMatmulL0 的策略随 memory planner 改变，应计入调用方已有 pipeline 的
   同时存活资源，或提供保留原分块策略、仅更换分配算法的正式配置。不要在应用侧
   临时篡改 PassContext 或删除硬约束绕过。
3. 先验收上面真实两段的源码完整编译，再验证完整 CSA；后者还有独立的区域 SSA
   代码生成问题。不能把本次“地址分配通过”扩写为“整层编译通过”。

对历史 `static_b4_cpu_v15` 的中间 IR 做相同诊断时，打印源码含已有的
`progress__ssa_v1__FREE_VAR`，在重新解析时失败。该记录为
`full_b4_fixed_tiles_dsa_v1`，不构成完整 CSA 的 DSA 分配成功或失败证据。
