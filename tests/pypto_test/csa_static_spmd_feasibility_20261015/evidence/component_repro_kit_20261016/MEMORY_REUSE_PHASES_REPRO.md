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
| Indexer 单段 | 代码生成及 PTOAS C++ 生成通过 | 未测 |
| O-proj 单段 | 代码生成及 PTOAS C++ 生成通过 | 未测 |
| 两段融合，13 参数 | Right 81920 B > 65536 B | canonical greedy 找不到容量内布局 |

结果目录位于 `../results/csa_static_spmd_feasibility_20261015/`：

- `memory_indexer_PYPTO/result.json`
- `memory_oproj_PYPTO/result.json`
- `memory_pair_compact_PYPTO/result.json`
- `memory_pair_compact_DSA_RP_v2/result.json`

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
