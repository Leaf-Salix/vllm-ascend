# CSA 历史静态单 SPMD 参考

按 2026-10-18 用户要求，将已验收静态源码在 `ops/pypto` 下独立保留，供持续比较调度、
依赖、Tile 复用与流水设计。这里是历史参考，正式动态实现位于相邻的
`deepseek_v4_flash_csa_spmd/`。本目录不自动注册或切换服务入口。

每份入口都把完整 CSA 的 incore 数学融合为一个 24 AIC + 48 AIV MIX 任务，包括
HC_pre、Q/K/V 与压缩投影、Indexer/Top-K、稀疏 Attention、输出投影和 HC_post。
数学形状和逐核编排按规格冻结；长度、位置、页表与 cache/state 仍从 device Tensor 读取。
这种 device metadata 动态性不等于当前目标要求的源级动态 shape。

## 阅读入口

| 历史长度 | Batch / S | PTO 入口 | 已记录静态 L1 中位数（μs） |
|---|---|---|---:|
| 128K | 4 / 6 | [entry_jit.py](variants/h131072_b4/entry_jit.py) | 541.92 |
| 128K | 8 / 6 | [entry_jit.py](variants/h131072_b8/entry_jit.py) | 641.56 |
| 128K | 16 / 6 | [entry_jit.py](variants/h131072_b16/entry_jit.py) | 807.16 |
| 128K | 24 / 6 | [entry_jit.py](variants/h131072_b24/entry_jit.py) | 1121.12 |
| 8K | 16 / 6 | [entry_jit.py](variants/h8192_b16/entry_jit.py) | 621.96 |
| 8K | 24 / 6 | [entry_jit.py](variants/h8192_b24/entry_jit.py) | 见 manifest 原始记录 |
| 8K | 32 / 6 | [entry_jit.py](variants/h8192_b32/entry_jit.py) | 905.72 |

8K/B24 属于旧验收集合，保留历史价值；当前性能目标已换成 128K/B28。
128K/B28 的静态尝试在旧 driver 的模块导入阶段失败，未执行数学，没有性能数据；
按用户要求不修复、不将它加入已验收归档。

2026-10-18 新增同进程、同根输入的 B16/S6 五窗对照：原多 task **869.78 μs**、
动态单 SPMD **889.40 μs**、本目录对应静态版本 **792.78 μs**。
输出、Top-K、cache/state、保护区和图重放均通过。该测量使用搬运前的等价数学源码；
本次归档仅改相对 import 和增加阅读说明，七份数学 AST 均保持一致。

随后对本目录进行独立复现，任务 `task_20261018_095114_228736420852` 已 exit=0：
128K/B16/S6 的 A→B→A 输出、Top-K/score、完整 cache/state、Native/arena 保护区、
72 核进度与只读调度表、图重放全部通过。三窗完整 L1 中位数为原多 task
**864.90 μs**、本目录静态单 SPMD **782.66 μs**。这轮仅验证搬迁后的 B16；
其余六档保留原验证证据，不声称已经逐档重测。
结果在 `tests/pypto_test/csa_static_reference/h131072_b16_v1/result.json`。

## 文件合同

- `variants/*/entry_jit.py`：各档完整冻结 PTO 入口；按源码顶部总览阅读。
- `variants/*/entry.bindings.json`：入口参数顺序、根/派生 Tensor、shape/stride/offset 合同。
- `variants/*/layout.json`：原工作区分区、调度 `records/waits`、同步协议及每核末次发布序号。
  运行相关字段保持原值；重复的逐成员诊断列表继续保留在 tests 原始 layout 中。
- `bindings/*_prepared.json`、`prepared_arguments.py`：原 Native 根 Tensor 的一次性别名绑定，
  原 NZ 权重保留原对象。仅显式构造时分配独占工作区；不能在每次图重放时清零进度。
- `signals.py`：原 A3 notify/wait 适配，保留缓存发布和业务流水约束。
- `wait_codegen.py`：历史 cached100 等待的生成源码变换。导入入口不会自动安装此回调；
  `reference.py` 在首次编译期间安装并恢复回调，核对实际 CCEC 源码，不能把默认等待
  误标为 cached100。测试不再承担算子编译实现。
- `reference_manifest.json`：每档选择、原始源码/结果路径及完整来源。没有权重或编译二进制。

## 独立调用与复现

完整调用入口为 [reference.py](reference.py) 的 `prepare_static_csa`，其加载路径、
数学入口、参数与调度表均位于本包，运行时不读取历史 tests 目录。

```python
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_static_reference.reference import prepare_static_csa

# native_args是NativeCSACall已有的同一批device Tensor；在capture前准备一次。
op = prepare_static_csa(native_args, history=131072, batch=16)
op()  # 已初始化pypto.torch环境；首次完整编译，之后复用。
```

仓库根目录下可直接提交独立测试。它加载正式第4层权重，检查A→B→A完整输出、
Top-K/score、cache/state、进度和保护区，再检查图重放并测量三窗完整L1：

```bash
task-submit --device auto --max-time 1800 \
  'bash tests/pypto_test/csa_static_reference/run.sh h131072_b16_example --batch 16 --history 131072 --l1-windows 3'
```

测试代码在 `tests/pypto_test/csa_static_reference/`。它复用通用fixture与原多task
PTO作为参考，不依赖历史静态源码或生成产物。`--l1-windows 0`只做功能和图重放；
可改为表内其他H/B，S固定为6。该入口需要既有A3依赖环境及扩容到256入参的
独立Simpler/PyPTO；测试脚本支持用 `CSA_STATIC_PYPTO_ROOT`、`HCA_SIMPLER_ROOT`
指定兼容安装，不修改共享依赖。

过程和性能证据仍在仓库根的 `tests/pypto_test/csa_multishape_spmd_20261016/`；
统一优化记录为相邻的 `../DSV4_CSA_SPMD_OPTIMIZATION_LOG.md`。
这些静态结果用于定位差距，不计入七档动态单 SPMD 降低至少 30% 的目标完成度。
