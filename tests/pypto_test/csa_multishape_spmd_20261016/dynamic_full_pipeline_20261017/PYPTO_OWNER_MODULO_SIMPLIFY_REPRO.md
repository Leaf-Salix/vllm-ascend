# 输出组首项计算：正数补偿模与有符号余数不一致

2026-10-17。在正式动态 CSA 输出组流水的 CPU 编译中发现，尚未向上游另提 issue。
本次未修改 PyPTO/Simpler；坏表达的完整 CSA 候选 v1～v3 未上卡。

## 结论与组件边界

源表达的取模被除数始终非负，简化后却可能为负，而 PyPTO 仍将 FloorMod
生成成 `arith.remsi`。最小例中生成的 C++ 按有符号余数执行，与源表达不等价。
这发生在 **PyPTO 简化和 Scalar 运算 lowering 的语义衔接**；PTOAS 按已收到的
`arith.remsi` 输出有符号 `%`，没有证据表明是硬件能力或 Scalar DCache 问题。
本记录只证明当前隔离 worktree 的行为，不声称核对过最新上游版本。

## 最小表达及证据

24 个 worker，每输出组 8 个逻辑任务。希望找到该组起点之后第一个属于本核的编号：

```python
group_first = group * 8
first = group_first + (worker + 24 - group_first % 24) % 24
```

`0≤worker<24` 时，模运算被除数最小为 1，源表达没有负数取模。
真实 `05_after_Simplify.py` 删除了补偿项：

```python
first = group_first + (worker - group_first % 24) % 24
```

其 PTO 为两次 `arith.remsi`，最终 C++ 中除数为 `const int64_t = 24`。
取 `group_first=8, worker=0`：源表达为 24；简化后 signed remainder 为 -8，
结果变成 0。若用于任务分配，会把本核前一组的任务错当成本组任务。

最小脚本仅编译，不运行 NPU。随后用宿主 C++ 执行生成的 Scalar 函数体原文，
只将设备修饰符及缓存 intrinsic 置空，在 8×24 个组合上得到：

| 写法 | 192 组合的错误数 | group=1 / worker=0 |
|---|---:|---:|
| 正数补偿模，经当前编译链 | 56 | 0，期望24 |
| 非负分子整除公式 | 0 | 24 |

宿主结果证明生成算术的差异，不是 NPU 缓存、同步或完整 CSA 的测试结果。
原始证据：[result.json](owner_modulo_codegen_v1/result.json)、
[生成 Scalar 的宿主核对源码](owner_modulo_codegen_v1/generated_scalar_host.cpp)。

## 独立复现

脚本：[owner_modulo_codegen_repro.py](owner_modulo_codegen_repro.py)。
在仓库根目录激活既有隔离依赖，然后执行以下命令，输出目录必须不存在：

```bash
source ../env-dsv4-0251rc1.sh
export CSA_STATIC_PYPTO_ROOT="$(cd ../.worktrees/pypto-csa-static-fixes-20261016 && pwd)"
export HCA_SIMPLER_ROOT="$(cd ../.worktrees/simpler-csa-core-args-256-20261016 && pwd)"
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python tests/pypto_test/csa_static_spmd_feasibility_20261015/bootstrap_isolated.py \
  tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/owner_modulo_codegen_repro.py \
  --output /tmp/csa-owner-modulo-repro-new
```

脚本分别编译两种写法，保存 pass IR/PTO/C++；宿主部分需要 `c++` 编译器。
不需要权重、NPU、Torch dispatcher 或服务。

## CSA 中的处理及修复建议

正式源码采用：

```python
first = worker + ((group_first + workers - 1 - worker) // workers) * workers
```

合法 worker 范围使除法分子始终非负；实际每核工作集与原设计相同。
O_B 的 NZ 路径从包含组起点的全局 wave 开始，只跳过首 wave 中落在前组的项，
保留 `work=worker+wave*workers` 的显式非负证明。没有把 NZ 权重当成 ND 使用。

组件修复时应统一 FloorMod 的源语言、简化规则及后端语义：若保留 signed remainder，
不能在无法证明新被除数非负时删除正补偿；若接口要求 Python floor-mod，后端须正确
实现负被除数。建议回归正补偿模、负被除数、嵌套模及带动态 group 的首项公式，
同时核对生成算术，而不仅比较简化前后的数学整数表达。
