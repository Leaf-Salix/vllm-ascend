# JIT 单 MIX Group：共享 InOut 参数在 ABI 统一后变为 free variable

日期：2026-10-16。本例无需模型、设备、while、NZ 或大参数，只有 5 个 Tensor。
问题首次出现于 PyPTO `ExpandMixedKernel`，尚未进入 PTOAS；与 Simpler 容量无关。
本会话没有修改 PyPTO 源码。源码基线为本机调试分支 `cb2484471`，使用该目录已有
`build/python/bindings/pypto_core.cpython-310-aarch64-linux-gnu.so`，不是新构建干净上游。
共享目录存在其他会话的修改，本结论仅针对实际加载的本地构建。

## 对照

- [repro_jit_group_inline.py](repro_jit_group_inline.py)：AIC 写 c，AIV 写 y，两个成员
  没有共同 InOut；完整 CPU 编译通过，生成一次 24 核组 MIX，sync_start 保留。
- [repro_jit_group_shared_inout.py](repro_jit_group_shared_inout.py)：AIV 额外接收并
  原样返回 c；它仍只写 y，核间没有通过 c 交换计算数据。wrapper 正确接回 InOut
  返回值，前端检查通过，但 `ExpandMixedKernel` 之后产生 free variable。
- 全量 B4 同源 JIT 也受影响，错误为 `Tensor view not found for parameter:
  progress__ssa_v1`；结果目录 `static_b4_jit_cpu_v6_alias`。不把小例子通过当作全量通过。

关键结构：

```python
@pl.jit.inline
def group_wrapper(a, b, c, x, y):  # 完整带类型签名见可运行文件
    with pl.cluster(name_hint="csa_group_probe"):
        c = cube(a, b, c)
        c, y = vector(c, x, y)
    return c, y

@pl.jit
def entry(a, b, c, x, y):
    with pl.spmd(24, sync_start=True):
        c, y = group_wrapper(a, b, c, x, y)
    return c, y
```

直接传调用前的 c 会被 InOutUseDiscipline 正确拒绝。不能用旧变量、弱化 InOut
声明或新增无用 block_idx 读取绕过。这里也没有拆成两次 runtime 提交。

## CPU 复现

当前工作区命令；通用组件环境可去掉 bootstrap 与 `dsv4_csa_env.activate()`：

```bash
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python tests/pypto_test/hc_pre_layer_integration_20261012/bootstrap.py \
  tests/pypto_test/csa_static_spmd_feasibility_20261015/repro_jit_group_shared_inout.py \
  --output /tmp/csa_group_shared_inout
```

实际失败证据：`../results/csa_static_spmd_feasibility_20261015/jit_group_shared_inout_v1/`。
成功对照：`jit_group_inline_v1/`。Meta 注册和返回别名检查通过于
`jit_group_inline_meta_v1/`；它不等于设备执行。

## 第一处 IR 破坏

`kernel/passes_dump/24_after_LowerAutoVectorSplit.py` 中 AIV 成员参数和返回均为
`c__ssa_v0`，Group 的调用参数为前一个成员的返回别名 `c__ssa_v1`。

`25_after_ExpandMixedKernel.py` 统一 AIC/AIV 签名之后：

```python
def vector(a__ssa_v0, b__ssa_v0, c__ssa_v0, x__ssa_v0, y__ssa_v0):
    # 省略只写 y 的正常 vector 计算
    return c__ssa_v1__FREE_VAR, y_v1__ssa_v0
```

`c__ssa_v1` 既不是新函数参数，也不是其局部定义。后续 codegen 报不能物化该符号，
错误提示建议传入 Scalar，但此处实际上是 Tensor 别名，不适用这个建议。

## 建议组件修复位置与验收

本地 `src/ir/transforms/expand_mixed_kernel_pass.cpp`：

- `BuildGroupAbiAdapter`（约 2521 行）构造 Group 参数到 fresh 参数映射。
- 约 2543 行直接把 `member.inner_call.args[i]` 替换到该映射后 DeepClone。
- 当前映射只有 Group 参数，没有把成员调用返回的 Tensor 别名追溯到 Group 参数；
  所以 `c__ssa_v1` 留在新函数体内，而调用本身已被改成统一参数表。

修复应先解析 Group 内部 SSA 返回别名，沿被调函数的 return-to-param 关系追溯，
再建立 member-to-adapter 映射。可参考现有 `return_lineage_utils.h` 的接口；
不能按变量名猜测身份，也不能把所有未知值都替换成第一个参数。
无法证明为边界别名的参数应明确拒绝，保留原语义约束。

最小验收：成功对照和失败例都生成一次 MIX，24 组、双 AIV、sync_start；
pass25 无 free variable，输出别名正确；补充重排参数、tuple 返回、多级别名与
无法追溯参数的负例。之后再验证真实 132/133 参数 CSA，不预记全量 PASS。

与 #3009 的关系：同处 `ExpandMixedKernel`，但本例是 Group ABI 统一中的
Tensor 别名映射；#3009 是 while 条件的 loop-carried scalar liveness。
修复其中一个不能代替另一个的验收。
