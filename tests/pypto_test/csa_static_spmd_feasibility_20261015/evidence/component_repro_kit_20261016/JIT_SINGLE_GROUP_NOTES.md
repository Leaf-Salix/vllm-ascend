# JIT 的单 MIX Group 表达：当前核对结果

目标仍为一次 24 组 MIX 提交，不用两个提交替代。本说明记录接口表达限制，
不是宣称所有 JIT 写法均不可实现，也不要求组件必须新增某一个具体装饰器。

## 已验证的两条路径

显式 program 原型包含 `FunctionType.Group`，根入口 SPMD 只调用这个 Group。
其部分 orchestration 产物已确认只有一次 MIX 提交；完整核编译问题另见
[COMPILE_AND_RESUME.md](COMPILE_AND_RESUME.md)。

直接将它改写为 JIT 的以下结构，会在解析阶段被拒绝：

```python
with pl.spmd(24, sync_start=True):
    with pl.cluster():
        c = cube(a, b, c)
        y = vector(x, y)
```

错误为 `dispatch body launches 2 kernels; only one kernel dispatch per SPMD scope`。
源于 parser 对 dispatch body 的检查；后端 `FindFirstInnerCall` 只能取一个 callee。
不能通过新增无用 block_idx 读取欺骗该检查。

进一步检查了先用 `@pl.jit.opaque` 包裹 cluster，再让 SPMD 只调用一次 wrapper：
[repro_jit_group_wrapper.py](repro_jit_group_wrapper.py)。它是 5 参数、无模型依赖、
无 while 的 CPU 编译小例子。两个物理计算核可生成，但 orchestration 拒绝：

```text
InferFunctionCoreType expects AIC or AIV (Spmd/Group are filtered upstream);
got Opaque on function 'group_wrapper'.
```

证据：`../results/csa_static_spmd_feasibility_20261015/jit_group_wrapper_v1/`。
这说明该 opaque wrapper 不能当作已支持的 Group kernel 入口。

## 本地源码依据与接续选择

- `python/pypto/language/parser/ast_parser.py`：SPMD 的单 callee 检查。
- `src/ir/transforms/outline_cluster_scopes_pass.cpp`：一个 cluster 只允许一个
  嵌套 SPMD，不能用两个 SPMD 分别包住 AIC/AIV 再冒充一次 MIX。
- `python/pypto/jit/decorator.py`、`specializer.py`：现有 JIT 类型没有原生
  DSL Group 装饰器；extern mixed 可以生成 Group，但需已有正确的物理核产物。
- `tests/ut/jit/test_roundtrip.py` 明确将模块级 `pl.function` 调用列为 JIT
  依赖发现不覆盖的模式；不能直接把 program 的 IR 函数塞给 JIT 当作完成接入。

后续可选组件正式支持的 Group 调用表达，或在 #3009 修复后验证自动 MIX 内联
表达。若采用 extern mixed 承接 PyPTO 生成的两份物理核，仍须验证同一参数布局、
双 AIV、一次 24 组提交及 kernel runtime/当前流，不能用 program runner 替代。

目前 `entry_jit.py` 是准备中的源码，不是已通过 `pypto.torch.register` 的生产入口。
