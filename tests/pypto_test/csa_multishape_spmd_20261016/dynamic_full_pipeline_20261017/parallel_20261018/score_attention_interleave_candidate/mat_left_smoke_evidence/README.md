# 固定数学 Mat/Left helper 参数与循环状态 CPU 检查

2026-10-18。v2通过CPU lowering/codegen/PTOAS，证实固定full-valid `Mat[384,64] FP16` 与 `Left[16,384] FP16` 可经当前隔离Tile注解overlay进入JIT Inline helper，再以tuple返回并在动态循环中传递。没有提供样例Tensor，没有设备执行或数值/性能测试。这不是整层Score/Attention资源规划通过。

本地原始目录：

`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-indexer-20261018/tests/pypto_test/jit_tile_mat_left_smoke_20261018`

依赖使用相邻 `jit_tile_parameter_overlay_20261018/pypto_overlay`；复用旧隔离依赖二进制，无新依赖修改、安装或重建。

| 用例 | 结果 | 范围 |
| --- | --- | --- |
| `mat_left_kernels.py` → `cpu_v1` | specialize FAIL | 将GM output加入显式 `pl.range init_values` 后，以 `out_state` 别名调用helper；JIT报 `missing inferred tensor metadata for parameter 'output' of 'one_step'`。还未进入Mat/Left lowering，不能称Tile失败。 |
| `mat_left_kernels_v2.py` → `cpu_v2` | compile PASS | 按主会话指示只取消GM Tensor的显式carry，直接传入并接回原output变量；Mat/Left显式carry及helper数学保持。 |

v2主函数以Mat create和固定全valid权重load/move初始化句柄；循环 `min(T//384,2)` 中helper先gather完整384行到Mat，再Mat→Right、Left×Right、store16行。T通过动态Tensor轴提供；T小于384时循环为空，没有读取未初始化Mat或越界加载。上述是源码及生成物分析，未执行T边界数值用例。

`cpu_v2/kernel/ptoas/mat_left_kernel.pto` 中：

| 空间 | 实际地址与容量 | 证据 |
| --- | --- | --- |
| Mat panel | 0，48KiB | 第14行；返回panel第36行仍同地址 |
| Mat weight | 49152，12KiB；Mat末端60KiB | 第15行 |
| Left coefficient | 0，12KiB | 第18行；返回coefficient第37行仍同地址 |
| Right | 0，48KiB | 第28行 |
| Acc | 0，4KiB | 第30行 |

PTO第2行保留runtime `%arg3`；第20–22行由T决定循环。两次helper返回只建立同地址句柄，未新增整Tile复制。C++保留TLOAD、TMOV、TMATMUL、TSTORE和相应局部事件；不能据此保证任意TileView、动态valid、更多并存状态或复杂嵌套调用均通过。

两次命令：

```bash
bash run_cpu.sh --output <本地目录>/cpu_v1
bash run_cpu.sh --module mat_left_kernels_v2 --output <本地目录>/cpu_v2
```

`archive_evidence.py` 只复制必要源码、失败/成功文本、小PTO/C++到主候选 `mat_left_smoke_evidence`。完整pass dump与依赖二进制不复制，仅记录绝对路径和大小。v1失败证据未覆盖。没有进一步编译。
