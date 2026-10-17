# QK 独立行 Tile 加载流水：CPU 负结果，停止上卡

2026-10-17。本次仅完成源码变换和完整 CSA 的 CPU lowering，没有启动 NPU，没有修改生产代码、共享 helper/driver、冻结基线或依赖仓。

**结论：当前 `stage=2` 写法不成立，不建议上卡，也不扫描 stage。** 它没有把相邻两行的 MTE2 加载提前，反而在每个有效行上增加了一次 1 KiB UB→UB 拷贝及其等待。更关键的是，现选 B16 基线原本就直接把每行 TLOAD 写到目的 Tile 的不同子行，生成 C++ 的 64 行循环内部没有逐行显式 wait。原先将“逐行 gather”推断为“每行必须等上一行完成”的假设，不能成立。

这只关闭本次具体变换。它不证明所有独立行流水结构都不可行，也不把 C++ 的提交顺序当成设备吞吐或耗时。

## 实际输入合同与变换范围

真实冻结函数是 `qk_pv_aiv`，在完整入口中为 AIV 的 `stage_30`。本次准备/完整 lowering 使用 128K B16/S6、`qk_vector_indices=True`、`row_dependencies=True`，在原 `qk_index_cache` 地址计算之后接入新变换。

**短 B32 当前策略是 `qk_vector_indices=False`。** 本报告不把长 B16 的缓存组合称为短 B32 基线。通用变换接口同时支持 `raw_scalar` 原地址计算，已完成该版本 AST 合同验证，保存在 `raw_transform_only/`；没有为 raw 版本再做完整 lowering。

[变换入口](row_load_pipeline.py)：

```python
changed_catalog, audit = use_row_pipeline(schedule, catalog, output_dir)
```

接口只改压缩 KV 的 64 行循环：原 `gather_row(dst, src, [row,0], [physical_row,0], [1,512])` 变成独立 `[1,512]` `tile.load`，再 `tile.assemble(dst, row_tile, [row,0])`；循环从 `pl.range` 改为 `pl.pipeline(..., stage=2)`。

以下合同保持原样，并有 AST 断言：

- Top-K 的原始索引、页表/物理行地址计算、原 `raw>=0` 有效性条件。
- 无效行保留初始零值，不读取无效地址；按原行号 assemble。
- SWA 分支、数学计算、QK AIC、输出写回与所有 FFTS 调用。
- 每轮 64 行、每行 512 个 BF16、原循环 carry；不新增 GM 分配。

[准备入口](prepare.py) 仅在自身 Python 进程内串联现有地址变换，不编辑共享 Python 文件。

## 完整 lowering 的实际证据

[编译结果](cpu_v1/result.json) 的 `full_codegen_passed=true`，`device_execution=false`。包括完整单 SPMD CSA，不是另写的简化 QK kernel。没有 CCEC 设备运行、精度或性能结论。

自动审计：[audit_lowering.py](audit_lowering.py)、[结果](lowering_audit.json)。

| 检查项 | 结果 |
|---|---|
| 软件流水展开 | 每次处理两行 |
| 两个行 Tile 的实际 UB 地址 | 都为 `[131072,132096)` |
| 相邻行操作顺序 | `TLOAD → wait → TMOV → TLOAD → wait → TMOV` |
| 下一行是否在上一行 assemble 前提交 | 否 |
| 每个有效行增加的 UB→UB 数据量 | 1024 字节 |
| 64 行全部有效时，单 AIV 半块增加的拷贝 | 64 KiB |
| 一个 query 的 512 个压缩候选全有效时，两 AIV 合计增加的拷贝 | 512 KiB |
| GM 读取字节数 | 未减少 |

内存规划的最高已分配结束地址（字节）：Vec **163872**，Mat/L1 **475136**，Left **65536**，Right **65536**，Acc **131072**。这是完整生成程序的静态地址上界，不是运行时采样的内存峰值。没有溢出，但两个新增行 Tile 并未形成两个同时存活的加载槽；其相同地址不应被解释成成功实现了双缓冲。QK 的 AIC/L1 tile 形状没有改动。

候选完整生成 C++ 的关键位置（路径在审计 JSON 中）：

- 第 **48807** 行：第 0 行 `TLOAD`。
- 第 **48816–48817** 行：等待 MTE2→V，然后 `TMOV` 到目的子行。
- 第 **48897** 行：此时才提交第 1 行 `TLOAD`。
- 第 **48906–48907** 行：第二次等待与 `TMOV`。

摘录：[candidate_generated_loop.cpp.txt](candidate_generated_loop.cpp.txt)。两行之间还存在 V→MTE2 的同步，不能把源码的 `stage=2` 当作已经发生重叠。

基线摘录：[baseline_generated_loop.cpp.txt](baseline_generated_loop.cpp.txt)。来自 `b16_cached_row_deps_v1/build_output/entry_q0pmqift/ptoas/csa_static_group.cpp`：第 **48777** 行的 TLOAD 直接写目的子行；整个 64 行循环没有显式 `wait_flag` 或 `pipe_barrier`，块末写回前才等待 MTE2→MTE3。这里只证明生成代码没有逐行加载完成等待，实际 DMA 排队深度、发射吞吐与访存延迟仍须设备证据。

## 为什么没有形成预期流水

两行的有效性 `if` 分别把本行 `load + assemble` 包在一起。流水展开后仍是两个完整条件块，IO 重排没有跨条件块提取两次 load；每个 load 的结果立即被 assemble 消费。因生命周期不重叠，内存规划也把两个临时行 Tile 合并在同一 UB 地址。

本地 PyPTO 的 `src/ir/transforms/canonicalize_io_order_pass.cpp` 将普通 `AssignStmt` 的 load 分类为 Load，递归重排每个 `SeqStmts`；但本次 `IfStmt` 仍作为整体留下。这与产物一致。这里没有声称这是编译器正确性错误，也没有修改编译器。

因此，这个写法同时具有两点不利因素：没有消除基线已有的等待（基线循环中并无该等待），却新增了每行 assemble 所需的 V 流水数据移动及同步。

## 若以后重开，最小结构要先满足什么

只能作为后续结构建议，本轮没有实现：显式把同一对行的**有效性判断/受保护加载**放在前半段，再在后半段按原行顺序分别 assemble，并让两份行 Tile 的生命周期重叠。仍须保留每行原有效性条件，不允许借无条件加载读取负索引地址；生成代码必须同时证明两份 UB 地址分离、第二次 TLOAD 在首次消费前提交、条件块边界没有提前 drain。

即使满足这些条件，也还需解释为什么能胜过基线的“直接 TLOAD 到目标子行”：仅获得两个临时槽不足以证明收益。必须有减少有效 DMA 指令数、降低地址生成成本或隐藏原有真实停顿的新依据，不能仅用“独立 Tile/双缓冲”命名来决定上卡。已有全有效块、MGATHER 等关闭方向不因此自动重开。

## 复现与边界

`prepare.py`、`row_load_pipeline.py`、`prepared_contract.json` 保留准备过程。`cpu_v1.log` 保留本次完整 CPU 编译日志。编译使用本 session 的独立 PyPTO/Simpler worktree，没有改变共享安装。

完整编译方式（路径按当前 workspace）：

```bash
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
export CSA_STATIC_PYPTO_ROOT=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016
export HCA_SIMPLER_ROOT=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/simpler-csa-core-args-256-20261016
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:$PYTHONPATH"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

随后使用冻结目录的 `bootstrap_isolated.py compile_jit_entry.py`，输入本目录的 `prepared/variants/h131072_b16/entry_jit.py`，参数 `--memory-planner PYPTO --kernel-mode`，输出到新的 CPU 目录。审计当前留存结果执行：

```bash
python audit_lowering.py
```

本次结论基于 B16 的完整产物。raw 地址版本尚未完整 lowering，更未测试其他档位。对其他规格可泛化的只有“保留的变换合同”和“当前 if 结构不会自动跨行提取 load”这一代码结构观察；不外推设备收益，不宣称达到任意一档 30% 性能目标。
