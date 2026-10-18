# JIT Inline Tile 参数的隔离修复与最小 CPU 证据

2026-10-18。结论限定为：在独立 Python overlay 中保留 `@pl.jit.inline` 的固定数学 Tile 参数注解后，普通 Vec Tile 参数/返回，以及 full-valid roots/segments 的 `assemble → tuple 返回 → loop carry` 已通过 CPU lowering、codegen 和 PTOAS。动态 Tensor 的 T 轴保留。没有编译整层 CSA，没有运行 NPU，也没有运行数值或性能测试。

此结果足以支持后续以普通 Tile 参数和显式返回组织小型数学 helper，但不等于任意 Tile 注解均已支持。实际 Left/Mat/Acc 数学 helper、任意 `memref` / `TileView` 注解、复杂跨步资源规划都未验证。动态 valid 的 tuple loop carry、同一 helper 的 constexpr 参数作 Tile 形状仍有明确未解决边界。

本目录绝对位置：

`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-indexer-20261018/tests/pypto_test/jit_tile_parameter_overlay_20261018`

**隔离方式与修复范围**

`prepare_overlay.py` 直接复制当时隔离依赖 `/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016/python`，保留该依赖既有 Python 修复；复制时逐文件直接比较字节，记录在 `OVERLAY_PREPARATION.json`。既有两个扩展 `.so` 仅建立只读使用的符号链接，没有重建或安装。`pypto_reference` 保存未修改的同源副本，`reference/specializer.py` 保存原文件。没有改主工作区、旧依赖源码或共享安装。

最终补丁 `jit_inline_tile_annotation.patch` 只有 `python/pypto/jit/specializer.py` 的六行新增：导入 `_LangTile`；在现有额外参数注解分支中，对 `is_inline and isinstance(ann_obj, _LangTile)` 使用既有 `_fold_const_names` 保留原注解 AST。Tile 的数学 shape、dtype、memspace 因而不再变成裸参数。原 Tensor/Scalar/Array 分支、返回推导、IR inliner 和后端没有修改。

原失败链为：特殊参数保留分支只识别 Array → Tile 未进入 `extra_anns` → `_build_params` 输出裸名字 → parser 报 `Parameter 'tile' missing type annotation`。标准模块级 `pl` 绑定的 `before_v2` 是正式因果对照；不需要依赖新的 C++ 二进制才能触发或修复该入口问题。

**保留的全部尝试**

| 目录 | 结果 | 解释与范围 |
| --- | --- | --- |
| `before_v1` | specialize FAIL | 首版 fixture 把 `pl` import 放在局部作用域，未修改依赖时丢失 Tile 注解。 |
| `after_v1` | specialize FAIL | 已加 Tile 分支，但该 fixture 的 helper globals 无 `pl`，注解求值仍失败；未通过扩大依赖逻辑修复此 fixture 问题。原脚本留在 `reference/reproduce_tile_v1.py`。 |
| `before_v2` | specialize FAIL | 换成与正式 CSA 一样的模块级 `pl` 后，未修改 reference 仍明确报 missing annotation。 |
| `after_v2` | CPU compile PASS | 固定 `[8,64]` Vec Tile identity 参数/返回；调用方 valid 为 `min(T,8)`，没有 tuple loop carry。 |
| `state_v1` | specialize PASS；compile FAIL | 模块数学常量和 tuple 注解已保留；但 caller 的 valid `min(T,3/6)` 与 helper 返回签名静态 valid `3/6` 在循环边界不一致，`pipeline_input` TypeCheck 报四项错误。没有放宽验证。 |
| `inout_v1` | specialize PASS | `pl.InOut[pl.Tile[...]]` 注解保留并成功解析；未执行该用例的 CPU compile，不宣称完整支持。 |
| `constexpr_v1` | specialize FAIL | `ROWS: pl.constexpr` 是同一 helper 的形参，Tile 注解引用 `ROWS` 时初始求值无法解析，仍报 missing annotation；没有扩大修复。 |
| `state_full_v1` | CPU compile PASS | 最终六行补丁；实际 full-valid 初始化，固定 `[3,1024]` / `[6,1024]` Vec 状态，assemble、tuple 返回、动态循环传递完整通过。 |

每次尝试的 `result.json`、`generated_preparse.py`、原日志，以及失败的 `traceback.txt` 均保留。`after_v2` 在加入 `is_inline` 限定前完成；其 helper 本来就是 inline，这一限定不改变当时命中的分支。最终补丁的完整 CPU 产物以 `state_full_v1` 为准。

**生成物核对**

`after_v2/kernel/ptoas/tile_kernel.pto:2` 仍有运行时 T 参数 `%arg2`；第10行计算 `min(T,8)`，第11、14行输入与返回 Tile 都在 Vec 地址0、使用相同动态 valid。只存在原一次 TLOAD 与一次 TSTORE，没有因 identity helper 新增 TMOV 或第二份实际存储。

`state_full_kernels.py` 使用模块数学常量 `GROUP_SIZE=6`、`PAIR_WIDTH=1024`。生成的签名保留 `[6 // 2,1024]` / `[6,1024]`，解析后折叠为 `[3,1024]` / `[6,1024]`。这与引用同 helper constexpr 参数不同。

| full-state 项目 | 实际 PTO/C++ 证据 |
| --- | --- |
| 动态 Tensor 轴 | `state_full_v1/kernel/ptoas/state_kernel.pto:2,13–15`：三个 GM Tensor 仍使用 runtime `%arg3`。没有 sample tensor，没有把 T 实值化。 |
| 动态循环 | PTO 20–21：`min(T,2)`，输入按 `step` 读取1行；不是冻结循环两次。 |
| roots | PTO 16、30：输入与返回均 `addr=0`，`3×1024×FP32=12 KiB`，valid固定3×1024。 |
| segments | PTO 18、31：输入与返回均 `addr=12288`，`6×1024×FP32=24 KiB`，valid固定6×1024。 |
| incoming | PTO 22：`addr=36864`，`1×1024×FP32=4 KiB`；全部实际 Vec 区间末端40960 B，即40 KiB。 |
| 修改与返回 | PTO 26–29只有原两次 assemble 对 row0 的 TMOV；返回句柄别名原 roots/segments，没有整 Tile 复制。C++ 170、178为这两次 TMOV，186、193仅重新 TASSIGN 同地址。 |
| 输出尾部 | PTO 33–42在 helper/carry 之外用 `min(T,3)` / `min(T,6)` 设置 store 有效行；动态 valid 没有跨 helper 边界。 |

这里的 full-valid 是实际测试合同：两个状态由 `tile.full` 初始化，输入每次在 `step < min(T,2)` 下读取完整1行。它证明了该固定状态接口的编译可行性，**没有修复或掩盖 `state_v1` 的动态 valid 循环问题**。T=0 的源码循环为空、输出 valid 为0；本轮没有设备执行，不能把源码/生成物分析写成零长度设备验证。

**运行入口与归档**

`run_cpu.sh` 加载旧隔离依赖提供的 testing 环境，并优先导入本目录 overlay；runner 移除会覆盖该选择的 editable finder 后断言 PyPTO 的实际来源。所有操作不创建样例 Tensor，也不调用设备执行。

已执行的标准模块基准和最终 full-state 命令形态如下。输出目录必须是全新路径；本轮已结束，不需要再次执行。

```bash
bash run_cpu.sh --reference --case tile --output <全新绝对目录>
bash run_cpu.sh --case tile --output <全新绝对目录> --compile
bash run_cpu.sh --case state_full --output <全新绝对目录> --compile
```

`ARCHIVE_FILES.txt` 是可复制的相对文件精确清单，含补丁、fixture、文本结果、必要 PTO/C++ 和失败记录；不含两个依赖副本，也不复制完整的原 `specializer.py`。`LOCAL_ARTIFACTS.json` 列出所有本地产物路径/大小、overlay 与原二进制位置；`SOURCE_COMPARISON.json` 记录直接字节比较，仅 `specializer.py` 不同。完整 pass dump 留私有目录，已有产物无需重新生成。`collect_archive.py` 是在当前私有目录对既有文件作归档的脚本，依赖本地冻结副本；不能把缺少 overlay 的归档目录当作完整可运行依赖环境。

前置只读接口与状态拆分分析保存在相邻 `score_attention_step_state_readonly_20261018`；其旧依赖缺口仍是事实，本补丁只存在于此 overlay，不代表共享依赖已更新。
