# A3 TDIV 高精度选项：未关闭问题与最小复现

此问题仍需跟进，因此保留一份必要证据；旧 CSA B40 试错、重复源码和归档包已删除。
下面的实测属于 **2026-09-22 的指定版本**，不表示当前工具链已重新验证 TDIV 精度。
当前工具链版本与整层验收状态见 [任务清单](DSV4_FLASH_CSA_TASK_CHECKLIST.md)。

## 问题与实测边界

A3 的 `pl.div(Tensor, Tensor, high_precision=True)` 在该轮复现中与默认 TDIV 逐 bit 相同，
两者均有部分输入比正确舍入 FP32 商相差 1 ULP；同输入的设备端标量除法全部一致。

当时 PyPTO 的 IR 与 PTOAS 生成代码保留了高精度属性。
PTO-ISA A2/A3 实现接受该选项但仍调用同一 `vdiv` 分支；其文档明确注明 A3 忽略该选项。
这是已声明的 A3 能力缺口，不能据此认定参数传递 bug，也不能推断硬件绝对无法实现更高精度。

| 比较 | 不同元素 / 4096 | 最大 ULP |
| --- | ---: | ---: |
| 高精度选项 vs 默认 TDIV | 0 | 0 |
| 默认 TDIV vs 正确舍入 FP32 | 244 | 1 |
| 高精度选项 vs 正确舍入 FP32 | 244 | 1 |
| 设备标量除法 vs 正确舍入 FP32 | 0 | 0 |

输入为 `[16,256]`、随机种子 20260922；分子、分母与商均为正、有限、normal FP32，分母非零。
参考用精确有理数检查相邻 FP32 的最近偶数舍入。
未覆盖负数、极值、subnormal、零/非有限值、FP16 或其他芯片；没有性能结论。

可固定的差异样例：`1.0 / 0.00003821843711193651`，默认/高精度选项得到
`0x46cc6ac4`，设备标量与正确舍入结果为 `0x46cc6ac3`。

## 保留证据与版本

原任务 `task_20260922_125910_368689920621`，设备 0，exit 0，状态 `REPRODUCED`。
报告与输入输出位于
[report.json](results/cann90_20260921/tdiv_high_precision_repro_v1/report.json) 和同目录的 `tensors.pt`。
`vector_input.pto` / `vector_generated.cpp` 保留高精度属性传递证据，
`isa_TDiv.hpp` / `isa_TDIV.md` 保留对应版本的实现与文档。
这些文件是问题证据，不是当前整模型验收结果。

| 项目 | 原复现版本 |
| --- | --- |
| 硬件 / CANN | Ascend910_9392（A3）/ CANN 9.0.0 |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post2 |
| PyPTO / Simpler | `02c00269` / `e914837d`，指定调试分支 |
| PTOAS / PTO-ISA | 0.61 / `03e45c4b` |
| Runtime | a2a3 / tensormap_and_ringbuffer |

Native QR 在行归约后以设备端标量计算 rstd、`127/amax`、`1/quant_multiplier`，
再做向量乘法；相关源码摘录保留在同目录。
这种差异可能影响 INT8 舍入边界及后续 Top-K，仍须与规约、乘法结合顺序和 sqrt 等因素分别定位。
本例未反汇编，不对最终标量机器指令或吞吐作结论。

## 当前复现入口

[repro_tdiv_high_precision.py](repro_tdiv_high_precision.py) 不依赖模型、checkpoint、vLLM 或 Native custom op。
从本仓库根目录提交单卡任务，使用新的输出目录：

```bash
source ../env-dsv4-0251rc1.sh
task-submit --device auto --max-time 600 \
  "bash $PWD/tests/pypto_test/run_tdiv_high_precision_repro.sh $PWD/tests/pypto_test/results/tdiv_repro_new"
```

用 `task-submit --status <ID>`、`--log <ID>` 查询；不使用 `--wait`。
脚本记录实际 PyPTO/PTOAS/ISA 版本、三条计算路径及误差。
`REPRODUCED` 表示观察到该行为，不能当作高精度验收通过。
独立运行时 `--isa-root` 只标记源码证据，必须与实际编译所用 include 相符。

下一步仅在确认官方新能力或需要定位相关数值差异时重跑本例，
先查 A3 实际支持范围，再考虑性能路径的精度策略；不修改 PTOAS/ISA 实现。
