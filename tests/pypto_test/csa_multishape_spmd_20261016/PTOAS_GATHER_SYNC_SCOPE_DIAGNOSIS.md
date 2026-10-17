# CSA 大 SPMD 中冗余流水同步消除的范围诊断

2026-10-16。本文记录实际 CPU 编译证据和设备对照，**不是已确认的整层性能根因**。
没有修改 PTOAS、PyPTO 或 Simpler 的共享源码/安装。

## 观察到的编译行为

只读本地 PTOAS 的
`lib/PTO/Transforms/InsertSync/PTOInsertSync.cpp`：

- `hasGatherScatterLikeOps` 检查整个函数是否含 TGather、TGatherB、TScatter、
  MGather 或 MScatter。
- 命中任一项时，跳过整个函数的 `RemoveRedundantSync`，随后仍做 EventId 分配。
- 附近注释解释的是 A5 上 gather/scatter 相关同步可能被过度消除的问题；
  这个跳过条件本身没有限定为 A5。

因此，A3 大融合函数中一个阶段使用 gather，也会让其他阶段失去这一步优化。
这只能说明编译范围不同，不能据此直接删除保守分支或认定同步可以安全减少。

## 已安装 PTOAS 0.66 的直接证据

真实命令保存在 `ptoas_sync_phase_evidence/commands.json`；没有依赖修改后的编译器。
输入为同一套 CSA IR 的整层函数和独立 widen 阶段，命令使用：

```text
ptoas INPUT.pto --enable-insert-sync --pto-level=level3 \
    --pto-insert-sync-debug=1 -o OUTPUT.cpp
```

两份输入都声明 `pto.target_arch = "a2a3"`。日志、摘要及生成 C++ 位于
`ptoas_sync_phase_evidence/`。

| 编译对象 | Sync Motion 后 set/wait 对数 | RemoveRedundantSync 后 | EventId 分配后 |
| --- | ---: | ---: | ---: |
| 整层 AIC | 3188 | 772 | 938 |
| 整层 AIV | 2584 | 2584 | 1060 |
| 独立 widen AIV | 46 | 28 | 33 |

整层 AIV 的 EventId 分配还把部分事件转为 barrier：日志中的 barrier 数从 1169
变为 2854。以上是编译 IR 中的静态操作数量；包含条件与循环，不是动态执行次数，
也不是耗时。C++ 文件中额外公共 helper 的定义还会影响文本计数，不应混用口径。

## 设备对照结果

使用正式模型第 4 层权重、128K/B16/S6；两侧均为完整 CSA PTO，不运行 Native
forward。L1 指标为整层 `min(incore start)` 到 `max(incore end)` 的三窗中位数。

| 方案 | 同轮原动态 PTO / μs | 单 SPMD / μs | 结论 |
| --- | ---: | ---: | --- |
| 原 balanced 单 SPMD | 902.28 | 1067.36 | 当前基线 |
| 只独立编译 widen，内部内联 | 894.20 | 1120.22 | 没有改善 |
| 全部 68 个阶段独立编译，内部内联 | 901.72 | 1153.34 | 没有改善 |
| 全部阶段独立编译，noinline | 893.54 | 1044.54 | 相比内部内联更快，未达到目标 |

后两项是 `stage_codegen.py` 的测试进程内诊断：从同一融合 IR 提取阶段，
保留原等待、数学、写回屏障及通知；分别经过原 MemoryReuse、AllocateMemoryAddr
和 PTOAS 后再组合。外部仍是一个 SPMD 任务，完整泳道仍只有 72 条核记录。

全阶段独立版本把内部参数裁剪为每函数 8～26 项。由于外层已不执行数学操作，
核对并移除外层原数学的成对 auto-sync 初始化/回收，避免与内部函数的事件编号
生命周期混用。内部函数的流水事件、MIX 跨核事件保持原样。

两项设备实验均通过输出、Top-K、六份状态、保护区、图重放及 30 代信号检查。
它们使用固定输入恢复，不代表连续 decode 轨迹；C++ 组合诊断也不作为正式接口
完成或 700 μs 目标验收依据。

## 后续判断边界

保留设备函数调用边界的 noinline 对照已经 PASS，见上表，三窗为 1037.44、1044.54、
1063.24 μs。外部 PyTorch PipeUtilization 采样也已完成，原始表在
`b16_pipe_profile_v1/pipe_profile/`；AIC MAC 及 AIV Vector 时间接近，单 SPMD
的 Scalar、MTE2 时间及 ICache miss rate 增加，尚不能据此确认单一根因。当前 kernel
模式不接受 Simpler `enable_pmu`，不绕过该限制；外部 PMU 覆盖完整持久核，
包括外围调度，其时间不能替代 L1 incore 指标。

现有证据支持“融合改变了同步消除范围”，**不支持“修复这一点即可降到 700 μs”**。
只有确认可安全消除的同步并获得设备收益后，才适合进一步提出组件修改建议。
