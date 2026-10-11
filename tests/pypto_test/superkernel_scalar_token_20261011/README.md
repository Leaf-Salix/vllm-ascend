# 过程061：AIV Scalar 填写通知源，功能通过但未获 CSA 收益

日期：2026-10-11。接续[过程060](../superkernel_callback_cost_20261011/README.md)。
八字和单字两个候选均不接入生产生成器；它们专用的 S→MTE3 event7 预留也已撤回并重建 PTOAS。
保留实验驱动、生成代码和原始结果，公共 CSA 与 Simpler 均未修改。

## 改动与边界

原 AIV 后到回调先把不可变的 +1/−1 通知源从 GM 搬到编译器独占的 64B UB scratch，
再把完整 MTE2/V 尾部汇合至 MTE3，写回通知并观察正常派发寄存器。
候选仅将第一步换成 Scalar 写 UB，随后 S→MTE3 同步；原有流水汇合、完整 Wait、
无后继收尾、fanin/cohort gate、FIN/ACK 和派发流程均保留。

PTO-ISA 的 Vec `Tile::SetValue` 使用 Scalar UB 写入，`MScatter` 有 S→MTE3 同步用法。
这里的 scratch 与普通 tile/FIFO 分离，不复用尚在计算的输入或输出。
AIC 保持原 GM→L1 路径，没有把 Vec 能力外推到 L1。

八字候选填满 64B。协议实际仅检查 `TaskBoundaryNotification.value[0]`，因此第二轮只写一个
64 位字，其余独占 scratch 字节仅随 DMA 传输、不被协议读取；没有减少完整尾部等待。
两种写法均只存在于测试 wrapper hook 中，不要求算子作者修改任务主体或新增属性。

PyPTO 驱动 `df6449d88`、私有 vLLM 驱动 `31939eaa`；Simpler `5085a6ce0` 不变。
PTOAS 实验提交 `2d3b6d14` 在新通知占用的 S→MTE3 边预留 event7。
生成代码 FileCheck 通过：Scalar 写入后以 event0 同步到 MTE3，再 TSTORE，未占 event7。
这是正向生成检查，不声称该测试能迫使旧分配器分配 event7。
撤回提交 `658401d2` 恢复原事件池；精确版本见 [versions.json](versions.json)。

## 小用例功能

八字任务 `task_20261011_215801_326370222401`，单卡0、退出0。
该任务先于通用 S→MTE3 预留，使用已检查不占 S→MTE3 event7 的固定小图；
随后所有 CSA 路径统一用带预留的新 PTOAS 编译，避免混用事件分配规则。

| 候选 | 核 | loops | 输入0后到交接 | 输入1后到交接 |
| --- | --- | ---: | ---: | ---: |
| 八字 | AIV | 1 | 0 | 0 |
| 八字 | AIV | 4 | 1 | 9 |
| 八字，AIC 未改 | AIC | 1 | 17 | 18 |
| 八字，AIC 未改 | AIC | 4 | 19 | 22 |
| 单字 | AIV | 4 | 1 | 14 |
| 单字，AIC 未改 | AIC | 4 | 4 | 15 |

单字任务 `task_20261011_221649_40360964333` 先做小用例，再做 CSA，均单卡0。
两候选的输出、完整 fanin 读取、保护区、无后继结束及图重放检查通过。
后到计数位于原运行时首次探测之后；尾等待含新增通知 DMA，不能等同于原计算有效重叠长度。
小用例本轮只做功能，不新增其性能结论。

## 第一轮：八字通知

CSA 任务 `task_20261011_221026_376910030391`，卡0、退出0。
四路同 SDK、同 PTOAS 资源规则，CANN9.2/TMR/NZ2/atomic0、deterministic level0、诊断关闭。
第二层真实权重，128K 合成历史，B16×verify6；完整 HC_pre+norm+CSA+HC_post 单次图重放，
NPU events 计时，reset 在区间外，预热5次后轮换各30次。单位 μs：

| 路径 | min | mean | max |
| --- | ---: | ---: | ---: |
| 普通 | 960.58 | 983.93 | 1085.54 |
| 同 MIX 标记、单次探测 | 952.66 | 976.73 | 996.00 |
| 原完整通知 | 956.04 | 978.79 | 997.48 |
| Scalar 八字通知 | 963.32 | 985.06 | 1006.24 |

八字比原通知 mean 多6.27μs、max 多8.76μs，仅11/30配对更快，配对差值中位数 +7.25μs。
不采用。普通路径 max 本轮也有长尾，保留全部样本，不删除异常点美化结果。

## 第二轮：仅填写有效单字

沿用同一任务 `task_20261011_221649_40360964333`，卡0、退出0。
四路参考复用上一轮本 worktree 的同 SDK 产物，仅新增单字候选；
来源见 `runs/csa_scalar_word_128k_b16_20261011/reused_artifacts.json`。
其他配置与计时方式相同，每路30次，单位 μs：

| 路径 | min | mean | max |
| --- | ---: | ---: | ---: |
| 普通 | 975.22 | 994.64 | 1011.66 |
| 同 MIX 标记、单次探测 | 948.14 | 985.22 | 1006.46 |
| 原完整通知 | 967.52 | 997.25 | 1205.90 |
| Scalar 八字通知 | 975.62 | 994.74 | 1016.36 |
| Scalar 单字通知 | 976.60 | 1000.74 | 1022.72 |

单字比原通知 mean 多3.49μs，10/30配对更快，中位数 +5.52μs；
比八字 mean 多6.00μs，9/30配对更快，中位数 +6.55μs，也不采用。
本轮原通知存在1205.90μs长尾，八字 mean 虽少2.51μs，但配对中位数仍慢5.47μs、
仅13/30更快，结合首轮结果不能据此宣称八字稳定受益。
没有设备指令级证据给出 Scalar 写入为何更慢的精确根因，不把指令数变少等同于路径必然更快。

两轮各路径9项输出/缓存/状态及改变输入后的重放与普通 PTO 逐bit一致，各31项保护区通过。
这不是 Native 数值、token 或16卡验收。所有卡已释放，未新测七档。

## 复现与后续

`runs/`保留完整结果、trace 和日志，两个 `*_samples.csv` 共270条 CSA 计时；
`source/`、`generated/`、`repro/`和 `*-experiment.patch` 保留源代码及复现入口。
patch 为零上下文，使用 `git apply --unidiff-zero`。复现 Scalar 候选应在私有 PTOAS 使用
`versions.json` 的 `tested_head=2d3b6d14` 并重建，不用已撤回额外预留的当前默认编译器。

CPU 编译步骤：进入对应 Git 仓后 source `../env.sh`。
先以 `task_boundary_scalar_token_probe.py --compile --compiled ...` 编译八字小例子；
单字加 `--words 1 --work 4`。CSA 首轮运行 `tmr_csa_scalar_token_compile.py --output ...`，
次轮加 `--words 1 --reuse <首轮compiled>`。设备执行见 `repro/` 脚本，经 task-submit 单卡排队。
曾有一处 Ruff 行长错误，修正后通过；单字 CSA 首次编译继承了只含 PyPTO 的 PYTHONPATH，
导致 `vllm_ascend` 导入失败，重新加载私有 env 后通过。均未占卡，失败日志保留。

当前默认保留过程059的完整通知实现，不继续围绕 Scalar 源写入反复测试。
接下来处理显式 pipe-event 的通用覆盖：当前一律拒绝过于保守，应区分与通知真实冲突的事件，
不能把无冲突的手工同步任务排除。局部 `wait_pre_task_end()` 和诊断串行回退按用户确认保留。
