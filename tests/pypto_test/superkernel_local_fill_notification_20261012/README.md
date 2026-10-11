# 过程069：保留片上常数填充，降低完整通知成本

2026-10-12。**两轮均快于原完整通知，保留为私有编译器默认实现；相对普通 PTO 仍没有稳定净收益。**
本轮不修改公共算子、不增加作者标识，仅验证单卡 TMR 128K/B16。

## 改动依据与边界

现行通知先从 GM 读取常数到 64B L1/UB，再汇合各流水并写回通知。
常数可以在片上生成：按固定 PTO-ISA pin `327cd5869` 的 `TExpandS` 实现，
AIC 使用 `create_cbuf_matrix`，配置 `0x10002` 表示 2 次 × 1 个 32B 块、无 gap；
AIV 使用 `vector_dup`，count mode 填 32 个 uint16，随后恢复 normal mode 和完整 vector mask。
与过程061的 Scalar 写 UB 不同，本候选无需新增 S→MTE3 同步。

依据见 [PTO-ISA 源码摘录](timing/pto_isa_fill_reference.txt)、
[填充合同](timing/fill_contract.json) 和 [AscendC InitConstValue 参数说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0239.html)。
生成后的 [AIC helper](generated/aic_notification.cpp) 与 [AIV helper](generated/aiv_notification.cpp)
保留原全部流水→MTE3 的事件汇合、64B GM 通知写回、DMB 探测及末尾完整等待。
只是移除通知源的 GM 读取，没有通过减少等待或缩小 task 覆盖来加速。

通知值改为 0/UINT64_MAX，始终选择不同于当前通知值的下一值；
前序使用旧 1/UINT64_MAX 通知时也适用。旧不可变 GM 源保留兼容旧回调和直接通知实验。
局部 Wait/Set、原 early_resolve、队列、派发、ACK、核选择与完整 fanin 均不变；
不做安全前缀自动识别，chip swimlane/PMU/参数 dump 仍串行回退。

PyPTO 实验驱动 `361138bee`，默认实现 `9e2865861`；私有 vLLM 驱动 `a04c06f9`。
Simpler 保持 `6174ec5db`，PTOAS 保持 `0c21f984`，无需重建 SDK。
协议仍为 `RuntimeCompiledPollClean=8`，不是过程068未采用的 metadata 直接通知。
[版本及增量](versions.json) 包含恢复所需基准。

## 功能验证

任务 `task_20261012_011347_256290128129`，卡 0，退出 0、已释放设备。
小例子 loops=4、额外 Scalar 循环=0，两组输入、图重放、输出保护区、真实 fanin 和完成顺序通过。

| 核类型 | 两组输入后到交接 | 重放后到交接 |
| --- | ---: | ---: |
| AIV | 2 / 0 | 0 |
| AIC | 4 / 23 | 16 |

计数来自专属插桩，不是未插桩净性能；AIV 第二组及重放未命中不等于功能不支持。
完整尾等待含通知，不能把它当作原计算独立结束时刻，也没有新证据判定某个 CSA task 为 Scalar bound。
MIX 两项通过，含跨核 FIFO、sync_start；两轮分别有 24/48 个 AIC/AIV 成员提前进入安全前缀。
见 [小例子](small/results.json)、[MIX 原始 trace](mix/mixed_chain_results.json)。

私有生成器接入相同实现后，显式事件分配相关 6 项 UT 通过；
原 PTOAS 事件分配仍作用于原来的同步边，不改显式事件。
CSA 生成产物含 25 个 AIC、46 个 AIV 通知 helper，均使用对应的本地填充。
确认任务 `task_20261012_012001_281191521194` 同在卡 0，退出 0、已释放设备。
两轮 CSA 的 9 项状态、换输入重放均逐 bit 一致，每路 31 项保护区通过。

## 同 SDK 的两轮性能

范围：第 2 层 HC_pre+norm+CSA+HC_post，真实权重、合成 128K 历史、B16×verify6。
CANN 9.2/TMR/NZ2/atomic_add=0/deterministic_level=0，关闭所有上述诊断。
每轮三路预热 5 次，各 30 次轮换单次 NPUGraph 重放；NPU Event 计时，reset 在区间外。
普通和原通知复用过程068同一 SDK 的产物；确认轮重新编译私有默认填充实现。
全部为 PTO 内部对照，单位 μs。

| 初测路径 | min | mean | max |
| --- | ---: | ---: | ---: |
| 普通 PTO | 963.74 | 993.47 | 1022.44 |
| 原 GM 读取通知 | 977.44 | 1001.52 | 1023.82 |
| 片上填充通知 | 963.74 | 993.74 | 1018.96 |

相对原通知 mean −7.79 μs、max −4.86 μs，19/30 配对更快，中位差 −11.45 μs。
相对普通 PTO mean +0.27 μs、max −3.48 μs。
[初测结果](timing/results.json)、[90 条样本](timing/samples.csv)、[配对比较](timing/paired_comparison.json)。

| 确认路径 | min | mean | max |
| --- | ---: | ---: | ---: |
| 普通 PTO | 966.18 | 982.17 | 1013.30 |
| 原 GM 读取通知 | 976.80 | 997.37 | 1165.48 |
| 片上填充通知 | 971.72 | 990.17 | 1013.32 |

相对原通知 mean −7.20 μs、max −152.16 μs，18/30 配对更快，中位差 −3.67 μs。
原通知 repeat=1 出现 1165.48 μs 长尾，保留该样本，不将其根因或全部 max 差归于本次改动。
相对普通 PTO mean +8.00 μs、max +0.02 μs；不能宣称整体 early_start 已经加速。
[确认结果](confirmation/results.json)、[90 条样本](confirmation/samples.csv)、
[配对比较](confirmation/paired_comparison.json)。

两轮对原通知的均值、最大值及配对中位差均改善，因此保留片上填充，不追加参数扫描。
当前仅证明通知路径相对改善，稳定胜过普通 PTO 的完整 CSA 目标仍未完成。
没有新增 Native、七档或 16 卡测试。

## 复现

本目录是过程068之后的增量归档，包含补丁、驱动、源实现依据及全部 180 条计时样本。
依赖既有私有环境、冻结 CSA 包及过程068的 baseline/mixed 产物。
全新重建时必须先用基准 PyPTO `44b4ffe76` 编译原通知对照，再应用本轮零上下文补丁
（`git apply --unidiff-zero`）；不能用新默认生成器重编“原通知”后仍沿用旧标签。
Simpler binding、runtime 与 torch adapter 都必须匹配 `6174ec5db`。

将 `repro/` 脚本放回私有根目录 `artifacts/`，执行编译，再通过 task-submit 排队执行：

```bash
bash "$private_root/artifacts/compile_local_fill_20261012.sh"
task-submit --timeout 900 --max-time 900 --device auto --device-num 1 \
  --run "bash $private_root/artifacts/run_local_fill_20261012.sh"
```

确认轮对应 `compile_local_fill_confirm_20261012.sh` 与 `run_local_fill_confirm_20261012.sh`，
只复测默认生成器接入后的 CSA，没有重复小例子和 MIX。
编译与测试日志在 `build/`，设备日志在各任务的 `timing/task.log`、`confirmation/task.log`。
