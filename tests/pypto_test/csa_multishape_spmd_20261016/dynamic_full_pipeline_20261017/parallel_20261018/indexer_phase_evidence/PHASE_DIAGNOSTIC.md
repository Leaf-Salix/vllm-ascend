# 8K/B32 半叶本地 score 候选：同轮阶段诊断

本轮结论是：**B32 的 Score 区间未显示稳定缩短；AIV 少做的本地工作与更长的
FFTS READY 区间同时出现，支持 AIC 供数限制使局部节省难以进入关键路径的解释。**
这不是完整 CSA 性能收益证据，也不能反向证明 B16 的 20.4µs 来自同一机制。
保留上一轮“不默认采用、不设 B16 白名单”的决定。

## 执行范围和检查

唯一新增设备作业为 `task_20261018_011739_393773027095`，经
`task-submit --device auto --max-time 1200` 分配设备 0。输出位于本目录
`phase_h8192_b32_pair_v1/`。旧源码为主测试 `parallel_20261018/baseline_source`
中的 593 默认 SPMD，候选为私有正式目录中冻结的 v6；候选和 B32 上轮已测
快照逐文件直接文本一致。未修改算法、主仓、共享安装或生产源码。

同一进程加载两套独立 JIT，共享正式权重、Native 输入/output/cache 根地址，
各自独立 workspace/signal。原动态 task 提供 A/B 精确 oracle；两侧各一次
eager 编译执行和图重放校验后，三轮按旧→候选、候选→旧、旧→候选执行，
输入为 A→B→A。总计 10 次完整检查均 PASS，包含输出、TopK indices/scores、
cache/state、Native/workspace 保护区和信号代际；两侧最终 generation=5。
三轮两侧共 6 份时钟。没有 DFX/L1 窗口，没有重跑 150 轮 helper 或其它完整档位。

`phase_clock_pair.py`、`phase_clock_ffts.py`、`run_phase_clock_pair.sh` 在提交前
冻结；runner 支持 `--baseline-only` 供主 agent 另行复用，本 agent 未提交其
长 B16 作业。Ruff、bash 语法与最终 C++ 插桩生效检查均通过。

## 时钟和 FFTS 含义

私有 hook 复用主 `compile_diagnostics.py`，保留 100 tick GM wait 退避。
原槽 16..19 保存阶段起止/累计 GM wait/count；同一第二条 cache 行的 20/21
补充 `wait_flag_dev` 原位两次 `get_sys_cnt()` 差及次数。每核寄存器累计，
只在原 notify 端点写回，没有新增 fence、通知或等待。两侧 C++ 各有 8 个
FFTS wait 位点、45 个 notify 位点，原有 cross-core sync、barrier、set/wait
flag 数量未变。当前单位为 50 MHz 计数换算的 µs。

字段保留保守命名 `ffts_issue_us`。兄弟 agent 的
`A3_FFTS_WAIT_TIMING_READONLY.md` 提供官方 A2/A3 语义、本地 CANN/PTOAS
证据：A3 的 `wait_flag_dev(id)` 会阻止后续全部流水指令，因此原位区间有依据
包含尚未到达的 READY/ACK 等待，而不只是把等待异步发给 MTE2 的成本。
原始文档及最终机器指令次序的核对由该审计提供；不能仅凭 C++ 顺序宣称零扰动。

最终审计没有取得可读的 A3 指令助记符：运行产物仅保存 raw text bin，等字节
包装后 CANN 9.2/8.3 反汇编均显示 `<not available>`，既有真实 A3 object
控制也相同。因此 **最终 SYS_CNT/WAIT/SYS_CNT 机器指令次序未验证**，不能
记作 ISA PASS。C++ 原位结构和 A3 阻塞语义已有证据，当前归因仍保留这一限制。

原 `wait_us` 仍只计 GM TWAIT；`other_us` 仍包含 FFTS、搬运、计算、核内同步
及探针成本。辅助事实文件中的“扣除已计 GM/FFTS 后剩余区间”逐核计算后再
取中位数，仍不称纯计算。跨核等待可以重叠，不能把累计值从完整层耗时直接扣除。

## 观测事实

下表用 phase32 最晚发布为共同起点，避免此前阶段的时钟偏移掩盖 Score 本身。
数值为旧→候选；全核最晚端点差不等同逐核 interval，也不作正式性能比较。

| 轮次/输入 | 32→34 Score AIC | 32→35 Score AIV | 35→36 TopK | 36→37 实际 plan owner | 37→39 Attention AIV |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 / A | 88.22→85.08 | 94.18→91.24 | 19.52→21.24 | 25.56→23.98 | 270.30→268.62 |
| 1 / B | 88.28→87.28 | 94.10→94.40 | 19.48→18.58 | 22.90→22.06 | 274.28→271.20 |
| 2 / A | 86.52→86.88 | 92.32→94.46 | 20.54→22.08 | 25.72→20.96 | 267.12→271.48 |

plan 的真实 owner 按 `ceil(T/8)` 和 `%48` 推导；本档 T=192，对应 core24..47，
没有把其它 AIV 的空发布当成 Attention 接棒时刻。三轮 Score AIV 窗口差为
−2.94/+0.30/+2.14µs；没有观察到一个稳定的大 Score 改善再由 Attention 抵消。

| 轮次 | Score AIV FFTS 区间中位数 | Score AIC FFTS 区间中位数 | AIV 扣除已计 GM/FFTS 后剩余区间中位数 |
| --- | ---: | ---: | ---: |
| 0 | 53.57→55.73 | 12.21→11.83 | 30.64→26.03 |
| 1 | 51.73→56.88 | 11.78→11.86 | 30.88→26.76 |
| 2 | 52.23→56.44 | 11.85→11.97 | 30.24→27.32 |

两侧每个 Score AIV 和 AIC 的 FFTS 次数均为 12，符合本档每 AIC 四组、每组
三个 score FIFO 迭代。AIV 的源码位点是 SCORE_READY；AIC 是双槽回收/排空
的 SCORE_CONSUMED。候选 AIV FFTS 区间三轮增加 2.16/5.15/4.21µs，同时
上述剩余区间减少 4.61/4.12/2.92µs。AIC 的 ACK 区间和 Score 结束窗口基本不变。

Attention 阶段两侧每核 FFTS 次数均为 80。旧/候选 AIC 中位数分别约
180–183/182–187µs，AIV 约 10–13/7–11µs，表明 Attention 与 Score 的等待
方向不同，不能混在一起解释。37→39 窗口差为 −1.68/−3.08/+4.36µs，没有
一致的额外 Attention 尾部代价。

完整插桩最后端点差为 −6.94/+7.04/+17.62µs，而 phase32 前的差已经是
−3.38/+6.48/+12.58µs。这里只说明全链插桩端点包含阶段外波动，**这些数字
不替代先前未插桩完整 L1 的 B32 1020.64→1025.04µs（无改善）结果**。

## 推断与局限

源代码删除了 AIV 私有 GM 往返和重复 visibility 工作；这次三轮观测到 AIV
剩余区间减少、READY 区间增加，而 AIC Score 供数窗口没有同步缩短。三者
共同支持“B32 的局部节省主要转成等待生产者”这一解释。FIFO 的 ACK/READY
和核间重叠仍在，没有足够证据把增加的等待当成可直接删除的气泡。

这次只诊断 B32，没有同轮 B16 阶段对照；插桩也可能改变流水平衡。不能因此
宣称 B16 的正式 20.4µs 改善已被解释，不能据两点设 B16/T96 启用阈值，也
不能得出所有长短区间都受相同瓶颈限制。现有生成代码另已确认 B16/B32
Indexer UB 布局没有容量跳变，故没有新的 UB 分界依据。

如果继续优化，证据更支持先研究 AIC 原生 Key 页解析/搬运和供数的可减少工作；
本报告不把该方向当成已实现或已验证的优化，不授权额外设备重测。

## 归档和复查入口

可复制目录为本目录 `phase_archive/`，含 runner/hook/汇总脚本、检查结果、
6 份原始 JSON 时钟、阶段摘要、事实表、设备日志和路径/大小清单。
完整生成 C++、二进制和源码快照留在私有 `phase_h8192_b32_pair_v1/`。
逐核原始值见各 `clock_round_*` 的 `phase_clock.json`；每核阶段 trace 也保留
在原输出内。`phase_summary.json` 按端点汇总；`phase_evidence_facts.json`
保存前述逐核差分定义和数字。归档不含校验摘要，不触发重新编译或设备测试。
