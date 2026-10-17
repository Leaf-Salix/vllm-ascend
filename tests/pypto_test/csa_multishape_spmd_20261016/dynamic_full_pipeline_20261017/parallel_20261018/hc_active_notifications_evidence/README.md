# HC 无有效消费者通知：四处发布 guard

基于 `parallel_20261018/adopted_source_d56c9eb6`，2026-10-18。本轮只给
`vector_schedule.py` 的 RAW/GATES/NORM/COMB 四条 `publish_phase` 加运行时
owner guard；所有数学调用、wait、INPUT、META、FFTS 与 72 核
`begin_invocation` 原样保留。不增加策略开关，不修改正式包或共享依赖。
候选完整 33 源位于 `source_candidate/`，diff 为 `candidate.patch`。

主优化日志 `DSV4_CSA_SPMD_OPTIMIZATION_LOG.md` 全文关键词检查没有发现这
四处只裁剪发布的同方案。有关旧尝试：16167–16188 行的消费者就地归约
删除真实 RAW 数学/写回并改消费者，不同于本轮；10017 行是 Attention
逐块 FFTS 空块协议；本轮 PAR/README 中的 Attention folding 只折叠消费
等待。静态成员裁剪与这些历史记录均不能代替本轮动态 guard 验证。

## 源级安全边界

令 `T=dim(x,0)`，AIV worker 为 0..47；guard 只依赖真实 T，不依赖有效
请求数量、compact count、Q/L 或 dummy 标记。令 R=ceil(T/16)、M=ceil(T/8)。

| 发布 | 保留条件 | 消费者与可见性 |
| --- | --- | --- |
| RAW | worker<R | 非 consumer_reduce 路径；gates/comb 的有效 M8 条带只读取对应 M16 的 RAW owner。HC_ROW_DEPENDENCIES 开启时按条带枚举，关闭时 `wait_row_owners(0,T,16,48)`；二者均不等被省略槽。 |
| GATES | worker<M | 当前无跨核 GATES wait；同核 mix_norm 按相同 M8 条带读取 pre。所有真实 gate writer 的排空与发布均保留。 |
| NORM | worker<M | AIC 只等 `wait_row_owners(0,T,8,48)` 的真实 owner。输入投影依赖未改。 |
| COMB | worker<M | 当前无直接 COMB wait；活跃 COMB 发布，以及随后全 48 SCORE_VECTOR→TopK→ATTN_PLAN→Attention/输出的原依赖链全部保留，最终 HCpost 的 post/comb 可见性不削弱。 |

删除的 phase 节点没有跨核等待边；保持其它事件逐核顺序，相当于只缩短
局部顺序链，未删真实 producer 的 release 或 consumer 的 acquire。
空 RAW 角色仍可能有后续 M8 工作，故不将其后等待或数学一起包进 guard。
没有任何改动涉及全 48 IDX_SCALE、SCORE_VECTOR、QUERY 通知。

失去的空发布所附 cacheinvalid 也不承担下一次真实消费的唯一 acquire：
有后续 M8 工作的空 RAW 角色仍执行所需 RAW wait；无 M8 工作的角色只
剩 immutable HC 参数的 Scalar 读取，并在真正的 QR 消费前保留 QA wait。
INPUT 发布全部保留，唯一 planner 的 META 发布也保留。

正式 `binding.py` 使用 `p16=ceil(T/16)*16`，raw 为 `[p16,32]`；
`hc_stages.py::workspace_contract` 同样要求实际 T 的准确 shape/子视图。
若单独讨论更大 P16，`hc_reduce_range` 会按容量执行额外尾部数学，本轮
完全保留这些调用，因此称其“无有效消费者的发布”，不称全部无数学工作。
有效 gates/comb 只读取 T 对应行，额外容量 owner 不会成为新消费者依赖。

跨调用只推进 phase0，不清零其它槽。上一代活跃、这一代失活的槽保留旧
值；当前等待集合只含当前活跃 owner，其本轮发布使用新的 phase0 代际。
再次活跃时旧值小于当前代，不能满足 GE 等待。混合 Q、全 dummy 和
padding 不改变 HC 按 T 的分工。T=0 仅作为协议结构边界模型，不扩展正式
绑定的正 T 输入合同。现成设备检查只要求 phase0 全 72 核及 padding；
输出组检查仅涉及 phase40/41、44..59，均不用修改或放宽。

## 有限模型与通知计数

`audit.py` 复用主 `check_schedule.py`，不导入 PyPTO/torch：原 576 组合
加 M8/M16、48-owner 饱和、多 wave、混合 Q/失活请求和 consumer_reduce
分支补充，共 780 对 baseline/candidate 通过。对每核删去预期空发布后，
其余数学、wait、发布与组握手事件顺序逐项一致；所有被省略槽与完整 wait
集合交集为空。另有 12 次不清槽的跨代回放、60 个 P16 容量组合通过。
剥离恰好四个新增 if 后 AST 完全相同，另外 32 文件直接文本一致。
证据：`audit.log`、`audit_result.json`、`source_manifest.json`。
模型不模拟硬件缓存/FFTS 或数值，不能替代完整设备验证。

| 场景 | T | RAW 省略 | GATES/NORM/COMB 各省略 | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 128K/B4/S6 | 24 | 46 | 45 | 181 |
| 128K/B8/S6 | 48 | 45 | 42 | 171 |
| 128K/B16/S6 | 96 | 42 | 36 | 150 |
| 128K/B24/S6 | 144 | 39 | 30 | 129 |
| 128K/B28/S6 | 168 | 37 | 27 | 118 |
| 8K/B16/S6 | 96 | 42 | 36 | 150 |
| 8K/B32/S6 | 192 | 36 | 24 | 108 |

以上仅计原 192 次四阶段发布，不能换算性能。潜在代价包括四个 Scalar
条件分支与编译后控制流变化；真实 producer 上通知成本不变，关键路径核
可能本来就在执行真实工作，削减旁路角色不保证完整 L1 改善。

## 唯一一次完整 CPU lowering

主 D 的 `reproduce_cpu.sh`，输出本目录 `cpu_v1`，参数
`--attention-query-metadata off --spmd-source 本目录/source_candidate`。
唯一一次编译已完成 exit=0/PASS，完整命令保存于 `cpu_command.txt`；复现须
换全新输出目录。58 动态轴、130 根 Tensor、single MIX、未提供样例 shape。
long G6/N64/SCORE1024；CPU 输出策略 OA256/OB128，不能将其说成 B4/B16
设备策略的全份分配验证。没有执行设备或测性能。

`codegen_audit.json` 与 `codegen_excerpt.txt` 给出实际 AIV C++ 证据：
RAW/GATES/NORM/COMB 分别为 label159/176/195/208，false 分支直接到
label160/177/196/209。前者比较 worker<ceil(T/16)，后三者比较
worker<ceil(T/8)。每块的 3 个 PIPE_ALL、2 个 DSB_DDR、整 DCache
invalid 与 TNOTIFY 都在条件内部；对应 notify 行为
34335/34853/35960/37540。PTO 的 RAW scf.if 从 5443 行开始。
原数学循环和其它阶段仍在原位置，编译器自身的其它同步仍然存在，不能把
四块发布省略理解为这些核已经无同步成本。未做机器码或设备缓存语义验证。

`source_candidate/` 保持编译时的 33 份源码，后续冻结不再修改。没有新增
policy flag；由主会话使用 source override 决定设备试验，参照为上述
adopted_source_d56c9eb6。没有新写设备读回脚本，没有重复 CPU 或计算 hash。
