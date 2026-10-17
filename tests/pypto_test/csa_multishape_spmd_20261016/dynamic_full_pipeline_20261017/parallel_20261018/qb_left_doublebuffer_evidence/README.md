# 主 Q_B 的 Left 双缓冲：独立候选与两次 CPU 生成审计

本目录以正式 `d56c9eb6` 的全部 33 个文件为冻结基线，只修改
`query_projection_stages.py::_q_b_nz_stage`。截至本记录，第二版完整 CPU
编译通过，生成物确认恢复两份 Left，并保留原两份 QR/weight Mat 窗口。
**本目录没有启动设备任务，没有设备精度、保护区、图重放或性能结论。**
第一版虽然完整编译通过，却让 L1 搬运退回单槽，已经否决；源码及生成物均保留。

## 1. 基线、范围与历史排重

工作树：

`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018`

证据目录，即本目录：

`tests/pypto_test/ultra_qb_left_doublebuffer_20261018`

正式基线冻结包：

`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/adopted_source_d56c9eb6`

`make_candidate.py` 在生成时逐字核对以上 33 文件和当时正式包一致。
本目录的 `source_before/` 保存这些输入，`source_candidate/` 保存第一版，
`source_candidate_v2/` 保存第二版。每版另有 32 文件逐字相同；Q_B 函数
签名、其他函数 AST、外部张量/阶段号与调度源码不改。

本轮是动态 M128/N256/K256、仅主 Q_B 操作数流水候选。它不是历史的
`qproj_l0_prefetch`：后者是静态 T96/M96、20 worker、K128，Left/Right
同时双槽，并把 QR 整行留在 L1。历史记录见 canonical：

- `tests/pypto_test/csa_multishape_spmd_20261016/qproj_l0_prefetch.py`
- `tests/pypto_test/csa_multishape_spmd_20261016/HC_MIX_QPROJ_EXPERIMENTS.md`
- `tests/pypto_test/csa_multishape_spmd_20261016/b16_row_deps_qproj_l0_prefetch_v1`

该旧候选曾执行 `task_20261017_054303_7422476042`，不能称本轮完全没有
类似思路；但不是相同 K、M、资源和动态实现，也不能拿它的时间当本轮基线。
本轮也不重复已经否决的 compact Hadamard Right 缓冲候选。

## 2. 原问题的生成证据

`parallel_20261018/pipeline_depth_audit/` 已对照同次原分 task 与融合版本：
原主 Q_B 的 Left 为 `32768/0`，融合版本 K256/K512 的 Left 均为 `0`。
Right 为 `[256,256] INT8`，占满单核 64 KiB，两边均只有一个槽。

本次精确采用 d56 已执行的完整动态编译物作为生成对照：

`parallel_20261018/compact_priority_h131072_b16_v1/build_output/decode_csa_spmd_ad5etrbf`

本目录 `generated_v1/baseline_qb.pto.txt`、`baseline_qb.cpp.txt` 保存带
原行号的摘录。`audit_generated.py` 只读已有产物，未为审计重编基线。
CPU 候选使用相同长档区间策略：G6、N64、score1024、OA128、OB96，T/B
仍为符号轴。不能因为命名为长档就声称 CPU 只支持 B16。

## 3. v1：Left 确实双槽，但 L1 窗口退化，拒绝上卡

冻结时间：UTC `2026-10-17T18:40:29.209509+00:00`，本地约 02:40。

v1 将高层 Tensor `matmul/matmul_acc` 改为明确的
`load(Mat) → move(Left/Right) → tile.matmul[_acc] → store`。
K0、256、512、768 四段 INT8 点积按原顺序累加到 INT32。
中间两份 Left 在 MMA0 之前同时活跃，从而得到两个实际地址。

`cpu_v1/result.json` 为完整 symbolic codegen PASS；但生成地址揭示伴随退化：

| 对象 | d56 原版 | v1 |
| --- | --- | --- |
| K256/K512 QR Mat | 196608 / 229376 | 65536 / 65536 |
| K256/K512 weight Mat | 65536 / 131072 | 65536 / 65536 |
| Left | 0 / 0 | 0 / 32768 |
| Right | 0 / 0 | 0 / 0 |

由于每个 QR Mat 被 move 后立即失活，下一份 Mat 可以复用同址。
最终 QR 与权重都串行复用地址 65536，生成额外 MTE1→MTE2 覆盖等待。
而且 v1 的 Left1 在 Right0 前搬运，MMA0 的 MTE1→M 事件也在 Left1 后，
不能把这版称为“只恢复 Left，其他流水不变”。

因此 v1 不提交设备。保留 `source_candidate/`、`candidate.patch`、
`cpu_v1.log`、`cpu_v1/`、`generated_v1/` 全部真实结果。

## 4. v2：仅重排现有 load/move，保持 Mat 活跃窗口

根代理批准一次有限修正，冻结时间 UTC
`2026-10-17T18:46:16.282342+00:00`，本地约 02:46。
`make_candidate_v2.py` 从 v1 新建独立目录，拒绝覆盖旧版。

只改变已有语句次序，不增加数学操作：

1. K0/768 先同时建立 QR Mat 与 weight Mat，再移入 Left/Right。
2. K256/K512 先建立 QR0、W0、QR1、W1 四份 Mat，合计
   `2×32 KiB + 2×64 KiB = 192 KiB`，与 d56 原 pair 窗口相同。
3. 后续顺序是 Left0 → Right0 → Left1 → MMA0 → Right1 → MMA1。
   Left0/Left1 在 MMA0 前同时存活；Right1 仍须等待 MMA0 释放单份 Right。
4. 将 Right0 放到 Left1 前，允许后端把 MMA0 所需的就绪事件放在
   Right0 之后，不必包含后续 Left1 搬运。是否真正形成这个事件位置，
   由生成 C++ 核对，不能只由 Python 顺序推断。

`source_v2_audit.json` 核对 v1/v2 的 Assign/Return AST 多重集合完全一致，
其余 32 文件完全相同。`candidate_v2.patch` 记录最小重排。

## 5. v2 完整 CPU 与生成事件审计

`cpu_v2/result.json`：完整后端生成 PASS；58 个动态维度、单次 MIX、
kernel 模式、sync_start 保留、没有提供实际 shape，也没有执行设备。

实际 Mat 地址恢复到 d56，Left 为 `0/32768`，Right 为 `0`：

| 对象 | d56 | v2 |
| --- | --- | --- |
| K0 QR / weight Mat | 131072 / 65536 | 131072 / 65536 |
| K256/K512 QR Mat | 196608 / 229376 | 196608 / 229376 |
| K256/K512 weight Mat | 65536 / 131072 | 65536 / 131072 |
| K768 QR / weight Mat | 131072 / 65536 | 131072 / 65536 |
| K256/K512 Left | 0 / 0 | 0 / 32768 |
| Right | 0 | 0 |
| Acc `[128,256] INT32` | 0 | 0 |

`cpu_v2/kernel/kernels/aic/cube_schedule.cpp` 的关键顺序：

```text
10701 TMOV(Right0, weight0_Mat)
10702 set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0)
10710 wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2)
10711 TMOV(Left1, qr1_Mat)
10712 wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0)
10713 TMATMUL_ACC(Acc, Acc, Left0, Right0)
10714 set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1)
10722 wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1)
10723 wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3)
10724 TMOV(Right1, weight1_Mat)
10725 set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0)
10727 wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0)
10728 TMATMUL_ACC(Acc, Acc, Left1, Right1)
```

因此 MMA0 使用的是 Right0 完成后的事件，未把 Left1 追加到该就绪事件前。
Right 的覆盖等待明确保留。这里只证明生成的依赖和地址符合设计，不能声称
实际硬件一定达到某个重叠量；最终完整 CSA 时间仍待真实对照。

精确排除 `_q_b_nz_stage` 后，**其他 915 个 allocation 分组的多重集合完全
一致**，包括 source helper、AIC/AIV、存储空间、dtype、shape、地址和容量。
这比初步排除整个 `query_projection_stages.py` 后统计的 812 组更严格；
两者不矛盾，915 组包含同文件未改的 RoPE/dequant helper。

全函数生成物的物理地址末端上界，d56/v1/v2 相同：

| 空间 | 最大末端字节 |
| --- | ---: |
| AIC Mat | 475136 |
| AIC Left | 65536 |
| AIC Right | 65536 |
| AIC Acc | 131072 |
| AIC Vec | 16384 |
| AIV Vec | 163872 |

该表是已生成 allocation 的地址上界，不是把不同阶段的瞬时活跃容量求和，
也不是理论容量替代编译证明。没有新增固定地址或全局 reserve。

## 6. 数值、动态尾行与同步边界

两版 PTO 中，Q_B 均为 8 次 GM load、1 次首次 INT8 matmul、3 次 INT32
matmul_acc、1 次 GM store；K 次序不变。NZ weight 分块保持
`[1,8,16,16,32]`，K256 对应 NZ K 维偏移 0/16/32/48。
没有把 K 改成 128，也没有改变原 cache policy。

M 循环仍为 `ceil(T/128)`，尾块 `m_rows=min(T-t0,128)`。生成的 Left、
INT32 Acc 和 GM partition store 使用同一个动态有效行符号，store 列数仍
为 256。写入范围为 `[t0:t0+m_rows,w_col0:w_col0+256]`，未把 pad128
容量全部当成有效输出。`source_audit.json` 的有限整数覆盖模型只是额外
检查，不能代替设备 pad/guard 验证。

只改独立 Q_B helper，原 24 worker 列块归属、外部 notify/wait、业务
fence/cacheinvalid、PHASE 数量、130 个根张量 ABI 和后续消费逻辑不改。
源代码与生成事件审计仍不代替真实 A3 协议/可见性验证。

## 7. 可复制命令与交付边界

```bash
E=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/ultra_qb_left_doublebuffer_20261018
D=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017

# 以下审计仅读已有生成物，可重复执行；不会编译或访问 NPU。
python "$E/audit_generated.py" v1
python "$E/audit_generated.py" v2

# 已执行过的 CPU 命令；复现应改成全新输出目录，勿覆盖现有证据。
bash "$D/reproduce_cpu.sh" "$E/cpu_v2" \
  --attention-query-metadata off \
  --spmd-source "$E/source_candidate_v2" \
  --controls-json "$E/controls.json"
```

源码生成脚本只接受新目录；再次执行需使用新的独立证据目录。每个 CPU
版本只完整编译一次。尚未进行 NPU 验证，尚未接入正式包，也未提交或 push。
根代理决定是否用 **v2** 做一次长 B16 完整同进程矩阵；v1 明确不作为
待测候选。不得将本轮结构改善写成整层性能改善。
