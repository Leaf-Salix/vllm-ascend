# Q_B 全局 N128：已有完整动态生成物的独立审阅

结论：**Ready，可交由主会话做完整设备对照**。未发现本次范围内的动态尾行、资源越界或跨核协议阻塞；这不是设备精度或收益结论。**Right 仍是单槽，不能把本候选描述为双 Right 流水。**另外存在非 Q_B 的 Acc 地址和局部事件变化，完整层性能不能被纯归因于 Q_B 任务均衡。

本次只读主会话已经完成的 CPU 产物，没有启动编译、设备任务或修改冻结源。全部逐项结果见 [review.json](review.json)，文本审计脚本为 [audit_generated.py](audit_generated.py)。未做 hash；大文件只登记原路径和大小。

## 输入与编译范围

`PAR=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018`。

- 候选：`PAR/qb_global_n128_evidence/source_candidate`；完整 CPU：同目录 `cpu_v1/kernel`。
- 同控制正式 668 参考：`PAR/oa_pair_manual_h131072_b16_v1/build_output/decode_csa_spmd_jp7o4g2r`。
- 已逐字段核对两者控制相同：G6、panel64、score1024、prefetch1、roots1、OA128/OB96，QA_M16_PARALLEL=False，Attention query metadata=False。
- CPU `passed/full_codegen_passed/single_mix/sync_start=True`；130 根 ABI、58 个符号轴，没有提供样例 shape，没有设备执行。C++ 中额外的派生 view 不增加外部根 ABI。

## 实际数学与动态任务

候选 `query_projection_stages.py:80,123–157` 只把 NZ 的 N256 改为 N128；函数体 AST 不改。`vector_schedule.py:763` 的等待任务数从常量 128 随动为 `QPROJ_N_BLOCKS`。ND N512 路径不变，本次 CPU 编译的是 NZ 路径。

实际 PTO `1477–1553` 保留运行时 T（`%arg132`）：外层每核执行 `ceil((256-worker)/24)` 个 N 任务，列起点 `(worker+24*round)*128`；内层 M128 到 `ceil(T/128)*128`，有效行 `min(T-t0,128)`。有效循环中 `t0<T`，无须把负行当有效尾行。256 个 N128 分区恰好覆盖 32768 列；NZ 分区实际为 `[1,4,16,16,32]`，K 偏移依次 `0/16/32/48` 个 16 行分形。

四次 K256 的顺序仍是 `TMATMUL` 初始化，再三次 `TMATMUL_ACC`，全程 INT32；每个输出的 K 累加顺序不变。有效行同时传到 QR 的 TLOAD、Left、Acc 和最终 TSTORE，未改成 FP32 中转。T129、255、256 分别为 `[128,1]`、`[128,127]`、`[128,128]`；T257 仍可动态跑第三块，不是冻结到最多两个 M。见 [candidate_qb_pto_excerpt.txt](candidate_qb_pto_excerpt.txt)。这是生成表达式审阅，不能代替这些尾形状的设备精度验证。

N 任务数翻倍使 workers 0..15 各 11 块、16..23 各 10 块；原 N256 为前 8 核各 6 块、其余各 5 块。最忙核列数从 1536 降为 1408，但全局 MMA/FIX 调用和 QR 逻辑读取次数翻倍。权重总逻辑字节、输出总字节不变；例如 T96 的 QR 有效元素逻辑读取由 12 MiB 变 24 MiB，不等于实际 HBM 流量，也不是收益估计。

## 实际地址和局部依赖

| Q_B 存储 | 正式 N256 | 候选 N128 |
| --- | --- | --- |
| Left | 4 个逻辑 tile 均 @0，各 32 KiB | 同左 |
| Right | 4 个逻辑 tile 均 @0，各 64 KiB | 4 个逻辑 tile 均 @0，各 32 KiB |
| INT32 Acc | @0，128 KiB | @0，64 KiB |
| QR Mat，K0/1/2/3 | 131072/196608/229376/131072 | 65536/65536/131072/65536 |
| B Mat，K0/1/2/3 | 65536/65536/131072/65536，各 64 KiB | 98304/98304/163840/98304，各 32 KiB |

候选 Mat 的 K1/K2 使用两组供数空间，实际 Right/Left 都没有交替地址。C++ `cube_schedule.cpp:10539,10694,10718,10804` 在覆盖同址 Left/Right 前保留 `wait_flag(PIPE_M, PIPE_MTE1, ...)`；四次 MMA 位于 `10562/10709/10732/10819`。所以源级 `stage=2` 不能证明 L0 双缓冲。TSTORE 后有 FIX→M drain，跨 M/N 复用和阶段末尾 DSB/DCCI/PIPE_ALL/PHASE_QB 通知均保留（`10858–10894`）。小摘录见 [candidate_qb_cpp_operations.txt](candidate_qb_cpp_operations.txt)。

AIV 的归一化 PTO 全文逐字一致，1812 条 allocation 的空间/地址/尺寸相同，3088 条局部事件序列相同。实际 `PTO:9441–9450` 仍遍历 owner `0..23`，信号 word 偏移为 `10*72*128+owner*128`，等待后 DSB/cacheinvalid/DSB 保留。一个完整 512 维 head 现在跨四个 N128 producer，但全 24 owner acquire 仍覆盖它们。见 [candidate_qb_acquire_pto_excerpt.txt](candidate_qb_acquire_pto_excerpt.txt)。72 核 begin、其余通知和 FFTS 未改。

## 全局资源与非局部差异

| 空间 | 参考与候选最大地址尾端，字节 |
| --- | ---: |
| AIC Mat / L1 | 475136（小于 524288） |
| AIC Left / Right | 各 65536 |
| AIC Acc / L0C | 131072 |
| AIC Vec | 16384 |
| AIV Vec / UB | 163872（小于 196608） |

两者均有 AIC 558 条、AIV 1812 条 allocation；以上是物理地址尾端，不是把不同时期的 tile 大小累加。

不能称“Q_B 外生成物完全不变”。去掉 Q_B 数学区域后，归一化 PTO **只有 8 条 Acc allocation 的地址不同**：Indexer compressor 的 score Acc 从 2048 移至 98304；主 compressor 的 score Acc 声明/窄化/初始化从 16384 移至 98304；Attention PV 第二 Acc 的四处声明从 32768 移至 98304。尺寸、数学、控制、GM 地址和同步 PTO 无其他差异。相关片段为 `PTO:169,772,1656,1710,3787,3792,3803,3808`；见 [aic_outside_qb.diff](aic_outside_qb.diff)。新地址仍不重叠该阶段另一个 Acc，也未增加全局容量上界。

C++ 的 AIC 局部事件静态序列由 1711 条变 1718 条：Q_B Mat 复用事件链净增加 4 条；此外 raw C++ `11045` 的 compact Hadamard Right TMOV 前、`11395/11517` 的主 compressor Left extract 前分别新增一个 `pipe_barrier(PIPE_MTE1)`。后者不在 Q_B 函数数学区域，属于真实非局部 codegen 变化；见 [candidate_non_qb_cpp_barriers_excerpt.txt](candidate_non_qb_cpp_barriers_excerpt.txt)、[cube_schedule_events.diff](cube_schedule_events.diff)。这些静态条数不是动态次数或周期，局部 wait_flag 也不能用于推断 Scalar 实际等待时间。

## 历史与结论边界

旧 `tests/pypto_test/csa_multishape_spmd_20261016/qproj_n128_codegen_audit.json` / `HC_MIX_QPROJ_EXPERIMENTS.md:37` 已有静态 20-worker、T96 的 global N128 CPU 实验，实际同样单 Right，没有当时的设备验证。本次是动态 24-worker 全局任务重分的实测候选，不能把 N128 本身包装为新双槽机制。旧 `qproj_l1_columns` 的同 owner 双 N128、以及 QR+B 常驻候选也不是本次代码。

本审阅止于现成 PTO/C++：未检查新增设备二进制，未声称真实流水重叠或性能改善。现有证据支持提交完整精度及同轮 L1 对照；收益若有，仍同时包含全局 N 任务重分、QR 请求增加、Mat 复用改变和上述非局部编译效应。

复现本只读提取：`python /data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_qb_global_n128_codegen_review_20261018/audit_generated.py`。该脚本只读取已存在生成物并写本目录的 JSON、有限差异与小摘录，不启动 CPU 后端或 NPU。
