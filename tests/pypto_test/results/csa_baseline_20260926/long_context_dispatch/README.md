# 六档未达标后的长上下文定位

2026-09-27。六档结果没有证明PTO快于Native，128K/B16是当前优先定位项。
用户要求总结后停下，当前已暂停；本文后续方向仅为恢复工作的依据。
本轮未提交16卡MoE路由采样；只复用六档原泳道，并对一个调度候选做两个代表档位的单卡对照。

## 原始泳道能证明什么

`analyze_existing.py`只统计Worker View，使用`kernel-duration-us`区分核内执行与setup等待，
并用`chip_swimlane_records.json`的物理核/任务ID计数核对重复派发。没有把Scheduler View重复加算。
数据是正式layer4权重、合成输入/历史、单次eager DFX，不等于整模型的无profiler耗时。

| H / B | score AIC实际覆盖 / 实例数 | AIC核内均值 μs | AIC整组计算首末 μs | key repack核内均值 μs |
| --- | ---: | ---: | ---: | ---: |
| 131072 / 4 | 24 / 24 | 186.42 | 198.64 | 43.89 |
| 131072 / 8 | 24 / 24 | 370.10 | 382.40 | 91.15 |
| 131072 / 16 | 22 / 24 | 703.21 | 1373.66 | 210.71 |
| 8192 / 24 | 24 / 24 | 96.34 | 102.48 | 30.95 |
| 8192 / 32 | 24 / 24 | 102.49 | 104.98 | 27.44 |
| 8192 / 40 | 24 / 24 | 118.84 | 145.40 | 30.64 |

B16/128K：AIC_0/1没有执行score，AIC_5/10各执行两份，第二份在1278～1279μs才接收，
到1938～1939μs结束；配套AIV也有4个重复核、4个未执行score的核。
这些任务已在约587μs派发到忙核，说明存在物理核队列上的串行化。
未执行score的AIV_24～27在592～596μs接收merge，约1948μs才开始其约20μs的核内计算。
**不能倒过来说merge启动导致了之前的重复分配**；两者的先后关系不支持该因果推断。
merge长条也不等于做了1.3ms的Top-K归并算术。

## 一次调度候选：没有足够收益，已撤回

仅给性能版 `indexer_score_topk_leaf` 增加 `sync_start=True`，尝试让24个MIX block整组启动。
保留24个worker、原有算术/缓存布局/early-resolve设置。依赖说明见PyPTO本地
`docs/en/user/tutorials/05-scheduling-tuning.md`：整组启动会失去逐block提前派发能力。

两侧均为单卡A3设备0、正式第二个CSA层权重、合成历史、mode2、atomic1、Native level0，
复用metadata；5次预热后20次完整图重放，恢复初态和诊断读取在计时外。
Native/PTO在同一进程测量，PyPTO初始化后两侧使用同一进程event设置。

| 档位 / 实现 | 原实现 p50 / p95 μs | sync_start候选 p50 / p95 μs |
| --- | ---: | ---: |
| 128K/B16 Native控制 | 1310.15 / 1315.48 | 1321.71 / 1326.00 |
| 128K/B16 PTO | 1792.34 / 1924.84 | 1786.43 / 1920.30 |
| 8K/B40 Native控制 | 1413.16 / 1416.34 | 1416.70 / 1423.18 |
| 8K/B40 PTO | 1501.15 / 1556.26 | 1527.74 / 1555.96 |

128K/B16的PTO均值1855.75→1816.93μs，但中位数仅下降0.33%，仍明显慢于Native；
8K/B40中位数增加1.77%。没有理由保留为默认优化，也不为这个失败候选再测16卡或补泳道。
两组图重放的保护区、索引结构、有限值检查通过，不据此宣称完整跨实现精度验收。
该候选源码已恢复原状。原始20次样本、Native控制与检查摘要见`timing_comparison.json`。

- 基线：`task_20260927_000141_5285552958`，completed/exit=0，源码`d9468c62`。
- 候选：`task_20260927_000407_55623329739`，completed/exit=0；仅上述一行差异。
- 入口：`run.sh baseline`、`run.sh score_sync_start`（标签不自动修改源码；重现候选须先应用这一行）。

## 下一步应改什么，为什么不能只继续调序

当前性能版score与pypto-lib `2164563`的默认路径均为单query的64个head、N384 tile、
INT32中间矩阵转到Vector后规约head。B16/H128K每个worker循环16个query/leaf组合，
仅score的核内均值已超过700μs；此外当前实现每步先重排Native页式key/scale，核内均值约211μs。
两类窗口会重叠，不能相加作为可节省的整层时间。

本仓Native A3 QLI代码使用M256（64个head下最多合并4个query）/S2=2048分块，
QK结果先转FP16到L1，再由Cube做head加权规约，随后才把规约结果写到GM；同一key tile可在
query组内复用。当前PTO的逐query key读取、跨核中间结果及规约位置都不同，
这些是需要实测的具体数据流差异，不能因为参考pypto-lib就默认其更快。
原整模型Native rank0 trace中QLI核本身在128K/B16的均值约366.21μs；这与单卡合成DFX不是
同一次调用，不能直接把约337μs差额当作已隔离的算法成本或收益承诺。

pypto-lib还保留N768、FP16中间传输/双缓冲路径，但入口条件是B≥64且压缩历史≥32768，
本轮B≤40不会命中。后续先比较query内key复用和中间结果传输/规约方式；若移植该路径，
须解释降低启用阈值的性能证据，以及FP16转换与当前性能版的算术差异。
不把该上游分支的存在当作本轮性能问题已解决，也不直接迁移到精度版。

Native参考：`arch32/quant_lightning_indexer_kernel.h`的M_BASE_SIZE/S2_BASE_SIZE，
`quant_lightning_indexer_service_cube.h`的ProcessQk、ProcessWs、FixpSToL1。
优先验证这条长上下文路径的完整span，再补对应整模型forward/token看护。
