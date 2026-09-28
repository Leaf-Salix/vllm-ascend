# Sparse同核串行：qk_pv整组准入单变量候选

当前决定：两档同卡对照完成，状态/重放通过，均值和P95没有显示收益，暂不合入生产。
本轮不扩EP16或完整矩阵，保留同核串行证据和候选，回到最新AscendC源码驱动的核内优化。

来源：[短档Score对照捕获的异常](../csa_short_score_sync_20260928/README.md)。
基线8K/B16的一个DFX窗口中，24个qk_pv逻辑AIC块只落在22个核，AIC_18/21各跑两份，
实际kernel启动分散124.86μs，Sparse首次接收→merge结束升至261.00μs，Worker首尾866.46μs。
该证据直接指向Sparse分配；不把单层独立窗口当成模型相邻12/14层的内部拆解。

候选工作树`.cache/csa-sparse-sync-e58ddc94`，冻结e58ddc94，只给性能版qk_pv增加`sync_start=True`。
24个逻辑块、原early_resolve=True、核内数学、buffer、Score全部保持；不叠加短档Score候选。
精度版和Native流程不改。原逻辑块按全局block_idx处理不同query，同核两份是串行分配问题，
不能据此称原实现漏算或重复执行同一逻辑块。

完整CPU lowering/PTOAS/CCE/链接已通过；该开关使整组等待资源齐备，可能产生drain成本，
必须同时检验启动、核内、Sparse整段、完整CSA均值/P95，不能只以24核各一份判成功。

任务`task_20260928_124129_133793627580`，通过task-submit申请一张卡，最长2400秒。
8K/B16与128K/B16分别两侧各100次正式图计时、5次预热、四个独立DFX窗口。
两档交换baseline/candidate进程顺序，Native控制原样保留；不拼接旧轮基线。
layer4真实权重/合成历史、S6、mode2、atomic0、det0、EPLB关闭、复用第二层metadata。
八类PTO状态、保护区、自重放和候选A→B→A要求精确；det0下不要求Native跨进程浮点精确。
仍不含未保存的idx_topk_scores，不代替真实模型token/DSpark验收。

只对当前明确捕获的调度问题做这项长短代表档，收尾后回到最新AscendC源码驱动的核内阶段。
有收益再覆盖受影响短档、尾块与必要EP16；无收益保留反例，不扩测矩阵。

[候选补丁](candidate.patch)、[CPU编译入口](compile.py)、[CPU编译日志](compile.log)、
[设备命令](run_layer.sh)、[复用的状态/计时/调度汇总](summarize.sh)。

## 两档结果

任务完成、退出0，两侧均device8；每档八类PTO状态零差异、保护区/自重放/A→B→A通过。
以下100次无profiler样本全部保留，P95采用nearest-rank。

| 档位 | CSA均值，基线→候选 μs | 变化 | P95 μs | max μs |
| --- | ---: | ---: | ---: | ---: |
| 8K/B16 | 777.328→779.639 | +0.30% | 795.00→796.56 | 863.42→804.96 |
| 128K/B16 | 1111.900→1117.114 | +0.47% | 1127.14→1132.56 | 1148.32→1145.70 |

两档P50分别775.54→778.36、1111.16→1116.26μs。
Native控制均值924.382→953.874、1299.821→1325.896μs，存在跨进程变化，
不能据此把0.3%～0.5%全部定性为sync_start的真实回退，也不按Native比值归一化宣称收益。
短档最大值改善，未显示P95或均值共同收益；这不足以证明真实EP16已经获益或偶发长尾已根治。

独立DFX四窗口与正式100次计时分开：

| 指标 μs | 8K基线→候选 | 128K基线→候选 |
| --- | --- | --- |
| qk_pv AIC核内均值，四窗口均值 | 119.03→123.83 | 147.03→146.16 |
| AIC启动分散，四窗口范围 | 0.36～8.52→0.36～0.64 | 0.36～18.40→0.30～0.44 |
| 全部drain标记并集，四窗口范围 | 0→8.98～13.08 | 9.88～13.80→19.78～23.74 |

本轮全部窗口两侧均24个AIC各一份Sparse，未再次捕获同核串行；不能删去上一轮的反例。
长档原本已有Score drain，上表是所有drain标记并集，不能全部归给新增Sparse；
标记也未覆盖无进展重试，不能当完整暂停派发时间。
候选能收敛启动，但当前证据没有证明它改善整体稳定性；不将长档约0.6%的核内变化称为新的核内实现收益。

[8K完整结果](h8192_b16/summary.json)、[128K完整结果](h131072_b16/summary.json)、
[8K基线泳道](h8192_b16/swimlane/baseline/dfx/merged_swimlane.json)、
[8K候选泳道](h8192_b16/swimlane/candidate/dfx/merged_swimlane.json)、
[128K基线泳道](h131072_b16/swimlane/baseline/dfx/merged_swimlane.json)、
[128K候选泳道](h131072_b16/swimlane/candidate/dfx/merged_swimlane.json)。
