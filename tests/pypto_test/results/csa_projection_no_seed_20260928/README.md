# 固定规约时移除 Q/KV 投影的冗余清零：独立候选

基底71153bb3，工作树`.cache/csa-no-seed-71153bb3`，不修改正在七档测试的冻结源码。
仅性能版：atomic0 时 QR_OK=KV_OK=1，每个 M/N 输出块只有一个写入者，完整 K 累加后以普通 store 覆盖输出。
原来的 qr_proj_seed（ND/NZ）和 kv_proj_seed 仍将这块临时 GM 清零，并让 matmul 等待。
候选只在 atomic1 时保留种子任务；atomic0 不改 K 遍历、cast、量化和 cache 布局，也不改精度版。

QA 的 dense 行块和末尾16行块覆盖所有有效输出，KV 的 M 分组写入互不重叠；
消费者对实际 token 行裁剪，padding 不作为下一阶段的有效输入。
这只是代码分析，必须以设备状态与 A→B→A 图重放确认没有依赖旧数据。

[单卡命令](run_layer.sh)：128K/B4（24行，含16行矩阵块尾部padding）、8K/B16（96行，dense+tail）。
双方固定规约、det1、mode2、第二个 CSA 层真实权重、可变物理行scale；8类输出/状态精确比较，
候选图重放、保护区和metadata检查，20次本体图计时。Native控制同时保留。
不将测试输入当真实EP16路由；未通过前不应用生产，不扩真实权重模型。

CPU完整lowering/PTOAS/CCE/链接已通过，生成任务表不再包含两类seed。
PyPTO条件要求Bool，源码使用`if ATOMIC_ADD == 1`，没有修改编译器或ISA。
编译结束时8K模型仍在初始化，未与正式forward窗口重叠。
单卡任务`task_20260928_053518_14737829545`已排队，在正在执行的七档EP16之后运行。

## 单卡完成，进入代表档 EP16

任务退出0。两档8类输出/状态跨版本逐元素精确一致、候选A→B→A、metadata/保护区检查通过。
未保存idx_topk_scores；Native浮点零容差仍不通过，本次不冒充Native精度全对齐。
[状态检查：128K/B4](h131072_b4/comparison.json)、[8K/B16](h8192_b16/comparison.json)、
[全部20次计时样本](single_report.json)、[只读汇总入口](summarize_single.py)。

| 档位 | PTO基线→候选 μs | PTO变化 | Native控制变化 | PTO P95基线→候选 μs |
| --- | ---: | ---: | ---: | ---: |
| 128K/B4 | 724.377→711.037 | −1.84% | −0.84% | 738.060→723.120 |
| 8K/B16 | 797.020→767.674 | −3.68% | −2.42% | 821.200→778.220 |

Native控制也变快；部分运行期间device1另采当前七档DFX，不能将全部下降量归因于免seed。
生成代码中确实减少两个清零任务，Q/KV数学遍历未变；不将此单卡表直接换算成整网收益。
生产仍为七档已验证的71153bb3核内实现，默认配置提交45412ba3单独选择atomic0。

`task_20260928_054914_3094321594`排队，只补128K/B8与8K/B16真实EP16。
[命令](run_model.sh)使用独立候选工作树、双方新控制、atomic0/det0/mode2，仍为8步预热后10步纯forward，
位置/token/DSpark/P95/慢卡同样检查；[收集器](collect_model.py)不放宽门禁。
候选尚未应用生产。未完成这项对照之前，不用新的单卡结果替代已有七档正式结论。
