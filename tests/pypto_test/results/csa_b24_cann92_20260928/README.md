# 新增128K/B24：CANN9.2基线、私有包与容量证据

用户于2026-09-28要求启用128K/B24，后续去除8K/B16，先核对9.2基线、实验副本及ring能力。
新矩阵：128K B4/8/16/24；8K B24/32/40。长短7:3，各组内等权；后续代表档为128K/B16与8K/B24。
旧结果不改标签，不将B24插入旧七档冒称同一次测试。

## 已核对与无需重复的工作

- 公共9.2切换后已经完成e33d842a统一七档；WO_A修正3b27c7fd也已完成长短B16及8K/B40局部A/B，
  两侧均9.2、PyPTO88f60598/Simplera54c05095/PTOAS0.66/PTO-ISA327cd586。无须重跑这些基线。
- 已检查另一个会话n92r1/r2/r3全部48份rank性能记录：均H131072/B24、mode2、atomic0、EPLB关闭、
  util0.97、每rank预热8步后10步sufficient，每步144 tokens/24 requests；48份加载日志无PTO_CSA_WEIGHT_RECAST。
  [复用证据及原始路径](model_capacity_evidence.json)。三轮PTO的容量结论有效，不重复16卡容量试验。
- Native B24整模型仍缺9.2基线；9.0 Native的88.7ms不能与9.2 PTO82.9ms相减判性能。
  当前先单卡CSA，整模型仍按用户优先级后置。按进程实际库版本判定环境，不仅看公共脚本21:59的修改时刻。
- memory ANALYSIS中“B24不可达”“WO_A恒ND”等早期推断已被后续实测推翻，本阶段只引用§19/20及原始记录。
- ring配置按形状显式指定，不当全局默认。大模型已用heap=[256,128,256,32]MiB、task_window4096，
  max_num_seqs24、capture[144]、token预算256、util0.97；本轮单层用相同ring检查当前算子。

## 新单卡入口

源码3b27c7fd冻结于`.cache/csa-b24-cann92-3b27c7fd`，整包复制性能版为
`dsv4_csa_b24_cann92_3b27c7fd`，使用`PTO_CSA_VARIANT=pkg:dsv4_csa_b24_cann92_3b27c7fd`。
私有包和公共依赖均位于该独立快照；排队后不再编辑。
`decode_csa_tp1_layer._get_dep_graph()`与测试根解析、完整PTOAS/CCE/链接/load均通过，见[prepare.json](prepare.json)。

沿用正式layer4权重、合成历史、反序页表及变化scale，TP1/S6/mode2/atomic0/det0。
先两侧20次图计时、A→B→A、metadata/保护区及独立PyTorch profile，成功后另采4个PTO DFX窗口。
历史单层B24曾报output_count断言，本轮当前冻结源码未复现；不据此声称旧断言根因已经修复。

任务task_20260928_233705_354148710045由auto分配card0，已完成退出0；[汇总](RESULTS.md)、[原始读数摘录](evidence.json)。
Native/PTO完整CSA均值1489.736/1362.396μs，PTO低8.548%；P95 1498.500/1374.220μs，
最大1500.380/1398.360μs。PTO P95比P50高0.895%，本批20样本无明显异常尾部，不能代替EP16长尾结论。
两侧各20样本、两份PyTorch JSON及四DFX窗口齐全；八类自身重放、图对eager、A→B→A、metadata/保护区和Top-K结构通过。
四张目标根权重均Native/PTO format29且同地址。配置的ring足以执行本档；未测出最小需求，不向其他形状外推。
跨Native零容差比较存在既有性能版算术差异，x_out max_abs=0.03125、RMSE=0.004002；
完整逐项差异保留于evidence.json，未声称Native逐bit或新增B24整模型token/DSpark验收通过。
[CPU准备](prepare.py)、[设备入口](run.sh)、[任务](task.txt)。
[运行时参数实现](../../../../vllm_ascend/ops/pypto/variant.py)；失败诊断位于
`timing/ascend/debug/device-N/device-*.log`，重点区分Heap ring耗尽与output_count等执行错误。
