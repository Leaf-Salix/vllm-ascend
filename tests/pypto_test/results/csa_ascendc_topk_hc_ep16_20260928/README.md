# 最新AscendC两项策略组合后的真实EP16验证

算子冻结为d1f170ff，工作树`.cache/csa-ascendc-topk-hc-ep16-d1f170ff`。
在e58固定K KV基底上，包含15cb35c3按实际长度选择二/四路Top-K，以及d1f170ff的HC输入/RMS复用。
两项单变量长短层、必要归并边界、T60尾行及精度版共用函数已先在单卡通过；
本轮验证组合后的真实模型，不把两项局部核内收益相加预测forward。

任务`task_20260928_133801_289312424331`，task-submit统一申请16卡，最长5400秒。
任务完成、退出0；本轮两个档位均通过位置、token及DSpark检查。

- 128K/B16和8K/B16；每侧max_num_seqs=40，capture 24/48/96/144/192/240。
- 固定DeepSeek-V4-Flash-0731-w8a8及既定128K/8K bank；TP1、DP=EP16、DSpark出5验6、EPLB关闭。
- Native/PTO两侧mode2、atomic0、det0、HCCL_DETERMINISTIC=false；记录worker实际配置。
- 每请求先96 token预热，正式轮生成128 token；跳过前8个满档step后取连续10个decode forward。
- 两侧事件均生成前预建/首次record，正式步不新建事件、不加额外同步；入场按整批enqueue/DP barrier/resume。
- 另采3步PyTorch profiling；正式forward不含加载、编译、预热、草稿和采样，保留forward内等待。
- 128K先PTO后Native，8K反向；样本全部保留，报告所有rank及逐步最慢rank的均值/P95/max。
- 核对实际请求位置、逐token和32组rank DSpark；有差异时不生成可比性能结论。

单变量DFX仅解释各自改动，不改名为本次组合源码泳道；阶段出口再补同一最终源码的七档与泳道。
若P95异常，复用本轮已采集的入场/主机分项定位，不剔除异常或先盲目重测。

[设备命令](run_model.sh)、[正式forward/token/DSpark收集器](collect_model.py)。

## 本轮实测

| 档位 | Native/PTO forward均值 ms | PTO变化 | Native/PTO P95 ms | Native/PTO最大值 ms |
| --- | ---: | ---: | ---: | ---: |
| 128K/B16 | 73.229/70.274 | −4.03% | 74.501/71.418 | 75.026/71.701 |
| 8K/B16 | 65.363/62.683 | −4.10% | 66.287/63.494 | 66.454/63.647 |

共131072输出token零差异、32组rank DSpark一致，正式请求位置相同。
逐步最慢rank均值分别73.690→70.711、65.725→63.157ms，20/20个配对步骤PTO更快。
每档160个rank×step样本相关，不能称160次独立实验；两个单变量核内收益也不能相加当作本轮模型收益。
没有同轮旧e58模型控制，不能从跨轮均值差单独归因四路Top-K或HC融合。

两侧正式窗口均未出现>2ms设备相对入场异常或GC，不剔除样本，不宣布旧尾部已修复。
[完整forward与所有rank样本](model/RESULTS.md)、[相对入场](ARRIVAL.md)、
[主机/GC](HOST.md)、[准备分项](PHASES.md)、[复用分析入口](analyze.py)。
完整主机标记保留在本地host.json/phases.json及原始rank记录，报告没有修改Runner或GC行为。

任务退出后才运行[离线profile导出](export_profiles.sh)，避免CPU解析与正式设备计时重叠；
从已有四份rank0采集提取模型CSA和相邻层差值，不加跑设备或混用旧泳道。

## 模型内CSA与原短档波动

每档独立三步rank0 profile，共63个HC_pre→norm→CSA→HC_post区间，包含首次metadata。
128K/B16 Native/PTO均值1294.32/1115.04μs（−13.85%），8K/B16为981.28/796.76μs（−18.80%）。
这不是正式十步forward的精确分账，也不是单卡DFX；8K/B16的750μs目标仍未达到。

短档第三步第12→14层仍出现769.82→807.28μs；设备Worker本身762.04→793.10μs，问题没有消失。
Native同位置976.04→1023.44μs，但其已记录任务区间并集938.78→932.98μs，
并集外间隙37.26→90.46μs；任务记录含控制事件，不代表纯计算忙时。
两侧同位置变慢不能证明同一根因，当前没有对应两层的逐incore DFX，不能归因到Score或声称sync_start已修复。

[完整模型分项](model/MODEL_GAP.md)、[相邻CSA及三步第12/14层展开](model/ADJACENT_CSA.md)、
[四份PyTorch JSON下载目录](download/README.md)。保留全部区间及相邻对，不剔除慢层。
