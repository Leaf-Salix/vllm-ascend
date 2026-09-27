# 当前原 cache 路径的 atomic 受控诊断

算子71153bb3，冻结目录`.cache/csa-forward-boundary-71153bb3`。不改算术源码或cache布局。
只切换已存在的`VLLM_ASCEND_PTO_CSA_ATOMIC_ADD`。Native控制、mode2、det0、HCCL=false、EPLB关。
QR/KV从跨核split-K atomic加法改为单核固定K顺序，因此此实验同时改变归约次序和分片量，
不能把结果只归因为硬件atomic抖动。

动机：七档GMM任务duration均增加；B8正式forward持续慢，8K/B16两个rank少接受一个草稿。
历史atomic0实验是source-split/ordered缓存和旧分组，未覆盖当前原cache版本的受影响两档。
本次不扩大七档，不先改生产默认值。

先运行[单卡命令](run_layer.sh)：128K/B8和8K/B16，各atomic1/0的20次图重放，
记录本体代价、P95/max和已有保护区/Top-K检查。固定归约全CSA曾在adaptive分支单卡输出状态通过；
此处补两档当前部署形状的性能成本，然后才做真实EP16干预。

是否保留取决于正式forward、token和DSpark共同结果，不以单卡或不同轮profile胜负决定。

## 单卡门禁完成，进入EP16

task_20260928_045034_378240623103退出0，[原始样本与数值诊断](single/report.json)。

| 档位/μs | atomic1 | atomic0 | 变化 |
| --- | ---: | ---: | ---: |
| 128K/B8 完整CSA均值 | 862.903 | 868.761 | +0.68% |
| 128K/B8 P95/max | 876.200/878.620 | 880.580/881.820 | |
| 8K/B16 完整CSA均值 | 805.604 | 780.862 | −3.07% |
| 8K/B16 P95/max | 827.860/831.900 | 789.960/793.220 | |

Native控制长档1003.695→1010.290μs、短档905.961→919.395μs，存在约0.66%/1.48%漂移；
上述是PTO自身观测差，不放大成严格收益承诺。metadata/保护区、有限值、Top-K结构通过；
Native浮点零容差仍FAIL。长档Top-K集合替换350→350，短档366→365，归约数值并非完全不变。

模型task_20260928_045503_382273916328已提交：只128K/B8与8K/B16，双方新采Native控制，
atomic0、det0/HCCL=false保持配对，正式8步后连续10步forward；独立profile沿用默认不加额外事件。
[模型命令](run_model.sh)、[严格收集器](collect_model.py)。没有改生产默认值、没有以单卡门禁冒充DSpark验收。

## 长档EP16先完成

128K/B8正式Native56.307→PTO55.830ms（−0.85%），P95 57.290→56.567、max57.511→57.028；
每步最慢rank56.831→56.416ms（−0.73%）。32768输出token零差异、16rank DSpark一致。
[当前正式结果](model/RESULTS.md)。收益尚小，短档仍在运行，不改默认或提前扩大七档。

rank0另轮profile的专家GMM总时长：Native4.749/PTO4.936ms，增量0.187ms；
上次atomic1同算子为4.802/6.325ms，增量1.523ms。
与固定归约减少下游专家工作差距相符，但没有在本轮直接采集专家索引，
不能把这两次profile差额当正式forward精确归因，也不能把其余算术差异视为已解决。
本轮profile CSA body21.391→18.497ms，FFN25.807→23.516ms；首层等待不同，继续分开记录。
