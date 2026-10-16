# TND精度版：按nalinaly低接受率方案对齐验收

日期：2026-10-16。分支`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
数学源码为`cbc810c8afab47c2425068049440424f014dcde3`，沿用128K修复结果；不修改性能版或HCA。
精度版保留Native HC/norm、WKV、Indexer权重投影及O-proj边界，CSA主体运行PTO TND kernel；本轮不调整实现。

## 参考与输入

[参考RESULTS](https://github.com/nalinaly/vllm-ascend/blob/f04320413804103fe31bedd5a2175b5021bcd91d/tests/pypto_test/low_acceptance_20261001/RESULTS.md)及同目录README、baseline与cases固定于`f04320413804103fe31bedd5a2175b5021bcd91d`。
本次只对照全Native与TND精度CSA＋Native HCA，不运行HCA替换矩阵。
公开40题的正文、顺序与本地准备入口完全一致；实际tokenizer/DeepSeek chat编码重新计算后，
8K主档40题与128K主档24题的实际token后缀逐一相同。

每臂恢复同一份自有Native导出的target/draft cache/state；未重新prefill或筛选简单题。
原nalinaly完整bank无读取权限，因此仅称同题、同编码及同验收协议，
不称完整prefix/cache状态与其原始bank逐字节相同，也不直接将两份报告的接受率差值归因于算法。

## 协议对齐与明确差异

| 项目 | nalinaly参考 | 本次TND对照 |
| --- | --- | --- |
| 主档 | 128K/B24、8K/B40，各192 token | 长档复用本日完整结果；短档补测 |
| 模型与并行 | DSV4 Flash W8A8、TP1、DP=EP16 | 相同，真实75分片权重 |
| 题目 | 固定40种不同问题池 | 24/40种题，不能将DP副本算成不同问题 |
| 采样 | greedy、DSpark提出5个草稿 | 相同，真实接受0–5，不强制接受 |
| 接受验收 | 全六种边界，按草稿轮加权均值3–4、目标约3.8 | 按相同口径检查，事件合计与原生计数核对 |
| 数值配置 | CANN9.2、TMR、NZ2、atomic0、det1、HCCL确定性、EPLB关闭 | 相同配置，具体组件提交另记 |
| 图 | FULL_DECODE_ONLY，长档[144]、短档[6,240] | 长档[6,144]，短档[6,240]；披露NONE首轮及实际FULL计数 |
| event | Native0、PTO1 | 两臂统一event1 |
| 容量与调度 | 自动KV容量，不保证跨臂轨迹相同 | 固定共同24000 blocks，逐轮核对实际请求/输入/DP轨迹 |
| 替换 | 原报告含多组CSA/HCA组合 | 只替换21个CSA，20个HCA保持Native |
| 重复性 | Native独立重复 | 先Native控制组，再Native witness，资格通过才交CSA |
| 对比 | 全token、DSpark统计/事件、跨DP、抢占 | 相同；额外rank0的43层边界与outer逐bit检查 |
| 性能 | 不作性能结论 | 诊断有同步/落盘，同样不测性能 |

三臂共享数学源码、模型、bank、确定性和容量。精度观测范围为全部16rank的实际输入与最终token/events，
以及rank0记录的浮点边界；不能扩展为全部rank的所有中间tensor。
每个NPU任务单独提交，`--max-time 780`；首差即停止扩大测试，排队时间与实际占卡时间分开。

## 当前结果

| 档位 | Native重复 | CSA对Native token | CSA对Native接受事件 | rank0模块边界 |
| --- | --- | --- | --- | --- |
| 128K/B24 ×192 | 一致 | 0/73,728位置不同 | 0/384副本请求不同 | 16,614条摘要逐bit一致 |
| 8K/B40 ×192 | 完整重复一致 | 待测 | 待测 | Native黄金10,863条已保存 |

128K为24种题、384条副本请求：输出、DSpark统计、事件不同题目均0/24；两臂跨DP均0/360副本差异，零抢占。
平均接受3.8264033264，0–5各长度均覆盖，实际在场峰值21。
长档完整轨迹/边界、环境及首次JIT与严格READY复用资格的区别见
[128K192详细记录](DSV4_FLASH_CSA_TND_192_ADMISSION_RETEST_20261016.md)。

8K Native首轮已完整输出640条副本请求、122,880个token，零抢占；实际并发峰值40。
平均接受3.7035909921，0–5直方图为1696/2432/2816/2592/2240/14512，符合主档要求。
完整生成窗口51轮，为1轮NONE/40 token及50轮FULL/240 token；完成后的原始空尾另行严格审计，不抹掉原始轮数。
首轮及Native重复的小型原始证据已保存；Native重复CPU资格通过：全部16rank输入、token、接受统计与事件一致，rank0 outer逐bit一致。
Native2保存51轮、43层/21CSA共10,863条模块边界，供CSA对拍；Native1没有内部stage，不称两遍Native的内部tensor全部比较过。

Native1为`task_20261016_105617_31836166643`，约7分钟实际执行，exit0；此前约40分钟仅排队。
Native2为`task_20261016_114627_49838428141`，约8分钟实际执行，exit0；此前长时间仅排队。
CSA为`task_20261016_125003_260178911192`，约5分钟执行后exit1：生成前的显存预检失败。
24000 blocks计划上界21,955,846,144 bytes，rank0可用23,840,948,224 bytes，
分配后预计仅余1.756 GiB，低于2 GiB预留；尚未开始生成，不能记为精度失败或通过。
原生profiling前后已执行gc及empty_cache，因此未增加重复清理或放松门禁。

另建共同22000 blocks三臂，保留原失败证据；其余40题、192 token、模型、数学源码、图档与观测不变。
重新完成Native两次重复后才提交CSA，不将不同容量的轨迹混为受控对照。
新的Native控制为`task_20261016_130947_40242198337`。
8K CSA补测未完成前不作精度结论。本页将补齐替换结果和首差信息。

[结构化记录](evidence_tnd_precision/nalinaly_protocol_alignment_20261016.json)保留协议、输入身份与两档结果。
