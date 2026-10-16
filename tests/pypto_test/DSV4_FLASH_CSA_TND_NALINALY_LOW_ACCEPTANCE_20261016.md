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
| 主档 | 128K/B24、8K/B40，各192 token | 长档复用本日完整结果；短档本轮完成 |
| 模型与并行 | DSV4 Flash W8A8、TP1、DP=EP16 | 相同，真实75分片权重 |
| 题目 | 固定40种不同问题池 | 24/40种题，不能将DP副本算成不同问题 |
| 采样 | greedy、DSpark提出5个草稿 | 相同，真实接受0–5，不强制接受 |
| 接受验收 | 全六种边界，按草稿轮加权均值3–4、目标约3.8 | 按相同口径检查，事件合计与原生计数核对 |
| 数值配置 | CANN9.2、TMR、NZ2、atomic0、det1、HCCL确定性、EPLB关闭 | 相同配置，具体组件提交另记 |
| 图 | FULL_DECODE_ONLY，长档[144]、短档[6,240] | 长档[6,144]，短档[6,240]；披露NONE首轮及实际FULL计数 |
| event | Native0、PTO1 | 两臂统一event1 |
| 容量与调度 | 自动KV容量，不保证跨臂轨迹相同 | 长档24000、短档22000 blocks，各档三臂容量相同；逐轮核对实际输入/DP轨迹 |
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
| 8K/B40 ×192 | 一致 | 0/122,880位置不同 | 0/640副本请求不同 | 10,863条摘要逐bit一致 |

128K为24种题、384条副本请求：输出、DSpark统计、事件不同题目均0/24；两臂跨DP均0/360副本差异，零抢占。
平均接受3.8264033264，0–5各长度均覆盖，实际在场峰值21。
长档完整轨迹/边界、环境及首次JIT与严格READY复用资格的区别见
[128K192详细记录](DSV4_FLASH_CSA_TND_192_ADMISSION_RETEST_20261016.md)。

8K为40种题、640条副本请求：输出、DSpark统计、事件不同题目均0/40；两臂跨DP均0/600副本差异。
全部16rank的实际生成输入一致，完整192 token及逐轮接受事件一致，零抢占，实际峰值40。
平均接受3.7035909921，0–5直方图为1696/2432/2816/2592/2240/14512，符合主档要求。
生成窗口51轮：1轮NONE/40 token及50轮FULL/240 token；没有观察到同轮请求query长度不等，
因此本页不能单独证明任意不等长FULL图的逐bit一致性。

Native2与CSA的rank0 outer hidden/aux/MTP逐bit一致，43层/21CSA共10,863条模块边界摘要逐bit一致。
Native1没有内部stage，因此Native重复资格仅称已比较outer，Native2内部边界用于CSA黄金。
16rank的21个CSA精度服务、图捕获选择及模型FULL replay检查通过，20个HCA保持Native；
选择计数来自warmup/capture/Python资格入口，不称其为每层运行时replay hook计数。

## 8K执行与容量调整历史

原24000 blocks组的Native两次运行通过：
`task_20261016_105617_31836166643`、`task_20261016_114627_49838428141`。
CSA `task_20261016_125003_260178911192`约5分钟后exit1，尚未开始生成：
KV计划上界21,955,846,144 bytes，rank0可用23,840,948,224 bytes，
预计仅余1.756 GiB，低于2 GiB预留；该资源失败不记为精度失败或通过。
原生profiling前后已gc及empty_cache，未重复清缓存或放松门禁。

新建共同22000 blocks三臂，保留全部旧证据；源码、SDK、bank、40题、192 token、图档和观测均不变。
新计划上界20,134,406,144 bytes；CSA rank0实际可用23,841,312,768 bytes，
预计剩余约3.45 GiB，通过原门禁。先重新验证同容量Native重复，再提交CSA，未混用旧黄金。
另外核对容量调整前后的Native最终token/events完全一致；该旁证不代替新三臂受控对照。

| 新任务 | 角色 | 状态 | 约实际运行时间 |
| --- | --- | --- | --- |
| `task_20261016_130947_40242198337` | Native控制 | exit0 | 6分15秒 |
| `task_20261016_131636_170271863` | Native见证 | exit0 | 9分44秒 |
| `task_20261016_132730_25928429019` | TND精度CSA | exit0 | 8分6秒 |

生成后Native原始forward数为55（rank5为52），CSA为54（rank5为52）；
全部51轮实际生成完整对齐，之后的空尾分别4/1与3/1轮。
空尾原始记录全部保留，严格核验空请求、FULL6及DP padding；没有将不同原始总轮数说成全程一致。

## 缓存资格与产物身份

8K数值资格通过；默认strict READY复用资格仍失败，两个结论分别保存。
各rank READY命中为0，rank15发生一次generation/binary构建，耗时15.7538秒，其他rank等待；
bypass、invalid、storage error、forced rebuild在全部16rank均为0。
没有为取得READY命中另占卡重跑，也没有把首次构建改写成复用成功。

CPU重新核验冻结源码、工具链和ABI，并用实际任务binary manifest唯一定位READY产物；
全部206项文件在本次构建与READY中均存在且SHA一致。
Specialization取自实际不可变manifest，未独立重新specialize。
实际构建产物身份检查通过，不改变strict READY失败记录；这些诊断运行不提供性能结果。

## 证据与结论范围

短档最终小型原始证据396份文件均验证SHA，三个任务均exit0；原始输入、token/events、边界及身份链已通过独立复核。
最终archive SHA256：
`8096508d7294128c3d35e5b5b3e514d771e1d31e369dac38af5dda952df72f7b`。
24000失败组单独封存SHA256：
`e5739d6f02f769ac4215a7c49a4787cbc5e27663c304f0918c51960f8772c380`。

已验证真实模型、同一自有bank下两档192 token的CSA替换输出与Native完全一致，
并且rank0记录的浮点边界逐bit一致；本轮没有新改数学源码或性能版。
不能扩大为任意输入、全部rank的所有中间值均逐bit一致，也不称原nalinaly完整bank已复现。

[结构化记录](evidence_tnd_precision/nalinaly_protocol_alignment_20261016.json)保留完整协议、输入身份、两档结果与失败历史。

完整可复现方案、冻结脚本与原始证据入口见[复现手册](DSV4_FLASH_CSA_TND_PRECISION_REPRO_20261016.md)。
