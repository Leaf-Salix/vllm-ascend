# 单入口纯DSL精度CSA：真实整模型性能（2026-10-18）

## 结论

两档预热及全部测量批次均与冻结Native黄金的完整192 token和接受事件一致、零抢占；单入口纯DSL精度版存在明显性能回退。三轮/两轮均稳定，不能用旧七调用/C++版或早期性能变体的速度代表当前实现。

下表是target整模型forward及解码周期，单位ms，不是单层CSA：

| 配置 | Native forward | PTO forward | PTO/Native耗时 | Native周期 | PTO周期 | PTO/Native周期 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K/B40 | 89.482 | 1218.433 | 13.616倍 | 105.575 | 1234.349 | 11.692倍 |
| 128K/B24，窗口并发21 | 79.730 | 758.039 | 9.508倍 | 92.293 | 770.728 | 8.351倍 |

本轮不改生产数学，未做模块profile，因此没有把总回退直接归因到某一阶段。HC-pre的HF32等价计算等实现差异需要另做受控分解；本轮不引入额外优化或改变精度边界。

## 实现与环境身份

分支 `dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`，数学提交 `a20bcdb76`。
HC-pre至HC-post单一PTO顶层入口、内部七阶段、62tensor ABI；15个Python数学文件、无手写C++、无编译器/SDK改动、保留真正TND。21层CSA完整替换，20层HCA及FFN保持Native。
[实现与上一轮逐bit证据](DSV4_FLASH_CSA_TND_SINGLE_ENTRY_DSL_20261018.md)独立保留；本轮不再开启逐步GPU输入/浮点tensor取证，不把token一致扩张为本轮所有内部tensor已逐bit观测。

冻结source manifest：`7b0650b926a557241b622eba4f032731e7b21d75d807c3b6baccfe689b29c960`，3362文件；归档15个生产文件与实测/本地数学逐字节相同。快照是历史完整源加本次数学覆盖，不将新增文档混入后伪称同一manifest。
Python3.12、Torch2.10.0+cpu、Torch-NPU2.10.0.post2、vLLM0.25.1、vLLM-Ascend0.25.1rc1；CANN/ATB9.2.0-beta.2、PTOAS0.66、原PyPTO私有3e87a843基线、Simpler a54c0509不变。调度 `--ptoas 0.63`不代表runner实际工具版本，显式PTOAS_ROOT和preflight为准。

## 与nalinaly方案的对应关系

用户指定的[nalinaly报告](https://github.com/nalinaly/vllm-ascend/blob/dsv4-flash-pto-v0.25.1rc1/tests/pypto_test/low_acceptance_20261001/RESULTS.md)在线下载SHA与冻结参考完全相同：`6f1820207580e74071a7ea0947367cf6edcfa2daf60876708338a279c2f939e9`，对应f043204138参考记录。
沿用40/24道不同问题、恢复同份Native导出的target/draft cache/state、greedy/ignore EOS、DSpark出5验6、每请求192 token，逐请求比较输出、接受统计及有序事件，检查全部16rank和零抢占。
使用自己已验收的固定Native bank；公开问题、顺序和chat后缀token已核验，不声称原nalinaly前缀/cache全部payload相同。DP16是同问题副本，不是640/384个独立题目。本轮只比较Native与CSA替换，HCA保留Native，不新增Native2精度任务。

| 参数 | 8K | 128K |
| --- | --- | --- |
| TP / DP=EP | 1 / 16 | 1 / 16 |
| 每卡提交请求 | 40 | 24，允许等待 |
| 计时窗口实际请求/真实token | 40 / 240 | 21 / 126 |
| FULL捕图档位 | 6、240 | 6、144 |
| KV块 / utilization | 22000 / 0.95 | 24000 / 0.97 |
| max-num-batched-tokens / NZ / event | 400 / 2 / 1 | 400 / 2 / 1 |
| 确定性 / HCCL / EPLB | level1 / true / 关闭 | level1 / true / 关闭 |
| 预热 | 完整192 token | 完整192 token |
| 测量批次 | 3×192 token | 2×192 token |
| 汇总样本forward / cycle | 60 / 57 | 40 / 38 |

128K减少重复轮数是因为8K纯DSL任务实际755秒，给更长档留出15分钟内余量；两臂相同轮数，192输出长度和精度门禁不变，独立v2目录，不改8K冻结助手。

## 计时和精度门禁

每臂独立16卡任务，先Native并审计，通过再PTO；先8K配对通过才进入128K。
每轮从同bank独立恢复192 token生成，跳过8个满档step后，测连续20个target forward和19个begin→begin周期。事件提前创建、批末统一同步；同一步取16rank最大，再合并各轮分布。
forward是target `_model_forward`设备时间；cycle还含logits、sampling、DSpark draft、调度间隙和CPU metadata观测。模型加载、编译、bank恢复不计入这两个主指标；批次wall另留原始记录，不能当算子时间。
周期输出计数以实际sample output及positions推进互证；推算的aggregate token/s只是代理，不是全局同步服务吞吐。

预热和每个测量批次结束后检查全请求192 token、完整有序接受事件、计数器和零抢占；首个已观察差异则停止后续批次/配置。这不是逐forward在线首差检测。测量窗口CPU请求顺序、query bounds、positions、真实/物理token、图桶和DP向量跨臂/轮次/rank相同；非权威CPU staging input IDs保留但排除等值门禁，未新增GPU D2H hook。
21层真实单根6/主档warmup/capture、namespace与62ABI、Native HCA、13包静态实际reselect一次及验证复用、测量期编译/cache无新增及错误0均独立审计。无DFX、profiler或逐层tensor hook。
各臂只有一个进程运行，轮次为进程内重复；未反转执行顺序或新增独立进程置信区间。

## 原始分布与各轮

### 8K/B40

| 指标ms | min | mean | max | P95 |
| --- | ---: | ---: | ---: | ---: |
| native forward | 87.546 | 89.482 | 101.336 | 90.311 |
| native cycle | 103.628 | 105.575 | 118.623 | 116.984 |
| pto forward | 1216.877 | 1218.433 | 1222.756 | 1219.164 |
| pto cycle | 1233.120 | 1234.349 | 1238.727 | 1237.201 |

| 轮次 | Native forward ms | PTO forward ms | Native周期ms | PTO周期ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 89.633 | 1218.263 | 105.775 | 1234.085 |
| 2 | 89.524 | 1218.373 | 106.187 | 1234.156 |
| 3 | 89.289 | 1218.663 | 104.762 | 1234.805 |

### 128K/B24

| 指标ms | min | mean | max | P95 |
| --- | ---: | ---: | ---: | ---: |
| native forward | 78.301 | 79.730 | 81.678 | 81.245 |
| native cycle | 90.915 | 92.293 | 95.327 | 94.176 |
| pto forward | 756.626 | 758.039 | 760.242 | 759.395 |
| pto cycle | 769.559 | 770.728 | 773.453 | 772.105 |

| 轮次 | Native forward ms | PTO forward ms | Native周期ms | PTO周期ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 79.644 | 758.042 | 92.348 | 770.752 |
| 2 | 79.816 | 758.037 | 92.238 | 770.704 |

## 完整输出与运行时间

8K两臂预热加三轮合计983,040个token；128K两臂预热加两轮合计442,368个token。全部对各自Native黄金完全一致，接受事件/统计一致，零抢占。不是将不同bank的两档或nalinaly历史接受均值作同输入精度比较。

| 任务 | ID | 排队 | 实际执行 | 状态 |
| --- | --- | ---: | ---: | --- |
| 8K Native | `task_20261018_115149_27340938505` | 3分12秒 | 7分23秒 | exit0 |
| 8K PTO | `task_20261018_121140_375232826319` | 45分41秒 | 12分35秒 | exit0 |
| 128K Native | `task_20261018_131246_277112432062` | 0分59秒 | 7分53秒 | exit0 |
| 128K PTO | `task_20261018_140959_9812614581` | 0分00秒 | 11分25秒 | exit0 |

实际执行按done记录START_TIME→FINISH_TIME，列表duration包含排队，不作占卡时间。780秒是原提交命令参数；done记录可能不保留MAX_TIME，归档将计划值与实际起止证据分开，不伪称缺失字段已观测。
所有任务实际均低于15分钟，结束后无本轮残余任务、无自动调度。

## 准备、失败与复现

CPU计时/单根资格11项测试在8K及128K-v2均通过，独立助手审查通过。无设备预检先修正新source manifest打印字段；原资产检查器需在其原冻结数学source下验证历史黄金，再恢复本次source。所有失败日志随8K证据保留，没有放宽精度阈值、改变gold或生产数学。收集器v2支持128K新后缀，并明确计划上限来源；这不是NPU重跑。

复现先还原前述单入口精度记录的模型75分片、固定bank、Native黄金、3362源manifest及原私有运行环境；证据包不包含大模型/缓存payload或SDK二进制。使用现有227独立目录或按相同身份还原新目录：

```bash
BASE=/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex
J=$BASE/single-dsl-real-model-perf-20261018
CASE=8k
# 128K使用以下独立v2，两臂均两轮，先8K配对通过再继续：
# J=$BASE/single-dsl-real-model-perf-128k-v2-20261018
# CASE=128k
bash "$J/preflight.sh" "$CASE"
ARM=native
# 下一行先提交Native，通过独立审计后才ARM=pto重新提交。
task-submit --device auto --device-num 16 --max-time 780 --ptoas 0.63 \
  --env PYTHONNOUSERSITE=1 \
  --env ASCEND_HOME_PATH=/data/pyptouser/yejia/vllm-cann92-main/env/Ascend/cann-9.2.0-beta.2 \
  --env PTOAS_ROOT=/data/pyptouser/yejia/vllm-cann901-csa-0251rc1/experiments/nalinaly-71153bb3/reference-env/bin \
  --env CCACHE_DIR=/data/pyptouser/yejia/vllm-cann901-csa-0251rc1/src/pypto/.ccache \
  --env CMAKE_BUILD_PARALLEL_LEVEL=4 "bash $J/run-case.sh $CASE $ARM"
```

未来复测须新建助手/输出标签、重新冻结helper SHA，不覆盖本轮路径；两臂source/助手/SDK/bank/参数必须一致，排队与运行期间不改助手或JIT源。先本机/CPU环境运行 `pytest -q helpers/test_perf_probe.py helpers/test_static_qualification.py`，再无设备preflight。
占卡后每5分钟以内监测状态/STOP/实际进度，不创建sche。Native终态后用归档 `helpers/analysis/analyze.py <native输出> --output <新报告>` 独立审计，PTO终态后追加 `--pto <pto输出>` 做配对；新报告路径必须不存在。只有8K配对通过才128K。分析器重读黄金和完整source/helper SHA，不只信任PASS标记。
单根资格和bootstrap保留原实际静态选择检查，不得删除重选/包身份验证来缩短时间。

## 证据

[结构化8K对照](evidence_tnd_precision/single_entry_dsl_performance_20261018/comparison-8k.json)、
[结构化128K对照](evidence_tnd_precision/single_entry_dsl_performance_20261018/comparison-128k.json)、
[任务实际时间](evidence_tnd_precision/single_entry_dsl_performance_20261018/task-times.json)、
[文件索引](evidence_tnd_precision/single_entry_dsl_performance_20261018/index.json)。

[8K原始归档](evidence_tnd_precision/single_entry_dsl_performance_20261018/pair-8k-evidence-v2.tar.gz)、
[128K原始归档](evidence_tnd_precision/single_entry_dsl_performance_20261018/pair-128k-evidence-v2.tar.gz)。
归档包含每rank三轮/两轮及预热完整token/events、计时原样本、Native黄金、冻结助手、实测15数学源、manifest、静态/运行身份、任务原日志及实际时间。全部载荷SHA/字节数本地复核，原脚本/JSON/log不格式化；索引记录tar及各结构化结果SHA。
本轮未改生产数学或旧性能/精度包；新增文档与证据单独提交，原基准备份保留。

## 最终本地检查

8K的229份和128K的221份原始载荷SHA/字节数核验通过，独立复算所有rank输出/事件、maxrank样本和上述分布、工作量、单根/静态/cache资格及实际运行时间通过。
本次新增Markdown/JSON的增量pre-commit和manual markdownlint通过，CPU两套11项测试通过。
全量 `bash format.sh ci` 在隔离临时clone执行exit1，历史tracked源码/原始诊断材料仍存在格式、拼写、导入等问题，本机缺shellcheck；错误未涉及本次新增结果文档/证据路径。没有把全量格式化改动带回正式源码，未修改历史原件来凑门禁。
[全量门禁日志](evidence_tnd_precision/single_entry_dsl_performance_20261018/full-format.log.gz)及增量、CPU日志均列入索引。
