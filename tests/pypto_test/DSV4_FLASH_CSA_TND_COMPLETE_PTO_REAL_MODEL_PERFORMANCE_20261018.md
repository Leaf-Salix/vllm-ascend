# 完整PTO精度CSA：真实整模型性能（2026-10-18）

## 结论

当前完整PTO精度版两档均保持既有Native黄金的192 token和完整接受事件；
整模型性能明显落后Native，不能沿用旧性能变体的提速结论。
本轮没有修改生产数学实现，也没有做新增模块profile，无法仅凭总时长判定具体回退模块。

| 配置 | Native forward ms | 完整PTO forward ms | PTO延迟增加 | Native周期ms | PTO周期ms | PTO周期增加 |
| --- | --- | --- | --- | --- | --- | --- |
| 8K/B40 | 89.114 | 182.108 | 104.35% | 104.625 | 197.812 | 89.07% |
| 128K/B24，实际并发21 | 79.238 | 138.248 | 74.47% | 91.442 | 150.612 | 64.71% |

每档每臂3轮，均值口径为每个相同步骤取16rank最大设备时长，
60个target forward和57个解码周期；不包括模型加载、bank恢复和编译。

## 范围

当前分支 `dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`，
完整数学实现 `425ae1f11`，记录前HEAD `32726e8c1`。
显式变体 `tnd_full_precision`，21层CSA从HC-pre到HC-post完整PTO替换；
HC-pre linear和inputNorm square-sum为自写PTO C++ extern，其余PTO DSL，
Native保留metadata producer、cache所有权、connector、FFN与20层HCA。
本轮不修改数学实现；不与10月16日保留部分Native数学的精度版混同。

## 方法

先8K/B40 Native→完整PTO，通过再128K/B24 Native→完整PTO。
每臂单独16卡任务，780秒上限。真实模型DeepSeek-V4-Flash-0731-w8a8，
TP1 DP=EP16，DSpark出5验6，greedy/ignore EOS，每请求192 token。
8K每臂96 token预热；128K每臂完整192 token预热，
随后均为3轮独立恢复相同bank的192 token生成。
每轮跳过8个满档step，采20个连续target forward、19个begin→begin周期；
计时事件预先创建，批末统一同步。按每步16rank最大设备时长再合并三轮分布。

forward是target整模型 `_model_forward` 设备时间，不是单层CSA；
周期包含logits/sampling/draft/调度间隙和CPU metadata观测开销，
跨rank最大周期推算的aggregate吞吐只是代理值，不是同步全局服务吞吐。
加载、bank恢复和启动不计入forward/cycle；另行记录批次wall但不作为主指标。
无DFX/profiler/模块tensor hook/逐步D2H位摘要。

精度门禁为每批完整token和完整接受事件对既有Native黄金结果，零抢占，
失败则停后续轮次和配置；不是逐forward在线首差停止。
Native和PTO两臂、三轮、16rank的CPU请求几何/positions/query bounds/图桶/DP向量一致；
非权威的CPU staging input IDs保留但不作为等值门禁，不宣称本轮重新逐步校验GPU输入或内部浮点bit。
测量轮前后编译/cache计数不得增长，不允许新增错误。

8K每卡40个不同问题，FULL [6,240]，KV22000/util.95；
128K每卡提交24条、实际并发由审计记录，FULL [6,144]，KV24000/util.97；
均max-num-batched-tokens400、NZ2、event1、确定性level1/HCCL true、EPLB关闭。
两臂相同官方13包静态选择：第一次实际reselect一次，后续验证后复用，禁止capture后再次真实reselect。

## 身份

- 冻结源3364文件manifest：`56d688b218d0d634024aed98a1568adac982f6005e5cd87373dd7553df8e6496`。
- 当前17数学文件逐字节绑定正式425ae实现，保留生产包SHA清单。
- helper22文件manifest：`a508197eb9a30e9e703b865b0527ec9c7288b319692938550bf2608562beb833`。
- 官方static metadata：`863a1e5daffc54aec0e9e8afb9f809f1bc6f4b3a2487e440615f3c942f36c602`。
- Python3.12，Torch2.10.0+cpu，Torch-NPU2.10.0.post2，vLLM0.25.1，vLLM-Ascend0.25.1rc1。
- CANN/ATB9.2.0-beta.2，PTOAS0.66；私有PyPTO3e87a843基线运行包，Simpler a54c0509。

## 准备验证

无设备preflight、bank检查、11项CPU门禁通过；独立审查无运行阻断。
历史源、黄金输出、性能版和精度基准均保留。

## 8K/B40结果

| 指标 | Native | 完整PTO | PTO延迟变化 |
| --- | --- | --- | --- |
| target forward均值ms | 89.114 | 182.108 | +104.35% |
| 解码周期均值ms | 104.625 | 197.812 | +89.07% |

| 轮次 | Native forward ms | PTO forward ms | Native周期ms | PTO周期ms |
| --- | --- | --- | --- | --- |
| 1 | 89.002 | 181.968 | 104.632 | 197.811 |
| 2 | 89.101 | 182.111 | 104.568 | 197.745 |
| 3 | 89.239 | 182.245 | 104.675 | 197.879 |

| 8K样本分布，ms | min | mean | max | P95 |
| --- | --- | --- | --- | --- |
| native forward | 87.259 | 89.114 | 90.094 | 89.892 |
| native 周期 | 103.638 | 104.625 | 105.530 | 105.357 |
| pto forward | 180.381 | 182.108 | 183.116 | 182.846 |
| pto 周期 | 196.898 | 197.812 | 198.986 | 198.625 |

两臂各3轮×16rank×40请求×192 token，合计737,280输出token，
全部与同一冻结Native黄金token及完整接受事件一致，零抢占。
每轮测量窗口FULL240、40请求、实际240 token，跨臂/轮次/rank的CPU工作量一致；
21层CSA在PTO臂确认为完整PTO，20层HCA保持Native，测量轮无新增编译/cache错误。
三轮都明显慢于Native；本轮没有模块分解，不能仅凭整模型时间将回退归因于某个具体模块。

8K原始证据173文件逐项SHA校验通过，独立复算确认统计/精度/工作量门禁。
8K归档SHA256：`6c3c7d3ed27237148663867c102694ebbd9d59c7d331f14ce26da61ab19591e7`，
3,117,686字节。三轮近似跨rank周期吞吐代理为Native28,718.12、PTO15,189.35 token/s；
该代理值含CPU观测，不能称服务端全局同步吞吐。

## 复现与证据规则

1. 校验每档证据tar的SHA，解包到全新目录；逐项核验 `evidence-manifest.json`。
   原始helpers、JSON/JSONL和日志保持字节不变，不把格式化副本当历史运行入口。
2. 还原[8K完整替换精度记录](DSV4_FLASH_CSA_TND_COMPLETE_PTO_8K_B40_20261018.md)
   和[128K完整替换精度记录](DSV4_FLASH_CSA_TND_COMPLETE_PTO_128K_B24_20261018.md)
   中的冻结step12源、bank、Native黄金输出和私有SDK。bank大tensor仅头部/stat校验，
   未宣称与nalinaly原bank字节相同；使用同一个已验收的自有Native bank。
3. 档案 `helpers/env.sh` 依赖原root-fix环境和旧private bootstrap；归档 `identity/` 保存其原始字节。
   完整source manifest、生产17文件SHA、runtime manifest和官方static metadata均随证据保存。
4. 为新复测新建独立job/输出目录，统一替换脚本中的实验标签/输出路径，不修改数学和测量参数。
   同一组Native/PTO必须使用完全相同的helpers和source；运行/排队期间冻结它们。
   输出目录已存在会拒绝，禁止覆盖旧基准。
5. 先无设备预检和11项CPU门禁，再按顺序分别提交Native、CPU审计、PTO、配对审计。
   8K配对通过后才继续128K。每个NPU任务 `--max-time 780`；不封装成长任务。
6. 每批生成结束检查全token/完整接受事件和零抢占，首个已观察差异停止后续轮次/配置。
   每轮计时检查连续FULL窗口、真实请求几何、确定性、完整替换资格、有限正计时、
   输出计数与positions推进一致、无新增编译及错误。

现有资产下本轮实际入口（在227；未来复测先换全新实验标签）：

```bash
export J=/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/full-pto-real-model-perf-20261018
export R=/data/pyptouser/yejia/lc128
bash "$J/preflight.sh" 8k
bash /data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/full-pto-real-model-perf-128k-v2-20261018/preflight.sh 128k
```

CPU门禁可在本机带pytest的Python环境对解包后的 `helpers/` 运行，
不需要vLLM或NPU，本轮使用workspace `.venv311`：

```bash
python -m pytest -q helpers/test_perf_probe.py helpers/test_static_qualification.py
```

不要在未分配NPU的shell直接执行 `run-case.sh`。
分别提交每一臂，`CASE=8k/128k`，`ARM=native/pto`；顺序遵守前述门禁：

```bash
export CASE=8k
export ARM=native
export SUFFIX=20261018
if [[ "$CASE" == 128k ]]; then
  export J=/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/full-pto-real-model-perf-128k-v2-20261018
  export SUFFIX=20261018-v2
else
  export J=/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/full-pto-real-model-perf-20261018
fi
task-submit --device auto --device-num 16 --max-time 780 --ptoas 0.63 \
  --env PYTHONNOUSERSITE=1 \
  --env ASCEND_HOME_PATH=/data/pyptouser/yejia/vllm-cann92-main/env/Ascend/cann-9.2.0-beta.2 \
  --env PTOAS_ROOT=/data/pyptouser/yejia/vllm-cann901-csa-0251rc1/experiments/nalinaly-71153bb3/reference-env/bin \
  --env CCACHE_DIR=/data/pyptouser/yejia/vllm-cann901-csa-0251rc1/src/pypto/.ccache \
  --env CMAKE_BUILD_PARALLEL_LEVEL=4 "bash $J/run-case.sh $CASE $ARM"
task-submit --status <返回的task-id>
```

Native单臂成功并验收后才能提交PTO；审计输出必须为尚不存在的新文件。

```bash
python3 "$J/analysis/analyze.py" "$R/perf-full-real-$CASE-native-$SUFFIX" \
  --output "$J/analysis/native-$CASE.json"
python3 "$J/analysis/analyze.py" "$R/perf-full-real-$CASE-native-$SUFFIX" \
  --pto "$R/perf-full-real-$CASE-pto-$SUFFIX" \
  --output "$J/analysis/comparison-$CASE.json"
```

CPU审计重新读取原黄金tokens/events和全部3364源文件SHA，不仅信任runtime的PASS标记。
归档和CPU预检不占用NPU。采用新实验标签时要一致修改审计路径。

## 128K短预热的失败与口径修正

原助手以96 token预热对192黄金前缀，在iteration0全部16rank首差均为q21；
按顺序检查的q00..q20前缀已通过。实际B24/峰值21意味着q21..q23需等待，
96上限可能改变后续请求入场时点，因此不能假定该预热与gold192使用相同执行轨迹。
现有证据不能确定是否由MoE或哪个具体算子引起，也不能将Native自身差异归为PTO回退。
原失败立即停止，无原助手128K PTO任务，无有效计时数据。
旧脚本在raise前未持久化实际token IDs，不能还原该次首差token索引；该观测缺口已在v2补上。

独立128K v2将两臂预热统一192，预热也硬比完整token和事件并保存原始requests；
配置/analyzer同步更新，STOP增加actual/gold tokens/events和首差索引。
数学、计时器、三轮192/8skip/20forward/19cycle不变。新11项CPU门禁/独立review/无设备预检通过。
v2助手22文件manifest：`a213991efbc3be5d3355ae1fd5ea4ce057f262a02ff940fc9e330cc877abda33`。
原失败证据106文件、527,020字节，SHA256
`6d2b48c4c04f7299040fe029505e0929f58d12c4e4710998ce0bbb8faf5a1839`。

采用完整192预热后，Native v2的预热与三轮测量均与黄金完整token/events一致；
同一冻结源/模型/bank下旧短预热未通过，而192对照通过，证明新方案闭合了所要求的验收口径。
不据此进一步推断旧差异的具体算子根因。

## 本地门禁

计时器/静态生命周期11项CPU测试通过（初版与128K v2各一次），bash语法与Python静态编译通过；
两版无设备preflight通过。全量 `bash format.sh ci` 在临时clone运行exit1：
已有源码/历史原件的ruff、codespell/typos、clang-format、Markdown/forbidden-imports问题，
以及本机缺shellcheck；没有将这些修改带回正式源码或改写冻结证据。
本次不改生产数学，新增文档和结构化JSON的增量pre-commit、两份Markdown的manual门禁均通过；完整全局门禁日志保留。

## 128K/B24结果（192完整预热v2）

| 指标 | Native | 完整PTO | PTO延迟变化 |
| --- | --- | --- | --- |
| target forward均值ms | 79.238 | 138.248 | +74.47% |
| 解码周期均值ms | 91.442 | 150.612 | +64.71% |

| 轮次 | Native forward ms | PTO forward ms | Native周期ms | PTO周期ms |
| --- | --- | --- | --- | --- |
| 1 | 78.973 | 137.803 | 91.079 | 150.101 |
| 2 | 79.714 | 138.251 | 92.076 | 150.657 |
| 3 | 79.027 | 138.690 | 91.169 | 151.080 |

| 128K样本分布，ms | min | mean | max | P95 |
| --- | --- | --- | --- | --- |
| native forward | 78.043 | 79.238 | 80.549 | 80.464 |
| native 周期 | 89.977 | 91.442 | 93.254 | 92.936 |
| pto forward | 136.961 | 138.248 | 140.275 | 139.720 |
| pto 周期 | 149.476 | 150.612 | 152.743 | 152.398 |

两臂3轮合计442,368输出token，与冻结Native黄金完整token/events全部一致，零抢占；
另两臂完整预热合计147,456 token及完整events亦一致。每卡提交24条，
计时窗口实际21条同时活跃、126真实token、FULL144，非24条同时活跃的性能。
两臂三轮的CPU测量几何和DP向量一致，21层完整PTO/20层Native HCA资格通过，测量轮无新增编译/cache错误。
三轮均明显慢于Native，没有独立进程重复或反转执行顺序的置信区间。
近似周期吞吐代理Native18,778.05/PTO11,400.55 token/s，不是服务端全局同步吞吐。

128K原始证据172文件、2,558,806字节，SHA256
`11905e132c92084280b05f37245f03ec33694d49448c46cc7d645f05feb70b64`。

## 最终复核与证据

独立复核8K的173份和128K的172份原始文件SHA、三轮60/57样本统计、
全部rank的token/events、实际工作量、完整替换和静态资格、测量期无新增编译门禁均通过。
128K另含完整192预热原始token/events验收。全局格式问题按前述保留，没有改数学源码。

[8K结构化对比](evidence_tnd_precision/full_pto_performance_20261018/comparison-8k.json)、
[128K结构化对比](evidence_tnd_precision/full_pto_performance_20261018/comparison-128k.json)、
[资产和任务索引](evidence_tnd_precision/full_pto_performance_20261018/index.json)。

[8K原始证据](evidence_tnd_precision/full_pto_performance_20261018/pair-8k-evidence.tar.gz)、
[128K原始证据](evidence_tnd_precision/full_pto_performance_20261018/pair-128k-evidence.tar.gz)、
[128K短预热失败](evidence_tnd_precision/full_pto_performance_20261018/failed-short-warmup-128k-evidence.tar.gz)、
[全量格式门禁日志](evidence_tnd_precision/full_pto_performance_20261018/full-format.log.gz)。

本轮全部NPU任务已完成，没有遗留排队或运行任务，也没有创建自动调度。

## 任务与时间

| 臂 | task | 排队 | 实际执行 | 状态 |
| --- | --- | --- | --- | --- |
| 8K Native | `task_20261018_084924_38888812273` | 0m00s | 6m54s | exit0 |
| 8K PTO | `task_20261018_085723_5788973517` | 0m00s | 8m43s | exit0 |
| 128K短预热Native失败 | `task_20261018_090811_8428314606` | 1m42s | 6m23s | exit1 |
| 128K Native v2 | `task_20261018_091952_121379118854` | 0m00s | 8m12s | exit0 |
| 128K PTO v2 | `task_20261018_092837_145150022505` | 5m33s | 9m58s | exit0 |

实际执行以调度done记录的START_TIME到FINISH_TIME计算；
调度列表的duration使用SUBMIT_TIME到FINISH_TIME，会把排队计入。
128K PTO列表15m31s=排队5m33s+执行9m58s；本轮最长实际9m58s，
每个任务提交上限780秒，全部小于15分钟。
[原始时间字段与计算记录](evidence_tnd_precision/full_pto_performance_20261018/task-times.json)。
