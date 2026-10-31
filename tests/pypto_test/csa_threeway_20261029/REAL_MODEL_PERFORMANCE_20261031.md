# 8K真实模型：Native、我方精度修复TND与nalinaly性能TND

## 测试目的与复用

比较完整CSA替换后的输出token、接受事件与性能。HCA、FFN、采样和调度框架保持公共版本。
复用已验收Native金标，不重复运行Native2或128K。
已有我方61,440个输出token与Native完全一致；原观测要求40请求同时活跃，实际采用8请求分波，
因此旧记录没有有效性能样本。本轮每方只新增一次模型运行，补采匹配实际入场的计时。

## 固定配置

| 项目 | 设置 |
|---|---|
| 模型 | DeepSeek-V4-Flash-0731-w8a8完整真实权重，43层 |
| 并行 | TP1，DP=EP8；本轮临时批准物理卡8–15，0–7禁用 |
| 请求 | 每rank相同40种不同问题，分5波×8；不是B40同时活跃 |
| 上下文/输出 | 固定前缀8192，加真实问题；每请求192输出token |
| 生成 | 温度0，DSpark出5验6；确定性level1和HCCL确定性开启 |
| 图/缓存 | FULL_DECODE_ONLY，capture sizes 6/240，KV blocks=4000 |
| 静态/SK | Static默认开启；未显式启用、验收新版SuperKernel，不标作SK=1 |
| 权重/并行优化 | NZ2、atomic add关闭、EPLB关闭 |
| 替换范围 | 完整21个CSA层HC-pre至HC-post；20个HCA层全部Native |
| 公共框架 | `41ba392f482bf88f7b6a797e2542443c81497338` |
| 我方CSA数学包 | `65bebf7f6b2e4f3f6492fd6463f73106506f3496` |
| nalinaly性能CSA包 | 准备阶段冻结`79ea6d3c8460338a172cb5570d14999d5a544262`，Python路径 |
| SDK | 共享已验收`csa-shared-20261024-group-dims`，不更换编译器或运行库 |
| Python栈 | vLLM 0.25.1、Torch 2.10.0+cpu、torch_npu 2.10.0.post2 |
| CANN | 9.2.0-beta.2 |
| 后端 | 两份PTO实现均使用Python提交路径（含AICPU），不是ACLNN后端对比 |
| PTO容量 | 双方同样window2048，heap=(256,512,256,256)MiB，默认dependency pool |
| 时间上限 | 每任务内部720秒、调度780秒；首个输出token差异停止 |

复用已生成的Native prefill bank，本轮测decode。三方共享同一fixed8192输入与缓存，
没有重新下载权重或生成新bank，也不声称与原作者原始bank逐字节相同。

## 计时与精度口径

1. **目标模型forward**：过滤真实非dummy、非draft、非capture的实际B8/FULL240步骤，
   丢弃8个预热步骤，采20个NPU Event样本/rank。每条保留实际T、padding T和positions。
   包含整模型CSA、HCA、FFN等；不是单层CSA耗时。
2. **公平性能门禁**：与本轮Native的实际请求数、实际token数、padding、positions逐样本核对；
   同时要求替换覆盖、完整有效采样、零抢占。缺样本或入场轨迹不一致时不计算公平速度比。
3. **分波客户端墙钟**：计时`generate_aligned_batch`调用，止于外部完成屏障之前；
   包含内部暂停/入队/入场协调/唤醒、草稿生成、验证、CPU逐token检查，不能称为纯设备生成耗时。
   每波取8个rank最大值后求和。仅作端到端辅助指标。
4. **不用相邻采样cycle作主要指标**：选中样本间可能夹有未选中步骤或分波等待。
5. **精度**：逐请求、逐token核对既有合格Native金标；token或输出长度首差停止。
   接受事件差异独立保存，不把事件不一致自动称为输出token不一致，也不把token一致称为内部逐bit一致。
6. **冻结**：只在快照旁新增内容hash的runner、observer、mixed副本；不改原冻结数学包、
   runner、scheduler或Native金标，不重装SDK。严格原postflight的事件失败仍保留。

`performance_qualified`只表示性能样本满足其门禁，不意味着token或事件验收通过；必须分列。
按各rank选中样本序号取max的辅助统计，没有独立公共forward步号证明，不能称为同一步EP8关键路径。
launcher逐token检查以既有v5金标为准；离线三方汇总的token比较以本轮Native计时目录为准，
两条证据分别记录，本轮Native还需离线核验与v5金标一致。

## 任务和复现

| 实现 | 任务 | 输出标签 |
|---|---|---|
| Native计时 | `task_20261031_085454_241324827611` | `last8-native-8k-wave-perf-v1` |
| 我方精度TND | `task_20261031_085811_249497130046` | `last8-ours-8k-wave-perf-v1` |
| nalinaly性能TND | `task_20261031_085811_24943951347` | `last8-nalinaly-8k-wave-perf-v1` |

同[主报告](RESULTS.md)的环境和数据根目录。先source公共环境及共享SDK环境。
`run_eight_card.sh`在进入模型前强制核对`TASK_DEVICE=8,9,10,11,12,13,14,15`。
Native使用`eightcard-source-v5/native`；我方使用同cohort的`ours`；
nalinaly使用`eightcard-nal-perf-v1/nalinaly`，由同Native框架副本只替换完整CSA包创建。
三方均传`--measure-waves`；两份PTO实现另传`--inspect-token-sequence`，
并显式设置`PTO_CSA_RING_TASK_WINDOW=2048`，reference指向`last8-native-8k-waves8-v5`。

以下在227的bash中执行；**未来复现仍须重新取得14卡授权**，本轮许可不构成默认权限。
`run_tag`必须选未使用的新值，启动器拒绝覆盖旧结果。Native金标和bank须已按主报告准备并验收。

```bash
r=/data/pyptouser/yejia/vllm-nal-tnd-native-align-20261027-codex
source "$r/src/tests/pypto_test/nal_native_alignment_20261027/env.sh"
unset PTO_CSA_RING_TASK_WINDOW
gold="$r/results/last8-native-8k-waves8-v5"
run_tag=repro-new-tag
run_arm() {
  local task_arm=$1 task_cohort=$2 task_label=$3 task_reference=$4
  local flags='--measure-waves'
  local options=(--env PYTHONNOUSERSITE=1 --env HCCL_DETERMINISTIC=true
                 --env PTOAS_ROOT --env ASCEND_HOME_PATH)
  if [[ "$task_arm" != native ]]; then
    options+=(--env PTO_CSA_RING_TASK_WINDOW=2048)
    flags='--inspect-token-sequence --measure-waves'
  fi
  TASKQUEUE_DEVICE_CONF="$r/csa-approved-last8-20261030.conf" \
  TASKQUEUE_DEVICE_POOL=8,9,10,11,12,13,14,15 \
  TASKQUEUE_IGNORE_WHITELIST=0 TASKQUEUE_IGNORE_GROUP_AFFINITY=0 \
    task-submit --device auto --device-num 8 --max-time 780 "${options[@]}" \
      "bash $r/src/tests/pypto_test/csa_threeway_20261029/run_eight_card.sh $task_cohort $task_arm $task_label $task_reference $flags"
}
run_arm native "$r/cohorts/eightcard-source-v5" "native-$run_tag" none
run_arm ours "$r/cohorts/eightcard-source-v5" "ours-$run_tag" "$gold"
run_arm nalinaly "$r/cohorts/eightcard-nal-perf-v1" "nalinaly-$run_tag" "$gold"
```

```bash
# 在共享环境下、无设备运行CPU门禁，bootstrap参数同主报告。
pytest -q tests/pypto_test/csa_threeway_20261029/test_wave_measurement.py \
  tests/pypto_test/csa_threeway_20261029/test_runtime_audit.py \
  tests/pypto_test/csa_threeway_20261029/test_token_diagnostic.py \
  tests/pypto_test/csa_threeway_20261029/test_summarize_waves.py \
  tests/pypto_test/csa_threeway_20261029/test_postflight.py

# 完成后无设备汇总；参数使用绝对结果目录。
python tests/pypto_test/csa_threeway_20261029/summarize_waves.py \
  "$native_performance_root" "$native_performance_root" "$ours_root" "$nalinaly_root" \
  --output "$summary_json"
```

CPU门禁已在227通过：观测/审计/token诊断7项，汇总及postflight17项。
独立审阅确认冻结与首差机制保留；上述客户端墙钟和非连续cycle限制已纳入口径。
最终rank统计更新后，在227复跑汇总/postflight共17项通过（0.15秒），
日志为`evidence/eightcard/threeway-summary-final-cpu-20261031.log`。

## 结果

三项均已结束并释放设备，未使用前八张卡。只有Native和我方完成了全部生成及有效计时。
主性能统计为各rank的20个目标forward样本先取均值，再取八rank中的最大均值；
这是保守的rank级统计，不是逐步EP关键路径或完整生成平均步长。

| 实现 | 与Native输出token | 目标forward，最慢rank均值 | 相对Native耗时 | 接受事件逐条一致 |
|---|---|---:|---:|---|
| Native `41ba392f` | 61,440/61,440与旧v5金标一致 | 73.494 ms | 基线 | 是，与旧v5一致 |
| 我方精度TND `65bebf7f` | 61,440/61,440与旧金标及本轮Native一致 | 77.079 ms | +4.88% | 否 |
| nalinaly性能TND `79ea6d3c`，Python | 首个输出token差异，停止 | 无有效完整采样 | 不计算 | 未完成 |

Native与我方各160个样本（20/rank），请求数、有效T、物理padding和positions逐样本一致，
各自八rank的上述输入签名也一致；coverage错误为空、零抢占。仅这些选中步骤通过性能门禁，
不表示整个192token期间的接受轨迹一致，也不表示hidden/cache逐bit一致。
均值、P95为合并160个rank样本的描述统计，不将相关rank样本当作独立重复实验。

| 计时细节 | Native | 我方精度TND |
|---|---:|---:|
| 合并样本均值 | 73.418 ms | 77.058 ms |
| 合并样本中位数 | 73.425 ms | 77.023 ms |
| 合并样本P95 | 74.112 ms | 77.997 ms |
| 分波客户端墙钟合计 | 40.990 s | 60.257 s |
| 对应客户端墙钟输出速率，仅辅助 | 1498.91 token/s | 1019.64 token/s |
| DSpark平均接受长度 | 3.695018226 | 3.695018226 |
| 生成子进程总运行时间，含加载及捕获 | 403.11 s | 393.60 s |

客户端墙钟含内部入场协调、CPU逐token检查、draft及采样，不将其约32%的速率差归因于CSA；
加载/编译/捕获总时间同样不作为性能。只运行一次/实现，未证明跨任务重复收益或退化的稳定性。

### 我方接受事件差异仍保留

全部输出token一致，但`h8192_q04`的第36个接受事件（零基35）为我方`[5,5]`、Native`[5,4]`。
接受长度0..5的全rank直方图分别为Native`[848,1224,1624,1048,1080,7344]`，
我方`[848,1224,1624,1056,1064,7352]`；均值相同不能替代逐事件验收。
因此我方模型生成exit0，原严格postflight仍因接受事件不同为false、调度exit1。
原postflight的timing错误来自比较旧v5无有效计时的记录；新版离线性能门禁改与本轮Native对照，
结果通过，未删除或改写旧postflight失败。

### nalinaly首差与结论边界

`FIRST_DIFFERENCE.json`记录rank0、`h8192_q02`、输出零基位置0：
nalinaly为token **3648**，Native为 **9854**。任务在679.86秒（约11分20秒）停止，
未产生完整rank结果或有效计时样本；不使用该679.86秒反推生成性能。
启动日志确认选择`spmd_tnd/python`、NZ2并准备21个CSA目标层，但首差停止前未完成全部覆盖后检，
不能补写为coverage通过。尚未通过同输入内部hook定位到具体误差模块。
本结果证明本次公共框架、bank和配置下的冻结Python组合未对齐Native，
不等于已证明最新ACLNN后端同样失败，也不能据此把全部速度差归因于精度。

按首差规则，本轮没有取消校验后继续跑nalinaly，也没有为补速度重跑大配置。
因此三方token关系已得到区分，但本轮**尚无nalinaly有效性能值**，
不能声称我方已达到或落后其最新ACLNN版本多少。

完整JSON、8rank输出/事件/观测、首差、命令、源码清单和日志归档：
`evidence/eightcard/threeway-real-model-wave-20261031-evidence.tar.gz`，
SHA256：`9e32a172eb43f195b0a2da913e7eaaa2fe440228e6a5bdabf436fd3664c0ecfa`。
其中`native_matches_frozen_gold.json`保留本轮Native与旧v5的独立验证；
`threeway-wave-summary-20261031.json`使用本轮Native作离线对照。

## 排队期间的上游更新

11:00左右复核时，nalinaly公开分支已推进到`4d5b11821a11da28f16a0e075608c8c316a22fe9`。
新提交包含94档设备选档ACLNN 0.4.0、静态/SK接入及Host/metadata优化；
`entry.py`及主要数学核没有变更，但执行和接入方式已有明显变化。
因此，当前冻结的79版本Python任务不能冒称最新ACLNN端到端性能。

当前共同环境未安装`csa_device_pto_ops`，源码归档没有wheel或.run；公开GitHub Releases为空。
新包原构建为Python3.10/aarch64，本轮共同环境为Python3.12，复用时仍须检查ABI；
不直接修改清单来绕过版本检查。已询问yejia可读的现成包位置，尚未重编译94档设备核。
旧Native金标、我方数学包、三项任务及结果均保留；是否切换新后端以实际包兼容检查为前提，
不把未运行的最新后端并入结果。

## 后续对照回退到无wheel版本

用户选择不追ACLNN wheel，改用引入它之前的版本。已核对Git历史：
`f589dc84854d48ad067905cdb5fae00e34e32cce`首次增加可选ACLNN后端，
其直接父提交`1574ed593af8b0f3aa6077cf13d599df281ac242`（2026-10-28）
是最近一个完全没有该后端的提交，选作后续nalinaly对照。

已在独立`cohorts/eightcard-nal-pre-aclnn-1574-v1/nalinaly/source`冻结完整CSA包，
仍搭配共同框架`41ba392f`及现有共享SDK，未改我方精度包、旧快照、bank或运行环境。
46份包文件已入来源清单，Python语法检查通过，确认无`backend.py`及ACLNN绑定分支。
来源凭证为`evidence/eightcard/nal-pre-aclnn-1574-rollback.json`。
这是冻结时的历史状态：当时仅完成源码冻结，尚未做该副本的导入、设备或整模型验证，
没有新增NPU任务。随后用户批准的实际补测见下一节。

前轮`79ea6d3c`虽已含可选ACLNN代码，但实际显式走Python，不需要wheel。
两版核心计算核没有差异；变更涉及后端选择、绑定、短于8K的支持范围及服务检查。
因此旧版是可用的无wheel对照方向，但不能推定回退会修复已观测token首差，
也不能将79版本的结果改标为1574版本。旧结果及失败现场继续保留，不重复编译相同核。

### 1574版真实模型补测

用户另行批准本轮物理8–15（含14），要求只补nalinaly1574，不重复Native和我方。
无设备preflight已通过，模型75分片、40题bank、公共工具链、vLLM0.25.1和
Native插件导入路径均核对，NPU未初始化。原Native计时结果作为本轮reference，
我方结果仍直接引用上一节；不改金标、数学或已验收环境。

补测任务`task_20261031_183419_15358237012`，结果标签`last8-nal1574-8k-wave-perf-v1`。
沿用同一`run_eight_card.sh`、`--inspect-token-sequence --measure-waves`，
仅cohort改为`eightcard-nal-pre-aclnn-1574-v1`，reference改为
`last8-native-8k-wave-perf-v1`。仍为每rank40题、5波×8、192输出token，
FULL6/240、window2048、内部720秒/外部780秒、首token或长度差异即停。
首次补测在第一个图捕获阶段触发`720s_limit`，子进程运行725.30秒（含终止宽限），
调度任务exit1并释放设备。未进入生成、没有首差或有效性能结果，不能记为精度失败。
日志出现新PyPTO源码生成记录，因此先将缓存准备移到不占卡任务，避免继续占八卡冷编译。

必要形状T6/B1、T240/B40、T8/B8分别通过现有`precompile.py`公共模块入口预热，
`--max-seq-len 9000 --task-window 2048`，各自780秒上限，不改SDK或数学。
对应无设备任务依次为`task_20261031_184958_20436819231`、
`task_20261031_184958_204372518229`、`task_20261031_184958_204379325810`。
预热通过后才补跑生成；CPU编译通过不作为设备精度或性能验收。

三项无设备预热均通过，T6/T240/T8用时141.07/15.08/140.62秒，NPU均未初始化。
补跑`task_20261031_185325_216936325056`，标签`last8-nal1574-8k-wave-perf-v2`，
只使用8–15；生成子进程422.71秒，在首个token差异时结束，设备释放。
`FIRST_DIFFERENCE.json`记录rank3、`h8192_q02`、零基输出位置0：
1574为3648，Native为9854，与前轮79版同问题同位置的差异相同。
rank是竞争写首差记录的捕获方，不据此断言只有rank3出错。
没有完整rank结果或有效计时，不补填性能数字，不继续后续生成。

为首差定位，仅放开测试launcher与八卡wrapper的`--first-step-probe`到nalinaly臂，
保留既有首个NONE forward、单次归属、首个不一致层停止、cache/metadata恢复及快照上限。
冻结生产包和`first_step_observer.py`均不修改，独立静态审阅确认1574的55参数/五carrier接口兼容。
227离线汇总17项通过；首步/审计回归首次误导入旧快照的postflight（缺少新审计函数），
修正CPU driver的测试导入优先级后9项通过11.34秒，冻结源码未改。

首差后单独提交`task_20261031_190352_26812776499`，标签`last8-nal1574-first-step-v1`，
同一1574快照、同一bank/Native参考，物理8–15及原时间上限；只传`--first-step-probe`。
该诊断只在真实首个普通目标forward按层做同输入Native半层shadow，
恢复PTO的cache/metadata/输出，首层不一致即停，不继续192token生成，也不作性能验收。

该诊断在345.20秒后停止于**首个CSA层layer2**，`forward_id=1`、NONE/T8/B8、
每请求真实query长度1，query边界`[0,1,2,3,4,5,6,7,8]`。
seq lens为`[8394,8411,8394,8467,8411,8225,8394,8411]`。
这证明在同输入普通forward中已出现CSA层差异，不能把误差全部归因于后续FULL replay。

| layer2同输入边界 | 结果 |
|---|---|
| BF16输出 | 131,072元素中26,756个不同，27,487个byte不同 |
| 输出relative L2 | 0.0025701371，即约0.257% |
| 输出max abs | 0.0029296875 |
| raw/SWA cache carrier | 1,263 byte不同 |
| main cache / main state carrier | 分别32,331 byte不同 |
| Indexer state / cache carrier | 分别8,110 byte不同 |
| 输入、PTO输出、cache、metadata恢复 | 均逐byte通过，只读输入无改写 |
| 共享TopK buffer | 逐byte相同；不能据此认定所有实际TopK中间结果一致 |

carrier可能包含共享allocation的不同view，差异数量不能相加当作独立错误总量。
指标比较整个物理carrier，不将其自动解释成有效slot错误数。
保存的首差tensor约1.28GB，仅留227，不下载或提交；小JSON保留差异指标和恢复证明。
目前已定位到第一个CSA层；尚未在该真实首步中逐子模块证明最早的内部误差来源，
不能将HC-pre、量化或attention中的某一项直接定为唯一根因。没有修改1574数学。

本轮最终关系如下，Native与我方两行完全复用已有结果，没有重跑：

| 实现 | 8K真实模型token | 有效目标forward计时 |
|---|---|---:|
| Native `41ba392f` | 与原金标61,440/61,440一致 | 73.494ms |
| 我方 `65bebf7f` | 与Native61,440/61,440一致；接受事件不同 | 77.079ms（+4.88%） |
| nalinaly `1574ed59` | q02第一个token不同，首差停止 | 无；未完成有效采样 |

因此本轮未取得可用于三方速度比的nalinaly计时；未放宽首差规则来继续生成或补性能数字。
两个完成的生成/诊断任务均已退出，物理8–15释放；首次冷编译超时记录也保留。

本轮完整小型证据：`evidence/eightcard/nal1574-real-model-20261031-evidence.tar.gz`，
374,931 byte，54份日志/JSON/清单，SHA256：
`572edfdd49d24e4a54331cbee07e8339a3c8e454824a5da996593632d2f7302a`。
复现使用新的未占用结果标签；补测沿用前述生成命令，首步诊断将两个生成观测开关
`--inspect-token-sequence --measure-waves`替换为单独的`--first-step-probe`。
仍需取得当轮14卡授权；不要直接覆盖本轮结果或修改旧金标。
