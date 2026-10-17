# WKV恢复：真实权重首差与Native归约顺序定位（2026-10-17）

## 起点与边界

分支`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
数学基准`e267e4e62`已通过真实8K/B40、192token的一次Native完整对拍，结果保存于`d7d9a5b5b`，
见[整模型通过记录](DSV4_FLASH_CSA_TND_INITIALIZATION_DIAGNOSIS_20261016.md)。
本轮从独立实验目录恢复WKV。下列早期实验发生时正式服务WKV仍为Native；
完成后面的全行数对拍后，当前精度分支接入PTO WKV，并通过文末真实8K/B40整模型回归。
没有改原混合精度基准，没有跑128K或整模型Native2，每个NPU任务max-time=780秒。

独立模块使用真实L4权重及Native HC-pre、输入Norm产生的T11输入。
这里的hidden是固定seed生成的合成BF16；与前述真实token整模型8K/B40测试分开。
B阶段将hidden加0.125，再通过Native HC-pre/Norm重新计算输入；其他层复核使用不同seed。
WKV实际为`AscendUnquantizedLinearMethod`，BF16权重[512,4096]，Native format29，bias=None。
只做一次布局整理，不改权重dtype或值；输出保护区、输入及PTO权重不改写均检查；输入没有单独分配保护区。
验收要求逐bit一致；Native先完成重复、A/B/A图重放及ND布局控制。
PTO eager一旦出现首差即停止，不继续PTO graph或其他形状。

## 当前结果

- 已通过的旧整模型基准及原混合精度/性能版本保持不变。
- 新WKV统一候选以N16、K256及INT32次序表执行，1..256每个行数都完成真实L4权重对拍，
  eager、Native/PTO Graph A/B/A及批内首末跨形状重放全部逐bit一致。
- 输入A复用真实L4权重下Native归一化行，T256追加前16行；B为列roll后加0.03125。
  这是固定权重与输入样本的模块资格，**不是任意权重、所有输入或整模型证明**。
- 新候选已消除本轮实际遇到的T240 31处、T64 4处误差，并覆盖简单分段未包含的T16/T65等变化。
- 本轮WKV验收时已将`tnd_full_precision`的WKV调用改为PTO；Indexer权重投影和O-proj仍为Native。
  真TND metadata与每请求1..6行契约未改，未增加每请求S6补位。
- 正式kernel数学AST与通过NPU的v17一致；次序常量逐T核对冻结Native表，另有CPU边界测试和独立审查。
  新接入版本真实8K/B40/192 token回归也已通过，未用百分比阈值放行，也未用FP64替代Native gold。

## 完整已结束实验

| 候选/诊断 | 任务 | 结果 |
| --- | --- | --- |
| v1早期原型，无NPU任务 | 无 | lower阶段Tensor/Tile混用失败，未产生数值结果 |
| v2，M16/N64/K128 NN | task_20261017_011532_35185295468 | 测试输入构造落在inference上下文外，进入WKV比较前退出；不是数值结果 |
| v3，同数学，修好测试上下文 | task_20261017_011745_355813513608 | Native控制全部逐bit一致；PTO首个eager有5/5632元素不同，停止 |
| Native实际矩阵核短profile | task_20261017_012304_367419917871 | ND、NZ、graph与保存Native快照逐bit一致；记录单个MatMulV2 AI_CORE、16blocks |
| v4，单因素NN→NT | task_20261017_012654_379758422661 | 仍为相同5处、相同误差，停止 |
| v5，单因素K128→K256 | task_20261017_013732_40788507243 | 仍为相同5处、相同误差，停止 |
| v6，单因素N64→N32 | task_20261017_014150_4623254 | 仍为相同5处、相同误差，停止 |
| v7，两个K2048 FP32 partial固定顺序相加 | task_20261017_014607_9101513078 | 4处不同，消除[2,289]，仍不通过，停止 |
| 同形状普通数值用例，原K256自然顺序 | task_20261017_015225_2656898699 | 前8例一致，第9例首差2464元素，停止 |
| v8，N32组、K256块、pair shift2、奇组反向 | task_20261017_015738_44873310122 | 第8例[0,128,256]出现2816差，停止，未进入真实权重 |
| v9，N32组、K512块、pair shift1、奇组反向 | task_20261017_020200_5312589961 | 九例全部逐bit一致；真实L4/T11 eager及Graph A/B/A全部逐bit一致 |
| v10，相同kernel，只参数化测试器 | task_20261017_020719_6713349597 | T6/T40 eager与Graph A/B/A逐bit一致；T240 eager首差31/122880，自动停止 |
| T240定位v1 | task_20261017_021026_7356899872 | profile.py遮蔽Python标准模块，导入阶段退出；没有数值测试，v2只改名修复 |
| T240定位v2 | task_20261017_021156_76294318743 | Native保存输入ND/NZ/graph逐bit一致；第8求和例首差，Native呈96列分组 |
| v11，T240列组96/K256/pair shift1 | task_20261017_021536_87961216107 | 第8例53760差，首差停止，未进入真实权重 |
| Native-only K256次序区分 | task_20261017_021936_9869504196 | 5组三块查询得到各列唯一循环次序，Native重复及240行一致性通过 |
| v12，按T240次序表 | task_20261017_022207_10831544297 | 九例、真实L4/T240 eager和Graph A/B/A全部逐bit一致，消除此前31差 |
| v13，同kernel、其他层 | task_20261017_022441_113199118090 | L2/seed29、L26/seed43、L42/seed61的T240 eager和Graph A/B/A全部逐bit一致 |
| 行数边界v1 | task_20261017_022907_125383020821 | T1/T18逐bit一致；T64 eager首差4处，停止，未测T128/T256 |
| T64 Native-only次序区分 | task_20261017_023128_131075518281 | 6组查询在44候选中收敛；实际MatMulV2/16blocks |
| v15，复用v8数学代码，仅验证T64 | task_20261017_023520_14604544981 | 九例、真实L4/T64 eager和Graph A/B/A全逐bit一致，消除4处首差 |

v3～v6：max_abs=`1.52587890625e-5`，relative L2=`3.04906711789954e-6`。
五个坐标为`[2,289]、[4,112]、[5,165]、[6,121]、[6,438]`，均为相邻BF16值。
这些误差虽小，仍未通过逐bit要求，不能接回服务或记成精度通过。

## 已定位到的内容

1. 本组Native的重复与A/B/A图重放稳定，ND/NZ控制一致。
2. NN/NT边界、K分段128/256、N分工64/32各自改变，没有改变这组首差。
3. 改变K归约组织能改变结果，但两个partial尚不足以对齐Native。
4. CPU FP64点积在本组全部5632元素上舍入为Native值；PTO有5处不同。
   FP64仅帮助理解误差，没有替代Native验收gold。CPU顺序FP32与半区树FP32均仍与Native不同，
   不能用它们的少量匹配来证明Native实现。
5. 同形状合法有限值用例进一步显示Native的列组相关归约路径，见下节。

Native短profile的唯一设备矩阵核为
`aclnnMatmulWeightNz_MatMulCommon_MatMulV2`，static、AI_CORE、16blocks、输入BF16 ND/NZ，输出[11,512] BF16。
profiler存在ACL→NPU flow关联解析警告；kernel_details.csv仍有唯一、形状匹配的矩阵核记录。
只用于算子身份，不称完整flow解析成功，也不作Native/PTO性能对比。
block数不能唯一推导微tiling，因此v6和v7都是待验证假设，其失败结果完整保留。

## 普通数值用例如何约束顺序

固定T11、K4096、N512、BF16。x全1，每一列权重完全相同，仅三处非零，分别为
`+1、2^-25、-1`。多组K位置用于区分累加顺序，不涉及异常硬件状态。
Native每例先重复确认稳定，再与PTO比较；第一个不一致即停止。

原自然顺序候选在第9例K位置`[0,2048,4095]`首差。
Native的输出按32列出现不同结果，而PTO全0：
Native输出`2^-25`的32列块索引为`[3,5,7,9,10,12,14]`，其他块输出0；11行均同此模式。
这提供了列组相关K次序的证据，不能单凭一个模式确定唯一公式。

v8的K256块轮转虽然能解释后一模式，却在更早的第8例`[0,128,256]`与Native不符，
因此双门禁阻止了真实权重测试。后续只能修正次序假设后再试，不能绕过定位用例。
v9改为K512粒度：列组`g=col//32`，第`step`轮的块号为
`(g//2 + (7-step if g%2 else step)) % 8`。本组九例及真实权重A/B/A均逐bit一致。
这证明候选通过当前测试，不代表已从Native源码证明该公式覆盖所有形状或权重。
T6、T40门禁同样通过，但T240出现31个差异元素，max_abs=`0.0009765625`、
relative L2=`4.352858741185628e-5`。其Native重复、ND布局及Native Graph A/B/A均通过；
PTO eager首差后，没有运行T240 PTO Graph或后续层。正式服务仍未改。

首批坐标含`[2,289]、[6,438]、[14,415]、[36,400]`。
T6、T11、T40通过不能外推至T240。

### T240首差定位

同一保存输入的Native ND/NZ与Graph均重新逐bit匹配原gold。
实际设备矩阵核仍为MatMulV2，但Block Num为24，而T11为16；不能只凭此数字推出完整tiling。
第8个普通求和例`[0,128,256]`首差：Native在32列块`[3,4,5,9,10,11]`输出非零，
即96列为一组的奇数组；所有240行模式一致。候选仍输出0，共46080处不同。
第9例按首差规则未运行。这与小形状的K512/32列组行为不同。
仅用于T240的K256、96列组候选v11仍失败。Native非零32列块为`[3,4,5,9,10,11]`，
该候选却为`[3,4,5,6,7,8,15]`，不能只调整组宽后直接合入。

T240 CPU辅助分析中，Native有20个元素不同于FP64点积舍入，PTO有25个；
再次说明验收目标必须是实际Native顺序，不能简单用高精度点积代替。
Native-only诊断用三处非零系数`+1、-1、2^-25`，放在不同的K256块，
每个三块组合轮流让三者充当小数项。若小数最后累加，输出保留`2^-25`，否则为0。
每次先检查Native重复一致、所有行相同，并检查仅一个块具有“最后累加”关系；
任一不满足即停止，不强行套循环顺序假设。

在16个K256块的32种正反循环顺序集合中，5组三块查询将每列收敛到唯一候选：

| 输出列区间 | K256块次序 |
| --- | --- |
| [0,96) | 0,1,…,15 |
| [96,192) | 15,14,…,0 |
| [192,288) | 5,6,…,15,0,…,4 |
| [288,384) | 4,3,…,0,15,…,5 |
| [384,512) | 0,1,…,15 |

这只是在指定循环假设内的唯一结果，尚不构成Native完整源码/tiling证明。
v12按该表实现，已通过原9个定位例、真实L4/T240 eager与Graph A/B/A。
v13保持kernel逐字节不变，在L2、L26、L42及不同seed上重复，也全逐bit一致。
在v13历史时点，正式服务尚未切回PTO WKV，不能将T240次序表推广到其他T。

### 行数边界首差

为避免每个形状重复加载整层，边界复核复用L4保存的真实权重与Native归一化输入行：
T不超过240时取前T行，T256原计划再拼前16行；每个形状以实际Native `F.linear` 的NZ权重为gold，
先检查保存的T240 gold未变，并做Native重复、ND布局控制与Graph重放。
小形状候选先假设用于T1/T18/T64/T128，T256计划使用大形状候选；这是待检验假设。

结果T1、T18的eager和Graph A/B/A通过，T64首个PTO eager有4/32768元素不同：
`[2,289]、[4,112]、[62,96]、[63,122]`，max_abs=`1.52587890625e-5`。
因此未执行T64 PTO Graph、T128或T256。B阶段只将输入乘0.5，主要验证图读入更新和写回；
与前面整层重新计算HC-pre/Norm的不同seed、hidden加0.125测试区分。
在行数边界v1的历史时点，T64仍不具备通用接入条件。其Native短profile仍为MatMulV2、16blocks，不能以block数等同于T11。

T64普通三块查询扩展到K256/512/1024/2048/4096块循环、块内自然顺序的44种唯一候选，
6组查询得到唯一结果：N32列组，16个K256块；组`g`的第`step`轮块号为
`(2*(g//2) + (15-step if g%2 else step)) % 16`。
这与早期v8代码逐字节相同；v8在T11失败不代表在其他T一定失败。
v15保持此数学源码不变，在T64重新通过九例、真实L4 eager和Graph A/B/A门禁。
在v15时点T128、T256等其他行数尚未验收，不能把单一候选当成通用替换；后续统一候选另见下节。

## 全行数定位与统一候选

Native分段用6组固定三块查询，在44种分块循环次序中区分各列顺序。
T1..64的32列检查通过，但第二批首个T65发生32列组内变化，诊断按规则停止；
v2只将粒度改为16列，之后补齐T65..256。这个失败是诊断假设不够细，不是PTO精度失败。
每次均检查Native重复、行一致性、列组一致性、唯一末块关系，超出假设立即停止。

| 阶段 | 任务 | 结果 |
| --- | --- | --- |
| Native T1..64 | task_20261017_024121_161163119545 | 64形状完成 |
| Native T65..128，旧32列粒度 | task_20261017_024313_165522130360 | T65首查询拒绝，停止 |
| Native T65..128，16列粒度 | task_20261017_024618_175013515280 | 64形状完成 |
| Native T129..192 | task_20261017_024814_182886828941 | 64形状完成 |
| Native T193..256 | task_20261017_024932_185614527723 | 64形状完成 |
| v16统一次序表，代表形状 | task_20261017_025118_189405514387 | T11/16/64/65/80/128/240全逐bit一致 |
| v17真实权重/双Graph T1..64 | task_20261017_025433_199456529496 | 全逐bit一致；首末跨形状重放通过 |
| v17真实权重/双Graph T65..128 | task_20261017_025554_205389430240 | 全逐bit一致；首末跨形状重放通过 |
| v17真实权重/双Graph T129..192 | task_20261017_025721_211811515483 | 全逐bit一致；首末跨形状重放通过 |
| v17真实权重/双Graph T193..256 | task_20261017_030029_21734781469 | 全逐bit一致；首末跨形状重放通过 |

实际芯片`Ascend910_9392`，环境沿用前述9.2标定。完整256形状归并为9种列布局、30条不同K次序。
例如T16/32/48用自然顺序，T64改变K粒度，T65又出现48列分组；不是一个简单T阈值。
独立T64/T240查询结果已与全行数表交叉核对一致。

[冻结Native次序表](evidence_tnd_precision/full_pto_20261016/wkv-native-order-map.json)，
SHA256 `ba7885227b29b521f7dcbde69ff79bc4bf6c706c1942c59985a01ee323ae7255`。
正式代码用不可变常量压缩同一份表，加载时准备9个INT32[32,16]持久tensor，
每步只按T选择view；不在服务中跑诊断，也不调用Native WKV或执行每步设备适配计算。
新kernel仍先FP32累加，最后一次BF16 RINT；只整理权重ND布局，不改dtype或量化。

[全行数证据包v3](evidence_tnd_precision/full_pto_20261016/wkv-evidence-v3.tar.gz)，
SHA256 `f8fcd4ae5603690b2401e1cf9f50dda5531f5f5d16c088b110bcb5d50e9de0c4`，207项文件逐项校验。
包含全部上述原始JSON、运行脚本、Native表来源SHA、次序汇总程序和统一kernel；不含权重快照。
复现先完成Native四批诊断并运行`summarize_native_orders.py`；v17每次仅给一段最多64行范围，
经`task-submit --device auto --max-time 780 --ptoas 0.63`执行
`wkv-module-v17-all-shapes/run.sh <新标签> <first> <last>`。首差停止，不保留后续形状的假通过。

## 复现与证据

[WKV精简证据包](evidence_tnd_precision/full_pto_20261016/wkv-evidence-v1.tar.gz)，
SHA256 `1447074d560591f6292c9780bdd65183ceebc5bcd1a22d22e8c09dea0b423da1`，70项文件各有SHA。
后续[完整WKV证据包v2](evidence_tnd_precision/full_pto_20261016/wkv-evidence-v2.tar.gz)覆盖本轮全部已结束实验，
SHA256 `11234624ba7635010801cfeb193f17149835340d3458b0b977bbb3e176349d4d`，173项文件均已本地逐项复核。
包括v1早期lower失败、v9/v12/v13/v15成功、T64/T240的原始首差、次序诊断、行数边界首差及任务映射。
包含各候选脚本、lower/CCEC日志、原始失败JSON/日志、CPU分析和Native kernel CSV。
不包含可能携带模型权重的first-difference.pt，只记录其SHA/大小；需要重新运行脚本或沿用远端原快照做张量复核。

沿用整模型复现记录中的精确环境、模型及step07源清单；脚本中的路径绑定原实验资产，
迁移须在新目录显式更新和重新冻结。单卡运行`experiments/<候选>/run-wkv.sh <新输出名>`；
普通用例运行`wkv-order-case-v1/run.sh`，Native短profile运行`wkv-native-profile-v1/run.sh`。
统一经`task-submit --device auto --max-time 780 --ptoas 0.63`提交，保留旧输出、不重装框架。

## 正式WKV接回后的8K/B40复测方案

数学提交`bdf14adb5`，只修改完整精度服务并新增`wkv.py`、`wkv_order_data.py`。
旧混合精度基准、性能实现和已通过的step07快照保持不变。
新step08在step07上只更新这三个文件，3359份文件逐项冻结；旧测试变体身份guard保留。
step08 manifest SHA256：
`d3525c39033a5c88933bb33177caae9fa75d351c561ff54e3b7b6e64a8500580`。
占卡前先校验实际Python、Torch-NPU、PyPTO/Simpler、CANN/ATB/PTOAS路径、全部源SHA和bank，
并完成正式WKV两个设备程序的lower/CCEC；没有重装或重编框架。

| 项目 | 口径 |
| --- | --- |
| 模型 | DeepSeek-V4-Flash-0731-w8a8，真实75个权重分片 |
| 输入 | 固定8192-token bank，40条请求；公开问题及chat后缀一致，前缀/cache为本实验Native构造 |
| 并行 | TP1、DP=EP16，每rank提交B40，共640次请求执行 |
| 解码 | 每请求192个输出token，DSpark出5验6；实际接受率由结果统计 |
| 图 | FULL_DECODE_ONLY，capture sizes=[6,240]，max_num_batched_tokens=400 |
| 容量 | num_gpu_blocks_override=22000，gpu_memory_utilization=0.95，分配前检查2GiB余量 |
| 确定性 | level1、Torch确定性、HCCL_DETERMINISTIC=true、atomic0、NZ2、event_work_mode=1 |
| 替换 | 21层CSA使用`tnd_full_precision`，20层HCA保持Native |
| Native基准 | 沿用task_20261016_192621_7066592194，没有运行Native2 |
| 新任务 | task_20261017_031709_24393218129，16卡，max-time=780秒 |

新任务使用正常异步初始化，没有初始化逐程序同步或内部层hook。
实时比较每rank实际输入、请求入场和rank0 target hidden/aux/MTP位摘要，出现首差即停止。
结束后独立CPU验收核对：完整640请求输出、全部接受事件、51轮生成窗口、16rank实际FULL240重放、
21层两档PTO选核与无回退，以及缓存错误计数。非活跃空尾须通过原判据，不能任意截掉活跃差异。
相对Native的源差异必须精确等于已验证Norm入口两行、原测试身份guard及本次WKV三个文件；
每个源文件、原始Native数据、派生参考和helper均校验SHA。
验收规则沿用旧production-v1，仅增加WKV源码身份白名单和绑定，已经过独立本地审查。

此处是依赖既存冻结源码、模型、bank和SDK的增量复现；不是全新机器可直接运行的自包含包。
迁移时先还原前文环境和原整模型通过记录中的资产，再核对源清单和Native1派生绑定。
只需在同一环境执行以下新任务和CPU验收，使用新的输出名，保留历史数据：

```bash
BASE=/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex
JOB=$BASE/full-pto-model-8k-b40-20261016
H=$JOB/wkv-production-v1
bash "$H/preflight.sh"
task-submit --device auto --device-num 16 --max-time 780 --ptoas 0.63 \
  "bash $H/run-case.sh precision-witness <新的CSA输出名>"
# 任务结束后在该环境中仅做CPU验收；run-case禁止提交Native phase。
source "$JOB/env.sh"
export SOURCE_ROOT="$BASE/full-pto-precision-20261017/step08/source"
export PYTHONPATH="$SOURCE_ROOT:$SOURCE_ROOT/tests/pypto_test:$JOB:$PYTHONPATH"
python "$H/qualify_production.py" \
  /data/pyptouser/yejia/lc128/full-pto8k-native-20261016 \
  /data/pyptouser/yejia/lc128/<新的CSA输出名> --job "$JOB" --output <新的验收JSON>
```

### 整模型验收结果

任务正常结束，exit=0，实际运行7分41秒，符合单任务不超过15分钟。
[完整验收JSON](evidence_tnd_precision/full_pto_20261016/model-8k-wkv-production1-qualified.json)中
`numeric_pass=true`、`wkv_restored=true`；不是仅根据任务退出码或接受率均值判定通过。

| 检查 | 结果 |
| --- | --- |
| 全部16rank实际输入与入场轨迹 | 与原Native1完全一致 |
| 完整请求/输出token | 640请求、122880token全部一致，每请求192token |
| 接受事件 | 全部一致，0..5计数[1696,2432,2816,2592,2240,14512] |
| 平均接受草稿数 | 3.7035909920876446，与Native1一致 |
| 有效生成窗口 | 51轮一致；raw记录的52/54轮空尾按原门禁验证后排除 |
| rank0 target hidden/aux/MTP | 51轮真实行的位摘要全部一致 |
| 实际并发/抢占 | 每rank峰值40请求，零抢占 |
| 实际FULL240重放 | 16rank均50次；首步T40为NONE/eager，不是FULL6 |
| CSA/HCA | 21层CSA两档均选择PTO，20层HCA保持Native |
| 缓存 | 全rank bypasses/invalid_entries/storage_errors/forced_rebuilds均0；strict_ready_reuse=false |
| 占卡前编译 | 正式WKV两个程序lower/CCEC通过，未重装或重编框架 |

当前结论：恢复PTO WKV没有破坏本组真实模型的完整token/事件与模型外层位一致性。
本轮没有内部逐层hook，因此不声称已逐项观测所有层内部WKV tensor；
固定8K bank、模型与上述环境的通过，也不等于任意输入或其他CANN版本的普遍证明。
本组活动生成首步每请求1行，后续每请求6行；本轮没有补测非等长请求矩阵。
该次回归时Indexer权重投影与O-proj仍为Native，HC-pre到HC-post全PTO替换尚未全部完成。
后续[Indexer投影恢复记录](DSV4_FLASH_CSA_TND_INDEXER_PROJECTION_20261017.md)单独保存其进展。
本轮只做8K，未跑Native2、128K或性能测量；未将初始化/JIT开销作为性能结论。

[两臂完整数值与复现证据](evidence_tnd_precision/full_pto_20261016/model-8k-wkv-production1-evidence.tar.gz)，
SHA256 `182fc3606e9350a1275ba8fc966f8df723ac87e822483f93e606a6f8d2e08146`，289项文件均已本地逐项校验。
包含全部16rank原始结果、请求事件与输入轨迹、rank0 outer摘要、Native1派生参考和绑定、
运行/预检/验收/归档脚本、step05～08源清单、正式WKV三个文件及Norm入口文件、Native次序表。
归档中的四个正式源文件另与当前提交逐字节核对一致。
不包含模型权重、venv、大型bank张量或完整冻结源码树；须沿用并核验前述既存资产。
