# Indexer权重投影PTO恢复与逐bit定位（2026-10-17）

## 基准与范围

分支`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
起点为`60733f506`，数学提交`bdf14adb5`已经恢复PTO WKV，
并通过[真实8K/B40、192token回归](DSV4_FLASH_CSA_TND_WKV_PRECISION_20261017.md)。
原混合精度基准、性能实现、step08冻结源码与Native1数据保持不变。
本轮只在`tnd_full_precision`恢复Indexer的64头系数投影；O-proj仍使用Native。
Native metadata、compact RoPE等生产者继续复用；本轮没有改变TND请求边界或添加每请求S6补位。

Native实际路径来自当前冻结源码：

```text
BF16 normed[T,4096]
  → AscendReplicatedLinear / AscendUnquantizedLinearMethod
  → unquantized_gemm → torch.nn.functional.linear
  → BF16 projected[T,64]
  → × (indexer_softmax_scale × indexer_heads**-0.5)
  → BF16 scaled[T,64]
  → DeviceOperator.prepare_dsa_indexer_weights
  → FP16 weights[T,64]
```

实际权重为原始BF16[64,4096]，无bias，Native format29，return_bias=False，
scale=`0.011048543456039806`。
L4原权重SHA256为`b46991cf5a8a032ec111ede4af54fe7df5e54fa32d167bedc315fca869d44a5e`。
PTO只在加载时把NZ布局整理为ND，不改变dtype、数值或量化方式。
计算保持FP32点积累加，随后BF16 RINT发布、FP32缩放、BF16 RINT发布、FP16 RINT发布；
三个输出分别对拍，不能跳过中间BF16边界。

## 首差与修正证据

v3先完成Native实际模块重复与Native ND布局控制，三阶段均逐bit一致。
旧顺序候选在T40的PTO首个eager中，投影BF16 `raw`有2/2560个元素不同：
`[19,62]、[25,58]`，max_abs=`0.0009765625`。
出现首差后立即退出，没有继续缩放/FP16验收或T240/T11。
因此这一首差发生在投影归约边界，尚不能归因后续缩放或FP16转换。

T40 Native-only定位使用普通有限矩阵：输入全1，权重仅三个K256块位置非零，值为
`+1、-1、2^-25`。6组三块查询区分44种分块循环家族；
每次检查Native重复、16×16输出分组、唯一末块关系，假设不符立即停止。
这是当前家族内的数值行为定位，不等同于Native完整tiling源码证明。

T40的4个N16列组、所有行一致，对应K256块次序为：

| 输出列 | K256块次序 |
| --- | --- |
| [0,16) | 0,1,…,15 |
| [16,32) | 14,15,12,13,…,0,1 |
| [32,48) | 8,9,…,15,0,1,…,7 |
| [48,64) | 6,7,4,5,2,3,0,1,14,15,12,13,10,11,8,9 |

v4仅换用实测次序输入，kernel逐字节不变。
同一真实L4权重与T40输入下，原2处差异消失；raw/scaled/output、eager、
Native与PTO各自Graph A→B→A全部逐bit一致。
本轮没有用误差百分比或容差阈值放行，也没有用FP64替代实际Native gold。

## 全行数与跨层验证方案

Native全部T1…256按64形状一项任务分段核对。
每个T、每个16×16输出tile的44家族候选交集在CPU独立重算，最终必须恰为一个候选，
并与原始查询记录/计数一致；T40另与独立诊断交叉核对。
[冻结顺序表](evidence_tnd_precision/full_pto_20261016/indexer-native-order-map.json)
SHA256=`700c26586fe104bf92ffbb5d6eba74d307ba4c67df9eeac08363c729d2ec71fc`，
共20种布局、30条不同K次序。

真实权重模块对拍另行执行，不能用上述顺序诊断代替精度验收：

1. L4实际Native投影模块，T1…256全部行数，分四项小任务。
2. 使用真实L4权重下Native HC-pre/RMSNorm产生的固定seed合成输入；T≤240取前T行，
   T241…256追加前16行。B阶段对列roll后加0.03125，A恢复原值。
3. 每个形状先比Native实际模块重复和ND布局控制，再比PTO eager三个阶段；首差立即退出。
4. Native和PTO分别捕获raw NPUGraph，A/B/A重放均重新写入同址输入、清空输出为NaN，
   Native B必须确实改变，恢复A必须恢复原gold。
5. 三个输出分别有首尾保护行；输入、ND权重和order逐bit不变。输入没有额外保护区。
6. 保留每形状图/输出/输入/order/原gold的owner；每批完成后重放首末形状，检查跨形状状态。
7. 再按每任务7层，覆盖21个CSA层L2、L4、…、L42的真实权重，检查T40/T240及相同图门禁。
   输入仍复用L4归一化行，不能称为各层在真实模型运行中的中间hidden。

全部module gold均由实际Native模块计算；诊断阶段Native重复不属于整模型Native2。
每个NPU任务都通过`task-submit --device auto --max-time 780 --ptoas 0.63`，
最长780秒，发现第一个差异便先定位，没有扩大到128K。

## 任务记录

| 实验 | 任务 | 结果 |
| --- | --- | --- |
| v1 | task_20261017_081626_330184012751 | 循环导入，数值前退出 |
| v2 | task_20261017_081821_33340835350 | fixture少一层wrapper引用，数值前退出 |
| v3 | task_20261017_082009_336753415542 | T40 raw首差2/2560，停止 |
| T40 Native顺序 | task_20261017_082204_340449113617 | 通过 |
| v4，T40修正 | task_20261017_082350_344147729318 | 三边界eager/双Graph ABA逐bit一致 |
| Native T1…64 | task_20261017_082518_346907324127 | 通过 |
| Native T65…128 | task_20261017_082657_350038520381 | 通过 |
| Native T129…192 | task_20261017_082830_35337757417 | 通过 |
| Native T193…256 | task_20261017_082959_356069917284 | 通过 |
| v5启动 | task_20261017_083252_361189319213 | --layer缺少数值，argparse退出，没有数值结果 |
| v6，L4/T1…64 | task_20261017_083612_366803025593 | 全部逐bit一致，首末图重访通过 |
| v6，L4/T65…128 | task_20261017_083800_37041628810 | 全部逐bit一致，首末图重访通过 |
| v6，L4/T129…192 | task_20261017_083954_374315417951 | 全部逐bit一致，首末图重访通过 |
| v6，L4/T193…256 | task_20261017_084154_378022611485 | 全部逐bit一致，首末图重访通过 |
| v7跨层启动 | task_20261017_084415_382827723944 | shell参数校验退出，没有数值结果 |
| v8，L2…14 | task_20261017_084750_388817520042 | 7层×两形状全部逐bit一致 |
| v8，L16…28 | task_20261017_084933_39190997299 | 7层×两形状全部逐bit一致 |
| v8，L30…42 | task_20261017_085114_394816028414 | 7层×两形状全部逐bit一致 |

v1→v2只修复循环导入；v2→v3只修复fixture对象路径。
v5的默认层号问题最初怀疑环境脚本修改参数，但独立CPU shell复核未复现，根因未确认，
不能据此认定环境脚本有问题；v6在source前保存入参并显式传层号4。
v7未用的第四参数校验被删除后形成v8，probe及kernel与已审查v7逐字节相同。
所有启动失败均不计为精度结果；v1…v8使用同一数学kernel，v4之后只改变已测order输入。

CPU严格验收共298项模块case：L4全部256行数，加21层×T40/T240的42项。
每项raw/scaled/output三阶段均零bit差、零max_abs；Native/PTO双Graph ABA和首末形状重访通过。
实际设备为Ascend910_9392，环境仍为既有CANN9.2、Torch2.10.0、Torch-NPU2.10.0.post2、
vLLM0.25.1及相同私有PyPTO/Simpler/PTOAS资产；精确路径和环境见前述WKV/整模型记录。
全部模块NPU任务实际耗时均小于2分钟，最长1分23秒；单任务上限仍为780秒。

[模块完整证据包](evidence_tnd_precision/full_pto_20261016/indexer-evidence-v1.tar.gz)
SHA256=`8787c67d3d7f9b04caa3a596521b1aee94d732b130796d8bbaef3a238af3acab`，78项文件均已本地逐项核验。
包含v1…v8源文件、编译日志、全部原始结果JSON、首差日志、Native查询表、CPU独立汇总/严格验收脚本、
查询来源SHA、任务映射与状态摘录、step08源清单。v1…v8 kernel字节相同。
Native/PTO张量快照可能含权重，因此仅保存其SHA/大小，不将PT内容加入GitHub。
此包依赖既存模型、输入快照、冻结源与SDK，不是全新机器可直接运行的自包含环境。

复现模块测试时，在同一227独立资产目录先按原环境核对import/PTOAS/CANN及step08源SHA，
再运行证据中的Native四段诊断和`summarize_native_orders.py`核对表SHA；
真实权重使用`v6/run.sh <新标签> <first> <last> 4`，每段最多64行；
跨层使用`v8/run.sh <新标签> <first-layer> <last-layer>`，每段最多7个偶数CSA层。
经`task-submit --device auto --max-time 780 --ptoas 0.63`提交，CPU运行`qualify_modules.py`。
脚本绑定历史label和路径，迁移或新label时须显式更新并重新冻结身份，不能覆盖历史结果。

## 正式接入与后续整模型门禁

正式服务加载时准备不可变次序表对应的INT32[16,4,16]设备tensor及每层私有输出buffer，
forward仅按总token行数T选择相应view，执行PTO投影；不在服务中运行顺序诊断。
对原权重dtype/shape/bias、Native unquantized方法、return_bias和scale做显式检查。
CSA内部历史参数名`native_indexer_weights`保留，实际producer改为PTO FP16输出。
保留T≤256准入和原生请求边界，未改变cache、metadata、连接器和O-proj。
正式kernel数学AST与NPU已测v6程序绑定，次序常量逐T核对冻结表。

模块完成后，将新源冻结到独立快照，先无设备环境/source/bank及正式kernel lower/CCEC预检，
再提交16卡8K/B40、192token回归。复用既有Native1，不跑整模型Native2。
采用原来相同的确定性设置、bank、容量、实际入场轨迹与实时outer首差门禁，
严格验收全部16rank输入/入场、完整640请求与122880输出token、接受事件及rank0 hidden/aux/MTP位摘要。
本轮不做128K或性能测试。整模型结果与最终证据见下文。

## 8K/B40整模型接入回归方案

数学提交`c1ef141ae`已正常推送，13项CPU合约/证据绑定检查通过，增量pre-commit和Markdown检查通过。
接入经过独立审查；正式kernel数学AST与已测v6完全相同，服务类AST与审查预览相同。
在step08上只修改`service.py`、新增`indexer_projection.py`和`indexer_order_data.py`，
独立step09共3361文件；保留已验证Norm入口与测试变体身份guard。
step09 manifest SHA256：
`d997d41080882209007134564ca488ce19cfe7f136397b6fd0e389c929001638`。
这是有明确白名单的旧版本测试overlay，不应称为整份Git tree的逐文件复制。

| 项目 | 参数 |
| --- | --- |
| 新CSA任务 | task_20261017_090030_412028016524，16卡，max-time=780秒 |
| Native对照 | 沿用task_20261016_192621_7066592194，没有提交Native2 |
| 模型/输入 | DeepSeek-V4-Flash-0731-w8a8真实75分片，固定8192-token bank/40条请求 |
| 并行/输出 | TP1、DP=EP16，每rank B40；共640请求，每请求192输出token |
| DSpark | 出5验6，接受事件照原协议实际统计 |
| 图/容量 | FULL_DECODE_ONLY，[6,240]，max_num_batched_tokens=400，KV blocks=22000，gpu util=0.95 |
| 确定性 | level1、Torch确定性、HCCL_DETERMINISTIC=true、atomic0、NZ2、event mode1，EPLB关闭 |
| 替换 | 21层CSA使用full_precision，Indexer三阶段改PTO；20层HCA保持Native |
| 输出目录 | `lc128/full-pto8k-indexer-production1-20261017` |

占卡前已核验相同环境/导入路径、SDK、全部源SHA、bank公开问题及chat后缀，
正式两个Indexer程序lower/CCEC成功，字节数117672/122448；没有重装或重编框架。
bank前缀/cache仍为本实验Native构造，与nalinaly原bank逐字节身份未核验，此限制与原基准相同。

启动与最终CPU验收位于独立`indexer-production-v1`；
与WKV版相比，只扩大Indexer新增文件/服务源码白名单和显式配置身份。
输入/入场、原始Native与派生SHA绑定、完整token/events、outer、21层捕获选择和cache门禁均不放松。
`indexer_module_real_weight_layers_qualified=21`引用本页298项模块证据；
整模型本次没有内部stage hook，不能将此字段读成21层真实运行中间tensor的逐项观测。

同资产复测用新的输出名，保留历史结果：

```bash
BASE=/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex
JOB=$BASE/full-pto-model-8k-b40-20261016
H=$JOB/indexer-production-v1
bash "$H/preflight.sh"
task-submit --device auto --device-num 16 --max-time 780 --ptoas 0.63 \
  "bash $H/run-case.sh precision-witness <新的CSA输出名>"
# 任务结束后仅做CPU验收；run-case禁止Native phase。
source "$JOB/env.sh"
export SOURCE_ROOT="$BASE/full-pto-precision-20261017/step09/source"
export PYTHONPATH="$SOURCE_ROOT:$SOURCE_ROOT/tests/pypto_test:$JOB:$PYTHONPATH"
python "$H/qualify_production.py" \
  /data/pyptouser/yejia/lc128/full-pto8k-native-20261016 \
  /data/pyptouser/yejia/lc128/<新的CSA输出名> --job "$JOB" --output <新的验收JSON>
```

脚本绑定历史Native1、bank、环境和源清单；迁移必须显式更新资产并重新冻结身份。

### 整模型最终结果

任务exit=0，实际运行7分35秒，符合每任务不超过15分钟。
[完整CPU验收结果](evidence_tnd_precision/full_pto_20261016/model-8k-indexer-production1-qualified.json)中
`numeric_pass=true`、`indexer_projection_restored=true`；验收不是仅按退出码或接受率均值判定。

| 检查 | 结果 |
| --- | --- |
| 全16rank实际输入与入场轨迹 | 与Native1完全一致 |
| 完整输出token | 640请求、122880token全部一致，每请求192token |
| 全部接受事件 | 完全一致，接受0…5计数[1696,2432,2816,2592,2240,14512] |
| 平均接受草稿数 | 3.7035909920876446，与Native1一致 |
| 有效生成窗口 | 51轮一致；raw CSA记录52/54轮，非活跃空尾通过原判据后排除 |
| rank0 target hidden/aux/MTP | 51轮真实行的位摘要全部一致 |
| 实际并发与抢占 | 每rank峰值40，零抢占 |
| 实际图执行 | 首步T40为NONE/eager；后续各rank50次FULL240重放 |
| 替换资格 | 21层CSA两捕获档均PTO，20层HCA保持Native |
| 缓存 | 全rank bypasses/invalid_entries/storage_errors/forced_rebuilds均0；strict_ready_reuse=false |
| 范围外 | 没有内部逐层stage hook、Native2、128K或本轮性能测量 |

**结论：本组真实模型测试中，把Indexer投影、缩放及FP16发布改为PTO后，
完整token、接受事件、实际输入/入场和模型外层位摘要继续与原Native1一致。**
模块精度证据为前述298项；整模型本次未逐项导出21层内部tensor，
因此不将外层摘要一致扩展成任意输入、未来CANN版本或所有内部中间值的普遍证明。
本组活动生成首步每请求1行、后续每请求6行；没有新增非等长请求矩阵测试。
TND metadata设计与原准入范围保持不变，当前投影数学调用中O-proj仍保留Native。

[两臂完整整模型证据](evidence_tnd_precision/full_pto_20261016/model-8k-indexer-production1-evidence.tar.gz)，
SHA256=`6a62b0f8e44387e35e7d293bb10cd36142e9ac348866ee586aaae96a65a9b14b`，
293项文件均已本地逐项校验。包含两臂全部16rank结果、事件、输入轨迹与审计、rank0 outer摘要、
Native1派生参考及绑定、step05…09清单、运行/预检/验收/归档脚本、两份Native次序表，
以及6份正式源文件；这些源文件另与本地数学提交逐字节核对一致。
不含模型权重、venv、大型bank张量和完整源码树，复现依赖前述既存资产。
原step08/WKV通过数据与原混合精度、性能实现均保留；本轮仅恢复Indexer，不修改O-proj。
