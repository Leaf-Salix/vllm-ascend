# 不同T与不等长请求的Native精度推广（2026-10-29）

## 基准与验收原则

- 分支：`dev/pypto-dsv4-csa-nal-tnd-native-align-20261027-codex`。
- 保留基准：`646acb28`；原验收只覆盖8K/B4/T24。七组通过的检查点为`6a2d1a60`，
  也完整保留，不把它们推成整模型逐token结论。
- 使用真实第4层权重、8K合成history/hidden，TP1。Native确定性、共用validated工具链、
  CANN和权重加载条件沿用[主记录](README.md)。不修改原SDK、runtime或生成C++，全部新算子用DSL。
- 新增case通过`--lengths`传每请求真实1～6行；`T=sum(lengths)`，不对请求填充S6。
  P8/P16/P64/P128仅为内部tile容量，观察只对已发布有效域比较。
- 先生产FULL/eager八状态及guards，再检查同输入、源码身份、观察/生产等价，随后逐项
  对拍Native；第一个不同元素立即停止后续模块和配置，不放宽逐bit判定。
- 设备显式限定在227物理8–15；首轮七组用物理8，WO-A微任务及后续修正/回归改用物理15。
  外层780秒、内层750秒及15秒清理；
  需要16卡整模型时另申请批准。单层门禁通过前不做8K整模型192-token测试。

## 探针推广

按实际T分配全部观察缓冲：mixed=P8；QA、KV、Indexer norm/headweights等=P16；
Q INT32 projection=P128；Indexer query accumulator=P64；query heads=T×64；
packed=8×P128；WO-A输出容量遵循原policy。Native metadata、cache/state和原始位置保持不变。

`closed_tokens`按实际position与ratio4推导，并与Native有效compact槽计数交叉检查。
空compact域仍检查形状、dtype、非有限值和逐bit判定，只令空域`max_abs=0`；全物理
cache/state、TopK、最终输出和保护区仍完整比较，不把未发布compact padding当金值。

原观察根82参数新增3个主Compressor中间量，成为85参数；只替换原`create_tensor`
分配，核心调用未改。新增pool/output源码身份后为27项，必须配对新生产快照。
观测先证明八状态相对未观测生产根一致，才解释中间量。公开服务55/debug57 ABI保持。

## 首轮T18混合请求

输入：`lengths=[1,5,6,6]`，B4，T18；使用相同seed1024和Native builder生成元数据。

| 验证 | 结果 | 诊断用时 |
| --- | --- | ---: |
| 原生产v1，FULL/eager八状态、guards | 全部通过 | 42.57秒 |
| 原观察v1，Native逐阶段 | 前11项0；首差主compressed cache | 34.77秒 |
| 新27源码生产v2 | 八状态及guards通过 | 33.42秒 |
| 三中间量观察v2 | 生产等价、阶段FULL/eager通过；首差原样复现 | 213.24秒 |

主Compressor state逐bit相同。主compressed cache只有一个BF16元素不同：
坐标`[134,0,0,180]`，PTO为`-0.00015544891357421875`，
Native为`-0.0001544952392578125`，相差1 ULP、绝对差`9.5367431640625e-7`。
维180属于NoPE，直接RoPE不参与该元素。闭合token行4/9/15对应有效compact行0/1/2，
槽为`[68,0] / [134,0] / [200,0]`。这是修正前结果，未通过时没有扩大其它T。

### RMS直接除法的单因素排除

本地Native源码的RMS采用`(pooled / sqrt(variance)) * gamma`，当前继承实现为
`(pooled * recip(sqrt(variance))) * gamma`。使用同次观察保存的pooled测试，
不先假定通用`npu_rms_norm`与融合Compressor内部归约完全相同。

| 同输入NoPE检查 | 相对实际融合Native结果 |
| --- | --- |
| 微根原算法与实际PTO producer FP32 | 逐bit0，证明微根复现有效 |
| 原recip乘算法 | 同一1元素、1 ULP差 |
| 仅改直接除sqrt | 同一1元素、1 ULP差 |
| 通用Native RMSNorm，作为校准 | 同一1元素、1 ULP差 |
| 两臂各3次NaN FULL/eager、guards及只读 | 全部通过 |

因此**单改Norm不能消除当前首差，没有合入正式Norm**。此负结果不证明两种Norm
全域等价，也不单独证明pool误差的全部来源。任务用时13.90秒，已退出。

微根第一次无设备编译因FP32 `[1,1]` col-major物理Tile未满足32字节对齐失败；
改为物理8行、实际3行valid shape后ABI4编译/load通过，没有修改SDK或生成代码。
在设备初始化前严格核验epsilon等于已编译的`1e-6`。负编译结果不算数值失败。

## 主pool单因素候选

对照本地Native源码：
[Softmax](../../../csrc/attention/compressor/op_kernel/arch32/soft_max.h)、
[ColumnSum/ColumnMax](../../../csrc/attention/compressor/op_kernel/arch32/compressor_vector_comm.h)、
[Compressor向量流程](../../../csrc/attention/compressor/op_kernel/arch32/compressor_block_vec_perf.h)。
这些是源码边界线索；未核验它们与冻结Native二进制的构建SHA，最终仍以实际融合op对拍判定。

当前候选仅把主pool在线递推改为：

1. 保留原请求/历史寻址，八位置按`i%4*2+i//4`交错排列。
2. max与exp和使用8→4→2→1树；先除总和，得到每个位置的概率。
3. 再乘对应value，按相同8→4→2→1树求和。

Indexer pool、Norm、state提交、请求owner、causal及负页检查不变。
CPU57/85编译/load和独立生成物审查通过；完整Vec峰值仍163872字节，两个16KiB
window及归约临时无覆盖；pool发布后才允许state提交。

设备复测：生产213.45秒、观察214.22秒，FULL/eager、八个生产输出/状态、guards及
观察等价通过。主compressed cache由1个BF16元素不同变成逐bit0；前33个Native边界
全部逐bit0，包括Indexer TopK和inverse RoPE后的attention heads。
**新的首差为WO-A BF16输出：30个元素不同，max_abs=6.103515625e-5。**
仍停在T18，不进入其它T或整模型。

## 动态输出后缀修正（七个有限配置已验收）

- WO-A保留已验T24的128列packet，仅将起点半旋转资格推广至实际T=17～32；
  不推断T33以上的Native累加次序。
- 整行量化推广到真实T和不足8行的尾Tile，先舍入BF16、计算完整8192列amax；
  保留原group/32行任务owner和唯一尾零writer。量化前等待八组WO-A。
- WO-B先在INT32归并八组，再转换FP32与联合scale相乘；保留原HC-post。
- 首次无设备编译发现新建`full([8,1])`常量与归约Tile布局不一致，检查未通过，
  未提交NPU任务。改回从amax派生127常量，继承物理布局与valid shape；
  该编译失败不计作精度结果。57/85参数根均无设备compile/load通过。

T18设备结果：生产203.38秒，观察206.08秒；8项FULL/eager输出/状态、保护区、
39项跨任务同输入、8项观察/生产等价、28项阶段FULL/eager均通过。
**35个Native边界全部逐bit0，包含WO-A和完整HC-post后的最终CSA输出。**
本结论仅覆盖真实权重单层、8K合成hidden/history、`[1,5,6,6]`；不代表整模型token
或其它T全域通过。原基准646acb28保留；T24全S6回归也已通过35项逐bit门禁（生产37.78秒、观察47.70秒），
主pool修正未破坏原已验点。T16 `[1,3,6,6]`也通过35项门禁，继续检查更短query和跨尾Tile配置。

## 推广验收进度

| 配置 | 实际query长度 | Native逐bit门禁 | 生产 / 观察任务用时 |
| --- | --- | --- | --- |
| 8K/B4/T18 | 1,5,6,6 | 35/35全部一致 | 203.38 / 206.08秒 |
| 8K/B4/T24 | 6,6,6,6 | 35/35全部一致 | 37.78 / 47.70秒 |
| 8K/B4/T16 | 1,3,6,6 | 35/35全部一致 | 41.38 / 37.99秒 |
| 8K/B4/T4 | 1,1,1,1 | 35/35全部一致 | 33.53 / 38.01秒 |
| 8K/B5/T25 | 1,6,6,6,6 | 35/35全部一致 | 32.68 / 40.33秒 |
| 8K/B6/T21 | 1,2,3,4,5,6 | 35/35全部一致 | 33.49 / 39.15秒 |
| 8K/B6/T32 | 2,6,6,6,6,6 | 35/35全部一致 | 35.14 / 39.85秒 |

表中用时取`result.elapsed_s`，包含首次编译和诊断，不含启动器环境准备，不是CSA性能计时。
每请求padding均为0，外层任务日志也均未超过15分钟。
七组使用完全相同的27项正式源码SHA，生产与观察输入、源码均成对核验；
检查点`6a2d1a60`的四个正式修改文件与这组已验SHA一致；后续T33修正单独验收。
T33切换OB_ROWS=96策略，后续已发现WO-A首差，见下节；不把上述七个点推成T1～256全域验收。
T4没有新闭合的compact输出，其相关中间量比较为空域；完整state/cache、TopK和最终
输出仍实际逐bit比对。
8K整模型尚未开始。

## T33边界首差与修复

输入`[3,6,6,6,6,6]`，实际T33。CPU57/85编译/load和双AIV通过。
生产201.97秒，八状态FULL/eager及guards通过；观察209.20秒，前33个Native边界
全部0（包括TopK、attention heads），首差为WO-A BF16：56个元素、
`max_abs=6.103515625e-5`。未跑更大T或整模型。

当前独立微根保留已推送的`6a2d1a60`正式源码：原臂必须复现同次实际FP32 o_r；
候选只把K256起点改为`5*((column128//3-group)%4)`，四种起点0/5/10/15。
M128、N128、64个逻辑任务和输入权重均不变。物理8微任务排队未运行时取消，
改投允许的物理15，仍只有一项单卡任务。先要求Native BMM复现捕获的WO-B
输入BF16，再比较候选，保留三次NaN FULL回放、保护区与只读检查。
这是旧精度分支标定提出的假设；实际使用Torch-NPU算子，不能把checkout中另一个
自定义BMM的shuffle源码当作当前二进制行为的证明。微根在物理15用时32.98秒，已通过：Native重放0、原算法复现实际FP32为0且
保留同样56个BF16差异；仅改起点后BF16差异为0。两臂三次NaN FULL/eager、
guards及只读均通过，证据将此次首差定位到WO-A的K累加顺序。

完整CSA候选随后只接入该起点族；17～32旧算术不变，33～48选择四种静态起点，
其它T/tiling保留原路径。T33完整根的CPU55/57/85和独立生成物检查均通过。生产203.44秒，观察215.39秒；
**35个Native边界全部逐bit一致**，最终CSA输出一致，8生产状态/guards和观察门禁通过。
这不是33～48全域验收。四起点修正后已完成以下回归：

| 配置 | 实际query长度 | Native逐bit门禁 | 生产 / 观察任务用时 |
| --- | --- | --- | --- |
| T33，修复点 | 3,6,6,6,6,6 | 35/35全部一致 | 203.44 / 215.39秒 |
| T32，原边界 | 2,6,6,6,6,6 | 35/35全部一致 | 206.17 / 216.07秒 |
| T18，原首差点 | 1,5,6,6 | 35/35全部一致 | 32.95 / 41.84秒 |
| T21，混合1～6行 | 1,2,3,4,5,6 | 35/35全部一致 | 33.88 / 39.35秒 |

生产和观察的27项正式源码SHA均匹配当前四起点实现；没有用旧生产快照替代回归。
T21混合1～6行已回归通过。T48端点发现新首差，已停止T47及其它配置。T48的全部constexpr与
已编译的T33相同，可复用55/57/85动态根；这不省略设备数值验收。
之后补全局bucket padding/dummy资格，再准备8K整模型。
原基准646acb28及七组检查点6a2d1a60均保留。

WO-A微测试的原臂要复现修正前的实际FP32输出，因此应在`6a2d1a60`的独立checkout
加入本次`woatnd_kernels.py`、`probe_woatnd.py`及对应selector，再使用原T33观察快照。
当前修正后的正式helper不再是原臂；不能直接在新源码上重跑并称作同一个单因素实验。

### T48端点的Indexer首差（停止扩大）

`[6,6,6,6,6,6,6,6]`、真实T48、不做请求padding。生产37.42秒，八状态FULL/eager及
guards通过；观察38.07秒，前13个Native边界逐bit0，包括主cache和Indexer state。
第14项`indexer_norm_rope`只有一个BF16元素不同：有效compact行7、维82，
`max_abs=9.5367431640625e-7`。维82属于RoPE，不能直接排除Norm或旋转舍入。
观察/生产八状态、28阶段FULL/eager、39同输入及保护区均通过；没有解释或验收首差之后的模块。
门禁阻止了T47提交，未扩大到其它T或整模型。

新增独立Indexer pool微根：用同次Native/PTO已逐bit相同的post-state恢复本轮投影，
历史部分取initial-state；保留真实state stride/offset并验证本轮48个写槽唯一。
state里的score已经含APE，不再重复添加。原online臂必须先复现实际FP32 pooled和BF16
首差，再解释仅改为Native八行交错、概率优先归约的候选；Norm/RoPE暂保持原算术。
先无设备编译/load及生成物审查，再做三次NaN FULL、guards及只读检查。正式Indexer pool
在这项单因素资格前保持不变。

微任务随后在物理15用时16.91秒通过：原online臂逐bit复现实际FP32 pooled与实际BF16
输出，并保留相对Native的同一`[7,82]`首差；仅改pool树后相对实际融合Native的BF16
不同元素数变成0。两臂各三次NaN FULL/eager、guards及只读全部通过。
因此此次T48首差有pool单因素因果证据；尚不能把微根通过等同于完整CSA修正通过。
接回正式pool前已保留并推送WO-A修正检查点`c6c8fdce`；随后须重新配对生产/观察，回归旧点。
微根须在`c6c8fdce`的27项正式源码上运行；修改正式pool后旧快照会被身份门禁拒绝。
无设备v1/v2仅诊断临时变量shape重名失败，变量分名后v3两臂ABI7/load通过，未修改SDK。

正式候选随后只将Indexer的两个64列片改成相同八行交错概率优先树；请求owner、
历史/本轮寻址、负页、causal、APE、state commit及Norm/RoPE均保持。
已通过7项CPU合同测试、静态检查和独立源级审查。完整55/57/85根均编译/load通过，
双AIV分派通过；生成物独立审查确认全Vector静态UB高水位155872B，pool发布及
全部owner等待后commit的顺序保留。T48新生产204.61秒，八状态FULL/eager与全部
保护区通过；Native观察227.62秒，**35/35边界全部逐bit一致**，最终CSA输出一致。

| 新Indexer pool版本 | 实际query长度 | Native逐bit门禁 | 生产 / 观察任务用时 |
| --- | --- | --- | --- |
| T48，修复点 | 6,6,6,6,6,6,6,6 | 35/35全部一致 | 204.61 / 227.62秒 |
| T18，混合回归 | 1,5,6,6 | 35/35全部一致 | 207.53 / 207.69秒 |
| T21，混合1～6行 | 1,2,3,4,5,6 | 35/35全部一致 | 32.58 / 39.95秒 |
| T24，原全S6基准 | 6,6,6,6 | 35/35全部一致 | 32.35 / 38.90秒 |
| T33，WO-A修复点回归 | 3,6,6,6,6,6 | 35/35全部一致 | 33.18 / 40.12秒 |
| T47，混合尾行 | 5,6,6,6,6,6,6,6 | 35/35全部一致 | 34.01 / 42.80秒 |

T48生产/观察的27项正式源码SHA一致，也逐项匹配当前pool修正；原T48首差证据保持。
T18、T21、T24、T33、T47回归全部通过；每项都重新生成配对生产快照。
六组使用相同27项正式源码SHA，逐项匹配本次Indexer pool修正。35项逐bit精度、
八个生产FULL/eager状态、八个观察/生产等价状态、28个阶段FULL/eager、39个同输入
及保护区检查均通过。T47真实行数47，短请求为5行，不补成6行。
这些任务均用物理15且已结束，最长诊断227.62秒；数字含编译和观测，不含启动器准备，
对应外层任务日志约4分5秒，不是算子延迟。
本轮结论限于表中六组单层真实权重、8K合成history/hidden；不外推全T、全输入或整模型token。

### 整模型前的未验收项

上述逐阶段fixture均为`sum(query_lengths)=physical T`，未覆盖真实服务的global bucket
padding。Native runner可能插入L=0的dummy请求；活跃B4对应的metadata请求容量可能5，
不能把max_num_seqs=4导致的Native fallback当成PTO token通过。
后续须覆盖物理T24/活跃T18的真实Native builder与图合同，不手改Native decode数量或
compact容量以凑形状。按实际有效token/slot比较数值，全cache/state和guards仍全量检查，
并观察正式55入口、FULL replay及fallback。旧8K/B40每卡40对应满档T240，本分支更大T的
多条数值路线仍未验收，因此目前不宣称8K/B40可用；16卡仍须另获批准。

## 复现方法

沿用主记录的env/bootstrap与公共工具链，所有label/目录必须全新：

```bash
python "$BOOTSTRAP" --shared-toolchain --native-plugin-root "$SOURCE_ROOT" \
  "$TEST_DIR/compile_rms.py" <新CPU目录> spmd-probe 1,5,6,6

task-submit --device 8 --max-time 780 \
  --env PYTHONNOUSERSITE=1 --env HCCL_DETERMINISTIC=true \
  --env PTOAS_ROOT --env ASCEND_HOME_PATH \
  "bash $TEST_DIR/run_rms.sh <新生产标签> spmd --lengths 1,5,6,6"
```

核对生产八状态和guards后，另提交观察任务：额外传
`--env SPMD_PRODUCTION_REFERENCE=<新生产目录>/production.pt`，使用同一lengths。
源码身份变动后不能复用旧快照。以最终JSON的`complete/pass/first_difference`和逐项
门禁判断，不能只看exit0；生产阶段`pass:false`表示未做Native验收。

RMS微测试无设备selector为`compressor-norm`、ABI4；显式传
`--env COMPRESSOR_NORM_REFERENCE=<三中间量观察目录>/observation.pt`，执行
`run_rms.sh <新标签> compnorm`。先复现实际NoPE FP32，再解释两臂结果。

Indexer pool单因素复现：在`c6c8fdce`独立源码目录，使用同次T48的生产/观察快照：

```bash
python "$BOOTSTRAP" --shared-toolchain --native-plugin-root "$SOURCE_ROOT" \
  "$TEST_DIR/compile_rms.py" <新CPU原臂目录> indexpool-0
python "$BOOTSTRAP" --shared-toolchain --native-plugin-root "$SOURCE_ROOT" \
  "$TEST_DIR/compile_rms.py" <新CPU树形臂目录> indexpool-1

task-submit --device 15 --max-time 780 \
  --env PYTHONNOUSERSITE=1 --env HCCL_DETERMINISTIC=true \
  --env PTOAS_ROOT --env ASCEND_HOME_PATH \
  --env INDEXPOOL_PROBE_REFERENCE=<four-start-t48观察目录>/observation.pt \
  --env SPMD_PRODUCTION_REFERENCE=<four-start-t48生产目录>/production.pt \
  "bash $TEST_DIR/run_rms.sh <新标签> indexpool"
```

必须先满足原臂FP32及BF16复现，再检查候选Native差异；不能修改expected SHA跳过身份。

## 原始证据

[初始CPU82](evidence/cpu-ragged-t18-v1/receipt.json)、
[生产v1](evidence/ragged-t18-production-v1/result.json)、
[首差v1](evidence/ragged-t18-observe-v1/result.json)、
[CPU85](evidence/cpu-compressor-observe-t18-v1/receipt.json)、
[生产v2](evidence/ragged-t18-production-v2/result.json)、
[三中间量首差v2](evidence/ragged-t18-observe-v2/result.json)、
[RMS CPU4](evidence/cpu-compnorm-v2/receipt.json)、
[RMS单因素负结果](evidence/compressor-norm-t18-v1/result.json)、
[主pool后首差WO-A](evidence/native-pool-t18-observe-v1/result.json)、
[动态输出CPU57](evidence/cpu-dynamic-output-t18-v2-spmd-debug/receipt.json)、
[动态输出CPU85](evidence/cpu-dynamic-output-t18-v2-spmd-probe/receipt.json)、
[T18最终生产](evidence/dynamic-output-t18-production-v2/result.json)、
[T18全部35门禁通过](evidence/dynamic-output-t18-observe-v2/result.json)、
[T24生产回归](evidence/generalized-t24-production-v1/result.json)、
[T24全部35门禁通过](evidence/generalized-t24-observe-v1/result.json)、
[T16生产](evidence/generalized-t16-production-v1/result.json)、
[T16全部35门禁通过](evidence/generalized-t16-observe-v1/result.json)、
[T4生产](evidence/generalized-t4-production-v1/result.json)、
[T4全部35门禁通过](evidence/generalized-t4-observe-v1/result.json)、
[T25生产](evidence/generalized-t25-production-v1/result.json)、
[T25全部35门禁通过](evidence/generalized-t25-observe-v1/result.json)、
[T21生产](evidence/generalized-t21-production-v1/result.json)、
[T21全部35门禁通过](evidence/generalized-t21-observe-v1/result.json)、
[T32生产](evidence/generalized-t32-production-v1/result.json)、
[T32全部35门禁通过](evidence/generalized-t32-observe-v1/result.json)、
[T33原首差](evidence/generalized-t33-observe-v1/result.json)、
[T33四起点单因素](evidence/woatnd-t33-d15-v1/result.json)、
[T33修正生产](evidence/four-start-t33-production-v1/result.json)、
[T33修正全部35门禁](evidence/four-start-t33-observe-v1/result.json)、
[T32修正后回归](evidence/four-start-t32-observe-v1/result.json)、
[T18修正后回归](evidence/four-start-t18-observe-v1/result.json)、
[T21修正后回归](evidence/four-start-t21-observe-v1/result.json)、
[T48生产](evidence/four-start-t48-production-v1/result.json)、
[T48首差](evidence/four-start-t48-observe-v1/result.json)、
[Indexer pool CPU原臂](evidence/cpu-indexpool-t48-v3-0/receipt.json)、
[Indexer pool CPU树形臂](evidence/cpu-indexpool-t48-v3-1/receipt.json)、
[Indexer pool单因素通过](evidence/indexpool-t48-v1/result.json)、
[接回完整根CPU55](evidence/cpu-native-indexpool-t48-v1-spmd/receipt.json)、
[接回T48生产](evidence/native-indexpool-t48-production-v1/result.json)、
[接回T48完整35门禁通过](evidence/native-indexpool-t48-observe-v1/result.json)、
[接回T18回归](evidence/native-indexpool-t18-observe-v1/result.json)、
[接回T21回归](evidence/native-indexpool-t21-observe-v1/result.json)、
[接回T24基准回归](evidence/native-indexpool-t24-observe-v1/result.json)、
[接回T33回归](evidence/native-indexpool-t33-observe-v1/result.json)、
[接回T47不等长通过](evidence/native-indexpool-t47-observe-v1/result.json)。
