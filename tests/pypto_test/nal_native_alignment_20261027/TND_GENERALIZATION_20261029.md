# 不同T与不等长请求的Native精度推广（2026-10-29）

## 基准与验收原则

- 分支：`dev/pypto-dsv4-csa-nal-tnd-native-align-20261027-codex`。
- 保留基准：`646acb28`；原验收只覆盖8K/B4/T24，不把它推成整模型逐token结论。
- 使用真实第4层权重、8K合成history/hidden，TP1。Native确定性、共用validated工具链、
  CANN和权重加载条件沿用[主记录](README.md)。不修改原SDK、runtime或生成C++，全部新算子用DSL。
- 新增case通过`--lengths`传每请求真实1～6行；`T=sum(lengths)`，不对请求填充S6。
  P8/P16/P64/P128仅为内部tile容量，观察只对已发布有效域比较。
- 先生产FULL/eager八状态及guards，再检查同输入、源码身份、观察/生产等价，随后逐项
  对拍Native；第一个不同元素立即停止后续模块和配置，不放宽逐bit判定。
- 全部NPU任务显式物理8，外层780秒、内层750秒及15秒清理。仅允许227物理8–15；
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

表中用时包含首次编译和诊断，不是CSA性能计时。每请求padding均为0。
七组使用完全相同的27项正式源码SHA，生产与观察输入、源码均成对核验；
本地待提交的四个正式修改文件也与这组已验SHA一致。
T33切换OB_ROWS=96策略，尚未完成其编译/设备资格；不把上述七个点推成T1～256全域验收。
T4没有新闭合的compact输出，其相关中间量比较为空域；完整state/cache、TopK和最终
输出仍实际逐bit比对。
8K整模型尚未开始。

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
[T32全部35门禁通过](evidence/generalized-t32-observe-v1/result.json)。
