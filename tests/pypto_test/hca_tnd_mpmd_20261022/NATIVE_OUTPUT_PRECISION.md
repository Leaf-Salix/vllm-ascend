# HCA TND：8K/B32输出差异定位与Native数学修正

全部使用正式W8A8第3层权重、NZ2、公共`csa-shared-20261022-logical-codegen`。
全部直接执行，不使用ACLGraph capture/replay。输出容差保持atol/rtol=0.01/0.01，
cache/state严格比较仍为1e−5/1e−5；没有用放宽阈值消除失败。

## 先排除TND迁移，再定位计算段

八档初始矩阵中七档输出通过；8K/B32/S6在token143、hc3的27/1558/2687列有3处超差。
最大绝对差0.03125，RMSE=0.001952652896，保护区和重复调用通过。
`short_legacy_b32_v1`还原旧HCA短历史入口，修正旧函数签名和INDEX转FP32的兼容写法后，
旧HCA与新TND输出及所有cache/state逐bit一致，Native仍为相同3处超差。
因此不能把问题归因于TND请求边界或长短历史迁移。

`precision_probe.py`从生产入口AST提取采集版，只把normalized/post/comb/packed
改绑到外部Tensor，并在Q生产者scope内增加一个依赖q_ready的复制任务。
不另写QKV或attention；采集版必须与原入口最终输出及所有cache/state精确一致。
Native侧包装实际hc_pre、RMS hook、attention算子和`_forward_o_proj`，采完立即恢复。

`precision_stage_h8192_b32_v3`（`task_20261022_193738_9212439512`）结果：

| 诊断拼接 | 超差数 | 输出RMSE |
|---|---:|---:|
| 原PTO全链 | 3 | 0.001952652896 |
| PTO attention及门控 → Native O投影/HC_post | 0 | 0.001737958109 |
| PTO attention → Native尾链及Native门控 | 0 | 0.001737699352 |
| Native投影输出 + PTO门控 → Native HC_post | 0 | 0.000108361038 |
| Native attention及门控 → 原PTO尾链 | 1 | 0.001846529281 |

最后一项仍在`[143,3,27]`失败，说明原PTO尾链自身就有数学边界差异；
仅修改上游RMS/attention或HC门控不能解释全部失败。

## 修正内容

实际Native链路见`vllm_ascend/attention/dsa_v1.py`的`_forward_o_proj`：
分组O_A落BF16，之后按完整token的8192列量化，再执行O_B。
旧PTO保留O_A的FP32结果，八组分别量化，最后八组点积各乘scale后进行FP32归约。
这两者既有BF16舍入边界差异，也有量化尺度定义的差异。

新候选`candidates/output_native.py`：

1. O_A仍复用原分组矩阵函数；量化入口先作BF16往返，恢复Native的可见边界。
2. 对每token完整8192列取amax，分别计算量化乘数127/amax及反量化scale=amax/127。
3. 一份统一INT8激活供O_B；完整K8192点积在INT32中累加，随后按channel scale、
   token scale反量化，落BF16，再执行原HC_post单行计算。
4. 每token的scale在内部工作区独占64B，避免不同AIV写入同一短行存储块。

最坏整数绝对和8192×128×128=134217728，低于INT32上限。
后续K分两段的候选必须先把两份INT32结果相加，才转FP32；不能先各自转FP32再相加。
外部ABI、Native权重、cache/state布局和生命周期均未改变。

`native_output_h8192_b32_v4`（`task_20261022_202943_293295729130`）证明：

- 原3处输出超差归零，RMSE降至0.001737914970；最大绝对差仍为0.03125，
  是否通过依照逐元素绝对及相对容差，不能只看全局max_abs。
- 新尾链接相同Native attention及门控时，超差归零，RMSE=0.000031872203，
  max_abs=0.0078125；不声称逐bit移植Native。
- 采集版与生产版完全一致；原cache/state、只读metadata及写保护保持。

后续八档输出、S1和混长/padding服务验证已补齐，见`NATIVE_OUTPUT_SUMMARY.md/json`。
这次修复只解决输出尾链的已定位问题。上游归一化造成的Native严格storage差异仍存在，
没有把`passed=false`改写为全面精度通过，也没有完成连续状态轨迹或整模型token验收。

## 编译尝试与组件问题

- v1：低层row_max缺少temporary tile，修正算子调用；不是SDK问题。
- v2：1×1 FP32列向量不满足A3的32B对齐约束，改为8段归约布局。
- v3：用Scalar FP32 max合并8段，在实际AscendC设备编译中触发`ptoas_bitcast`
  helper缺少`AICORE`。已有[PTOAS #1667](https://github.com/hw-native-sys/PTOAS/issues/1667)，
  已补充[实际设备编译证据](https://github.com/hw-native-sys/PTOAS/issues/1667#issuecomment-6340763602)，
  正文保留于`PTOAS_1667_COMMENT.md`；没有重复建issue、提PR或改共享SDK。
- v4：8段行max后复制成8×8并做列max，全程Vector计算，规避Scalar helper问题。
  输出8份相同amax并用于各段量化，保持整个token共享一个尺度。
- 中间值采集首次把scope内产生的Tensor SSA别名用于scope外消费，生成的编排C++
  引用越界别名。最小复现`repro_scope_alias.py`已提交
  [PyPTO #3107](https://github.com/hw-native-sys/pypto/issues/3107)，说明及建议在
  `SCOPE_ALIAS_ISSUE.md`。把诊断复制放回生产者scope后通过；没有修改生产数学。

早期`--cpu`的结果只表示PyPTO/PTOAS生成码通过，不能代表AscendC实际目标文件编译。
当前CPU门禁已经补上编排C++和每个incore C++的真实编译；v4、split及区间版本均按新门禁验证。
目标文件、`.so`、本地`stage_values.pt`只作本地证据，不入Git；不做重复hash扫描。

## 性能取舍

Native数学版本相对旧分组算法在B1/B28/8K B32有改善，其余档略慢，不能当作全档提速。
采用它的首要依据是修正已证实的输出算法差异。
新泳道显示HC_post核占用减少，但整行量化需等齐所有O_A分组，O_B开工推迟。
进一步采用N256、K分两段是在减少激活重复供数和保持并行度之间取舍，
没有恢复旧分组量化，也没有改变整数点积结果。是否采用由完整层配对结果决定。

Native/PTO外部eager报告、PTO完整L1和各核占用是不同统计定义，不混用分母计算30%目标。
旧Native profiling含框架派发空隙，旧PTO外部profiling仅预绑定调用，尚不代表服务入口的最终比较。
