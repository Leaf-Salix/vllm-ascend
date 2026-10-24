# HCA融合计算的BF16边界差异与整模型观察

日期：2026-10-24。此记录区分**数值差异来源、Native一致性和答案质量**。
正式HCA算子及共享SDK没有修改；恢复BF16边界的版本仅保留为诊断候选，未采用、未push。

## 用户确认的验收原则

用户明确：不一定要恢复Native的BF16边界，最终看decode答案是否符合问题要求；
这笔数值差异必须记录。Native逐bit、逐token一致不是硬门槛，Native输出也不是标准答案。
此前报告中的64/256请求token一致仅是相似性统计，不能直接解释成正确率25%。

保留FP32通常减少中间舍入，相对高精度数学参考一般更准确；但经过INT8量化及后续
非线性计算，不能保证所有最终值或生成答案都更好。恢复BF16是在复现Native的数值
行为，不是提高浮点精度。是否采用须看答案质量和性能，不根据不同元素数下降直接决定。

## 第一处分歧已排除图重放和输入不一致

使用正式W8A8权重、同一个模型壳、同一离线KV、TP1/DP/EP16、NZ2、确定性level1。
框架冻结基础为`15b64665acafe64e3a851cba29a6fd78378d06e7`，共享SDK为已验收的
`csa-shared-20261024-group-dims`。这轮为功能/精度诊断，沿用原功能配置，
静态编译及SuperKernel关闭；不用于Native性能比较。

两侧只生成一个token，容量仍保持原192token配置。Native控制通过同一HCA模型壳
执行真实Native计算；PTO执行原HCA候选。两侧分别复现各自历史256/256个首token。
真实首轮为B16/S1、T16、graph mode NONE，尚未进入ACLGraph replay。
input_ids、positions、query_start_loc、seq_lens完全一致。

rank0在实际模型中逐层配对：先保存将写入的KV/state物理页，执行PTO，再恢复初态、
执行Native同输入对照，最后恢复PTO的输出与写入页继续原轨迹。
20层均验证输入未改、实际PTO输出未改、状态恢复逐bit一致；诊断没有改变历史首token。

第4层（layer index=3）是第一个HCA层：输入逐bit一致，Native同输入对照也与
Native完整轨迹的该层输出逐bit一致。PTO输出从这里开始不同：max_abs=0.00390625，
RMSE=0.000391288586，57063/262144个元素不同，原0.01容差下没有超差。
后续误差累积，最后HCA层的轨迹输出RMSE约0.30130；不能把累积误差全归给最后一层。
报告：`FIRST_TOKEN_ANALYSIS.json`。

q08两个实际候选的logits：

| token | Native | 原PTO |
|---|---:|---:|
| 9854，代码围栏开头 | 29.000 | 29.125 |
| 3648，“好的”开头 | 28.750 | 29.125 |

PTO出现并列最大值，argmax选择较小ID3648；Native选择9854。
`topk`对并列项的顺序不代表实际采样顺序，分析脚本使用真实输出ID和argmax核实。
该结果解释首token分叉，不代表其中一个答案更正确。

## 五处BF16边界

Native路径为`vllm_ascend/attention/dsa_v1.py::_mla_prolog_multistream`。
PTO主要在`ops/pypto/deepseek_v4_flash_csa/qkv_proj_rope.py`及
`ops/pypto/deepseek_v4_flash_hca/q_projection_streamed.py`。

| 位置 | Native实际行为 | 原PTO融合行为 |
|---|---|---|
| Q_A投影 → QR归一化/量化 | 投影结果BF16，再由融合RMSNorm动态量化读取 | FP32累加结果直接进入RMSNorm/量化 |
| KV投影 → KV归一化 | 投影结果BF16 | FP32投影结果直接归一化 |
| KV归一化 → RoPE | 归一化结果BF16 | RoPE部分保留FP32归一化结果 |
| Q_B反量化 → 每head RMSNorm | quant_matmul输出BF16 | FP32反量化结果直接归一化 |
| Q的RMSNorm → RoPE | RMSNorm输出BF16 | RoPE部分保留FP32归一化结果 |

诊断候选在已有tile上插入`pl.cast(pl.cast(value, pl.BF16, mode="rint"), pl.FP32)`，
保留Native的BF16舍入后继续FP32计算，不增加Host取值，不必落GM或新增task。
没有在Native融合QR RMSNorm和INT8量化之间凭空增加BF16边界。
原有tile、循环和数学函数复用不变，候选只在私有算子副本中生成。

## 同一个真实输入的三组单卡结果

输入来自实际模型第4层，先执行Native HC_pre与输入RMSNorm，固定该归一化结果后
调用生产QKV函数。它是实际QKV输入诊断，不是随机输入，也不重写QKV数学。
CPU事后检查确认三组归一化输入和Native参考相同；不依赖完整历史KV的数值。
因此本表仅证明QKV段，不代表整个HCA或整模型答案通过。

| 实现 | QR INT8不同元素 | KV不同元素 | Q不同元素 | Q RMSE | QR scale最大差 |
|---|---:|---:|---:|---:|---:|
| 原FP32融合 | 781/16384 | 2309/8192 | 381955/524288 | 0.00537905655 | 8.61287117e-6 |
| 只恢复Q_A BF16边界 | 0 | 2309 | 147473 | 0.00293782088 | 9.31322575e-10 |
| 恢复五处边界 | 0 | 0 | 77 | 0.000112857527 | 9.31322575e-10 |

恢复五处后q08的Q/KV/QR全部一致；QR scale仍有4/16行的FP32末位差。
余下77个Q差异分布为row7/head24的75个、row10/head48的1个、row14/head18的1个。
追加Native Q_B/RMS中间值采样没有改变QKV结果，供以后需要时继续分析；本轮不继续
为逐项对齐追求77→0。详见`QKV_BOUNDARY_COMPARISON.json`及各任务原始result。

恢复边界候选另做了B16/S1和B16/S6的10步完整单卡图验证：graph/eager输出及
全部cache/state相同，未写区域通过；每场景的Native输出原0.01容差通过。
**这不代表候选整模型答案质量或性能更优。候选未提交16卡整模型测试，未合入正式算子。**

## 已有答案不能以Native为真值

`audit_answers.py`只用CPU按题意计算标准数值，分析原来的192token实测输出，
没有运行新模型或执行模型生成的代码。完整结果为`ANSWER_AUDIT.md/json`。

- q02第一行正确值为`[1,9,5,24]`；Native为`[1,20,92,63]`，HCA为`[1,20,100,71]`，
  两者都错，Native相似性不等于答案正确性。
- q03正确排序的第一条应是`n0/net3069/score965/tagA`；两侧都先输出了错误的f1记录。
- 两个重复句子问题的已生成前缀符合题意，但192token内未完成要求的80/81次重复，
  以前的简单请求PASS只能解释为前缀检查通过，不能当成完整答案通过。
- 大多数复杂问题只生成到解释或部分列表。不能凭尚未出现可评分记录就宣布正确，
  也不能据“Native可解析错误5题、HCA2题”得出HCA更好；两侧完成度和可评分覆盖不同。

16个问题在16rank重复，不是256个独立质量样本。本次数据是功能轨迹而不是完整
答案质量基准，尚不能给出全面质量验收结论。下一次若做答案验收，应允许足够生成
长度并按题意评分，把模型两侧共同错误、HCA新增错误及仅措辞差异分开记录。

## 实验和复现索引

所有设备任务走task-submit；具体命令保留在`jobs.json`，终态在`STATUS.json`。
小型结果、首token各rank输出和逐层诊断在`precision_evidence/`。
大Tensor及编译产物只保留本地，路径/大小见`PRECISION_ARTIFACTS.json`，不做hash扫描。

| 任务 | 任务号 | 结果/作用 |
|---|---|---|
| 首token Native控制 | task_20261024_142414_13039461815 | 256/256复现历史Native |
| 首token原PTO | task_20261024_142414_13039861207 | 256/256复现历史PTO，20层同输入配对 |
| QKV初版探针 | task_20261024_145230_227533728876 | 探针误把ND wkv标成NZ，编译拒绝；非SDK bug |
| 原FP32融合探针 | task_20261024_145345_231458123848 | 完成上述基线比较 |
| 只恢复QA边界 | task_20261024_145345_231474732285 | QR INT8差异归零 |
| 恢复五处边界 | task_20261024_145346_231513923637 | KV差异归零，Q剩77 |
| 候选单卡S1/S6 | task_20261024_145553_238205414837 | 10步graph/eager与保护区通过 |
| QB/RMS额外采样 | task_20261024_145627_240587426246 | 原始中间值留本地，采样前后结果一致 |

准备路径：`prepare.py`按原基础提交生成`experiments/v04_graph_service/source`；
`prepare_precision.py`派生首token模型诊断；`prepare_rounding.py`从v04生成三种私有
算子包，`qkv_probe_entry.py`只包装原生产QKV函数。最小候选差异另存
`QKV_BF16_BOUNDARIES.patch`。所有历史结果保留原工具链版本，不跨版本计算性能。

`analyze_first_token.py`、`summarize_qkv.py`和`audit_answers.py`是CPU分析入口。
不得将本记录中的`COMPARED`误写成Native逐bit通过或答案质量通过。
