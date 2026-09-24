# G9：Indexer原生数值边界对齐

基于正式3d05566d1源码和G8五QKV对齐实验；本轮未修改正式服务算法。
环境仍为CANN9.2.0-beta.2、Torch2.10.0+cpu、Torch-NPU2.10.0.post4、vLLM0.29，PyPTO/Simpler同G8。
真实第2层C4权重、TP1/B4/S6/block128、seed62合成hidden/history，原生level1与HCCL确定性开启。

## 固定native QR的模块诊断

8K八组均通过权重/guard/finite及跨组输入、native输出、源码hash核验。百分比为relative L2。
此处输出是混合诊断：Indexer消费固定native QR，不能称完整CSA精度。

|因素|同QR RoPE|同RoPE量化query|同RoPE scale|weights|混合最终输出|
|---|---:|---:|---:|---:|---:|
|combined|0.000626%|0.000000%|0.000000%|0.000000%|0.696881%|
|control|0.168307%|0.670446%|0.231535%|0.237341%|1.199971%|
|hadamard|0.168307%|0.000000%|0.020883%|0.237341%|1.145535%|
|proj|0.000626%|0.673864%|0.236205%|0.237341%|1.200513%|
|proj_hadamard|0.000626%|0.000000%|0.020921%|0.237341%|0.955035%|
|scale|0.168307%|0.670446%|0.232187%|0.237341%|1.201901%|
|weight_fp16|0.168307%|0.670446%|0.231535%|0.237341%|1.202116%|
|weights|0.168307%|0.670446%|0.231535%|0.000000%|1.199648%|

proj仅补Indexer投影BF16；hadamard改变query与key的矩阵乘→BF16→归一化→BF16顺序及H绑定。
weights包含投影与scaled输出BF16两个边界；scale为query scale FP16；weight_fp16仅改打分入参权重FP16。
proj_hadamard为组合，combined含上述全部因素。所有组以QKV五边界对齐为基础。
Hadamard组index key和cache scale均逐元素一致；combined同RoPE的query/scale、weights也逐元素一致。
投影BF16观察值始终仅差1元素，但control内部RoPE仍消费FP32，不能以观察值误判实际边界。

## 去插桩、去native QR注入的完整单层Graph

所有权重与guard通过。12轮×100次固定metadata replay，顺序轮换，编译/reset/CPU检查在计时外。
baseline是G8 QKV aligned，aligned是再增加Indexer组合；**本表baseline不是正式未对齐CSA**。
单位ms，输出误差为relative L2。

|长度|原生|QKV基线|QKV+Indexer|基线误差|候选误差|候选allclose 1e-2|
|---|---:|---:|---:|---:|---:|---|
|8K|0.581960|0.652328|0.643850|1.203559%|0.736326%|True|
|128K|0.805926|0.905396|0.902204|1.427412%|0.775025%|True|

两种长度最终误差均改善，但仍非逐元素一致。耗时差较小，不据此宣称稳定性能收益。
此结果不验证整模型、动态padding或真实请求推进。

## 剩余误差分解（8K，混合诊断）

原生attention重复Graph输出逐元素一致。以下均比较heads：

- 原生attention只换CSA Q，相对原生heads误差0.019974%。
- 只换CSA cache：0.011313%；只换CSA TopK：0.170271%。
- 全部换成CSA Q/cache/TopK：相对原生0.171760%，相对CSA heads仍0.089956%。
- 原生TopK使用相同CSA query/scale/weights/cache，集合重合率99.967448%，2141个位置不同。

单项变化不可相加解释总误差。TopK的2141位置差不等于2141个集合元素不同，还含排序变化。
同heads的O-proj仅差1元素、relative L2为0.000020709%；当前应继续检查score与attention内部边界。
源码后续确认weights*qscale的FP16乘积、QK/1024的FP16舍入，以及attention输出到inverseRoPE的BF16边界仍待独立验证。

完整结果和源码hash见[JSON](DSV4_CSA_INDEXER_20260924.json)。
