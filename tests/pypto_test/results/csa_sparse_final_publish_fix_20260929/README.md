# Sparse末块发布：统计量切片修复

原候选的两档最终x_out未通过零容差，生产算子仍保持4ffccb7b。
本目录只针对明确的后16-head统计量地址问题，不放宽容差。

原候选MemoryReuse IR的mi/li切片带64字节偏移，但生成C++的后半块仍读取前半块基址。
显式TEXTRACT ColMajor→ColMajor在A3不支持；采用完整[32,1]→[1,32]的零偏移reshape，
ND显式extract后半块，再恢复[16,1]。其他计算、分块、缓冲寿命、调度属性不变。

完整包复制到`.cache/csa-sparse-final-publish-fix-4ffccb7b-{baseline,candidate}`，
使用`pkg:dsv4_csa_sparse_final_publish_4ffccb7b`；原失败副本保持不动。
[构建](prepare.py)、[单变量补丁](fix.patch)、[编译](compile.py)。

先使用已有Native固定Q/cache/Top-K的前16个序列，cos=1/sin=0，
比较原生产基线、失败融合版和修正版的完整Sparse输出及每16-head分组。
这不是新增B40测试，也不能替代完整CSA逆RoPE、尾块/padding和整模型token/DSpark验收。
只有修正版零容差通过，才恢复完整CSA验证。
[设备入口](run_sparse.sh)、[输出检查](compare_sparse.py)。

task_20260929_090812_141352622666已完成（exit=0），auto设备12，CANN9.2。
修正版完整编译/load通过，独立Sparse的3145728个BF16输出与4ffccb7b零容差一致。
失败版有1556014个元素不同；前16-head及32–47完全一致，差异只在16–31和48–63，
与生成代码丢失统计量半块偏移的定位吻合。最大误差0.3974609375；显式ND抽取后完全消失。
对历史Native输出的原有算术差异仍明确保留，不冒称Native逐元素一致。
[完整分组检查](sparse_check.json)。

完整CSA在[新对照目录](../csa_sparse_final_publish_fixed_pair_20260929/README.md)单独排队，
本诊断源文件与两侧私有包均冻结，不在排队/执行时修改。
