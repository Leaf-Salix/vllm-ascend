# 累计阶段诊断生成物审查

v2最终四根CPU通过：ours64、Nal完整/TopK/heads均57参，NPU未初始化，收据绑定347033c7最终manifest。两份cut实际C++与受审v1逐字节一致；PTO只有已声明v1/v2绝对源码loc变化，见CUT_GENERATED_PROOF.json。

独立审查确认heads保留QK/PV TMATMUL、softmax EXP/归并、DIV/BF16/inverseRoPE及packed TSTORE；PTO有809条attention_stream_stages定位。FFTS保持AIV(MTE3)set2/4→AIC(MTE2)wait2/4，AIC(FIX)set3/5→AIV(MTE2)wait3/5，同原有效块/三槽排空。TopK无attention_stream或packed写，只有Indexer0/1，没有残留attention/OA/OB等待。两cut编排与完整Nal一致：70tensor/33alloc/单scope/1MIX，216.133789MiB对齐payload。

TopK cut终点实际包含slot plan；heads包含packed发布。尾删会改变完整函数机器码资源调度，累计差只能视为增加阶段后的整体增量，不能当完整单SPMD内部的精确阶段耗时。7外边界不包含内部heads，不能据此新增heads逐bit验收声明。正式计算代码未改。
