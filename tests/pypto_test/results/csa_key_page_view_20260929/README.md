# Native页指针式Key读取：核内GM切片表达检查

状态：轻量CPU lowering失败，没有提交设备任务，也没有改生产或Native cache。
当前PyPTO88f60598；本地origin/main参考f997db72，未声称已查询远端最新提交。

ops-transformer28f40354 A3 QLI V2 KeyNd2NzForPA直接构造
`keyGm + physical_page * keyStride0 + row_offset * headDim`，以ND→NZ搬入L1。
当前PTO把原融合cache作为唯一可写根，页起点可能有64字节余数，分别使用原始和偏移64字节的二维Key视图。
若在核内先切出单页4096字节再作为32×128的GM视图，本可少做逐页视图分支；是否加速仍未知。

本探针使用真实页字节步长4160、32×128 INT8 Key、动态物理页、INT8 QK消费者：
`tensor.slice -> reshape -> tile.load(Mat)`。只执行默认IR pipeline/PTO MLIR生成，没有CCE或NPU。
结果在ConvertTensorToTileOps后出现“tile.load输入已是TileType”，未产生可执行候选。
源码确认convert_tensor_to_tile_ops_pass.cpp的PreservesTensorLike只保留tensor.dim/tensor.view，
tensor.slice由后续加载转换处理；tensor.view当前API接受shape/valid_shape/layout，没有显式页字节偏移参数。
本地main的view接口也相同。这证明当前这条表达链不成立，不证明所有替代写法都不可行。

暂保留原双视图分页读取；不能把一次lowering失败当成需要拆cache、改Native或整个核内优化已完成的理由。
若后续工具链有支持带动态offset的GM别名视图，再按实际生成搬运/同步和单卡性能评估。
[探针](lower_probe.py)、[错误与范围](lowering.json)。
