# 固定L1 Tile循环更新：CPU接口探针

2026-10-17。只编译CPU/PTOAS，不占NPU。目的是为后续N64循环流水判断写法，
不是当前CSA优化、正确性或性能结果，也没有修改依赖仓。

- `tile_inplace_loop_cpu_v1`：小例子遗漏SPMD block index，被前端拒绝。
  按报错在单worker中显式读取block index并用于输出行偏移；不是组件缺陷。
- `tile_inplace_loop_cpu_v2`：`pl.tile.assemble(pool, product, offset)`忽略返回值，
  编译器不能将结果复用到原Mat，要求不支持的保留区域Mat→Mat拷贝，完整编译失败。
  `probe_tile_inplace_loop.py`保留该写法，供查看失败原因。
- `tile_inplace_loop_cpu_v3`：改成`pool = pl.tile.assemble(pool, product, offset)`，
  完整代码生成通过。代码位于`probe_tile_inplace_loop_assigned.py`。

实际IR：`pool`为512B地址处的16×32 FP16 Mat，循环内TINSERT原位写入两个16列窗口，
循环结束后的TEXTRACT读取同一pool；没有Mat→Mat搬运。生成代码在消费前有FIX→MTE1
同步。见`tile_inplace_loop_codegen_audit.json`和完整`.pto.txt`、`.cpp.txt`。

边界：这里只覆盖16×16 FP16矩阵、无分支循环、无pre_quant。尚未证明原Indexer的
M384/INT32到FP16量化、动态Native页分支、双缓冲/FFTS或设备数值。小例子不执行设备。
不能据此宣称之前的Right→Right循环回边问题已解决。

后续可在新候选中保持固定Key/score L1池并回写DPS返回值，让Right提取留在循环内部，
避免把Right作为循环携带值。与旧展开流水相比，下一Key的Right预取可能提前量减少，
要检查实际分配和同步后决定是否值得测128K/B16。旧失败候选和现选均保留。
