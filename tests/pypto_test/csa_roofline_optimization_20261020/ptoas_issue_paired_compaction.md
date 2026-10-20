## 需求

希望 Tile 层支持：使用同一个 Scalar 比较掩码，保序压缩源值及其列索引，
同时返回真实匹配数。可以扩展现有 `tgather`，也可以提供独立原语。
这是性能能力请求，不是已经确认的主线正确性 bug；也不预先承诺性能收益。

场景是完整动态单 SPMD CSA 的精确 Top-K。现有做法先
`gather_compare -> indices/count`，再填充/物化短索引、随机 gather score、
加全局索引偏移，最后排序。希望去掉中间索引搬运和 score 的随机 gather。
GE 保留阈值同分项，不能改成近似 Top-K，也不要求本原语负责排序。

## 建议的合同

本地原型暂命名为 `pto.tcompact.compare`：

```mlir
pto.tcompact.compare ins(%src, %threshold, %offset : !src, f32, i32)
  outs(%values, %indices, %count, %scratch : !values, !indices, !count, !scratch)
  {cmpMode = #pto<cmp ge>}
```

- 首版 A3、FP32 src/values，INT32 indices/count，UINT8 scratch；EQ/GT/GE。
- valid rows/cols 是运行时值；output capacity 是物理 Tile 容量。
- 输出严格保序，score 搬运位表示；索引是 `offset + 原列号`，全程整数。
- offset 是运行时 INT32 Scalar，不能以 FP32 中转，也不应强制变成编译属性。
- count 是完整匹配数；每行输出只有 `min(count, capacity)` 个前缀有效，
  尾部未定义。超容量不能越界，调用方可根据真实 count 回退原算法。
- count 按32B对齐，至少 `[1, align_up(source_rows, 8)]`；src物理列数64对齐，
  output capacity为8的倍数且不大于src物理列数。
- 三个输出必须显式建模，禁止用调用方 scratch 充当隐含返回值。

## 建议实现

Vector 构造INT32索引，`vcmpvs_ge/gt/eq`产生一次比较掩码；
使用相同mask分别对索引和FP32的32位表示执行 `vreducev2`。
通过 `get_rsvd_cnt()` 获取真实 count，分别做容量受限的 `vcopy`。
零有效长度/零匹配不应执行非法的零长度mask操作。

可复用scratch：源物理宽度为C时，先预留4*C字节索引；
后区在索引构造完成后复用为比较掩码和压缩临时区。
原型最小容量为 `4*C + max(1024, align_up(C,256)/8 + 4*C)` 字节。
两个结果不相互别名，scratch与输入、输出不能发生未声明的覆盖。

IR/编译器还需同步：

1. DPS输出与资源校验。
2. values/indices是Vector写，count是Scalar写，scratch的读写效果明确。
3. 后继Vector、Scalar、MTE消费者的自动同步不能只按外层`PIPE_V`推断count。
4. 源/输出视图的物理容量与动态有效范围不能混淆。

## 本地验证范围

已有独立本地原型，未提PR、未push依赖分支。底层辅助模板的三种比较模式
已通过本机A3 CCEC编译；PyPTO三输出接口、类型与拒绝边界测试共56项通过。
这些不等于新PTOAS完整编译、真机数值或整层CSA性能通过。
完整PTOAS仍在构建，随后会验证实际生成代码、直接输出与图重放，再做整层配对。

新增的一遍连续score压缩及结果复制也有成本，必须计入完整CSA收益。
若抵消了删除随机gather的收益，算子侧候选会停止；能力是否值得上游采纳可独立讨论。

配套PyPTO接口请求：https://github.com/hw-native-sys/pypto/issues/3085
