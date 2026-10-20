补充本地实现、真实设备验证及后续优化建议（仅issue讨论，未提PR、未push依赖仓）。

原成对压缩路径已完成实际PyPTO→PTOAS→CCEC编译和A3真机验证：直接原语65组×GE/GT/EQ×3次变输入图重放、完整阈值/同分排序归并59组×3次重放均通过。完整CSA B16/B28严格输出、Top-K/score、状态、保护区和图代际均通过；当时完整L1净省分别-1.88/+10.28μs，不能仅凭局部排序减少声称整层大收益。

进一步检查发现两个可消除的内部成本，本轮在原本地PTOAS提交b27d5a0f上实现：

1. INT32索引先构造64项，再复用前缀倍增。2048列从32条独立vadds变为首块+5轮倍增，每层显式PIPE_V屏障；非2幂容量按64项尾块裁剪，repeat限制240。基址仍走INT32，不经FP32。
2. 当输出容量等于源列数时，vreducev2直接写value/index的DPS输出；较小容量仍先暂存，再按min(count, capacity)复制，保证真实count及不越界合同。

核心分支建议：
```cpp
constexpr bool directOutput = D::Cols >= S::Cols;
constexpr unsigned collectBytes = directOutput ? maskBytes : maskBytes + indexBytes;
constexpr unsigned scratchBytes = collectBytes > 1024 ? collectBytes : 1024;
auto indexTarget = directOutput ? outIds + row * I::RowStride : packed;
vreducev2(indexTarget, indices, mask, /* 原相同参数 */);
// 读取真实count，较小容量保留原有界复制；score使用相同mask。
auto valueTarget = directOutput ? values + row * D::RowStride : packed;
```
对应verifier最低scratch也必须同步：`indexBytes + max(1024, directOutput ? maskBytes : maskBytes + indexBytes)`。不要只缩小调用方scratch而保留旧helper访问。

当前满容量2048列scratch为9216字节，相对旧CSA的18432字节scratch减少9216字节；两个输出各增4096，局部总占用净减1024字节。容量截断路径不变，不把满容量快路径用于不足容量。

新验证已完成：
- 2048→2048直接输出、2048→1024有界输出、192→192非2幂列数，各65组×三比较模式×三次同址重放PASS；验证动态有效长度含0、整数基址±16777217、FP32位模式、保护区和输入不变。
- 59组阈值/同分排序归并回归PASS。
- 六项既有PTOAS验证/同步/负例PASS；新增9216接受、9184拒绝PASS。
- 完整B28不提供样例shape的符号编译保留58动态轴及一个MIX提交。
- 完整CSA B16/B28严格检查通过。同轮原SPMD/新候选L1分别789.46/790.04、1160.92/1139.70μs。B16持平，B28净省21.22μs且5/5窗口更快，正在交换实例角色复核，尚未正式采用。

另观察到同源B1双实例存在约13μs角色相关偏移，原因尚未证实。因此以上中位数原样报告，不把所有小差异视为稳定优化收益。此优化仍为A3本地原型，未宣称其它架构或全部规格通过。
