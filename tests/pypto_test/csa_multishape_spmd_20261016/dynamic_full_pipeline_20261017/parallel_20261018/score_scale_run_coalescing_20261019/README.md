# 2026-10-19：Indexer Native scale连续物理页合并搬运

## 假设与算法

balanced Score每处理1024个候选，两AIV各逐页读32份64B Native scale。
已有72核模型中对应小搬运每核出现704次，是值得用真机验证的供数成本；
不能把模型时间当作可节约时间。候选在每个叶开始时用Vector读取实际页表，
比较整个有效前缀与first±arange；物理页连续才允许一次二维stride搬运多页。
每叶只证明一次，碎片/重复/非法页仍走原逐页gather。原生cache仍是每页
4096B INT8 key+64B FP16 scale；没有Host回读、GM缓存转换或额外工作区。

Native页表既可能正序，也可能逆序。核对原七档fixture发现页号是逆序连续，
因此没有改测试输入去迁就升序实现：MTE2统一从低物理地址向高地址读取，
逆序时在UB用TGATHERB按页恢复逻辑顺序。每页64B用两个32B块，64个UINT32
字节偏移重排1024个FP16元素，页内顺序保持。A3已有对应ISA实现。
尾块只加载真实页，未加载行补零；倒序索引先钳制到已加载区域，最终按有效
页数补零。CPU索引核对覆盖1..32尾页数，共16896个有效FP16元素及所有偏移。

所有T/B、页数、地址和动态尾部保留device计算；仅固定1024/32等算法tile。
只有在物理页容量不超过2^24时才用FP32无损证明整数关系，否则直接回退。
前缀完整比较包括每一页，不能仅因首尾符合就误放行中间空洞。Cube数学、
AIV缩放/Top-K、FFTS和GM通知/等待不变，未叠加WS或Attention候选。

## 尝试记录

| 版本 | 结果 | 处理 |
|---|---|---|
| cpu_v1 | 快慢路径PadValue元数据不同，类型检查失败 | 保留source_candidate_v1，将已有零初始化统一为FP16 zero padding，字节仍全零 |
| cpu_v2 | 当前后端不支持INDEX到FP32直接cast | 保留source_candidate_v2，在已证明<2^24的条件下经INT32无损转换 |
| cpu_v3 | 无样例shape完整编译PASS，仅支持正序 | 核对fixture后保留source_candidate_v3_ascending，补逆序支持，不上卡测无效快路径 |
| cpu_v4 | 正逆序完整符号编译PASS，58动态轴 | 当前source_candidate；TGATHERB存在，待真机 |

上述前两次是算子表达/类型约束修正，没有修改PyPTO/Simpler/PTOAS。
本轮正式开发文件是ops/pypto下的indexer_score_stages.py；完整候选均已快照。

## 真机验证安排

四个任务已提交，通过统一队列分配设备，当前等待空卡：

- 原输入128K/B16、B28/S6：相同Native页表，三方五窗完整CSA L1。
- B16升序历史页：只反转完整历史前缀，当前写入页及slot_mapping保持；
  三方三窗，另验metadata A→B→A和padding图重放。
- B16碎片历史页：交换每请求历史页1/2，当前写入页保持；同样检查图重放。

扩展输入同时给原多task、当前单SPMD与候选，不能混进七档原输入性能表。
槽映射没有被误留给被改的当前页。驱动记录真实输入模式与源码差异，编译器
三方均为E17冻结reference_toolchain，避免叠加GM同步优化。
这些是已提交任务，不代表通过；结果以DEVICE_SUBMISSIONS和最终报告为准。

大编译目录、完整IR错误、二进制及泳道图只本地保留路径，不计算hash。

## 原输入真机结果与扩展输入修正

原B16/B28两项全精确/Top-K/score/cache/state/图/保护/协议PASS。
五窗完整L1：B16多task871.08、参考828.56、候选836.28μs（回退0.932%）；
B28多task1292.52、参考1271.86、候选1264.92μs（降低0.546%）。无稳定实质
收益，当前不整体采用，不把静态DMA次数减少当作真机必然收益。

扩展页序v1/v2/v3的测试构造问题及处置见NATIVE_FIXTURE_CONSISTENCY.md。
v4已改为在Native fixture的BlockTable.add_row前配置历史页序，所有CPU/GPU
表、slot、compact metadata及保护快照由同一布局生成。默认original的页号
表达式AST与原版相同，测试默认输入不变。v3仅为诊断，不作为规范Native输入
的完整功能证据；v4两项待最终结果。保护检查和阈值均未放宽。

v4两项最终均completed、exit=0：精确输出/Top-K/score、cache/state、保护区、
代际、普通图、metadata和padding图重放全部通过。扩展三窗中位：升序参考
822.40→828.94μs；碎片参考840.88→845.02μs。它们不属于原七档输入，不能
写入七档性能表。未获稳定收益，不采用；接下来验证已有META阶段统一证明
是否能减少Score中的重复工作，设计已单独保存，当前尚未实现。
