# HC 前段流式供数候选：同步与数值合同

本候选在 `hc_stream_pre.py` 中，只改变实验生成的 CSA 单 SPMD 的 HC 前段。
目前仅验证 T96（B16/S6），不是各档统一策略，也没有替换生产入口。

## 数值保持方式

- 保留 M16、K4096、4 份投影部分和。每份仍依次执行 K=0、256、…、3840 的16次矩阵乘。
- AIV 每次对8行、K512的 BF16 输入做无损 FP32 转换，与原加宽的数学 Tile 相同。
- RMS 的每个 K512 `row_sum` 单独写出。归约阶段从0初始化，按原32个K块顺序逐项相加，继续使用原缩放、epsilon、rsqrt。
- 不将32项改为四个局部和，不替换 standalone HC_pre 的另一套分片/归约策略。

## 核分工与工作区

24个AIC组各处理一个原M16/K4096分片，每组两颗AIV各处理8行。
全部48颗AIV供数。仍写入原 FP32 加宽区；后续HC混合阶段会读取它，不能删除。
32份RMS小结果占12288字节，使用T96原本无尾行时闲置的 `temporary_5`。
没有新外部参数、GM分配或 Scalar 信号。

原 `hc_pre_linear_reduce` 的6个AIV成员除了归约四份投影，还计算对应16行的RMS。
检查了两个inv_rms消费者均等待所有归约成员，因此原加宽完成标记不会被误当成新RMS已完成。

## 组内交接

事件8/9为两个READY槽，10/11为对应ACK槽。每个组独立，两个AIV分别发布，AIC等待成组就绪。

1. AIV在复用某个事件前消费上一轮ACK，加载BF16并转换。
2. AIV写原FP32加宽区、写RMS分片，再在MTE3发布READY。
3. AIC先等待READY，再在MTE2加载该K512；随后在MTE2发布ACK，继续两次K256矩阵乘。
4. AIV末尾消费剩余两个ACK，不能带着未消费事件进入后面的Indexer/QK。

原AIC整块GM等待由这一组内协议替代；后续阶段GM进度、代际、DCCI/DSB及数据维护仍保留。
128个随机交错×连续3代的CPU事件模型检查无覆盖、无死锁、无残留。模型假定现有FFTS API语义成立，
不能证明真实硬件pipe或缓存可见性；完整真机exact/图/保护区检查仍为必需。

## 编译中遇到的问题

1. CPU_v1：直接将store结果重新赋给根参数，触发MIX整参别名检查。改用原地store表达式，保持原参数返回。
2. CPU_v2：pipeline内动态event_id在PTOAS生成C++时报operand不在scope。候选改为偶/奇静态ID分支。
3. CPU_v3：静态ID后编译通过，但IR和C++均显示自动pipeline把输入TLOAD移到READY等待之前。这版未上卡。
4. CPU_v4：两个有手工FFTS的循环改用pl.range；仅纯数学RMS累加保留pipeline。CPU编译与指令顺序审计通过。

证据：`hc_stream_pipeline_order_audit.json`、`hc_stream_prepare_v4/codegen_audit.json`，以及各CPU目录的日志/源快照。
这次没有修改PyPTO、PTOAS或Simpler，不把作用域错误和pipeline顺序问题混为同一原因。
当前规避方式不支持把任意手工同步循环直接改成自动pipeline。若以后需要恢复该优化，应先补最小复现并修正编译器副作用依赖。

## 真机验证

`task_20261017_032959_27383327818` / `b16_cached_hc_stream_v1` 已完成exit0，固定fixture完整exact、图重放、30代进度和全部保护区通过。
对照只执行原动态PTO；正式Layer4 W8A8权重、固定fixture、原exact阈值；完整CSA L1 incore首尾3窗。
不得用该候选的局部阶段时间代替整层性能。结果以该目录result.json为准。

三窗L1：834.18/844.40/847.60μs，中位844.40；同轮原动态907.64。
未优于保留815.84μs，未采用，不扩其他档。不依据单次整层差值断言具体哪个新增阶段造成回退。
汇总见 `hc_stream_result.json`。
