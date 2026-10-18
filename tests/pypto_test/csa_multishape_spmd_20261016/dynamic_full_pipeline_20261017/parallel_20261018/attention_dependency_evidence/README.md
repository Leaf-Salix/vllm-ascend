# 恢复 Score→Top-K→计划→Attention 的动态行依赖

2026-10-18。优先恢复静态单SPMD已经采用、正式动态版本尚未保留的有效并行。
协同前段之后，配对插桩中Score AIV端点动态/静态514.36/514.62 μs，
Top-K为534.60/526.60 μs，Attention为729.58/694.16 μs；第三轮计划端点
为554.70/529.34 μs。它们用于确定调查方向，不是净计算时间或完整L1成绩。

## 候选改动

正式动态源码位于 `vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/`，
本目录冻结 `source_reference` 与 `source_candidate` 供独立对照。
新 `attention_dependencies.py` 明确完整中文总览与缓存/所有权约束。

1. Top-K保留原query%48分工，在每条query归并前，根据本轮group前缀、leaf
   布局与balanced/双根策略，只获取该query实际读取的根写者。复用归并已算
   的请求边界和可见根数，不再次扫描Q/L。B16/G6/六叶时每query只有3个
   不同AIV写者，原先统一等待48个。
2. 每个八行plan tile只等待该tile最多8个Top-K writer。没有plan工作的核
   不读取Top-K，也不等无关writer，仍按原协议发布自己的阶段完成。
3. AIC和两个AIV对query=group+k*24读取同一组计划行，只等这些行的plan
   owner。Native cache可能共享页，原cache写者与query/逆RoPE等待保留。

不冻结B/S/history，不增加根参数或phase槽，不减少数学/候选/排序，不改变
已有FFTS三槽协议。完整阶段发布仍先排空并维护缓存，消费者获取后仍执行
fence/cacheinvalid/fence；没有把ACK误用为根完成，也没有假设Scalar缓存一致。

## 已完成检查

`check_dependency_coverage.py` 提取真实候选两个等待函数，以生产者正向枚举
作为oracle：107457项根依赖、4992项计划依赖、600项mixed query分组映射
通过。覆盖G1/G2/G6、少/多group映射、1～32叶、截短可见前缀、尾部、空T
及多轮strip。移除新增等待后，Top-K/plan数学AST与旧版相同，Attention
核内流水、Score与排序源逐字相同。它不是硬件同步或数学精度证明。

完整无sample shape CPU编译 `cpu_v1` 通过，58个动态轴、130根Tensor、
一个MIX和sync_start保持。三个代表任务已提交，见 `SUBMISSIONS_V1.json`：
128K/B16、128K/B28、8K/B32，先查完整功能/精度，再看同轮五窗L1；尚不
预记通过或收益。已有协同前段的七档、S1/S5、metadata结果不作为本候选的新验收。

## 完成与采用

七档同进程五窗均改善，完整精确、Top-K/score、cache/state、图、保护与代际通过。结果见 `SEVEN_RESULTS_V1.md/json`；相对旧动态降低0.95%～6.39%，相对原多task降低30%的目标仍0/7。S1/S5对旧动态精确通过，另完成六次metadata和四次padding同图回放，不代表全部mixed-query/空rank/连续轨迹或Native/token验收。十个任务均exit0，终态写回各submission登记。

正式代码采用冻结候选；已按Git暂存版本逐字核对五个改动文件，见ADOPTION_AUDIT。工作树随后开始前置几何候选，不把它混入本轮成绩。

## 新静态配对诊断

`task_20261018_174436_80108314392` exit0，完整数值、图和保护通过。动态/静态端点中位：Score AIV518.58/522.02，Top-K534.78/528.78，Attention722.46/733.26，OB840.30/846.70 μs。新的Attention端点已不再晚于静态，不能继续沿用旧轮的35μs差值；两侧插桩数仍不同，不用于L1验收。

静态计划共16成员，只有12个发布，原通用覆盖检查仍正确标记全成员端点不完整。本轮进一步核对：全部计划业务写入都位于range(block*8,96,128)，未发布的block12～15没有数据工作。因此12个通知覆盖全部实际写者，见STATIC_PLAN_PAYLOAD_COVERAGE.json；该数据就绪端点动态/静态548.70/536.36μs，不能称为所有成员退出时刻。

下一候选只前移不依赖Top-K的请求几何和压缩页判定，并保留所有候选检查。原metadata/padding检查对下一候选不自动有效。
