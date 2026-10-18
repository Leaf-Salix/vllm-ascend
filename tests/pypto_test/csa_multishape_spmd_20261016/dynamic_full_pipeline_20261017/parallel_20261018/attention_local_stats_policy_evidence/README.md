# Attention本地统计区间策略：CPU通过，尚未采纳

基线25de3f32；无本目录的NPU结果。七档全域候选的结果在
`../attention_local_stats_evidence/`，不能冒充本区间版已验证。

候选沿既有Score历史/G与O_A T96分界选择本地统计：long G6且T<=96，
short G2且T>96，其余保持GM。对应四个已有改善点；计划补128K/B15/T90与
8K/B17/T102边界，当前尚未提交这两个设备任务。

## CPU与未解决项

v1五种策略完整CPU通过。开启时实际PTOAS可执行C++（只忽略整行注释）与
已测全域版本相同；关闭时仍有768B死统计环在loop carry/if-phi中相互引用，
改变其它UB窗口放置。因此不能称“constexpr=False即零额外成本”。
完整CODEGEN_EQUIVALENCE.json保留失败比较，两个diff没有删除。

v2令关闭路径直接传递原iter_arg，断开内层缓存phi；long-on、long-off、
short-off三次完整CPU仍通过，但纯alias传递本身也未消除，关闭代码仍不同，
开启路径也因新增alias而有变化。此表达未上卡、未采用，不反复测噪声掩盖问题。
两版完整冻结source_candidate_v1/v2及编译源留存。

## 上游登记

在主线14736f65的派生树72cadb1e5运行最小CPU复现，最终分配仍包含
`ghost[6,32] FP32`、地址128、768B；它只被if/loop传递，没有数据、条件、
地址、store或返回值消费者。相关Simplify与dead_code_elimination源未本地改动。
已提PyPTO #3037（性能/资源优化问题，不称数值bug）：
https://github.com/hw-native-sys/pypto/issues/3037

issue含完整复现、实际分配、以语义live roots求可达性的建议，以及TaskId、
while条件、attrs/deps、调用副作用和真实循环数据的保留要求。无PR、无标签。
尚未实现组件修复，不能把源码分析称为已经解决。

## 当前正式状态

本目录所有尝试保留；正式四个临时改动恢复到已验证25de3f32，避免默认启用
尚未验证成本的区间版本。没有丢弃已测四档收益，后续修复死状态消除后再按
区间接回。目标仍是完整动态单SPMD七档各比原多task降低至少30%，目前0/7。
