# 长档S6按query分配AIV

基线为生产算子4ffccb7b；私有整包`pkg:dsv4_csa_score_query_split_4ffccb7b`。
本候选只处理长档S6的Score Vector分工，未合入生产，没有提前声明设备收益。

## 最新AscendC依据与本次变量

参考ops-transformer 28f40354的
`attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_vector_arch22.h`
中ProcessVec1（328行起）：两个AIV按S1/query分工，每个query处理完整S2段。
这不是声称当前release Native二进制已由该源码重建。

当前PTO两侧AIV各处理六个query的一个候选半区；候选改为各处理三个query的两个半区。
每轮同样处理6×512=3×1024个FP32分数，将六次512列逐行TMUL改为三次1024列TMUL，
一次搬入的Score元素总数不变。Cube原QK/WS、Key读取和两半区交接顺序保持。
代价是每个AIV都读取两个候选半区的scale，分页读与scale转换量翻倍，必须实测判断。

为单独衡量分工收益，两个half根/query、原2048提前排序与UB根、跨leaf归并、
Top-K同分顺序和根输出ABI保持。没有把idx_topk发布并入Score，
因此不是重试验证日志§182因Out/InOut及scope别名失败的旧融合发布版本。
后续是否减少根数须先看此候选的Score核内、merge和完整CSA证据，不先叠加第二个变量。

pypto-lib 2164563的dspark路径按单query×leaf执行，无法直接套用本接入的S6 query划分。
本接入保留六query复用Key和第二次Cube加权规约，并直接读取Native交错物理cache；
没有恢复入口历史复制、源头分离或外部写回。短档与长档小query路径保持原分工。

## 编译与验证范围

两个私有副本从当前六档的冻结公共源码复制；整包、公共依赖、设备runner均独立。
两侧的两根_get_dep_graph解析通过；候选PTOAS/CCE及KernelArtifact.load的CPU检查已通过。
首轮发现常量分支两侧复用不同shape变量名、slice后缩小validshape的表达限制，
已分别改为独立局部名及slice自身的动态valid_shape；没有修改工具链或关闭ABI检查。

候选长S6生成的AIV代码确认TMUL调用点为3；这些静态数量不作为性能结论。
当前六档任务完成后提交auto单卡：128K/B16与8K/B24，CANN9.2/mode2/atomic0/det0，
5预热/20次无profiler真实编译图计时，各四个独立level-4窗口。
两侧完整八类状态跨版本零容差、图/eager、Top-K结构及保护区检查；不新增B40，
不为此局部候选启动六档或16卡。idx_topk_scores未列入八类状态，不宣称其独立验证。

核内实际收益、CSA 8:2变化及P95/max分别报告；短档源路径相同也保留实测回退。
没有核内收益不扩大测试，长档明显收益/短档明显代价时再在同一算子内按场景处理。
正式设备任务尚未提交；当前六档原任务继续执行，不修改其冻结源码。

[候选补丁](candidate.patch)、[冻结来源](source.txt)、[CPU编译](compile.py)、
[编译结果](compile_candidate.json)、[生成代码计数](lowering.json)、
[设备入口](run.sh)、[采样入口](run_side.sh)、[复用的状态/官方泳道收集器](collect.py)。
