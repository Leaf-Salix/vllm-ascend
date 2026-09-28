# 长档缩放分数按2048段驻留UB

基线4ffccb7b，冻结私有整包`pkg:dsv4_csa_score_segment_ub_4ffccb7b`。
只改变长档S6/balance_leaves路径的缩放分数保存位置，保留Cube算术、query分组、
半leaf划分、scale分页读取、Top-K顺序、最终pair布局和任务边界。
短档和长档双query维持原路径，不改Native cache、精度版或调度开关。

## Native依据与历史区别

参考ops-transformer28f40354的
`attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_vector_arch22.h`
ProcessVec1：Cube结果经GM送Vector后，key scale乘法、SortAll与累计根都留UB，
最终才发布根。当前PTO此前仍将缩放结果写score_arena，再读回排序。
pypto-lib2164563也保留该GM中转；本次不直接照抄其输入布局或旧725μs泳道结论。

已复核验证日志§182/§236：旧query分工并融合最终发布受ABI/scope阻塞，
旧完整半leaf驻留则实际变慢。后者用2×4224物理tile、大形状加载/乘法及UB gather，
没有当前长档1024步长和2048流式排序。这次不原样重试：
每AIV使用[6×4,512] FP32，48KiB仅缓存一个2048段；
每query的512分数写入等宽UB行，四行连续，reshape为[6,2048]后直接TEXTRACT排序。
不用大物理形状加载/乘法，也不使用UB gather；已有24KiB累计根继续复用。
未采用按query划分AIV，因为那还涉及重复scale读取和根布局，应独立评估。

完整四步或最后不足四步时排序。最后一段有效长度同时截断到query可见范围、
当前AIV半leaf末尾及2048；初始根为负无穷，未写的尾区在排序前按有效形状填充，
不能让下一个半leaf或旧UB内容参与Top-K。512/1024/2048排序分支和尾段优先合并规则不变。

## 编译与生成码

初次CPU检查暴露短档死分支中的reshape大小不匹配；改为固定512物理行宽。
后续SSA检查发现条件内创建缓冲却在外层循环携带，将创建提升到条件外，
短档未使用的缓冲由死代码消除。不改PyPTO、PTOAS或PTO-ISA，两个问题均未上卡。
当前两根依赖图与完整候选编译/load通过。

以下为生成C++中各自有效CUBE/VEC预处理区间的静态调用点，不能当作每次执行指令数。
同一源码文件含两种核的函数，未将它们混算：

| 长档S6 AIV调用点 | 基线 | 候选 |
| --- | ---: | ---: |
| TLOAD | 74 | 2 |
| TSTORE | 60 | 6 |
| TMOV | 18 | 24 |
| TEXTRACT | 48 | 54 |
| TGATHER | 0 | 0 |
| TSORT32 / TMRGSORT | 72 / 222 | 18 / 54 |

新TLOAD只剩scale及Cube结果输入，TSTORE只剩六个query根。
排序静态调用点减少主要来自统一段循环，实际候选数量与排序规则不变，
不能据此宣称排序算力开销按该比例下降。新增UB搬运和提取需由实测评估。
所有AIC及其他三个AIV特化的上述调用点数量不变。

## 单卡对照

auto任务task_20260929_060528_18437739541，128K/B16与8K/B24。
两侧CANN9.2/mode2/atomic0/det0、正式layer4权重和独立合成历史，EPLB关闭，
ring=[256,128,256,32]MiB/task_window4096。长基线→候选、短候选→基线，
每侧5预热/20次无profiler真实编译计时，另采四个level-4窗口。
复用八类完整状态、图/eager、Top-K和保护区检查；预期本次数据移动改动零容差一致。
不扩整模型或七档，不以指令计数代替核时/CSA/P95证据。
候选尚未合入，结果待收集。

[候选补丁](candidate.patch)、[生成调用点](lowering.json)、[冻结来源](source.txt)、
[编译入口](compile.py)、[设备入口](run.sh)、[收集入口](collect.py)。
