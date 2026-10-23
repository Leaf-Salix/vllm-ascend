# TND-SPMD CSA 数值差异验证与条件性修改建议

日期：2026-10-24。依据 [数值策略差异审计](CSA_SPMD_NUMERICAL_DIFFERENCES.md)，
文档提交7824ba4bb；算子快照仍为df57259b8，当前采用output_rectangle_production_v1。
本文中的N/S/R编号对应原审计，M编号是验证项目及其条件性修改方案。

**我的建议：先验证真实误差，再决定是否修改。数值策略差异不等于精度缺陷，
对齐Native也不等于数学上更准确。本文没有新增确认当前版本的精度缺陷。**

优先验证Attention概率舍入、输出投影量化、Indexer/QKV中间舍入、
QR amax和Compressor池化；HC二分累加也作为对照候选。
保留FP32、分组量化和在线softmax都可能是合理策略，不因与Native不同就替换。

这是建议清单，不是已采用的改动或新Goal。当前只新增本文，没有改算子、
没有调整验收阈值、没有运行新的CPU数值实验或NPU任务，性能优化仍暂停。

## 1. 判定原则：先区分策略差异、兼容性与精度缺陷

建议先判断当前实现是否违反明确的数学/量化合同，或者造成超过既定要求的误差、
数值不稳定或错误选择。需要固定相同输入、前态和权重，隔离具体因素后才能归因。
历史存在对Native超出容差的记录，但本轮尚未将当前版本的各项差异逐一证明为其根因。

Native是实际实现的兼容性参考，不是数学真值。建议分别报告：

- 对同一数学定义、同一输入构造的高精度参考误差。
- 对性能测试实际使用的Native输出的差异及既有兼容性门槛。
- 对冻结旧SPMD的变化，以及状态、语义和保护区是否出现新增问题。

局部高精度参考须保留该实验约定的输入dtype和量化定义；比较不同量化算法时，
另用共同的目标数学计算或模型级指标评价，不能选择对某一方有利的参考。
CPU高精度计算用于评价误差，不声称复刻Cube的全部有限精度行为。

增加BF16舍入只是复现Native行为的兼容性选项，可能比保留FP32损失更多信息。
分组量化也可能降低量化误差。只有定位出不满足要求的问题，或者存在明确的
Native数值兼容需求，才据此选择修改；不能将“已确认存在差异”写成“已确认存在缺陷”。

下列优先级是**验证优先级，不是生产修改优先级**，依据历史线索、可隔离性和传播范围，
不是对当前八档误差贡献的排序。验证前保留现有实现和原验收规则。
如果后续决定有意改变算法，应单列旧SPMD差异；未修改的独立部分、写入范围、
metadata和保护区仍须保持原有约束。本建议不自动改变原性能Goal的验收口径。

## 2. 验证清单与条件性修改方案

| 编号 | 建议 | 处理意见 | 对应审计 |
| --- | --- | --- | --- |
| M01 | 完善精度报告与相同中间输入的归因入口 | 先做；复用现有诊断，避免另写一套算术 | 全部；尤其旧结果解释边界 |
| M02 | 比较Attention概率的rint与round | 优先验证；两种合法舍入规则的差异不直接定性为缺陷 | N30 |
| M03 | 比较O_A舍入、分组/整token量化、整数/浮点合并 | 优先验证；Native路径作对照，分组量化不预判为更差 | N33–N35 |
| M04 | 比较Indexer各处FP32保留与Native的BF16舍入 | 优先验证；添加BF16是兼容性对照，不是默认修复 | N23、N24、N26 |
| M05 | 比较主Q/KV及Attention逆RoPE前的中间舍入 | 优先验证；保留FP32本身不是精度缺陷 | N09、N14、N16、N18、N32 |
| M06 | 检查QR的amax变形公式与实际归一化结果的差异 | 优先验证内部一致性；尚未证明造成异常量化 | N11；关联N10/N12 |
| M07 | 比较Compressor在线/八行池化及乘倒数/直接除法 | 优先验证；已有历史归因，但未证明当前在线算法有缺陷 | N21、N22、S06 |
| M08 | 比较HC四份partial左折叠与二分，以及其它独立归约 | 先验证再决定；不整体照搬独立HC_pre | N01、N03、N05、N06 |
| M09 | 比较Attention的128/512 softmax及sink更新 | 固定Q/KV后评价真实误差与结构成本 | N28、N29、N31 |
| M10 | 核对归约、sqrt/div、scale与反量化的实际分支 | 定位误差或明确兼容需求后，才选择对应修改 | N07、N10、N12、N13、N15、N19、N25 |
| M11 | 核对矩阵乘K顺序与HF32模式 | 不以复制Native为默认目标，只追已隔离出的剩余误差 | N02、N08、N17、N20、N27、R07 |
| M12 | TopK、mask、状态、同步、别名与边界防护 | 补合同/定向验证；发现具体反例再修实现 | S01–S08、R01–R08 |

## 3. 第一批：优先验证的差异及对照方案

### M01：先完善精度报告和归因入口

现有 [run_probe.py](../../../tests/pypto_test/csa_spmd_tnd_20261020/run_probe.py)
已经有native_comparison、native_frozen_output、阶段采集和原SPMD归属对照。
建议在其基础上整理可复用的报告，而不是重新实现数学参考或删除旧FAIL。

需要明确呈现：

1. 当前候选对实际Native：原输出容差是否通过，超差数量/位置、max_abs、RMSE；
   TopK集合与顺序分别统计；cache/state逐项保留。
2. 当前候选对冻结旧SPMD：哪些变化由本项修改有意产生；未修改独立边界是否一致。
3. 相同中间输入的局部对照：例如同一Q/KV/indices、同一O_A输入、
   同一8行Compressor窗口，避免把上游误差归到当前阶段。
4. 适合CPU分析的局部计算再对照FP64或更高精度参考；明确它不模拟Cube的全部行为。

诊断代码应采集或复用真实生产helper。增加采集后，先验证诊断版最终结果与未插桩版一致。
对比点应选在双方语义等价的接口：例如QLI系数须在双方实际转FP16后比较，
不能将SPMD的FP32暂存与Native的FP16入参直接判成“算错”。

**完成条件**：一份报告能区分“真实误差是否改善”“是否更接近Native”
和“是否出现新的状态/语义问题”；不能只有一个笼统PASS。
本项不自行修改任何既有精度阈值。

### M02：验证Attention概率舍入的影响

- **对照位置**：[attention_stream_stages.py](deepseek_v4_flash_csa_spmd_tnd/attention_stream_stages.py)
  的qk_probability，当前mode="rint"。
- **对照方案**：保持rint作为现有路径，另以mode="round"复现A3 Native概率发布处的
  CAST_ROUND；最终Attention输出、inverse RoPE输出和HC_post仍保留rint。
- **为什么优先**：差异明确、改动局部，原CSA记录§71已有固定输入归因；
  不是仅依据API名字推测。历史0.595703125正好能区分两种中点规则。
- **如何判断**：先固定qk_exp比较两种舍入误差；再固定Q/KV/indices对照同一高精度
  Attention参考和Native，最后进入完整链。记录超差坐标变化，不能仅看全局RMSE。
- **代价与限制**：预计实现改动小，不预先保证性能无变化。
  当前128局部softmax的exp本来就可能与Native不同，因此同一cast模式不代表整段已经对齐。
  rint不因不同于CAST_ROUND就成为缺陷；历史Native对照改善也不证明round数学上更准。

### M03：验证输出投影的量化定义与舍入边界

**这是值得优先隔离的算法差异，不是已确认的量化缺陷。**
当前每组1024列独立量化，Native对每token的8192列统一量化。
分组尺度可能降低量化误差；仅凭Native采用整行尺度，没有理由直接替换现有算法。

对照入口为
[output_projection_a_stages.py](deepseek_v4_flash_csa_spmd_tnd/output_projection_a_stages.py)
和
[output_projection_b_stages.py](deepseek_v4_flash_csa_spmd_tnd/output_projection_b_stages.py)。
建议保留当前路径，按三个可归因子步骤构造Native兼容性对照：

| 子步骤 | 对照方案 | 需要保留的条件 |
| --- | --- | --- |
| M03a O_A边界 | O_A的FP32结果先按Native落BF16；量化计算需要FP32时再无损加宽。 | 可在UB表达，不要求增加GM往返或外部kernel。保留实际有效行和尾块边界。 |
| M03b 整token量化 | 合并8组的amax，让同一token的8192列使用同一量化乘数/反量化scale。 | 合并结果在量化前真正就绪；不能各组仍用旧scale却仅修改最终归约。零/极小行规则先核对Native，不能简单删除floor。 |
| M03c 整数合并 | 在已经统一token scale的对照分支中做O_B；可以保留K分片，各片先在INT32合并，再转FP32和反量化，最后BF16进入HC_post。 | 不同组仍使用不同scale时，不能直接合并原始INT32。先证明整数范围及唯一写者；channel/token scale结合顺序按实际分支核实，不用atomic浮点累加。 |

K8192最坏整数绝对和为134217728，当前INT8输入界下不溢出INT32；
但先将各片转成FP32与整数合并后再转，不保证等价。

已有可参考实现是本仓HCA的
[NATIVE_OUTPUT_PRECISION.md](../../../tests/pypto_test/hca_tnd_mpmd_20261022/NATIVE_OUTPUT_PRECISION.md)
及其candidates/output_native.py。它证明另一条链曾通过相同修改消除指定超差，
**不证明当前CSA也会获得相同误差改善或性能结果**。只借鉴算术和证据方法，
不复制HCA任务调度，也不额外引入PyTorch/Native运行时调用。

**验证重点**：同一Attention输出和同一post/comb下，隔离比较O_A、INT8/scale、
O_B及最终输出；评价两种量化方案对共同目标计算的误差，再看Native差异和完整八档。
只有实际误差/模型指标或明确兼容需求支持时，才选择替换。量化尺度合并会引入组间依赖，
要评估它对现有OUTPUT_GROUP_PIPELINE的影响，不能沿用旧的逐组提前消费条件。

### M04：验证Indexer保留FP32与BF16舍入的影响

这一组位于离散TopK之前，应拆成独立对照。下表是复现Native边界的表达式，
不表示当前保留FP32有错，也不建议未经证据统一降为BF16：

| 子项 | 对照位置 | Native兼容性对照表达式 |
| --- | --- | --- |
| M04a Query投影→RoPE | [indexer_query_stages.py](deepseek_v4_flash_csa_spmd_tnd/indexer_query_stages.py)，indexer_query_dequant_rope_stage | INT32反量化后先BF16，再扩FP32做RoPE；反量化scale结合顺序另按M10核实。 |
| M04b Query Hadamard | 同文件，indexer_query_quant_stage | Hadamard Acc先BF16，缩放后再按Native落BF16，再作为量化输入。 |
| M04c Key Hadamard | [indexer_cache_stages.py](deepseek_v4_flash_csa_spmd_tnd/indexer_cache_stages.py)，indexer_compact_quant_cache_stage | 当前已经有Hadamard结果的BF16往返，只补评估缩放后的BF16边界；不要重复添加前一处cast。 |
| M04d head weights | indexer_query_stages.py，indexer_weights_reduce_stage | 投影结果先BF16，乘模型scale后再BF16，之后保留QLI入口FP16转换。四份partial的累加树暂不同时改。 |

**验证理由**：这些边界差异由实际Native源码和C04 trace支持。
原§52还证明head系数的很小变化可能改变排序；不能只看最终输出误差很小就忽略。

**验证重点**：同一QR INT8/scale、同一RoPE输入、同一weights输入下逐边界比较，
同时评价局部高精度参考误差，再检查最终INT8 query、FP16系数、key/scale、score及TopK。
先解释分数变化，再判断同分规则；保留请求内所有有效token及尾行。

### M05：验证主Q/KV与Attention逆RoPE前的中间舍入

保留FP32可以减少中间舍入，不能称作“缺失BF16导致的精度缺陷”。
仅在归因实验或明确Native兼容需求下，考虑下列BF16对照分支；
各边界单独处理，不一次性加完所有cast后只报告一个最终误差：

| 子项 | 位置 | Native兼容性对照边界 |
| --- | --- | --- |
| M05a | [qkv_stages.py](deepseek_v4_flash_csa_spmd_tnd/qkv_stages.py)，q_r_stage输入 | Q_A结果进入QR RMS/量化前先BF16。 |
| M05b | [query_projection_stages.py](deepseek_v4_flash_csa_spmd_tnd/query_projection_stages.py)，q_dequant_rope_stage | Q_B反量化后先BF16，再做每头RMS；RMS结果先BF16，再做RoPE。两处拆开验证。 |
| M05c | [kv_stages.py](deepseek_v4_flash_csa_spmd_tnd/kv_stages.py)，kv_norm_rope_stage | 原始KV投影先BF16再RMS；RMS结果先BF16再RoPE。两处拆开验证。 |
| M05d | [attention_stream_stages.py](deepseek_v4_flash_csa_spmd_tnd/attention_stream_stages.py)，最终publish | inverse RoPE消费BF16 Attention输出扩展出的FP32，而不是直接消费未舍入的n_full。NoPE已落BF16，不重复改。 |

**对照实现原则**：在既有incore内表达舍入，保留当前存储和分核结构；
不要为实现边界增加独立task，也不要统一将所有中间量都改成BF16。
融合Compressor的norm→RoPE本来是FP32，不属于M05c的原始KV路径。

**验证重点**：由相同阶段输入评价每个边界的真实误差与Native差异；Q_A边界改变会传到主Q与Indexer，
SWA新写入值改变会影响后续decode，必须看cache/state连续性。
当前input RMS前的mixed和HC_post输入都已有BF16边界，维持现状。

### M06：检查QR的amax变形是否造成不可接受的误差

当前q_r_stage用
inv_rms × max(abs(x×gamma))估计归一化后的amax，
实际量化输入却按(x×inv_rms)×gamma计算。实数等价不保证FP32逐bit等价。

建议固定同一Q_A输入后，比对变形公式与从实际归一化结果取max的差别。
对照候选可从实际按原顺序算出的归一化值取amax，
让第二遍量化消费相同值，或严格复算相同表达式。
M05a用于区分输入舍入的影响，不要求先把生产Q_A改为BF16。
不在同一候选里同时更换归约树、sqrt、scale公式及1e−4 floor。

**验证理由**：这涉及量化器内部算术一致性，值得优先核实，
但目前只确认了FP32表达式不同，未证明导致溢出、异常量化或超出既定要求的误差。
只有固定输入的反例和传播证据成立，才可将调整称为修复；
没有反例时，不以两个amax存在末位差为由自动替换。

**代价与验收**：可能增加归一化后的max归约或保留UB值的开销。
固定Q_A、gamma、rstd，检查实际amax、极值量化结果、INT8和scale，
再核对完整QR与TopK，不能通过额外clamp掩盖异常。

### M07：比较Compressor在线池化、八行池化与除法位置

对照入口：
[compressor_pool_stages.py](deepseek_v4_flash_csa_spmd_tnd/compressor_pool_stages.py)、
[compressor_output_stages.py](deepseek_v4_flash_csa_spmd_tnd/compressor_output_stages.py)、
[indexer_compressor_output_stages.py](deepseek_v4_flash_csa_spmd_tnd/indexer_compressor_output_stages.py)。

在线softmax是合理算法，数值顺序不同不等于缺陷。
建议分成两类对照，主/Indexer各自验证：

1. **八行池化**：按Native窗口顺序取得value/score，统一max，
   Exp后按Native树形归约总和；先归一化概率，再乘value、树形求和。
   与当前“最新行起步、逐行在线更新、最后除分母”路径比较。
2. **RMS除法**：保留当前已对齐的平方折叠树，
   比较x×(1/sqrt)与直接x/sqrt后乘gamma的误差。

原CSA记录§75有固定窗口的Native差异归因，支持优先验证这组策略。
它不证明当前在线算法数学上更差；当前版本、正式权重与旧实验不同，
仍须对相同窗口的高精度参考和实际Native分别评价，结果满足修改目标后再决定采用。

**必须保留**：Native state布局、ape加入位置、TND窗口覆盖、compact输出顺序、
旧state全部读完再commit、无关闭窗口时的行为。
不要把池化顺序修改变成状态调度改造，也不要恢复旧的额外GM窗口适配。

**代价与验收**：八行Tile增加UB需求，可能影响流水；先查实际内存规划。
先固定同一旧state/当前投影比较pooled与compact KV，再从共同初态连续执行，
同时检查完整FP32 state与未写区域，不能用“换成Native前态后通过”替代连续验证。

## 4. 第二批：按剩余误差决定是否深入的验证

### M08：HC二分累加值得验证，但不建议直接移植为默认

建议首先只做四份投影partial的
((p0+p1)+p2)+p3 → (p0+p1)+(p2+p3)单变量候选，
保持四个K4096分片、片内K顺序、RMS、sigmoid与Sinkhorn不变。

需要分别回答：

- 给定相同FP32 partial，对高精度和更接近了吗？
- 给定相同hidden/weights，完整投影对高精度点积更接近了吗？
- 对实际Native的mixed/post/comb、完整输出和TopK有什么变化？
- 是否只是某个shape或随机种子改善，是否出现相消/量级悬殊反例？

**我的取舍**：可以做候选，当前证据不足以列为必改项。
独立HC提交的构造样例中两种顺序各有更准确的情况；
Native自身也是顺序合并，而且分片数不同。
不能把“换成二分”当作Native精度对齐，也不能仅据一组max_abs决定采用。

HC的其它归约分别处理：N03是32段平方和及Cube/Vector策略；
N05是Sinkhorn列归约；N06的四路mix当前已经是pairwise。
不要以“一律二分”同时修改这些不同的数学边界。
更不建议未经HF32/K顺序隔离就为贴近Native重写整个HC前段。

### M09：Attention的512累计softmax是重要候选，但结构成本更高

在M02/M05d的归因完成后，若同一Q/KV/indices下仍有未解释的误差，
建议比较以下三个独立因素：softmax逻辑块128→512、
局部max→累计max、sink从最终补分母→从初始sum=1参与更新。

若构造Native策略的对照，可以继续使用128候选的Cube搬运/计算tile，
但必须在**同一个512候选逻辑softmax和统一max/sum边界**下生成概率。
不能仅在循环外合并4份已经各自舍入过的BF16概率/PV，就宣称等价于512策略。

该改法会触及probability/PV交接、UB占用和并发关系。
需检查全部存活Tile、环槽与事件，并保留可能降低并发度的实测结果。
目前不建议直接把ATTN_K_TILE改成512，或凭历史精度版源码跳过当前单SPMD的资源验证。

### M10：核实归约、Scalar除法与scale结合顺序的误差

确认Native分支只解决“实际比较了什么”，不自动产生修改理由。
下表中的方案以已定位的误差超出要求，或明确需要Native数值兼容为前提。

| 项目 | 修改前必须取得的证据 | 条件成立后的建议 |
| --- | --- | --- |
| QR平方和 / rstd（N10/N12） | 相同Q_A输入下的sum、sqrt、rstd；区分归约误差和Scalar/Vector差异。 | 仅对已定位的步骤采用Native对应归约或设备Scalar路径；不先移植旧整套sqrt补偿。 |
| input/KV RMS（N07/N19） | 当前CANN9.2各档实际tiling，而非旧CANN9.0或另一模型的模板。 | 已对齐的64列lane树保持；只改实际不一致且影响结果的分支。 |
| 主Q RMS（N15） | 本次实际triton_rms_kernel的中间dtype、归约与指令路径。 | 按证据对齐，不能仅从tl.rsqrt名字推出需要更换PTO算法。 |
| DynamicQuant scale（N25） | 实际选中amax/127还是1/(127/amax)等路径；零/极小行的行为。 | 逐路径对齐；不得把一个模板的公式批量替换到所有量化点。 |
| Q_B/O_B两尺度结合（N13/N35） | 当前实际QuantBatchMatmul采用先channel后token，还是先合并scale。 | 在同一整数Acc/scale上验证对应次序，再修改相应反量化helper。 |
| floor与特殊值（N12/N25/S08） | 零行、极小行、NaN/Inf/subnormal是否在支持合同内，以及Native实际规则。 | 明确后单独修改和验证；不要把删除floor或统一clamp当作常规精度提升。 |

### M11：矩阵乘K顺序与HF32最后按剩余误差处理

对Q_A、KV、两路Compressor、head weights、QLI WS，
建议先固定输入并完成舍入/量化边界归因，
只有剩余差异明确来自Acc时再追具体Mmad序列。

理由是Native的M/N/NZ和shape分档可能改变遍历顺序。
复制旧B40/ND的K公式到当前八档NZ2，可能既不能对齐又破坏性能。
HC的HF32更不能因为Native启用了就直接开启：
它涉及实际乘法精度和模式生命周期，可能与“更接近高精度参考”的目标相冲突。

同理，QLI拼接query的加零/分段WS已有自身回归证据；
在没有相同FP16输入下的差异之前，不建议改回逐query M16/K64或重新引入多task。

### M12：TopK及正确性风险先补合同和反例，不重写已验证协议

- **TopK**：先将评分差异与同分规则拆开；固定score检查集合、顺序和第512名并列带。
  不建议先关掉精确阈值筛选，也不把ge改为gt。若业务要求与Native的tie规则一致，
  先确认其实际规则及可依赖范围，再改排序/根合并顺序。
- **mask/sentinel**：分别验证普通有限输入、全masked/空请求及非法metadata合同。
  不将所有−FLT_MAX统一替换为−∞；某些现有算式对非有限值不等价。
- **别名**：在准备/绑定阶段确认哪些根Tensor必须不重叠。
  若这些禁止重叠的区间目前未检查，可增加Host描述符级拒绝；
  不将合法共享cache页或只读权重一律禁止，不在forward热路径回读设备。
- **代际/并发**：明确同一workspace串行使用和generation上界；
  如支持域内能复现旧代满足新等待，应修协议，不能作为数值容差问题处理。
- **scale页、state及环槽**：保留已验证的单写者、512B信号槽、
  64B scale读改写归属、pool读完再commit和RAW/WAR依赖。
  有具体反例再修，不因“可能影响精度”批量增加全核屏障或重写调度。

## 5. 我建议保持不变的部分

| 部分 | 保持理由 |
| --- | --- |
| HC sigmoid的recip写法（N04） | 当前A3实际也是填1后VDIV，没有仅由名称证明的算法差异。 |
| Q完整/尾块的两种求逆平方根写法（N15的一部分） | 当前带tmp rsqrt和recip(sqrt)均展开为VSQRT+VDIV；不为统一写法改数学。 |
| HC_post本体 | 当前A3乘加顺序及最终rint已有源码和同输入历史证据；不照搬A5 MulAddDst。 |
| FP32中间值、分组量化、在线softmax | 未证明有缺陷，验证前继续保留；不能仅为更接近Native而默认替换。 |
| QLI已经存在的低精度边界 | QK/1024→FP16+ReLU、FP16系数、FP16 key scale均保留；先定位它们之前的输入差异是否造成问题。 |
| 已对齐的RMS折叠树 | 原始KV的lane累加和Compressor的折半树不重复“修复”；Compressor的x/root位置另行验证。 |
| 实际加载的权重dtype、cos/sin和cache布局 | 不用checkpoint中更高dtype覆盖Native加载后的参数，不重算RoPE表，不增加布局转换掩盖问题。 |
| 固定K、非atomic的Q_A/KV | 不为恢复旧性能改回浮点atomic累加；显式partial并不等于atomic。 |
| SDK及流水协议 | 无新复现证据不改PyPTO/PTOAS/Simpler；也不把A3 high_precision=True当作必然提高TDIV精度。 |

## 6. 推荐验证顺序和修改决策条件

建议按以下顺序推进；这不是本轮已经启动的任务：

1. **M01建立清晰报告**，冻结当前采用版、实际Native、输入与前态。
2. **M02概率舍入对照**；**M03输出尾链对照**独立做，先不叠加。
   这两处分别有原CSA直接证据和同类尾链证据，定位边界也清晰。
3. **M04、M05逐边界验证**；先固定相同局部输入，再看全链传播。
   固定M05a的输入条件后推进M06，避免将Q_A边界和QR amax的影响混算。
4. **M07验证会进入cache/state的数值路径**，随后覆盖连续轨迹。
5. 对尚未解释的差异，再按**M08–M11**做归约、softmax结构、实际tiling与指令级诊断；
   M12的保护/语义检查贯穿各阶段。

决定修改时，先明确理由属于“违反既定数学/量化合同”“真实误差不满足要求”
还是“满足明确的Native数值兼容需求”。只有第三种理由的改动应标为兼容性调整，
不能称为数学精度修复；高精度参考更差、仅Native差异更小的结果不能混写成全面改善。
历史只读差异和理论风险都不足以直接判定当前缺陷。

每个候选的决策证据建议至少包含：

- 局部同输入因果对照与完整链对照分别成立；没有替换golden或隐藏原FAIL。
- 对同一目标数学计算的高精度参考报告误差，并检查变化是否满足预先明确的要求；
  涉及不同量化算法时补相应模型级证据。不能用同一个实现自作golden。
- 对实际Native沿用原输出0.01/0.01门槛逐项报告，保留旧FAIL；
  旧超差、剩余超差和新超差分别展示。它是既有Native兼容性门槛，
  不是数学真值，也不推广到所有中间张量或自动转化为新的修改目标。
- TopK不能只检查合法/无重复：保留分数、集合、顺序及并列边界；
  不自动将不一致改成可接受。cache/state对Native的差异沿用各自原检查门槛，
  有意修改的有效写入值不再要求与旧SPMD逐bit一致；写入范围、
  未写区域、只读metadata和保护区仍必须满足原严格合同。
- 有意变更的数值不再以旧SPMD逐bit作为唯一门槛；
  无关分支/不受影响张量仍需证明无新增变化。
- 覆盖[性能规范](CSA_PERFORMANCE_TEST_SPEC.md)八档，特别是B1/T6尾块、
  C05 joint/O_A矩形分档、C07短历史、C08混长；受影响动态边界和连续状态另验。
- 精度阶段通过后才报告性能取舍：同公共SDK、完整L1首尾、至少10个有效窗口、
  中位数并保留全部样本。能复用的未变实现历史数据继续复用；
  不重新定义原性能分母，不把局部算子更快当作整链收益。

本建议没有设定新的性能回退额度、真实误差或整模型通过阈值。
本文列的是待验证差异和条件性方案，不是已确认缺陷清单；
在证据支持之前，FP32中间值、分组量化和在线归约继续作为现有合理实现保留。

未来实施仍在本会话worktree隔离候选，使用公共manifest和bootstrap；
NPU只通过task-submit，SDK升级不原地干扰排队/运行任务。
本文本地提交，不push；当前生产实现和暂停状态均保持。
