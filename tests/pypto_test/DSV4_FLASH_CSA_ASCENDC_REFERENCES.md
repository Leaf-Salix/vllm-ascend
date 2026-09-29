# CSA核内优化：本地AscendC源码参考

更新：2026-09-29。按用户修正后的目标，最新AscendC实现是后续核内优化的主要依据：
首先研究ops-transformer，结合ops-nn、ops-math等ops仓库；当前Native作性能/行为对照，pypto-lib作PTO实现参考。
以下是本地已读取版本，不表示当前CANN二进制已包含这些实现，也不表示这些实现已测得比PTO快。

| 本地仓库 | 本次读取HEAD | CSA相关入口 |
| --- | --- | --- |
| [ops-transformer](../../../ops-transformer) | 28f40354 | QLI/QLI V2、SparseFlashMla、mHC |
| [ops-nn](../../../ops-nn) | 19614968 | RMSNorm、动态量化及融合路径 |
| [ops-math](../../../ops-math) | 361722c0 | 排序、Top-K及基础向量操作 |

2026-09-28 20:48 CST核对官方GitCode远端master，三仓均以`fetch --depth=1`更新源码参考，
保留原分支并切换到上述提交；提交时间分别为20:43、20:28、20:03。
没有安装这些仓库、升级CANN或改Native流程，实测Native仍是本阶段固定二进制。
后续优化先查这些最新AscendC实现，再查pypto-lib的PTO表达；每项记录具体文件、版本、A3适用性、
采用策略及保留差异的原因，不能将新源码推导直接称作Native新版本的实测性能。

本次与上午版本b5b33e14/7a71d54e/81802185的定向源码差异：

- QLI、QLI V2的A3核函数、QLI V2 AICPU metadata，以及SparseFlashMla arch22核函数没有变化；
  当前均衡leaf候选所依据的2048粒度、cost分配、UB累计Top-K仍有效。
- QLI V2有较大的arch35内核与host重构，不能把arch35策略视为A3现成能力；README明确列出A3支持，
  具体策略仍须追到对应构建入口及指令。
- MhcPreSinkhorn在另一条通用Matmul路径的`IterateAll`后新增`SetHF32(false)`；
  这不构成当前PTO mHC核内变快的依据，不据此改变精度策略。
- 本轮所参考的ops-nn RMSNorm/动态量化，以及ops-math TopKV2/Sort目录没有源码变化。

最近完成统一七档模型的组合为554b3bca，包含长S6 Key L1预取；token/DSpark通过，forward七三−2.979%。
B8均值无收益、B4仍有尾部代价；模型profile完整CSA七三−15.673%，21份JSON已汇集。
B8 profile的CSA本体节省2.991ms/步，FFN增加1.377ms，主要为首层MoE Dispatch；
该profile仍快2.320ms，不能解释为正式十步无收益的完整根因，独立核内收益也不能冒充单项模型因果。
[七档模型区间及下载](results/csa_key_l1_seven_20260928/README.md)。
pypto-lib参考为73078d0；后续更新源码时记录实际使用版本即可，不做全仓hash扫描。
先按产品支持表、构建入口与指令确认A3适用性，不能单凭`arch22/arch32/arch35`目录名称类推。

## 1. QLI V2跨分片Top-K：已保留四路，继续减少中间搬运

来源：QLI V2 A3路径
[ProcessLD](../../../ops-transformer/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_vector_arch22.h)。
其`ldProcessLen=4`，每轮将累计根与三个新分片做MrgSort。该V2源码的BASE_TOPK=2048、
BASE_TOPK_VALUE_IDX_SIZE=4096，UB内部保留2048对，最终按sparseCount输出；
PTO按当前模型需求保留512对，借鉴四路/UB累计结构，并未照搬Native缓冲尺寸；
尾部分别使用二路或三路，`validBit`区分有效输入。

e58基线PTO的
[indexer_topk_query_merge_one / merge2_top512_pairs](../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_indexer.py)
逐份二路归并，每轮把累计根写回pair_arena，下轮再读取。
有H份半leaf时，当前H−1轮；四路累计归并可降至ceil((H−1)/3)轮。
例如H=8时7→3轮，H=2仍是一轮。实际H取可见候选数，不能按128K标签硬编码。

本项实施范围及保留判据：

1. 仅在性能版跨leaf合并采用四路分组；Score算术、cache布局和任务数不动，短档保留原二路路径。
2. 先核对PyPTO四输入`mrgsort`的有效输入、截断及物理tile形状；原AscendC的耗尽暂停语义不能直接假定等价。
3. 明确相同score的输入优先次序。当前PTO约定新块在前，四路操作数需相应排列；
   若无法保持精确顺序，则作为有规则的Top-K策略差异独立记录，并满足token/DSpark验收。
4. 首轮先减少归并次数；UB循环携带根的形式必须检查生成的TMOV成本，不能把省GM字节数直接当作收益。
5. 单卡长档看merge核内与累计工作，短档作不退化检查；有明确收益才补受影响尾leaf及真实EP16。

这不是重复旧的[二路UB累计根候选](results/csa_topk_register_20260928/README.md)：
旧候选没有减少归并轮数，核内11.814→12.068μs，未保留；本项的新变量是四路归并。
QLI V2的metadata和`ProcessDecode()`还带全核同步，当前PTO已有任务依赖，不整体移植这套调度。
首轮完整CPU编译、五组单卡merge边界及长短八类状态/重放通过。
128K/B16的merge核内17.63→13.83μs（−21.53%），整层1106.84→1108.84μs尚未改善；
8K/B16核内8.59→9.20μs、整层762.96→783.50μs，P95上升，通用核不直接合入。
[首轮实测](results/csa_topk_fourway_20260928/README.md)。
按实际cache长度生成独立二路/多路核后，两档状态/重放通过，长档四窗口核内18.37→13.88μs（−24.44%）。
按核内规则保留分核实现；短档生成代码恢复原二路，实测均值/P95仍略升，不标不退化或整网完成。
[保留实现、完整反例及局限](results/csa_topk_fourway_adaptive_20260928/README.md)。

继续对齐ProcessLD的UB累计根：旧二路候选因view底层存储而携带2048-float并多出TMOV；
本轮用显式extract只携带1024-float精确前缀，四/三/二路尾块统一根形状，最终直接发布。
CPU生成代码确认循环根尺寸正确、没有额外TMOV；短档二路核正文保持一致，未修改工具链。
独立五组边界及长短B16八类状态/保护区/图重放通过。
长档merge四窗口均值13.072→11.539μs（−11.73%），分布重叠、不是每窗口都更快；
按核内规则保留性能版多leaf UB根。短档二路核内基本不变，完整CSA/P95小幅回退，保留反例。
两档完整CSA均值781.237→786.449、1111.950→1109.682μs，不把核内降幅当成整层收益。
该项与修正后的QR组合2d2f9ca0已通过长短B16真实EP16；包含长S6 Key L1的554b3bca七档结果见当前阶段记录。
[四路UB根实测及保留范围](results/csa_topk_ub4_20260928/README.md)。

## 2. 已经采用的策略与仍需核对的差异

| 对象与源码 | 本次核实 | 后续重点 |
| --- | --- | --- |
| [QLI V2 Cube](../../../ops-transformer/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_cube_arch22.h) | 与本仓Native QLI Cube的主体流程相同，主要为名称、布局枚举及stride字段差异；FIXPIPE的1/1024缩放、ReLU、FP16第二次Cube规约并非新发现，性能版已采用 | 对照Q/Key/S的实际驻留周期与流水等待；不要再次把已采用的Cube head规约当作新优化 |
| [SparseFlashMla CSA Cube](../../../ops-transformer/attention/sparse_flash_mla/op_kernel/arch22/sparse_flash_mla_csa_block_cube.h) | Q/P四份L1区、KV三份L1区及DataCopyPA；当前Native Sharedkv已有同类配置 | 核对PTO QK/PV复用、跨query流水中实际等待的位置，再决定搬运/缓冲改动；旧16行UB双缓冲与成对DMA没有稳定收益，不原样重测 |
| [MhcPreSinkhorn M分块](../../../ops-transformer/mhc/mhc_pre_sinkhorn/op_kernel/mhc_pre_sinkhorn_m_split_core.h) | Stage1 AIV加宽输入，一份写给Cube，原UB值直接计算平方和；已据此保留性能版输入/RMS融合，精度版原归约不变 | 两档输入/RMS累计核时间下降27%～30%，但HC区间慢2.5～4.1μs；T60双入口及组合源码B16 EP16通过，Cube启动延迟仍需处理 |
| [MhcPreSinkhorn Cube](../../../ops-transformer/mhc/mhc_pre_sinkhorn/op_kernel/mhc_pre_sinkhorn_cube_compute.h) | 另一条M/K分块路径的`ComputeDecode/MmadA2/MmadAB`在L1/L0A复用输入，用Cube计算平方和及投影 | 与上述Vector RMS路线区分；当前PTO未采用Cube A2，不把读取复用候选称为原样移植此算法。后续须单独评估规约变化与纯AIC开销 |
| [RmsNormDynamicQuant](../../../ops-nn/norm/rms_norm_dynamic_quant/op_kernel/rms_norm_dynamic_quant_normal_kernel.h) | A3支持，多行UB处理、权重驻留、归一化与量化融合；FP32→INT32 RINT→FP16→INT8 TRUNC链与当前PTO一致 | 检查QR的两遍输入/gamma读取能否减少，先算UB生命周期；当前性能版已把平方和与amax合在第一遍，不能把“融合”本身重复计为新改进 |

RMSNormDynamicQuant的新旧文件差异还包含单/双量化输出、smooth及beta接口，
不能把删去另一条输出分支带来的代码简化称为当前CSA的确定性能收益。
精度版仍以现有Native舍入及规约合同为准，不能直接套用性能版的代数化简。

Sparse的“三缓冲”还需要区分实际驻留对象。最新AscendC `InitBuffers/ComputeMm1/ComputeMm2`：
Q/P占4×64KiB，KV占3×64KiB；QK沿K256分片加载，PV又按K256/N128读取KV。
当前PTO则让三个完整128×512 BF16 KV块跨QK/PV保留，仅KV就占384KiB，减少GM重读但压缩了L1余量。
因此两个实现都写“三缓冲”不代表相同布局，也不能据此认定该项已完全对齐。
旧的QK预发2→3候选曾因Mat使用606208字节超出524288而编译失败（验证日志§154），
不原样重跑；后续若验证更深流水，应先明确分块驻留/重新加载的取舍，量化额外MTE2开销和等待。
这是源码差异和候选方向，尚无该布局改造的设备收益，不调整现有预发深度。
同时核实`SparseFlashMlaCsa`的`PRELOAD_NUM=2`、`SMLA_PRELOAD_TASK_CACHE_SIZE=3`：
后者缓存本轮、上轮和上两轮的RunInfo；不能将三个任务描述槽或三个KV L1槽误称为预发深度3。
因此下一步先比较驻留粒度与实际流水等待，不能凭“三缓冲”直接扩大PTO预发深度。

QR的另一项差异是输入/gamma复用：当前8行×1024列的FP32输入按256列两遍读取，
最新RMSNormDynamicQuant在UB保留输入和权重。驻留输入32KiB加FP32 gamma 4KiB只是基本容量，
还需计入归约、量化和流水临时量；采用时优先保留现有256列归约/乘法/舍入顺序。
已有HC单变量DFX中该任务每worker约6.54～7.15μs、B16共12worker，
而Sparse AIC约119～148μs；不能把省一遍GM读取直接说成完整CSA的大幅收益。
本项后续单卡及尾块修正结果如下；核内保留不代替最终组合模型验收。

2026-09-28继续核实PV生成代码：当前两份L0C已生效，但四个N128 Right tile仍复用L0B偏移0，
下一块TEXTRACT等待前一MMAD。最新AscendC `ComputeMm2`则按`abL0BufIter % 2`使用两个L0B槽。
据此做仅调整核内读取生命周期的候选，编译确认Right偏移0/32768交替，不改L1 KV驻留或softmax。
QK已是K128双缓冲，不再重复调整该参数；本项也不是旧PV两份Acc试验的重跑。
[候选、来源、编译证据与长短单卡结果](results/csa_sparse_pv_l0b_20260928/README.md)：
两档状态/图重放通过，AIC均值仅下降2.44%/1.01%、四窗口分布重叠；
完整CSA短档−1.17%、长档+0.30%，长档P95略升。没有明确收益，暂不合入，也不扩测EP16。
QR输入/gamma UB驻留候选已按最新ops-nn实现，完整CPU编译/链接通过。
生成代码确认每8行只读一次输入，显式extract避免多使用点重复UB提取；256列规约和量化顺序不变。
[QR两档结果](results/csa_qr_ub_20260928/README.md)：核内短档6.801→6.333、长档7.138→6.685μs。
满档状态/重放通过，但必要T60发现量化尾行回读早于写回；修正版从当前UB直接裁有效行发布，
复测T60八类状态/保护区/图重放通过后保留性能版。首版性能不自动视为修正版新测量；修正版已进入后续组合模型验证。

继续核实Sparse Vector的`SoftmaxFlashV2Compute/DealBmm2ResBaseBlock`：Native概率先按累计最大值生成，
PV更新只缩放旧结果。历史累计softmax候选仍对新PV乘beta，未利用该单调性；旧整层回退结论不撤销。
新独立候选删去冗余beta、两次新PV缩放及局部分母乘法，保留N128/三槽/跨query流水。
完整CPU编译通过，生成代码TEXP 3→2处、TROWEXPANDMUL 4→2处；独立两档单卡已完成，暂不合入。
概率累计最大值和round是明确算术变化，不能作为数值中性搬运优化验收；
固定Native输入7864320个元素与旧累计算法零容差一致，B3解析尾块通过。
128K/B16完整CSA+0.03%、8K/B16+1.36%，7:3综合+0.427%，短档P95升10.20μs。
长档Sparse AIC均值下降1.76%，但四窗口中位数反升0.30%，分布重叠，尚无稳定长档收益依据。
另外七类状态/保护区/图重放通过，x_out改变算术，误差单列且均有限；不标Native精度通过，不扩测EP16。
[累计softmax/PV候选与旧实验区别](results/csa_sparse_online_pv_20260928/README.md)。

128K优先的新候选：QLI V2的`ProcessQk/LoadKeyToL0b`按四个16KiB L0B槽轮换，
当前PTO长档S6在稳态用地址0的8KiB Key Right；下一次TMOV需等待前一QK释放。
首块Key在WS开始前临时使用8192地址，不构成稳态Key双缓冲；基线L1最大分配末端仅96KiB。
WS Right位于8192起、占48KiB，尝试利用剩余8KiB提前加载下一Key面板，保持S6算术与QK/WS形状。
初版在SSA修正后仍因Mat分配638976>524288失败；改为独立prologue后完整CPU编译通过，Mat恢复96KiB。
实际L0B确认两个8KiB Key槽交替、48KiB WS位于16384起，L0B合计64KiB；保持16次QK/WS。
未启用预取的四组AIC/AIV及长S6 AIV共9份生成二进制一致，其他策略没有新增核内指令。
单卡task_20260928_170324_293095832121完成，两档状态/图重放通过，但长档Score AIC+14.87%，
完整CSA七三+2.917%，本版不合入、不扩测EP16。L1复用产生MTE1→FIX保护，缺逐指令stall归因。
进一步核对`InitBuffers`：Native分别分配双槽Key L1和双槽Score L1，当前PTO候选没有隔离两者。
下一候选先表达持久Key L1池并核查分配/依赖，再决定是否上卡；不因L0B已双槽就宣称Native流水已复现。
[来源、四窗口回退与状态证据](results/csa_score_key_prefetch_20260928/README.md)。
后续独立Key L1池已按AscendC的缓冲寿命实现：Key独占16KiB、Score双槽各48KiB，MTE1→FIX等待15→0。
两档状态/图重放通过，128K/B16 Score AIC四窗口285.446→269.906μs（−5.44%），完整CSA−2.12%；
短档完整CSA−0.66%、P95+1.96μs，完整CSA七三−1.682%，保留到性能版。
短档生成核未变但DFX核内读数变慢，原始Score AIC七三+0.155%，不能写成综合核内已改善。
包含该项的554b3bca已完成统一七档EP16，模型收益不作本项独立归因；
[完整范围、设备启动失败/补采与证据](results/csa_score_key_l1_20260928/README.md)。

继续按QLI V2缓冲寿命检查长档小batch：双query/M128/N128可放入两份16KiB Key及32KiB WS；
三query/M192/N128则需80KiB L0B，不能直接打开同一预取。
已准备仅双query的独立554b3bca候选，CPU完整编译/load通过；实际Key稳态L0B 0/16384、WS 32768，
Key L1池32KiB与稳态Score双槽分离，尾段复用死Key池并增加一条MTE1→FIX等待。
后续128K/B4与短B16对照已完成，保留范围及尾部限制见本节末尾；未重复未改长B8/B16。
[候选、生成地址及定向入口](results/csa_score_key_l1_pair_20260928/README.md)。

### QLI分组与L0B容量的具体差异

再次读取QLI V2 `ProcessWs/LoadSToL0b/ComputeWs`：Native按query逐个取gSize=64行Score，
K64/N128的FP16 Right每份16KiB；Key N128 INT8也为16KiB，二者共用四个16KiB L0B槽轮转。
PTO用分块对角系数把多query的WS合为一次Cube，减少调用次数，但一次驻留整个query组的Score。
因此能否预取不能只看都叫“双缓冲”，必须结合WS Right实际尺寸。

| 554b3bca代表档 | query组 | QK M/N | Key稳态L0B | WS Right | 现有预取 |
| --- | ---: | --- | --- | --- | --- |
| 128K/B4 | 2 | 128/128 | 单16KiB | 32KiB | 554无，后续双槽按核内收益保留 |
| 128K/B8 | 3 | 192/128 | 单16KiB | 48KiB | 无，L0B没有第二Key槽余量 |
| 128K/B16 | 6 | 384/64 | 双8KiB | 48KiB | 独立Key L1及L0B预取已保留 |
| 8K/B16、B32 | 2 | 128/128 | 单16KiB | 32KiB | 无 |
| 8K/B24、B40 | 6 | 384/64 | 单8KiB | 48KiB | 无 |

这些是同一算子内部按实际cache长度和query数选择的分组，不是脚本换版本。
正式单卡B8的Score AIC均值189.48μs、包络229.50μs；核内包含流水等待，不能用差额直接当作调度开销。
三query下一候选保留一次K192的WS，只把下一Key提前到独立L1池，使用前再提取到单份L0B；
不直接退回Native的三次K64 WS，也不同时改N128分块，先隔离MTE2预取的贡献。
这仍与pypto-lib连续cache/query组织不同，保持Native分页零复制视图；源码方案尚不等于重叠已经发生。
[B8独立候选及待验证项](results/csa_score_key_l1_only_20260928/README.md)。

B8后续独立单卡已完成，两档状态/保护区/图重放通过。长档Score AIC 188.230→167.660μs（−10.93%），
四窗口分布不重叠；完整CSA 846.216→833.711μs，短B16 784.743→783.843μs，
完整CSA七三−1.069%，两档P95/max均下降。保留性能版L1-only分支，与B4合并后CPU全链编译/load通过。
短档未改核也观测到较快读数，不计为预取的短档因果收益；新B4/B8未包含在554b3bca七档模型中。
组合冻结为8e176285，已提交受影响长B4/B8加短B16控制的定向EP16；
只补新增策略的模型与尾部验收，不重新跑完整七档。[入口](results/csa_key_prefetch_final_20260928/README.md)。

双query独立对照已完成：128K/B4 Score AIC 135.158→132.958μs（−1.63%），四候选窗口均低于四基线窗口，
最近边界只有0.07μs，幅度有限；状态/图重放通过后按核内规则保留。完整CSA长短七三仅−0.087%，
短档P95增加31.28μs、最大值增加40.26μs，不能标稳定性或整网已通过；短档生成核不变，但观测仍有变化。
全部控制值、Sparse/Score同核串行窗口及来源见[B4完整结果](results/csa_score_key_l1_pair_20260928/README.md)。

### 后续长档分工：分片粒度与最忙核工作量

继续核实最新QLI V2的
[kernel常量及SplitCore](../../../ops-transformer/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_kernel_arch22.h)：
`S2_BASE_SIZE=2048`在`InitTilingData`实际写入constInfo，不只是Cube中的注释；
`SplitCore`从metadata读取每核的batch、query组和S2起止。
[metadata的AssignByBlock](../../../ops-transformer/attention/quant_lightning_indexer_v2_metadata/op_kernel_aicpu/quant_lightning_indexer_v2_metadata_aicpu.cpp)
按剩余代价逐S2块分配，A3所用910B分支同样取2048。
Vector在每个S2块排序后把累计Top-K留在UB，和PTO按8192 leaf、两个AIV各排半leaf的组织不同。

当前128K/B8（三query）和B16（六query）都形成16个query组；只计32768候选的完整部分，
每组4个8192 leaf，共64份分到24个worker，16个worker做3份、8个做2份。
这说明核内循环工作量本身不齐；不能把最慢核与均值的差额全部称为运行时调度开销。
若仅改4096，最忙核仍是6×4096=24576候选，不能据“块更小”推断关键路径变短；
2048时理论最忙核为11×2048=22528，完整部分的工作量上界减少8.33%，
但会增加Q/系数读取、通信尾部排空和跨分片Top-K归并。上述是静态计数，不是设备收益。

旧日志§152的全局8192→4096失败来自单leaf publish写死的切片范围，不能原样重试全局常量替换。
若后续实施，须把长档工作分片与score arena容量、短档分支、pair槽数/归并计数分别表达，
并保留tie顺序及边界；不能沿用旧失败的全局常量替换。

B8预取与七档profile完成后，已在独立8e176285快照准备另一种均衡方案，后续完整CPU已通过、单卡已提交：
不把所有leaf改成2048，而只对16组/24核的长档把leaf数向3的倍数取整，均分现有N1024步。
32769候选时8/8/8/8/1步变为6/6/6/5/5/5步，最忙核25→22步；score/pair arena容量不增加，
超过最多32个leaf时回退原计划，短档不改分片。这里改变核内循环工作量，任务数和调度标志不变。
long merge按同一计划计算各query可见根前缀，避免短请求读到未发布槽；长短B16八类状态/图重放现已通过。
128K的根从10个增到12个，四路归并三轮变四轮，更多排序/重复Q加载可能抵消最忙核收益。
[独立候选、CPU账本及实测代价](results/csa_score_balanced_leaves_20260928/README.md)。

Native `CalcCost`实际是`6×ceil(M/16)+10×ceil(S2/64)`，`AssignBlockToCore`按
未分配cost/剩余核数更新限额；不是单纯数S2块。Vector `AlignS2`还按32/128/512元素分段对齐，
最大处理2048候选，避免一律padding至2的幂。PTO新候选的2560/3072半leaf仍走4096排序，
因此先测当前单因素的AIC/AIV最大核、包络和完整CSA，不把新增排序padding同时修改后混合归因。

进一步核对[Native SortAll](../../../ops-transformer/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_vector.h)
与当前[PTO-ISA A3 TMRGSORT](../../../pto-isa-src/include/pto/npu/a2a3/TMrgSort.hpp)：
Native按剩余有序段数选择二/三/四路尾部，PTO单输入`block_len`形式要求有效列数整除`4×block_len`。
所以不能把4096候选tile直接改为3072后沿用三次相同mrgsort；最后一级6144个value/index元素不满足4096整除。
如后续实测排序成为瓶颈，应在算子侧显式组织二/三路尾部，不改PTO-ISA，也不能只凭少padding断言收益。
此前整半leaf Score留UB已在日志§236因额外子视图搬运及gather无收益而撤回；不原样重复该实验。

均衡候选已测得长B16 AIC最慢核321.480→281.750μs（−12.358%），均值却274.548→277.766；
AIV均值287.238→306.229、最终merge 11.721→14.187，增加的排序/归并抵消了大部分收益。
完整CSA长短七三仅−0.410%，短档P95增加9.18μs；Native控制也变化，不能声称稳定整层加速。
据此下一项单独采用2560/3072候选tile，仅对前2048作完整四路归并，最后把512/1024尾部与前段Top-512做二/三路归并。
保留原后2048段优先、段内顺序；需要独立边界/tie检查，不能将理论等价作为已测正确性。

该排序改动已完成十组边界/tie及长短B16状态检查，均通过。
相对均衡leaf基线，长档Score AIV均值302.662→271.441μs、包络316.840→280.970，四窗口分布不重叠；
完整CSA长档−3.322%、短档+0.648%，七三−2.131%，两档P95/max下降。
组合另补128K/B8对8e176285：CSA−3.294%，状态/图重放通过；主要改善最慢AIC/AIV，平均AIC基本持平。
两项已保留到性能版；不同基线的阶段数据不合成新的七档加速比。
[组合结果、代价和原始泳道](results/csa_score_balanced_sort_20260928/README.md)。

继续核对Native metadata准备与核内消费：当前PTO编排已经扫描kv_seq_lens得到max_topk_cache_len，
Score每worker仍扫描一次，新增长merge又扫描一次。已在长merge生成C++确认该循环及GM标量读取留在核内，
不是只看Python重复代码推断开销。标量复用候选已经在CANN9.2完成长短B16：
生成扫描消失且状态/图重放通过，但长档Score核内持平，短档AIC/AIV增加8.276%/7.259%，不合入。
[代码和核内反例](results/csa_maxlen_reuse_20260928/README.md)。不能把少读GM推导成设备收益。

下一项核查`ProcessVec1`的`innerS1Idx`循环：Native复用一份SortAll/归并正文，
当前PTO按query_group_size完全展开half-leaf排序，六种长度分支在S6中复制六份。
仅把末尾query排序的pl.unroll换为pl.range，完整编译/load通过；S6 AIV实际.text从41200降至9596字节，
2/3query同样减少重复代码。前面的scale展开、Cube算术、任务分工及调度不变。
无I-cache stall证据，不将代码尺寸降幅等同于性能降幅。两档状态/图重放已通过，
但长档Score AIV+0.644%、短档+15.343%，核内七三+5.054%，不合入生产或扩测。
CSA七三−0.681%不能代替核内目标；短档P95+7.54μs，控制漂移与全部窗口保留。
[唯一补丁、生成码及完整反例](results/csa_sort_query_loop_20260928/README.md)。

## 3. ops-math的适用边界

- [TopKV2入口](../../../ops-math/math/top_k_v2/op_kernel/top_k_v2_apt.cpp)此次读到的实现引用arch35路径。
  README新增`sort_policy=1`的Bitonic Small TopK针对2≤k≤32，当前CSA为Top-512；本轮不直接套用。
- [experimental SortV2](../../../ops-math/experimental/math/sort_v2/op_kernel/sort_v2.h)提供A3相关排序参考，
  但读取到的是通用Concat/Sort/Extract流程，并未发现可以直接替换当前Top-512的确定收益。
- 后续优先比较A3可用的排序/归并原语及中间搬运；不因仓库更新日期新就切换算法或运行时。

## 4. 当前执行顺序

用户2026-09-29要求优先128K：长短档取舍改按耗时变化率8:2评估，七档先在各上下文内平均。
总体有收益即保留；单侧明显退化时在同一套算子内按场景分支，不要求所有档位同时加速。
核内、完整CSA及最终forward各自计算，异常P95和功能约束单列，不能用权重掩盖。

当前正式七档为128K B4/B8/B16/B24、8K B16/B24/B32，B40退役；筛选代表档是长B16/短B24。
完整4ffccb7b/CANN9.2七档已包含WO_A原生NZ、长S6均衡排序、Key预取及三项系数优化，
[当前CSA/核内差距](DSV4_FLASH_CSA_INCORE_NATIVE_GAP.md)保留测量边界与原始证据。
该七档Native仍属vLLM编译包装、static开启、superkernel关闭的旧入口。
显式torch.compile backend=npugraph_ex的新Native已采用superkernel：设备直接replay长B16 1122.652μs、短B24 940.762μs，
[校准证据](results/csa_native_graph_replay_20260929/RESULTS.md)。不再做开关试验，阶段出口统一更新七档。
新profile中QLI→Sparse已经合成一个SuperKernel，不能把该PMU数字当成单个QLI核时。
旧拆分profile仍可作已注明配置的核内参考，不能把它改名为新SuperKernel内部实测。

短档Score及Sparse整组准入两项定向对照已结束，未证明整体收益，暂不采用；
[同核串行证据](results/csa_short_score_sync_20260928/README.md)及[Sparse对照](results/csa_sparse_sync_20260928/README.md)保留。
QLI V2四路Top-K及mHC M分块输入复用均已按核内规则保留，必要单卡状态/尾块通过，
组合源码d1f170ff长短B16真实EP16的forward分别比同轮Native快4.03%/4.10%，P95更低，token/DSpark一致；
[完整模型证据](results/csa_ascendc_topk_hc_ep16_20260928/README.md)不能证明每项独立整网收益，也未完成新版七档。
新增Top-K UB与QR尾修正版组合2d2f9ca0的长短B16状态/图重放通过，
长档单卡计时有CPU编译重叠，不据微小差额判断净收益；独立真实EP16已完成，长短B16 forward快4.34%/6.72%，
7:3变化率−5.054%，token/DSpark一致。两档P95虽较低，长档仍有一次metadata准备增加约3.6ms、
相对设备入场晚3.408ms的尾部，不能用均值收益关闭问题；当前不是新版七档验收。
[组合范围、数据限制及任务](results/csa_ub_combined_20260928/README.md)。
核内研究优先128K的Indexer Score，继续核对QLI V2实际驻留与流水；已有Sparse两项候选结束，不原样重测。
按七档实际热点继续审查最新AscendC策略；不能因本次只找到一个新候选，就将整个核内阶段标完成。
每项分别记录核内耗时、调度等待、完整CSA/P95与最终forward，解释与pypto-lib的任务和输入差异。
先单卡代表档，明确收益后再补必要的真实权重EP16；没有新证据不重跑旧失败方案或整矩阵。


## 5. Sparse末块直接发布：已保留的核内优化

来源为ops-transformer28f40354的
[Sparse SCFA Vector](../../../ops-transformer/experimental/attention/sparse_attn_sharedkv/op_kernel/arch22/sparse_attn_sharedkv_scfa_block_vector.h)：
DealBmm2ResBaseBlock的最后S2分支调用RowDivs后直接发布；RowDivs使用Div，不采用倒数乘假设。
此前PTO已在QK/PV中完成累计规约，但仍把FP32 mi/li/oi发布到GM，让独立merge_norm读回归一化和逆RoPE。
pypto-lib2164563的dspark仍使用该独立merge。新私有候选消除这一最终交接，把原16-head的归一化、
逆RoPE、BF16转换与O_A分组发布移到每个query最后PV块，保留现有性能版算术。

融合初版UB197632>188416字节，8-head版本仍189440，未降低工具链检查。
MemoryReuse IR确认softmax临时区16KiB全循环常驻；缩短到softmax子阶段后，原16-head版本181248字节，
完整PTOAS/CCE/load通过，独立merge_norm消失。不是重复旧累计softmax或PV L0B候选。
正常auto单卡任务task_20260929_083719_62875131423已结束，冻结后未编辑源码。
长B16/短B24的x_out检查失败，其他七类状态精确一致；首版不得采用。
生成代码后16-head统计量切片丢失64字节偏移，先在独立Sparse中验证显式抽取修复，尚无有效收益结论。
后续比较Sparse加原merge的总AIV核时和发布跨度，避免融合范围变化造成错误归因。
[候选、缓冲证据与设备入口](results/csa_sparse_final_publish_20260929/README.md)。

后续显式ND抽取修复在独立Sparse314万元素及完整两档八类状态中均逐bit通过；
B3/H127和active-B=3/2/1/3 padding亦通过。已保留在性能版Sparse单文件，
长/短CSA−2.960%/−1.495%，8:2−2.667%；Sparse加原merge的AIV总核时8:2−11.551%，P95均下降。
它吸收AscendC的末块数据交接策略，保留PTO现有算术，与pypto-lib的独立merge边界仍有意不同。
[保留结果](results/csa_sparse_final_publish_fixed_pair_20260929/README.md)。

## 6. HC_post残差常驻：吸收Permanent-X的数据重用

ops-transformer28f40354的`mhc/mhc_post/op_kernel/arch22/mhc_post_arch22.h`
在ComputeCopyOutAllX中按USE_PERMANENT_X复用UB残差，tiling按可用UB选择；
release Native的HcPostDSplit同样先整组读取。PTO此前沿pypto-lib2164563每输出通道
重读四行的循环；上游残差FP32，本接入为Native BF16，另多出重复的BF16→FP32转换。

新实现每token四行残差load/cast各一次并复用，保留post*x后0/1/2/3顺序mul/add及BF16 RINT，
未使用AscendC Axpy改变舍入。每token残差读取/转换16→4；任务分工和外层pipeline保持。
两档核内长/短−17.423%/−24.785%，四窗范围不重叠；CSA长−1.090%/短+0.859%，8:2−0.700%。
短档P95增加14.140μs，记录为区间回退；候选P95/P50最高1.0197，没有按异常点删除任何样本。
八类完整状态零容差及T18尾块/同图padding通过，按核内收益规则保留共享实现；
性能版继续reexport，两版依赖图通过。完整精度版/Native逐token及EP16验收仍后置。
[测量、边界和采用证据](results/csa_hc_post_resident_20260929/README.md)。
