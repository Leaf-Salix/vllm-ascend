# CSA核内优化：本地AscendC源码参考

更新：2026-09-28。按用户修正后的目标，最新AscendC实现是后续核内优化的主要依据：
首先研究ops-transformer，结合ops-nn、ops-math等ops仓库；当前Native作性能/行为对照，pypto-lib作PTO实现参考。
以下是本地已读取版本，不表示当前CANN二进制已包含这些实现，也不表示这些实现已测得比PTO快。

| 本地仓库 | 本次读取HEAD | CSA相关入口 |
| --- | --- | --- |
| [ops-transformer](../../../ops-transformer) | b5b33e14 | QLI/QLI V2、SparseFlashMla、mHC |
| [ops-nn](../../../ops-nn) | 7a71d54e | RMSNorm、动态量化及融合路径 |
| [ops-math](../../../ops-math) | 81802185 | 排序、Top-K及基础向量操作 |

三个HEAD提交日期均为2026-09-28。本次只读源码，没有安装这些仓库、升级CANN或改Native流程。
当前冻结生产算子对照为d1f170ff，各单变量候选另记实际基底；pypto-lib参考为73078d0；后续更新源码时记录实际使用版本即可，不做全仓hash扫描。
先按产品支持表、构建入口与指令确认A3适用性，不能单凭`arch22/arch32/arch35`目录名称类推。

## 1. QLI V2跨分片Top-K：已保留四路，继续减少中间搬运

来源：QLI V2 A3路径
[ProcessLD](../../../ops-transformer/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_vector_arch22.h)。
其`ldProcessLen=4`，每轮将累计Top-512与三个新分片做MrgSort，保留前512对在UB，最后发布结果；
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
这是新实现依据，不能据此认定性能收益。独立边界及长短B16设备任务待执行，尚未合入。
[四路UB根候选](results/csa_topk_ub4_20260928/README.md)。

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
本项列作后续小范围候选，优先核实Sparse等更大热点；未新建NPU测试或宣称收益。

2026-09-28继续核实PV生成代码：当前两份L0C已生效，但四个N128 Right tile仍复用L0B偏移0，
下一块TEXTRACT等待前一MMAD。最新AscendC `ComputeMm2`则按`abL0BufIter % 2`使用两个L0B槽。
据此做仅调整核内读取生命周期的候选，编译确认Right偏移0/32768交替，不改L1 KV驻留或softmax。
QK已是K128双缓冲，不再重复调整该参数；本项也不是旧PV两份Acc试验的重跑。
[候选、来源、编译证据与长短单卡结果](results/csa_sparse_pv_l0b_20260928/README.md)：
两档状态/图重放通过，AIC均值仅下降2.44%/1.01%、四窗口分布重叠；
完整CSA短档−1.17%、长档+0.30%，长档P95略升。没有明确收益，暂不合入，也不扩测EP16。
QR输入/gamma UB驻留候选已按最新ops-nn实现，完整CPU编译/链接通过。
生成代码确认每8行只读一次输入，显式extract避免多使用点重复UB提取；256列规约和量化顺序不变。
[冻结候选与单卡任务](results/csa_qr_ub_20260928/README.md)等待设备，未并入生产或宣称收益。

## 3. ops-math的适用边界

- [TopKV2入口](../../../ops-math/math/top_k_v2/op_kernel/top_k_v2_apt.cpp)此次读到的实现引用arch35路径。
  README新增`sort_policy=1`的Bitonic Small TopK针对2≤k≤32，当前CSA为Top-512；本轮不直接套用。
- [experimental SortV2](../../../ops-math/experimental/math/sort_v2/op_kernel/sort_v2.h)提供A3相关排序参考，
  但读取到的是通用Concat/Sort/Extract流程，并未发现可以直接替换当前Top-512的确定收益。
- 后续优先比较A3可用的排序/归并原语及中间搬运；不因仓库更新日期新就切换算法或运行时。

## 4. 当前执行顺序

短档Score及Sparse整组准入两项定向对照已结束，未证明整体收益，暂不采用；
[同核串行证据](results/csa_short_score_sync_20260928/README.md)及[Sparse对照](results/csa_sparse_sync_20260928/README.md)保留。
QLI V2四路Top-K及mHC M分块输入复用均已按核内规则保留，必要单卡状态/尾块通过，
组合源码d1f170ff长短B16真实EP16的forward分别比同轮Native快4.03%/4.10%，P95更低，token/DSpark一致；
[完整模型证据](results/csa_ascendc_topk_hc_ep16_20260928/README.md)不能证明每项独立整网收益，也未完成新版七档。
下一步继续筛选最新AscendC差异，先核实Sparse大热点的片上布局与等待，再安排QR输入驻留等小范围候选。
按七档实际热点继续审查最新AscendC策略；不能因本次只找到一个新候选，就将整个核内阶段标完成。
每项分别记录核内耗时、调度等待、完整CSA/P95与最终forward，解释与pypto-lib的任务和输入差异。
先单卡代表档，明确收益后再补必要的真实权重EP16；没有新证据不重跑旧失败方案或整矩阵。
