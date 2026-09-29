# Indexer：当前PTO与最新AscendC的差异

更新：2026-09-29。当前完整七档为 **55b89ee2 / CANN9.2**，任务已退出0：
128K B4/B8/B16/B24、8K B16/B24/B32。长短8:2完整CSA为−6.498%，长档−8.536%、短档+1.653%。
Native已采用最新标准：显式torch.compile/backend=npugraph_ex、dynamic=False/inplace_pass=True/static开启，
主性能SuperKernel开启；另采关闭SuperKernel的独立核内profile，不把诊断计时放入正式主表。
PTO采用已有验证路径和数据，不要求复刻Native编译选项；版本、环境与测量边界逐项标明。
本轮已有系数优化和Sparse末块发布；生产随后保留d93bba14的HC_post残差常驻，未倒写进七档。
生产另已保留长S6完整query单根：长B16 Score AIC/AIV−8.555%/−9.484%、CSA−4.390%，
短B24 CSA+2.941%、8:2−2.924%；两档及B4/H65535尾段/padding完整状态通过。
这一局部A/B不倒写到55b89ee2七档；新源码复用已测两档、补其余五档，Native基线保持。
[七档完整数据](results/csa_native_inplace_seven_20260929/RESULTS.md)、
[实际任务明细](results/csa_native_inplace_seven_20260929/TASKS.md)、
[当前核内差距](DSV4_FLASH_CSA_INCORE_NATIVE_GAP.md)。
旧版本和局部A/B留在[验证日志](DSV4_FLASH_CSA_VALIDATION_LOG.md)，不拼接为同轮七档。

## 实际接口与主路径

Native调用 `_indexer_qli`，消费原分页cache、页表、长度、query及head权重，返回Top-K索引。
PTO保持Native分配与更新：一个可写物理cache根入参，在编排里建立零复制字节/Key视图，
按页表直接加载Key和scale。没有入口历史复制，没有外部拆分/写回，不把GM视图当成连续请求历史。

压缩历史至少2048行走FP16片上QK及第二次Cube head加权规约；更短历史保留Vector路径。
长短选择及2/6 query策略均在同一份 `decode_indexer.py` 内，调用者不选历史版本。
以下为S6七档的实际分支，Q表示有效query总行数，worker数为24 AIC/48 AIV。

| 历史 | 档位 | 每组query | QK N面板 | Score tile | Key预取 |
| --- | --- | ---: | ---: | ---: | --- |
| 128K | B4/B8/B16/B24 | 6 | 64 | 1024 | 独立L1双槽，连到L0B |
| 8K | B16/B32 | 2 | 128 | 768 | 无长档预取 |
| 8K | B24 | 6 | 64 | 768 | 无长档预取 |

长档Q<24用双query，Q≥24用S6；四/八/十六个query组按24worker均衡leaf，超出pair容量时保留原分片。
短档比较整请求与双query最忙核的工作量，
整请求覆盖24worker且工作量增幅不超过25%时选S6，不硬编码单个batch标签。
长档Score保留sync_start与禁止early_resolve，短档没有照搬该开关。

## 与最新ops-transformer A3路径对照

参考ops-transformer **28f40354** 的 `quant_lightning_indexer_v2/op_kernel/arch22`，
本地运行Native仍为release custom二进制，源码参考不等于已替换Native算子。

| 环节 | 最新Native A3策略 | 当前PTO | 剩余问题/证据 |
| --- | --- | --- | --- |
| query/Key复用 | L1 query最多256行，L0按128行M面板；同一Key L1面板跨M子块复用 | 长B≥4用S6 M384/N64；更小长档保留双query | 长B4/B8 S6已取得明显核时收益；分组相似仍不等于L1/L0流水相同，旧4+2拼接退化不重试 |
| Query/系数驻留 | ComputeMm1只在isFirstS2InnerLoop加载Query及Weight，后续S2块复用L1 | 每个leaf重新加载Q和系数，再移至L0A；leaf内部多个N面板已复用 | B24固定组跨leaf驻留已测，AIC约+3.72%、AIV持平；当前候选不采用，不能把少读字节直接当收益 |
| Key读取 | ProcessQk首个M子块加载Key，末个M子块后释放；独立buffer事件 | 长档独立Key L1槽及提前一个面板预取，连到L0B | 本轮B4/B8 AIC均值为70.539/139.297μs，后续按当前核时继续看等待与重复move |
| 系数生成 | ProcessVec0在QLI内由偶数AIV成块加载FP16 weight/qScale，相乘、Brcb后GM交接 | 独立SPMD成块加载FP32输入，分别转FP16相乘；在UB补入对角行后一次GM写回，提交min(48,query组数)，stride48不变 | 空worker、批量准备和UB一次发布已保留；独立任务/转换/对角布局开销仍在，朴素融合失败；Native QLI不含Query Hadamard量化 |
| QK→WS | FIXPIPE把QK INT32缩放转FP16入L1；Cube完成head加权 | 同样采用FP16 QK和Cube WS | 此项已经采用，不再把旧Vector head规约写成当前差异 |
| WS归约粒度 | ProcessWs逐query循环，ComputeWs的M=16、K=gSize=64，N最大128；Brcb的16行重复结果只发布有效行 | S6以[16,384]对角系数同时规约六个query，N64，发布六个有效行 | 同六query/128候选的FP16 MAC数均为6×16×64×128=786432；不能只看到PTO的零系数就认定额外算力浪费。调用数与L0/流水布局不同，应连同QK分块评估 |
| scale | Vector按物理页加载scale并解量化score | Vector直接从原cache物理页取scale | 两侧均有分页读取，不存在Native恒为单次连续scale读取的依据 |
| Vector分工 | ProcessVec1的两个AIV分摊query/S1；每个query内处理整个S2段 | 长S6两个AIV各处理三个完整query，Cube传输连续候选；短档仍分候选半区 | 完整单根策略长B16 AIC/AIV−8.555%/−9.484%；scale每AIV全读的翻倍代价仍存在。旧只改query分工但留两根的失败候选不采用 |
| 本地Top-K | 2048候选分段排序，BASE_TOPK=2048的UB累计根，最终按sparseCount输出 | 长S6连续两次1024分数在UB组成2048段，保留Top-512根，每query/leaf发布一根 | 长S6已消除scaled-score GM暂存且consumer根数减半；短档仍沿原GM/half根路径，最终归并任务仍独立 |
| 分片平衡 | metadata按工作成本切S1/S2并给最终归并核分工 | 长档按query组与24worker平衡leaf；B16从8/8/8/8/1分成6/6/6/5/5/5个tile | 最忙核下降已保留；新增root数量和AIV排序成本单列 |
| 最终归并 | LocalTopK/Merge/MS式四路归并，可在同融合kernel结束 | 四路归并与UB累计根已采用，但仍独立merge task | 固定tie/量化规则需保持；任务融合是后续调度/结构调整，不能仅凭少一个任务声称收益 |

PTO Native cache适配没有device重排，但页内Key/scale错位、动态有效长度和更新依赖仍需正确表达。
最新本地pypto-lib 2164563的dspark Indexer也按页表读取独立key/scale cache，不能再描述成请求历史恒连续。
它仍按单query×leaf分工，普通路径AIV做FP32 head规约，大batch长档用FP16双缓冲；
当前PTO的S6复用、第二次Cube规约和Native交错cache视图需要分别评估，不能套用旧725μs泳道的输入假设。
本轮长S6进一步按最新AscendC处理完整query及UB单根，不直接照搬pypto-lib的单query/leaf组织；
这保留六query共享Key的Cube收益，并减少本接入原来的两根及scaled-score交接，取舍由同卡核内/CSA实测决定。
已检查Native页指针式切片替代双视图的写法：当前PyPTO默认核内转换会把tensor.slice变为Tile，
后续GM reshape/load链不能成立；该轻量探针未产生设备候选，保留原路径，见[表达限制](results/csa_key_page_view_20260929/README.md)。
当前实际核时与七档Native对照见[核内差距](DSV4_FLASH_CSA_INCORE_NATIVE_GAP.md)；
[完整四窗口、最慢核与包络](results/csa_native_inplace_seven_20260929/TASKS.md)保留全部原始读数。

WS粒度的源码依据为同目录`quant_lightning_indexer_v2_service_cube_arch22.h`：
ProcessWs（179行）逐gSize调用，ComputeWs（496行）设置M/N/K，
FixpResToGm（552行）只发布各query的有效行。上述MAC计算不包含DMA、FIXPIPE、同步、
尾部padding及Cube指令吞吐差别，不能据此判定两实现等速；不原样重试已失败的4+2/固定组驻留。

## 下一步按依赖推进

1. 已保留的S6 Key复用、2048分段排序、UB中间根和WO_A NZ已由同源码七档覆盖。
   当前长B16 Score AIC/AIV为250.640/256.646μs，merge9.677μs；
   长B24为349.266/365.861μs、merge10.844μs。Native独立QLI的PMU参考分别为
   242.293/241.730和376.467/376.063μs；不能忽略PTO独立系数、scale提交与merge。
2. 系数任务仅取消空worker已保留：固定stride48，提交数min(48, query组数)。
   长B16/短B24真实编译A/B的8:2均值−1.885%、两档P95下降；短档一次最大值和连续诊断分别保留。
   这是任务提交优化，不能把Score核时略升隐去或称为算术加速。
   [实测、历史区别和诊断](results/csa_coefficient_active_workers_20260929/README.md)。
   后续按组加载/乘法已保留，长B16/短B24系数核时4.579→3.318、4.559→3.165μs，
   CSA 8:2−0.887%；短档CSA/P95回退仍记录。[本轮证据](results/csa_coefficient_group_20260929/README.md)。
   再保留UB构造后一次发布，长短系数核时−25.798%/−36.454%；
   CSA−0.543%/+1.453%，8:2仅−0.143%，P95两档略升，按核内收益采用。
   [一次发布与分项](results/csa_coefficient_publish_20260929/README.md)。
3. 按现有level-4数据分别解释系数/scale依赖。Score已有early派发不能写成一直等待AICPU，
   cache scale的64字节读改写也不能在未证明页面所有权时直接并行。
   后续研究独立归并、AIV数据交接及关键链，保持长短8:2；明显顾此失彼时在同一算子内分场景。
4. 有真实核内收益且必要功能检查通过即保留，CSA和P95单列；算术或量化改变则另记精度影响。
   当前精度版新增中性优化迁移、新CANN9.2/B24整模型token/DSpark验收尚未完成。

固定组跨leaf Query/系数驻留已否定：生成代码虽证明TLOAD/TMOV移出leaf循环，
但长B24 AIC约+3.72%、AIV持平，没有明确收益。理论少搬运不能代替核时证据。
去除4个dummy也只有约0.1%的加权CSA差异，未合入；不继续原样扩测。
系数融合进Score已测，长B16/短B24 CSA分别+2.772%/+0.778%，8:2为+2.373%，未合入。
长档Score提前19.220μs启动，但系数从每组一次变为每leaf一次，核时增加。
后续Native式批量读入/乘法及UB一次发布已用于独立系数任务；再次融合仍须解决每leaf重复构造，
不能直接复用先前失败版本；
[结果、Native/pypto-lib差异及泳道](results/csa_coefficient_fused_20260929/README.md)。

## 有效证据与排除方向

- [均衡leaf与部分排序](results/csa_score_balanced_sort_20260928/README.md)：已保留，旧单项基线不与本轮9.2混算。
- [四路Top-K与UB根](results/csa_topk_ub4_20260928/README.md)、[Key L1预取](results/csa_score_key_l1_20260928/README.md)：已保留。
- [编排最大长度复用](results/csa_maxlen_reuse_20260928/README.md)：生成码扫描减少但无明确核内收益，不采用。
- [query排序循环](results/csa_sort_query_loop_20260928/README.md)：代码体积下降，短档核时恶化，不采用。
- [Score矩阵scale广播](results/csa_score_scale_matrix_20260929/README.md)：长档6个TMUL调用点合为1个TCOLEXPANDMUL，
  状态精确、CSA略降，但长B16 Score AIC/AIV约+7%，没有核内收益，不采用。
- [缩放分数按2048段驻留UB](results/csa_score_segment_ub_20260929/README.md)：固定512等宽行/48KiB缓冲，
  状态零容差通过，生成码去掉排序前GM中转；长Score AIC/AIV仅+0.065%/−0.294%，
  首轮长CSA−2.081%，同源码反向复测仅−0.151%，未形成稳定收益，暂不采用。
- [源码入口和版本](DSV4_FLASH_CSA_ASCENDC_REFERENCES.md)：A3 arch22优先，不套用不兼容的arch35能力。
- [当前PTO源码](../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_indexer.py)、
  [Native调用接口](../../vllm_ascend/attention/dsa_v1.py)。

[长S6 query分工候选](results/csa_score_query_split_20260929/RESULTS.md)已完成真机验证，未采用。
八类完整状态零容差通过；长B16 Score AIC/AIV为251.849→254.705、257.647→260.349μs，
没有核内收益。长CSA−1.054%，未改代码路径的短CSA+1.506%，8:2仅−0.542%，
不能用区间波动代替incore收益。该结果只否定保留两个half根的简单分工候选，
不外推为Native完整query内规约无效，也不原样扩测。

## 尚未验证的结构差异：每query的候选根数

重新对照当前ops-transformer28f40354 `quant_lightning_indexer_v2_service_vector_arch22.h`：
ProcessVec1按query/S1分摊两个AIV（约328–365行），每个query对完整S2段执行缩放、SortAll，
再合入自己的globalTopkUb（约395–416行）；无跨核归约需求时直接提取最终索引（约419–432行）。
当前PTO每个AIV处理全部六query的一个候选半区，每query/leaf产生两份Top-512根，
独立merge再消费这些根。新版七档长B4/B8 Score AIV80.560/146.302μs，
Native独立QLI参考64.369/126.021μs；这只是不同边界的差异，不是根数造成差距的因果证据。

此前query分工候选仍保留两份half根，且每AIV读取scale翻倍，因此其约+1%负收益
只否定那一版，不能外推为Native的完整query内排序/单根组织无效。
后续若进一步试验，必须成套处理完整候选排序、单根写入和consumer根数，不能只改AIV分工；
还须保持固定tie规则或明确记录Top-K规则变化、UB容量、scale额外读取与跨leaf合并工作量。
Native的BASE_TOPK=2048与本接入Top-512不同，也不直接照搬UB大小或声称相同MAC就等速。

HC调度对照已完成且8:2回退1.380%，不采用。随后已实现[完整query单根候选](results/csa_score_single_root_20260929/README.md)：
长S6使用Cube连续候选、AIV各三个query、UB内2048排序和单根consumer；短档原路径保持。
两入口解析/完整CPU编译通过，Vec地址覆盖88KiB；11:30正常auto提交两代表档
task_20260929_113003_337101716086已退出0，源码冻结。长B16 Score AIC/AIV下降8.555%/9.484%，
四窗均值范围不重叠；完整CSA长−4.390%、短+2.941%、8:2−2.924%，短档回退如实保留。
两代表档及B4/H65535尾段/padding八类完整状态精确通过，已采用到性能版；七档缺少的五档待补，模型token/DSpark仍未覆盖。
避免重复旧两half根版本，也不把理论少根数作为采用依据。
