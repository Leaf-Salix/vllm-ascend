# Indexer：当前PTO与最新AscendC的差异

更新：2026-09-29。最新完整新七档为c93ec723 / CANN9.2，
Native/PTO已按同配置真实编译半层重测；长档B4/B8/B16/B24、短档B24/B32/B40全部完成。
主线随后保留系数空worker优化，代表两档8:2−1.885%；该项不与七档旧样本拼表。
旧V7/V10及入口cache拆分结论移出当前说明；实验历史见[验证日志](DSV4_FLASH_CSA_VALIDATION_LOG.md)。

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
| 8K | B32 | 2 | 128 | 768 | 无长档预取 |
| 8K | B24/B40 | 6 | 64 | 768 | 无长档预取 |

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
| Key读取 | ProcessQk首个M子块加载Key，末个M子块后释放；独立buffer事件 | 长档独立Key L1槽及提前一个面板预取，连到L0B | 本轮B4/B8 AIC均值为76.619/135.495μs，后续按当前核时继续看等待与重复move |
| 系数生成 | ProcessVec0在QLI内由偶数AIV完成FP16 weight×qScale、Brcb及每核GM交接 | 独立SPMD生成对角块系数，提交min(48,query组数)，stride48不变 | 取消空worker已保留；仍有独立任务，后续研究融合时须说明Native的QLI不含Query Hadamard量化 |
| QK→WS | FIXPIPE把QK INT32缩放转FP16入L1；Cube完成head加权 | 同样采用FP16 QK和Cube WS | 此项已经采用，不再把旧Vector head规约写成当前差异 |
| scale | Vector按物理页加载scale并解量化score | Vector直接从原cache物理页取scale | 两侧均有分页读取，不存在Native恒为单次连续scale读取的依据 |
| Vector分工 | ProcessVec1的两个AIV分摊query/S1；每个query内处理整个S2段 | 两个AIV分摊候选范围，各自处理全部query并复用同一份scale | PTO每leaf有两个half根；改成query分工可能减少根归并，但会增加scale重复读取，未验证前不能判收益 |
| 本地Top-K | 2048候选分段排序，BASE_TOPK=2048的UB累计根，最终按sparseCount输出 | 长档先排序前2048候选并将Top-512根留UB，尾段完成后合并；短档保留原路径 | PTO缩放后score仍经GM，最终归并仍独立；临时根GM往返已消除 |
| 分片平衡 | metadata按工作成本切S1/S2并给最终归并核分工 | 长档按query组与24worker平衡leaf；B16从8/8/8/8/1分成6/6/6/5/5/5个tile | 最忙核下降已保留；新增root数量和AIV排序成本单列 |
| 最终归并 | LocalTopK/Merge/MS式四路归并，可在同融合kernel结束 | 四路归并与UB累计根已采用，但仍独立merge task | 固定tie/量化规则需保持；任务融合是后续调度/结构调整，不能仅凭少一个任务声称收益 |

PTO Native cache适配没有device重排，但页内Key/scale错位、动态有效长度和更新依赖仍需正确表达。
最新本地pypto-lib 2164563的dspark Indexer也按页表读取独立key/scale cache，不能再描述成请求历史恒连续。
它仍按单query×leaf分工，普通路径AIV做FP32 head规约，大batch长档用FP16双缓冲；
当前PTO的S6复用、第二次Cube规约和Native交错cache视图需要分别评估，不能套用旧725μs泳道的输入假设。
已检查Native页指针式切片替代双视图的写法：当前PyPTO默认核内转换会把tensor.slice变为Tile，
后续GM reshape/load链不能成立；该轻量探针未产生设备候选，保留原路径，见[表达限制](results/csa_key_page_view_20260929/README.md)。
当前实际核时与七档Native对照见[核内差距](DSV4_FLASH_CSA_INCORE_NATIVE_GAP.md)；
[完整四窗口、最慢核与包络](results/csa_compiled_seven_20260929/RESULTS.md)保留全部原始读数。

## 下一步按依赖推进

1. 已保留的S6 Key复用、2048分段排序、UB中间根和WO_A NZ已由同源码七档覆盖。
   当前长B16 Score AIC/AIV为254.344/260.119μs，merge13.511μs；
   长B24为346.587/363.192μs、merge13.815μs。Native融合QLI的PMU参考分别为
   248.899/248.300和370.853/370.460μs；不能忽略PTO独立系数、scale提交与merge。
2. 系数任务仅取消空worker已保留：固定stride48，提交数min(48, query组数)。
   长B16/短B24真实编译A/B的8:2均值−1.885%、两档P95下降；短档一次最大值和连续诊断分别保留。
   这是任务提交优化，不能把Score核时略升隐去或称为算术加速。
   [实测、历史区别和诊断](results/csa_coefficient_active_workers_20260929/README.md)。
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
先按Native ProcessVec0的批量读入/乘法优化独立系数任务，再考虑融合；
[结果、Native/pypto-lib差异及泳道](results/csa_coefficient_fused_20260929/README.md)。

## 有效证据与排除方向

- [均衡leaf与部分排序](results/csa_score_balanced_sort_20260928/README.md)：已保留，旧单项基线不与本轮9.2混算。
- [四路Top-K与UB根](results/csa_topk_ub4_20260928/README.md)、[Key L1预取](results/csa_score_key_l1_20260928/README.md)：已保留。
- [编排最大长度复用](results/csa_maxlen_reuse_20260928/README.md)：生成码扫描减少但无明确核内收益，不采用。
- [query排序循环](results/csa_sort_query_loop_20260928/README.md)：代码体积下降，短档核时恶化，不采用。
- [Score矩阵scale广播](results/csa_score_scale_matrix_20260929/README.md)：长档6个TMUL调用点合为1个TCOLEXPANDMUL，
  状态精确、CSA略降，但长B16 Score AIC/AIV约+7%，没有核内收益，不采用。
- [源码入口和版本](DSV4_FLASH_CSA_ASCENDC_REFERENCES.md)：A3 arch22优先，不套用不兼容的arch35能力。
- [当前PTO源码](../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_indexer.py)、
  [Native调用接口](../../vllm_ascend/attention/dsa_v1.py)。
