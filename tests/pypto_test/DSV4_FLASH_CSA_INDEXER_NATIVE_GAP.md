# Indexer：当前PTO与最新AscendC的差异

更新：2026-09-29。当前实现c93ec723；完整旧矩阵仍为CANN9.2/e33d842a，
局部长B4/B8 S6、B16分段排序/UB根及近期组合长B24的状态/图和核时验证已补齐。
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
| Query/系数驻留 | ComputeMm1只在isFirstS2InnerLoop加载Query及Weight，后续S2块复用L1 | 每个leaf重新加载Q和系数，再移至L0A；leaf内部多个N面板已复用 | B24恰为24个S6组，可研究固定worker处理同一组的全部leaf并保持Q/系数驻留；不等同已验证收益 |
| Key读取 | ProcessQk首个M子块加载Key，末个M子块后释放；独立buffer事件 | 长档独立Key L1槽及提前一个面板预取，连到L0B | B4/B8 AIC均值已降至71.036/137.435μs，后续按当前核时继续看等待与重复move |
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
[完整四窗口、最慢核与包络](results/csa_cann92_incore_seven_20260928/RESULTS.md)保留全部原始读数。

## 下一步按依赖推进

1. WO_A NZ长短B16及8K/B40集成对照已通过并取得核内收益；新128K/B24的Score AIC/AIV为383.890/412.053μs，
   merge14.260μs，四窗口均每核一份；完整CSA1362.396μs，详见[B24报告](results/csa_b24_cann92_20260928/RESULTS.md)。
2. 长B4/B8 S6已保留：相对d8627207，CSA−2.108%/−3.145%，Score AIC/AIV明显下降；
   短B24 CSA+3.291%单列，三档8:2 CSA−1.443%，状态/图/保护区通过。
   [实测与Native/pypto-lib差异](results/csa_small_long_s6_20260929/README.md)。
3. [2048分段排序与UB中间根](results/csa_stream_root_ub_20260929/README.md)已保留：长档AIC/AIV约−4%，
   长短8:2核时受益、CSA+0.818%单列；长B8已随S6补齐，长B24近期组合也已通过。
   B24 Score AIC/AIV383.660/411.575→349.409/366.048μs，merge16.852→13.960μs，
   CSA1376.883→1312.313μs；同配置Native编译半层1396.274μs，详情见[B24补测](results/csa_b24_integrated_20260929/RESULTS.md)。
   仍有缩放后score的GM中转；Native也有Cube WS结果GM交接，不能把二者混称为同一种复制。
   历史完整4096 Score UB驻留因gather/copy退化，不能原样重试。
4. 局部核内收益通过必要状态/图检查即保留，CSA与P95分别记录；更换候选划分或舍入才按算术差异单列验收。
5. 核内阶段之后，再按当前DFX处理独立归并、系数依赖、准入和物理核分派。短B16/B32/B40同核两份仍有记录，
   但已测短Score sync_start、Sparse sync_start都没有综合收益，不能把开关当作已证明的修复。

后续核内候选的具体约束：长B24当前走query-major枚举，32769候选产生5个leaf，
一个worker会轮换处理5个不同query组，因而不能直接把Q/系数加载移出leaf循环。
若改成每worker固定一组，理论上可消除该组后4次的48KiB Q和12KiB系数读取，
但必须同时改变两侧AIC/AIV枚举、保持根槽位置，并对不均匀请求长度保留原负载平衡策略。
先确认生成代码的驻留/同步，再做长B24和短档定向对照；不为“少读了字节”提前记收益。
此前只复用编排最大长度的失败候选不等同这一跨leaf驻留策略，不能混为同一次优化重试。

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
