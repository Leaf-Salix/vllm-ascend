# Indexer：当前PTO与最新AscendC的差异

更新：2026-09-28。完整性能基线为CANN9.2/e33d842a七档，后续3b27c7fd只改WO_A布局及告警，Indexer代码未变。
旧V7/V10及入口cache拆分结论移出当前说明；实验历史见[验证日志](DSV4_FLASH_CSA_VALIDATION_LOG.md)。

## 实际接口与主路径

Native调用 `_indexer_qli`，消费原分页cache、页表、长度、query及head权重，返回Top-K索引。
PTO保持Native分配与更新：一个可写物理cache根入参，在编排里建立零复制字节/Key视图，
按页表直接加载Key和scale。没有入口历史复制，没有外部拆分/写回，不把GM视图当成连续请求历史。

压缩历史至少2048行走FP16片上QK及第二次Cube head加权规约；更短历史保留Vector路径。
长短选择及2/3/6 query策略均在同一份 `decode_indexer.py` 内，调用者不选历史版本。
以下为S6七档的实际分支，Q表示有效query总行数，worker数为24 AIC/48 AIV。

| 历史 | 档位 | 每组query | QK N面板 | Score tile | Key预取 |
| --- | --- | ---: | ---: | ---: | --- |
| 128K | B4 | 2 | 128 | 1024 | 独立L1双槽，连到L0B |
| 128K | B8 | 3 | 128 | 1024 | 独立L1双槽，L0B按需读取 |
| 128K | B16 | 6 | 64 | 1024 | 独立L1双槽，连到L0B |
| 8K | B16/B32 | 2 | 128 | 768 | 无长档预取 |
| 8K | B24/B40 | 6 | 64 | 768 | 无长档预取 |

长档按Q<48、48≤Q<96、Q≥96选择2/3/6 query；短档比较整请求与双query最忙核的工作量，
整请求覆盖24worker且工作量增幅不超过25%时选S6，不硬编码单个batch标签。
长档Score保留sync_start与禁止early_resolve，短档没有照搬该开关。

## 与最新ops-transformer A3路径对照

参考ops-transformer **28f40354** 的 `quant_lightning_indexer_v2/op_kernel/arch22`，
本地运行Native仍为release custom二进制，源码参考不等于已替换Native算子。

| 环节 | 最新Native A3策略 | 当前PTO | 剩余问题/证据 |
| --- | --- | --- | --- |
| query/Key复用 | L1 query最多256行，L0按128行M面板；同一Key L1面板跨M子块复用 | 2/3/6 query组，QK为M128/192/384；长S6改N64控制L0C占用 | 分组数相似不代表L1/L0流水等价，旧4+2直接替换已退化 |
| Key读取 | ProcessQk首个M子块加载Key，末个M子块后释放；独立buffer事件 | 长档独立Key L1槽及提前一个面板预取，S6/双query连到L0B，三query受L0B容量约束 | 长B4/B8核时仍高，需看生成指令的等待和重复move，不能只数逻辑读取字节 |
| QK→WS | FIXPIPE把QK INT32缩放转FP16入L1；Cube完成head加权 | 同样采用FP16 QK和Cube WS | 此项已经采用，不再把旧Vector head规约写成当前差异 |
| scale | Vector按物理页加载scale并解量化score | Vector直接从原cache物理页取scale | 两侧均有分页读取，不存在Native恒为单次连续scale读取的依据 |
| 本地Top-K | 2048候选分段排序，UB中维护Top512并滚动合并 | 每AIV收集half-leaf score，再按512/1024/2048/2560/3072/4096排序；已减尾块padding | Native逐段UB累计与当前half-leaf GM交接仍不同 |
| 分片平衡 | metadata按工作成本切S1/S2并给最终归并核分工 | 长档按query组与24worker平衡leaf；B16从8/8/8/8/1分成6/6/6/5/5/5个tile | 最忙核下降已保留；新增root数量和AIV排序成本单列 |
| 最终归并 | LocalTopK/Merge/MS式四路归并，可在同融合kernel结束 | 四路归并与UB累计根已采用，但仍独立merge task | 固定tie/量化规则需保持；任务融合是后续调度/结构调整，不能仅凭少一个任务声称收益 |

PTO Native cache适配没有device重排，但页内Key/scale错位、动态有效长度和更新依赖仍需正确表达。
pypto-lib的连续私有cache、量化顺序、函数分界仅供PTO写法参考，不能消除Native ABI约束。
当前实际核时与七档Native对照见[核内差距](DSV4_FLASH_CSA_INCORE_NATIVE_GAP.md)；
[完整四窗口、最慢核与包络](results/csa_cann92_incore_seven_20260928/RESULTS.md)保留全部原始读数。

## 下一步按依赖推进

1. 先完成正在准备的WO_A NZ两档集成对照，建立当前主线的完整CSA关键链，避免在旧WO_A布局上调度。
2. 针对长B4/B8，逐项核对Native的query L1驻留、Key跨M子块复用与PTO生成的L1/L0同步。
   新候选必须改变真实搬运或等待，不再做无依据的query分组扫描；先长短B16，加受影响B4/B8必要项。
3. 研究2048候选滚动Top-K能否在保留固定tie规则下消除half-leaf score GM落地。
   明确UB容量、两AIV分工与跨轮累计依赖后才实现；历史完整4096 Score UB驻留因gather/copy退化，不能原样重试。
4. 局部核内收益通过必要状态/图检查即保留，CSA与P95分别记录；更换候选划分或舍入才按算术差异单列验收。
5. 核内阶段之后，再按当前DFX处理独立归并、系数依赖、准入和物理核分派。短B16/B32/B40同核两份仍有记录，
   但已测短Score sync_start、Sparse sync_start都没有综合收益，不能把开关当作已证明的修复。

## 有效证据与排除方向

- [均衡leaf与部分排序](results/csa_score_balanced_sort_20260928/README.md)：已保留，旧单项基线不与本轮9.2混算。
- [四路Top-K与UB根](results/csa_topk_ub4_20260928/README.md)、[Key L1预取](results/csa_score_key_l1_20260928/README.md)：已保留。
- [编排最大长度复用](results/csa_maxlen_reuse_20260928/README.md)：生成码扫描减少但无明确核内收益，不采用。
- [query排序循环](results/csa_sort_query_loop_20260928/README.md)：代码体积下降，短档核时恶化，不采用。
- [源码入口和版本](DSV4_FLASH_CSA_ASCENDC_REFERENCES.md)：A3 arch22优先，不套用不兼容的arch35能力。
- [当前PTO源码](../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_indexer.py)、
  [Native调用接口](../../vllm_ascend/attention/dsa_v1.py)。
