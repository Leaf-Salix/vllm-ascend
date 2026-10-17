# 正式668长历史Indexer内层：增量只读复审

**结论：暂无有充分新证据、值得实施的大额省工候选。** 本轮不重开旧方案，也不把信号或等待次数减少当性能收益。没有修改算子、编译、设备作业或新工作量扫描。

当前正式668的 `indexer_score_stages.py`、`indexer_topk_stages.py`、`indexer_group_stages.py`、`indexer_publish_stages.py`、`indexer_query_stages.py` 与593快照逐文本相同。`policy.py` 后续只增加QA的M16分档，Indexer的G、panel、score_tile、prefetch、balanced及root选择未改。直接对照见 [evidence.json](./evidence.json)、[policy_context.diff](./policy_context.diff)，可用 `python direct_compare.py` 复核。这里只证明相关算法源码未变，不把旧性能视为正式668性能。

## 现有内层和排重依据

以下源码行号对应正式668冻结目录 `../ultra_idx_query_quant_owner_20261018/source_reference/`：

- `indexer_score_stages.py:141,165,174,243,342`：Q/系数每叶加载；每1024候选16个N64面板，当前QK与上一面板WS已交错，保留FP16 FIX/ReLU边界。长G6每步50,331,648个INT8 QK MAC及6,291,456个FP16 WS MAC，前者均为实际head/candidate dot。删候选或head没有现成严格上界证明；近似剪枝不符合exact合同。数字复用既有审计，未再建模型。
- `indexer_score_stages.py:431,481,560,583`：长balanced的分数段和累计Top512根已经留UB；不存在短G2候选曾删除的同一scratch GM往返。热段sort32/三级merge仍需保留平分次序；最终根合并仅1–2次，不是对32768候选重新全排序。
- 将WS放AIV会让每步交接从24KiB变768KiB，单AIV原始QK384KiB大于184KiB UB预算；FP16乘积/FP32归约次序也未证明逐bit。此方向已只读否决，不能以AIV等待空隙为收益。
- AIV预整理Key再供AIC虽可用约6.1875MiB既有私有arena及约284KiB局部L1，但长B16额外产生132MiB逻辑Key读写。A3当前gather/TPush仍经GM FIFO，不能声称删掉搬运。旧静态页提交11.81μs/704次不是当前动态瓶颈证据。

精确排重入口（均已有中文结论和原始路径，不在此复制全部历史）：

1. [大额省工审计](../ultra_indexer_work_reduction_readonly_20261018/README.md)：WS、短尾、排序耗尽与最终根前缀认证；主MASTER约16947行已收录。
2. [区间复审](../ultra_indexer_20261018/interval_strategy_reaudit_20261018/README.md)及[query/leaf审计](../ultra_indexer_20261018/query_group_audit/README.md)：G4/3/6、N128/双Acc/N80/N64roll、16K叶、B28旋转、短G2 UB/prefetch等，不重做已覆盖模型。
3. [Key设计](../ultra_indexer_20261018/key_prefetch_design/KEY_PREFETCH_DESIGN.md)、[A3传输层核对](../ultra_indexer_20261018/key_transport_audit/README.md)、[WS搬迁核对](../ultra_indexer_20261018/ws_aiv_readonly_20261018/README.md)。

## HC_pre没有带来尚未使用的新内层依据

当前 `vllm_ascend/ops/pypto/hc_pre.py:224–238` 复用加宽输入供RMS和投影，并显式交错K256 Left/Right。Indexer的输入已经是INT8 Key，不能省掉同类BF16→FP32重复转换；其Query/系数已跨score步常驻，QK/WS也已有交错。HC的 `aic_gather` 仍经GM，主MASTER约14657行及上面的A3核对已明确这一限制。HC末段按token重新分核不能消除Indexer的有效dot或排序输入，也不能照搬HC的跨K归约：这里有固定FP16/ReLU和TopK平分语义。

旧593阶段时钟支持关注Score AIC供数链，但无法分离当前668的矩阵、页解析、DMA和等待读数，不能给新方案的微秒收益上限。更改关键数学、片上驻留或数据所有权若没有新增证据，只会重复上述风险；本轮到此收口。归档本目录文本文件即可，无新增大产物。
