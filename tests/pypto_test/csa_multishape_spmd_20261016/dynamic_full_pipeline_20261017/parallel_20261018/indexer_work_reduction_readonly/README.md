# Indexer 大额省工方向：有限只读复审

**结论：本轮不提出新实现候选。** 当前接口、逐 bit 合同和已有证据下，尚未找到既未尝试、又有充分依据减少关键链大额工作的路径。已覆盖的高价值方向应保持排重，不再重开 G/leaf/panel/预取扫描；这不是证明所有未来算法都不可能优化。仅检查源码、旧结果并复算算术，未生成候选、未编译或上卡；进行中的 DQ 连续区间源码未改。

## 真实大头与可消除量

对正式 `9d5327e5` 读取五个 Indexer 文件及 policy，与已有593快照逐文本相同，见 [evidence.json](./evidence.json)。因此原数学/计划的工作量可沿用；compact前移和前驱时序已改变，旧阶段耗时不能当当前版本性能。

以真实长 G6/N64/1024 为例，当前 [Score:342](</data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/indexer_score_stages.py:342>) 已把 QK/current 与 WS/previous 交错，FIX 的 FP16/ReLU 舍入留在两者之间：

| 每1024候选块的工作 | 数量 | 本轮判断 |
| --- | ---: | --- |
| QK：384×128×1024 INT8 MAC | 50,331,648 | 有效 query/head/candidate 的 dot 都在用；无已证明可删子集 |
| WS：16×384×1024 FP16 MAC | 6,291,456 | 有效对角加权乘积393,216，零项/填充最多可删5,898,240；但不能直接改归约 |
| 现最终score交接 / 原始FP16 QK交接 | 24 / 768 KiB | 把WS迁AIV会变成32倍载荷，单AIV原始输入384 KiB超过UB |
| 2048段排序 | 64个sort32 run＋3级merge | 已保留UB段和根；不能把整个历史当一次全排序 |

现有128K/B16/S6计划每AIC22步，即约11.073亿 INT8 QK MAC、1.384亿 FP16 WS MAC，各352次matmul。WS的所有MAC只占二者**计数**的11.11%，可去零项占10.42%；两种dtype、硬件吞吐与流水不同，这不是时延上界。QK规模最大，但省略dot需要可证明的候选上界；当前没有原样可用的紧上界或命中率证据。ANN、减少heads/候选、改量化或归约树均不满足本次 exact 合同。

## 已有高价值方向，不再换名重试

| 方向 | 证据与阻碍 |
| --- | --- |
| 删WS零项、归约迁AIV | [ws_aiv_readonly](../ultra_indexer_20261018/ws_aiv_readonly_20261018/README.md)：当前FP16系数/FP16 QK后Cube FP32累加，Vector乘积与归约树未证明逐bit；每步额外读写合计1488 KiB。旧分task由Vector迁Cube的实际历史也不支持直接逆转；不是声称当前反向消融已测失败。 |
| AIV整理Key，AIC大块读取 | [设计](../ultra_indexer_20261018/key_prefetch_design/KEY_PREFETCH_DESIGN.md)及[A3传输审计](../ultra_indexer_20261018/key_transport_audit/README.md)：现能力仍经GM，长B16额外132 MiB逻辑交接；不是删搬运。旧Scalar提交11.81 μs/704次来自静态旧版，不能据此宣称当前页解析主导。 |
| 跳过短尾多余矩阵 | [INDEXER_SHORT_TAIL_EXPERIMENT](</data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_SHORT_TAIL_EXPERIMENT.md>) 已对剩1..64候选只算N64；matmul32→2，B16/B24完整PASS但未采用。旧模型B16最忙核不减、B24面板528→513；这部分大百分比局部节约不能推广到整个Score。 |
| 提高query重用、减少面板/叶 | [query_group_audit](../ultra_indexer_20261018/query_group_audit/README.md)已覆盖G4/3/6、N128、双Acc、N80、16K叶及owner；N80虽32→26次matmul却牺牲Right并存，旧完整无收益。G6小batch已在正式中，不是新方向。 |
| 删score私有GM往返 | [半叶UB候选](../ultra_indexer_20261018/README.md)已完整测B16/B32，B32局部省工变成更多READY等待，未采用。长balanced路径本来已把段与累计根留UB（Score431–432行），没有同一往返可再删。 |
| 最终根前缀认证 | [final_root_prefix_readonly](../ultra_indexer_20261018/final_root_prefix_readonly_20261018/README.md)：真实仅4/5/6根、1–2次merge，每query16/20/24 KiB；128前缀R4需各根恰贡献128，R5短根只有1个有效项。接口可读UB阈值，但认证/回退无命中率和收益证据，已收口。 |

## 最后一项未实测想法也不足以推进

热段2048分支现在 [Score:560](</data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/indexer_score_stages.py:560>) 做sort32、block_len64/256/1024。把末级改为四个512根的 `exhausted=True`，即便保持输入次序和前512 exact，也只是“任一输入耗尽”，不能要求输出512。理想情况下末级从2048降到512；前三个merge级总输出计数最多减1536/6144=25%，sort32输入完全不减。四根分数交错时，首根耗尽已输出2045项，几乎没有省工；硬件批量写出还不能按理想逐项计数直接计时。

此前最终根已有四路耗尽实现，当前API没有输出数量上限，见同一 TopK 只读报告的 PyPTO/PTO-ISA 精确路径。没有当前score分布的有效证据支持段内耗尽常常提前；重新组织排序、压缩候选或radix筛选还必须另证原 sort32/mrgsort 的同分优先序。**本轮否决这一未测想法，不把它作为候选安排编译。**

## 关键核与证据限度

[CURRENT_PHASE_DIAGNOSTIC](</data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/CURRENT_PHASE_DIAGNOSTIC.md>) 的593长B16三轮，Score AIC原位FFTS中位16.05/16.60/16.15 μs，AIV107.90/106.69/107.63 μs；AIV在等Score READY，AIC在等ACK。短B32半叶UB对照也表现为AIV省工主要转成READY等待。它们支持优先关注供数AIC链，不能证明其中纯matmul、页解析或DMA各占多少，更不能把这些等待直接扣成可省时间。

当前没有足够分项证据给出任何新方案的微秒收益上限。上述大额算法/搬运路线已分别被 exact、传输成本、现有驻留或负历史约束；本轮到此收口，不为30%目标制造没有依据的参数试验。

复算只读证据：`python readonly_evidence.py`。归档本目录三个文件即可；没有新的大生成物或设备产物。
