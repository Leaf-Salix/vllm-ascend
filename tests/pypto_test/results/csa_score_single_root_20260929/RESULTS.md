# 长档按query分工、UB连续排序及单根发布

完整八类状态零容差：PASS。

同卡CANN9.2/mode2/atomic0/det0，inplace_pass=True；5预热20次正式事件，单位μs。

| 档位 | CSA基线→候选 | 变化 | P95 | max | indexer_score_topk_native_pair_aic | indexer_score_topk_native_pair_aiv | indexer_topk_query_merge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128K/B16 | 1021.805→976.950 | -4.390% | 1035.800→998.520 | 1039.260→1007.640 | 247.420→226.253 | 253.189→229.177 | 12.296→9.696 |
| 8K/B24 | 929.231→956.560 | +2.941% | 947.300→981.440 | 963.000→1002.480 | 33.812→30.781 | 50.081→44.526 | 10.686→11.130 |

长短8:2 CSA变化：-2.924%；核内变化：indexer_score_topk_native_pair_aic -8.637%、indexer_score_topk_native_pair_aiv -9.806%、indexer_topk_query_merge -16.087%。

核时包含核内流水等待，独立DFX不能与正式CSA样本直接相减。
只改变长S6的候选组织与Top-K结构；FP16/Cube缩放和量化不变。同分顺序可能不同，仍先用八类完整状态零容差揭示差异，不自动放宽验收。
采用还需依据四窗口分布判断真实核内收益，状态失败则禁止采用；不是Native或模型token/DSpark验收。
[四窗核时及原样本](summary.json)、[原始解析与完整状态](evidence.json)。
