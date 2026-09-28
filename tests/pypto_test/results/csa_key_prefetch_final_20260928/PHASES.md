# EP16入场分项

同一正式10步的可选主机分项，不从设备forward中扣减；不是kernel计时。

准备段含已有设备同步/忙等，墙钟与线程CPU之差不能单独证明OS抢占。gap是两个既有标记之间的范围，不能自动归因为单一函数。builder父子区间嵌套，不能相加；重复调用分别编号。

下面仅展开设备相对入场异常>2ms的rank；所有正式样本仍用于性能统计。

| 档位/侧/step/rank | 设备迟到ms | 主机段 | 当步墙钟/线程CPU ms | 同rank墙钟中位ms/出现次数 |
| --- | ---: | --- | ---: | ---: |
| 128K/B4/native/15/11 | 3.407 | input_sync | 0.031/0.019 | 0.032/10 |
| 128K/B4/native/15/11 | 3.407 | state_update_gap | 0.211/0.211 | 0.195/10 |
| 128K/B4/native/15/11 | 3.407 | inputs | 23.787/23.728 | 23.640/10 |
| 128K/B4/native/15/11 | 3.407 | batch_coordination | 7.212/0.299 | 4.171/10 |
| 128K/B4/native/15/11 | 3.407 | attention_metadata | 4.179/4.167 | 4.103/10 |
| 128K/B4/native/15/11 | 3.407 | preprocess | 0.047/0.047 | 0.045/10 |
| 128K/B4/native/15/11 | 3.407 | forward_context_gap | 0.245/0.246 | 0.227/10 |
| 128K/B4/native/15/11 | 3.407 | observer_prepare | 0.014/0.019 | 0.014/10 |
| 128K/B4/native/15/11 | 3.407 | event_and_submit | 0.251/0.245 | 0.238/10 |
| 128K/B4/native/15/11 | 3.407 | execute_to_input_sync_gap | 0.028/0.027 | 0.028/10 |
| 128K/B4/native/15/11 | 3.407 | inputs_to_coordination_gap | 0.033/0.032 | 0.034/10 |
| 128K/B4/native/15/11 | 3.407 | coordination_to_metadata_gap | 0.152/0.151 | 0.113/10 |
| 128K/B4/native/15/11 | 3.407 | metadata_to_preprocess_gap | 0.134/0.123 | 0.117/10 |
| 128K/B4/native/15/11 | 3.407 | metadata_builder_g0_a0_build | 1.193/1.180 | 1.159/10 |
| 128K/B4/native/15/11 | 3.407 | metadata_builder_g0_a0_build_decode_metadata | 0.753/0.753 | 0.707/10 |
| 128K/B4/native/15/11 | 3.407 | metadata_builder_g0_a1_build | 0.304/0.304 | 0.298/10 |
| 128K/B4/native/15/11 | 3.407 | metadata_builder_g0_a1_build_decode_metadata | 0.165/0.165 | 0.161/10 |
| 128K/B4/native/15/11 | 3.407 | metadata_builder_g1_a0_build | 0.389/0.388 | 0.388/10 |
| 128K/B4/native/15/11 | 3.407 | metadata_builder_g1_a0_build_decode_metadata | 0.253/0.254 | 0.253/10 |
| 128K/B4/native/15/11 | 3.407 | metadata_builder_g2_a0_build | 0.388/0.388 | 0.384/10 |
| 128K/B4/native/15/11 | 3.407 | metadata_builder_g2_a0_build_decode_metadata | 0.252/0.252 | 0.247/10 |
| 128K/B4/native/15/11 | 3.407 | metadata_builder_g3_a0_build | 0.293/0.293 | 0.290/10 |
| 128K/B4/native/15/11 | 3.407 | metadata_builder_g3_a0_build_decode_metadata | 0.153/0.153 | 0.156/10 |
| 128K/B4/native/15/11 | 3.407 | metadata_builder_g4_a0_build | 0.285/0.285 | 0.285/10 |
| 128K/B4/native/15/11 | 3.407 | metadata_builder_g4_a0_build_decode_metadata | 0.156/0.156 | 0.154/10 |
| 128K/B4/native/15/11 | 3.407 | metadata_builder_g4_a1_build | 0.285/0.285 | 0.280/10 |
| 128K/B4/native/15/11 | 3.407 | metadata_builder_g4_a1_build_decode_metadata | 0.153/0.153 | 0.153/10 |
| 128K/B4/native/15/11 | 3.407 | metadata_builder_g5_a0_build | 0.304/0.303 | 0.299/10 |
| 128K/B4/native/15/11 | 3.407 | metadata_builder_g5_a0_build_decode_metadata | 0.155/0.155 | 0.155/10 |
| 128K/B4/pto/12/11 | 5.066 | input_sync | 0.033/0.022 | 0.031/10 |
| 128K/B4/pto/12/11 | 5.066 | state_update_gap | 0.215/0.215 | 0.201/10 |
| 128K/B4/pto/12/11 | 5.066 | inputs | 21.150/21.070 | 21.929/10 |
| 128K/B4/pto/12/11 | 5.066 | batch_coordination | 8.817/0.313 | 3.750/10 |
| 128K/B4/pto/12/11 | 5.066 | attention_metadata | 4.021/4.003 | 4.018/10 |
| 128K/B4/pto/12/11 | 5.066 | preprocess | 0.044/0.044 | 0.040/10 |
| 128K/B4/pto/12/11 | 5.066 | forward_context_gap | 0.227/0.228 | 0.218/10 |
| 128K/B4/pto/12/11 | 5.066 | observer_prepare | 0.014/0.014 | 0.014/10 |
| 128K/B4/pto/12/11 | 5.066 | event_and_submit | 0.255/0.255 | 0.250/10 |
| 128K/B4/pto/12/11 | 5.066 | execute_to_input_sync_gap | 0.030/0.029 | 0.030/10 |
| 128K/B4/pto/12/11 | 5.066 | inputs_to_coordination_gap | 0.035/0.034 | 0.033/10 |
| 128K/B4/pto/12/11 | 5.066 | coordination_to_metadata_gap | 0.121/0.120 | 0.120/10 |
| 128K/B4/pto/12/11 | 5.066 | metadata_to_preprocess_gap | 0.122/0.122 | 0.115/10 |
| 128K/B4/pto/12/11 | 5.066 | metadata_builder_g0_a0_build | 1.107/1.112 | 1.135/10 |
| 128K/B4/pto/12/11 | 5.066 | metadata_builder_g0_a0_build_decode_metadata | 0.700/0.700 | 0.698/10 |
| 128K/B4/pto/12/11 | 5.066 | metadata_builder_g0_a1_build | 0.298/0.298 | 0.292/10 |
| 128K/B4/pto/12/11 | 5.066 | metadata_builder_g0_a1_build_decode_metadata | 0.153/0.153 | 0.155/10 |
| 128K/B4/pto/12/11 | 5.066 | metadata_builder_g1_a0_build | 0.384/0.384 | 0.381/10 |
| 128K/B4/pto/12/11 | 5.066 | metadata_builder_g1_a0_build_decode_metadata | 0.251/0.251 | 0.250/10 |
| 128K/B4/pto/12/11 | 5.066 | metadata_builder_g2_a0_build | 0.407/0.389 | 0.372/10 |
| 128K/B4/pto/12/11 | 5.066 | metadata_builder_g2_a0_build_decode_metadata | 0.253/0.253 | 0.242/10 |
| 128K/B4/pto/12/11 | 5.066 | metadata_builder_g3_a0_build | 0.285/0.284 | 0.276/10 |
| 128K/B4/pto/12/11 | 5.066 | metadata_builder_g3_a0_build_decode_metadata | 0.155/0.155 | 0.150/10 |
| 128K/B4/pto/12/11 | 5.066 | metadata_builder_g4_a0_build | 0.272/0.271 | 0.269/10 |
| 128K/B4/pto/12/11 | 5.066 | metadata_builder_g4_a0_build_decode_metadata | 0.146/0.146 | 0.145/10 |
| 128K/B4/pto/12/11 | 5.066 | metadata_builder_g4_a1_build | 0.267/0.267 | 0.261/10 |
| 128K/B4/pto/12/11 | 5.066 | metadata_builder_g4_a1_build_decode_metadata | 0.145/0.145 | 0.142/10 |
| 128K/B4/pto/12/11 | 5.066 | metadata_builder_g5_a0_build | 0.289/0.288 | 0.288/10 |
| 128K/B4/pto/12/11 | 5.066 | metadata_builder_g5_a0_build_decode_metadata | 0.148/0.148 | 0.144/10 |

全部rank原始标记与分项：[phases.json](phases.json)。正式统计：[model/RESULTS.md](model/RESULTS.md)。
未复现不能证明已修复；本轮8e176285仅验收长B4/B8新预取及短B16控制；保留原始尾部，不扣除EP等待，不将跨轮差额归因于单项。
