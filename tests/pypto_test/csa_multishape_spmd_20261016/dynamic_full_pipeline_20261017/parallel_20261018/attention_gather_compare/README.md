# 同轮原 task 与 d253：压缩 KV gather 有界核对

本轮没有找到可据此推进的新候选。**核心采集算法、逐行 DMA 粒度、候选顺序、GM ring 往返及三槽流控相同**；融合新增的是已检查过的动态请求几何与页保护。实际 UB 布局确有差异，但没有“只在首压缩块触发”的新增串行边证据，不能把首块约 6.2 µs 归因于 cache 或 MTE 队列。

范围限定于已完成的 [attention_guard_elision_h131072_b16_v1](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/attention_guard_elision_h131072_b16_v1)。依据该次 `source_matrix/result.json` 的 kernel_configs，选原 task `_decode_csa_tp1_layer_sh8x34u9` 与未删 guard 的 d253 reference `decode_csa_spmd_mqgq0rdn`；不把已否决候选 `lj39c6t5` 当基线。只查看现成 C++/PTO、前次 tick 结果与历史，不编译、不上卡、不改正式源码/依赖，也没有重复整份 106 操作提取。

小型 [evidence.json](evidence.json) 保存原物及实际 wrapper 的路径/大小、选定片段行号、直接文本匹配、物理分配与地址公式。所核对片段在实际编译 wrapper 中逐字存在；未复制大型生成物。

| 对照点 | 原分 task | 融合 d253 |
| --- | --- | --- |
| 高层采集源 | [decode_sparse_attn_csa.py:482](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa/decode_sparse_attn_csa.py:482) | [冻结 attention_stream_stages.py:261](/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/ultra_attention_guard_elision_20261018/source_before/attention_stream_stages.py:261) |
| 64 行压缩循环，实际 wrapper | [qk_pv_aiv.cpp:1797](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/attention_guard_elision_h131072_b16_v1/build_output/_decode_csa_tp1_layer_sh8x34u9/kernels/aiv/qk_pv_aiv.cpp:1797) | [vector_schedule.cpp:58235](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/attention_guard_elision_h131072_b16_v1/build_output/decode_csa_spmd_mqgq0rdn/kernels/aiv/vector_schedule.cpp:58235) |
| 原始 PTO 行循环 | [qk_pv.pto:509](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/attention_guard_elision_h131072_b16_v1/build_output/_decode_csa_tp1_layer_sh8x34u9/ptoas/qk_pv.pto:509) | [dynamic_csa_full.pto:12927](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/attention_guard_elision_h131072_b16_v1/build_output/decode_csa_spmd_mqgq0rdn/ptoas/dynamic_csa_full.pto:12927) |
| 块末 join/ring store，实际 wrapper | [qk_pv_aiv.cpp:1852](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/attention_guard_elision_h131072_b16_v1/build_output/_decode_csa_tp1_layer_sh8x34u9/kernels/aiv/qk_pv_aiv.cpp:1852) | [vector_schedule.cpp:58328](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/attention_guard_elision_h131072_b16_v1/build_output/decode_csa_spmd_mqgq0rdn/kernels/aiv/vector_schedule.cpp:58328) |

两版每 lane 的压缩半块都顺序遍历 row=0..63。每有效行先读 INT32 index，再读 INT32 page；以 INDEX 计算 `physical_row=32*page+index%32`，TLOAD 一段连续 512 BF16，直接写最终 64×512 UB Tile 的第 row 行。没有 FP32 地址中转，也没有逐行临时 Tile 再 assemble 的拷贝。核心地址可统一为：

```text
index = indices[t, (block-1)*128 + 64*lane + row]
page  = compressed_table[request, index//32]
GM cache byte address = cache_base + 1024*max(32*page + index%32, 0)
UB row byte address   = KV_UB_BASE + 1024*row
GM ring byte address  = ring_base + 1024*(128*(3*group + tick%3) + 64*lane)
```

原 task 的 request=`t//6`；融合从真实 Q/L 验证唯一 owner 与 causal position，且每个有效块重复调用。这是实际动态能力带来的额外工作，不能直接把除法写回动态源码。几何缓存已有同进程负/混合证据；此次两层页 guard 删除也已完整通过但无收益。除此之外，压缩行循环两边均仅有一个循环内 TLOAD 位点、两个间接 Scalar 读取，**循环内没有 wait_flag、pipe_barrier、TSTORE 或 TMOV**。因此没有证据说融合把原来的批量读取退化成逐行等待；原 task 本来也是逐行读取。既有 stage2 独立 rowTile 试验加临时拷贝且未形成两槽，不能再次用于解释或修复本路径。

SWA→首压缩块的物理重用与事件链如下：

| UB Tile 起点 / 大小（B） | 原 task | 融合 d253 |
| --- | ---: | ---: |
| KV 半块 | 69888 / 65536 | 65536 / 65536 |
| running/reset left | 4352 / 32768 | 0 / 32768 |
| running/reset right | 37120 / 32768 | 32768 / 32768 |
| softmax reduce_tmp | 69888 / 16384 | 65536 / 16384 |
| score/exp | 135680 / 16384 | 81920 / 16384 |
| probability | 69888 / 8192 | 65536 / 8192 |
| PV left/right 临时 | 69888 / 32768 | 65536 / 32768 |

两版所有候选块都复用同一 KV UB 区；row 子视图只改变 `base+1024*row`。融合 KV 与运行中 left/right 不重叠；softmax、probability、PV 临时则按阶段复用。原 task 的 score/mi/li 在 KV 区外，融合将其安排在 KV 区内，这是实际差异，不能宣称所有物理分配完全相同；但两版的 reduce/probability/PV 原本就与 KV 区重用，后续仍有完成依赖。现成时钟没有分开这些别名的硬件完成时间，因此不把该布局差异直接定为瓶颈或新增候选。

融合 [原 C++:57516](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/attention_guard_elision_h131072_b16_v1/build_output/decode_csa_spmd_mqgq0rdn/ptoas/dynamic_csa_full.cpp:57516) 的 KV TEXPANDS 受 `MTE3→V EVENT_ID2` 保护；该 event 来自上一拍 [KV TSTORE 后:58313](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/attention_guard_elision_h131072_b16_v1/build_output/decode_csa_spmd_mqgq0rdn/ptoas/dynamic_csa_full.cpp:58313)。原 task [1372](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/attention_guard_elision_h131072_b16_v1/build_output/_decode_csa_tp1_layer_sh8x34u9/ptoas/qk_pv.cpp:1372) 与 [1871](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/attention_guard_elision_h131072_b16_v1/build_output/_decode_csa_tp1_layer_sh8x34u9/ptoas/qk_pv.cpp:1871) 有同一生产消费关系。融合在 [57781](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/attention_guard_elision_h131072_b16_v1/build_output/decode_csa_spmd_mqgq0rdn/ptoas/dynamic_csa_full.cpp:57781) 的六路 V→MTE2 wait 位于 SWA/压缩分支之前，块末 MTE2→MTE3 join 后才写同形半块 ring。首压缩块确实继承前拍 SWA 的 UB 复用约束，但 **block 1..4 均进入同一循环与相同事件链**；没有 block==1 专用的多一轮初始化、额外 Tile、wait 或 ring 搬运。

新 tick 证据可以排除两个更具体的解释，而不能给出 cache/MTE 根因：

1. AIV gather=tick、merge=tick−3。当前 query q 的首压缩块在 `5q+1`；前一 query 的末块 `5q−1` 在 `5q+2` 才 merge/finalpublish/reset，即发生于当前第二个压缩块那一拍。首压缩块先于它一拍。首 query 的 tick=1 根本没有前一 query 最终发布，也同样偏慢。**“前拍 finalpublish/reset 直接使首压缩块慢”与当前执行时序不符。**
2. 四个首压缩块的 tick 为 1/6/11/16，ring 槽为 1/0/2/1；它们并非固定一个槽。原诊断中四个 query 次序均重复首块偏慢，不能单指某个 ring 槽或其固定地址。

旧 [逐 tick 分析](../attention_tick_clock_evidence/ANALYSIS.md) 的单半块中位数为：SWA 0.66–0.68 µs，压缩 block 1 约 6.12–6.20 µs，block 2 约 3.82–3.90 µs，block 3 约 2.64–2.68 µs，block 4 约 2.92–2.98 µs。这些是融合代码内包含清零、几何/地址、局部依赖和搬运发射的区间；没有同轮原 task 的相同逐 tick 插桩，不能把此差异说成“融合独有”。局部 wait_flag 的 Scalar 端点近零不证明 MTE/UB 真正等待近零。

跨 query 重用也没有现成证据。在本次 S6/T96/24 组所有权下，同一组处理 `t=group+24*q`，对应请求 `group//6+4*q`，四次 query 来自四个不同请求；同一请求的相邻 token 分布在不同组。不同请求可能有共享物理 prefix，也可能完全不共享，当前证据没有记录候选物理地址交集。不能把“相邻 query KV 相同”当合同或用本核旧行直接替代；改组所有权和新增跨组重用会扩大协议/内存范围，未得到本轮证据支持。

还需更正此前一条清零说明。当前原 task C++ 的 [1373](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/attention_guard_elision_h131072_b16_v1/build_output/_decode_csa_tp1_layer_sh8x34u9/ptoas/qk_pv.cpp:1373) 在 SWA/压缩分支**之前** TEXPANDS，压缩 else 内 [1773](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/attention_guard_elision_h131072_b16_v1/build_output/_decode_csa_tp1_layer_sh8x34u9/ptoas/qk_pv.cpp:1773) 又 TEXPANDS；PTO 对应 [345](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/attention_guard_elision_h131072_b16_v1/build_output/_decode_csa_tp1_layer_sh8x34u9/ptoas/qk_pv.pto:345) 与 [508](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/attention_guard_elision_h131072_b16_v1/build_output/_decode_csa_tp1_layer_sh8x34u9/ptoas/qk_pv.pto:508)。压缩路径确实执行两次初始化，融合 d253 仅一次。因此旧 `attention_readonly_audit/README.md` 的“两个互斥分支各一次”解释需要纠正；本轮不改旧归档，只在此保留直接证据。该事实没有构成新候选，重复清零与全有效跳零均已有历史试验且明确关闭。

本轮收口：未找到融合独有、可通过一个新动态算法删除的主要采集工作。保留已证实的相同搬运路径、不同元数据/布局和排除性时序证据；不重开 physical row plan、direct L1、MGATHER、跳清零、mi/li 三槽、wait folding、32 行分块或 rowTile stage2，也不据未验证的 cache/队列推测安排新测。
