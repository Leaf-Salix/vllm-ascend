# HCA attention后续候选与混长参考核对

共享SDK csa-shared-20261022-logical-codegen；以下均为已有结果，不新增设备测试。
本轮已收口，用户再次要求结束后停下；归档提交后暂停。所有新候选均未采用，正式算子仍为37bbd0f3c的区间策略。

| 候选/场景 | 控制μs | 候选μs | 性能提升 | 更快配对 | Native输出 |
|---|---:|---:|---:|---:|---|
| attention_online_b16_v1 | 531.32 | 524.38 | +1.32% | 4/7 | PASS |
| attention_online_b28_v1 | 746.44 | 748.60 | -0.29% | 4/7 | PASS |
| attention_online_n64_b16_v1 | 526.54 | 534.10 | -1.42% | 3/7 | PASS |
| attention_pair512_b16_v7 | 518.04 | 593.60 | -12.73% | 0/7 | PASS |
| attention_pair512_b28_v7 | 749.40 | 896.36 | -16.40% | 0/7 | PASS |
| attention_pair512_wait0_b16_v8 | 512.88 | 605.06 | -15.23% | 0/7 | PASS |
| attention_pair512_wait0_b28_v8 | 756.52 | 867.38 | -12.78% | 0/7 | PASS |
| attention_pair512_score_ack_b16_v9 | 528.90 | 593.66 | -10.91% | 0/7 | PASS |
| attention_pair512_score_ack_b28_v9 | 737.94 | 820.36 | -10.05% | 0/7 | PASS |
| attention_pair512_early_b16_v10 | 510.24 | 563.40 | -9.44% | 0/7 | PASS |
| attention_pair512_early_b28_v10 | 756.10 | 836.06 | -9.56% | 0/7 | PASS |
| attention_gather_mask_b16_v1 | 520.48 | 517.60 | +0.56% | 3/7 | PASS |
| attention_gather_mask_b28_v1 | 772.92 | 762.22 | +1.40% | 7/7 | PASS |
| attention_gather_policy_h131072_b24_v2 | 692.80 | 692.04 | +0.11% | 5/7 | PASS |
| attention_gather_policy_h131072_b28_v2 | 769.52 | 768.50 | +0.13% | 5/7 | PASS |
| attention_gather_policy_h262144_b16_v2 | 659.04 | 655.28 | +0.57% | 2/5 | PASS |
| attention_gather_policy_h131072_b16_v2 | 517.40 | 521.16 | -0.72% | 2/5 | PASS |
| attention_gather_policy_h8192_b32_v2 | 597.52 | 612.46 | -2.44% | 1/5 | PASS |
| attention_gather_dispatch_h131072_b28_v3 | 757.08 | 764.48 | -0.97% | 3/7 | PASS |
| attention_gather_dispatch_h8192_b32_v3 | 584.24 | 597.88 | -2.28% | 1/5 | PASS |
| attention_gather_dispatch_h131072_b24_v3 | 673.54 | 681.60 | -1.18% | 2/7 | PASS |
| attention_gather_dispatch_h262144_b16_v3 | 664.48 | 663.82 | +0.10% | 2/5 | PASS |
| attention_direct_kv_b16_v3 | 504.08 | 641.22 | -21.39% | 0/7 | PASS |
| attention_direct_kv_b28_v3 | 745.84 | 961.44 | -22.42% | 0/7 | PASS |

N64/H8的控制是累计max候选，不能把该增量当作相对正式算子的收益。
全部严格Native cache/state检查尚未通过，任务正常退出或输出PASS不等于整层验收。

## 尚未结束的设备任务

无。已提交设备任务全部终态；不再启动下一轮，按用户要求暂停。

## 混长padding参考的纠正

旧Native输入Q=[15,64,512]而cu_q包含padding至24，产生无效参考或MTE错误。
v5只裁metadata导致RoPE维度不匹配；v6同时裁metadata、hidden、position和output，
PTO仍测试原T24/B8容量。v6不再出现MTE，输出剩1个超差[12,0,17]，属于history=0/Q1。
给Native Q或Native cache单独替换时输出均通过；两者都用PTO时复现该项。
Native attention输入PTO输出尾部逐bit一致，定位剩余误差在前置Q/KV的组合而非O尾部。
源码复核：公共HCA mHC已采用(y0+y1)+(y2+y3)，也有BF16舍入，不能声称这两项遗漏。

CPU编译、输入快照审计和每项失败详情见JSON及evidence/。快照和编译产物只本地保留，
清单记录路径与大小，不做hash；不把失配参考的旧失败改写为PASS。
