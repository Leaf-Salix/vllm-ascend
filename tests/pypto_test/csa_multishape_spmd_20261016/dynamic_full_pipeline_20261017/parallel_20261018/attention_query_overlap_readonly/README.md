# 已有 Top-K 的跨 query 压缩 KV 交集：只读数据审计

**仅能确认随机 fixture 的复用分布，不能证明或否定真实服务复用价值。** 本轮没有实现候选、编译、上卡、读取模型权重或大 KV payload，也没有做 hash。当前证据不足以据此启动跨 query 共享实现。

## 找到什么，未找到什么

10/18 完整 S6 成功矩阵保存了数值/保护区摘要，没有保存实际 Top-K 数组；本次采用两份 9/30 历史 S6 的已保存 PTO Top-K。两份同次报告均为 `MEASURED`，第 4 层正式 W8A8 权重，报告 scope 明示 synthetic history。保留的 `compiled_case.py:114,134` 调用正式模型和 `make_fixture(...,seed=1024)`；后者 `hidden=torch.randn`，历史 KV 为 normal/random 填充。因此这些不是自然语言请求经过整网形成的 hidden/KV。

优先检索了 canonical `tests/pypto_test/results` 的真实 P→D bank 和 CSA args/reference 快照：bank 有 prefill cache/manifest；找到的 args 快照注明 constructed/synthetic fixture，未发现同时具备这两档 S6 实际 decode Top-K 与相应 metadata 的真实服务层快照。没有读取 bank 的 cache.safetensors payload，未新提交采集。检索范围有限，不声称机器其它目录绝无此类文件。

| 档位 | 读取的原保存文件 | Top-K shape | 仅读取 Top-K payload |
| --- | --- | --- | ---: |
| H131072/B16/S6 | [states.pt](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_accept_20260930/h131072_b16/pto_a/states.pt) | [96,512] | 196608 B |
| H8192/B32/S6 | [states.pt](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_accept_20260930/h8192_b32/pto_a/states.pt) | [192,512] | 393216 B |

脚本使用受限 pickle 元数据解析和 ZIP 单 storage 读取；没有调用 torch.load、更没有 materialize 整份数百 MB 的 states。除小型 data.pkl/byteorder，实际只读取上述 196608+393216=589824 B 索引。所有其它 storage 只解析 shape/dtype/stride/ID。

## 逻辑实测与物理推导的边界

`states.pt` 只保存输出 Top-K 和若干状态/cache，没有 Q/L/positions/页表数组。**直接实测的是逻辑索引交集**。物理结果由当时保留的 fixture 源构造合同推导，不能写成从设备页表数组测量所得：

- Q[r]=6r，L[r]=H+6，position[q]=H+(q mod6)。
- compressed 每页 32 行、压缩比 4；每请求 C=ceil((H+6)/128)+1 页。页号 `1+r*C+(C-1-col)%C` 是请求内反序置换，页 0 留作保护。
- 对候选 v，先按原 causal 条件保留 `0<=v<floor((position+1)/4)`；物理行推导为 `32*page[r,v//32]+v%32`。本次值远小于 FP32 整数精度边界，原 plan 的浮点 ±1 往返不会改变索引。
- 脚本显式检查全页列：请求内无别名、请求间页集合不交；实际保存 cache descriptor 为长档 `[16417,32,1,512]`、短档 `[2113,32,1,512]`，与 C=1026/66 完全相符。
- 原 report 的 `metadata.positions`、compressed Q/L/page table 只读比较均 PASS、0 mismatch，页表实际 shape 同为 `[16,1026]` / `[32,66]`。这些是当时比较结果，不是今天重新读取的 metadata 数值。
- 两份真实 Top-K 每 query 均 512 项，全部非负、未越 causal/page 范围、无重复。因此在同请求的单射页映射合同下，逻辑交集恰等于推导物理交集；跨请求相同逻辑整数则不能等同相同物理行。

详细来源、读取字节、原 descriptor、原只读检查和统计均在 [overlap.json](overlap.json)；fixture 原文摘录见 [fixture_source_excerpt.py.txt](fixture_source_excerpt.py.txt)。

## 同请求相邻 S6 query

每请求取 5 对 (q,q+1)，不跨请求。集合保留实际有效候选，统计唯一行交集；均值比例以 512 为分母。

| 档位 | query 对 | 交集均值 / 中位 | min～max | 平均重合比例 | 同序号128块交集均值 / 中位 |
| --- | ---: | ---: | ---: | ---: | ---: |
| H131072/B16 | 80 | 27.3375 / 25 | 5～85 | 5.339% | 1.8094 / 1 |
| H8192/B32 | 160 | 153.1625 / 153 | 69～241 | 29.915% | 10.7453 / 10 |

下面每格为相邻两 query 的平均交集，行是前 query 的压缩块 1..4，列是后 query 的压缩块 1..4；每块原样取 128 个排序位置，**没有重新按索引排序**。矩阵 16 格相加等于整 512 项平均交集，不把同块交集冒充全部共享行。

**H131072/B16**

| 前块 \ 后块 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.875 | 2.000 | 1.688 | 2.025 |
| 2 | 2.250 | 1.762 | 1.400 | 1.462 |
| 3 | 1.863 | 1.525 | 1.337 | 1.312 |
| 4 | 1.850 | 1.312 | 1.413 | 1.262 |

**H8192/B32**

| 前块 \ 后块 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 15.575 | 11.394 | 9.306 | 8.412 |
| 2 | 11.488 | 9.894 | 8.919 | 8.531 |
| 3 | 9.100 | 8.963 | 9.037 | 8.450 |
| 4 | 8.281 | 8.925 | 8.412 | 8.475 |

长档仅 579/2187≈26.5% 的共享行落在相同块号；短档为 6877/24506≈28.1%。即使复用整个前 query 的行集合，仍需处理不同 score 排名位置；当前同块三槽不是直接按行号可查询的 cache。两者均不能据交集直接跳过数学、改变 Top-K/bias 次序或去重掉乘法重数。

## 当前 q%24 分核的相邻 query

把这两份历史数组应用到当前正式 Attention 的 owner 公式，仅比较同核前后相邻 query `(q,q+24)`：长档 72 对、短档 168 对。S6 下二者属于相隔 4 个请求的同一 query 偏移，fixture 请求间的页集合完全不交。

| 档位 | 逻辑整数交集均值 | 推导物理行交集 | 各128块物理交集 |
| --- | ---: | ---: | ---: |
| H131072/B16 | 12.1667 | 所有对均0 | 全部4×4均0 |
| H8192/B32 | 125.6250 | 所有对均0 | 全部4×4均0 |

这里的物理 0 是 fixture 独占页构造与当前 owner 公式的联合推导，不能外推到真实服务的 prefix 共享页。反过来，短档同请求约30%的重合分布在不同 AIC 组间，也不是现有单核 L1 三槽可以免费复用的证据。

## 对后续设计的判断

本数据可以排除“这些 fixture 在当前 q%24 owner 下，上一条同核 query 的 KV 可以大量直接复用”的前提；它没有否定真实请求内相关性或 prefix sharing。短 fixture 的跨组重合存在，但单 query 已由唯一 AIC 组消费，复用需要额外地址查找/重排、跨组可见性和多消费者 ACK/回收合同。现有 AIC Mat 上界475136 B（其中 KV三槽384 KiB），AIV还承载在线状态和半块采样，容量与协议仍按此前 [多消费者只读审计](../attention_kv_multiconsumer_readonly_20261018/README.md) 约束，不能把交集直接换算成节省的DMA或µs。

在没有真实服务同层/同调用 Top-K 与 Q/L/page arrays 之前，本轮不提出实现或性能任务；本结果保留为随机 fixture 的明确数据边界。

复现只读分析：

```bash
python3 /data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/attention_query_overlap_readonly_20261018/analyze_saved_topk.py
```

脚本不会导入 torch/PyPTO、编译、访问 NPU 或读取权重；仅写本目录小 JSON 和源码摘录。

归档小脚本首次生成路径清单时漏定义仓库变量，报告和分析JSON已写出；仅补齐路径清单，没有重读张量或改变结果。
