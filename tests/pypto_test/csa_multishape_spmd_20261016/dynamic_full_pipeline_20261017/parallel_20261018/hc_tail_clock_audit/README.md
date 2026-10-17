# HC / pre-QA / 输出尾部：已有阶段时钟只读收口

2026-10-18。**本轮没有找到足够新证据支持的实现候选。** 这不等于数学上没有
继续优化空间；只是当前可见的关键区间，要么已有同类实验，要么不能从已有端点
分离出可消除成本，不应为并行而再制造一次编译/设备尝试。

本目录只读已有三轮完整 CSA 诊断、同作业源快照及 PTO 文本；没有编译、NPU
执行、参数扫测、正式源码修改或依赖修改。仅记录路径和文件大小，不生成 hash。

## 同轮证据与解释边界

唯一时钟来源为主仓
`tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/attention_tick_h131072_b16_v1/`，
任务 `task_20261018_032708_262386210870`，128K/B16/S6，三轮 A→B→A。
使用该目录 `sources/spmd_reference` 的正式 d253 源快照；生成 PTO 为
`build_output/decode_csa_spmd_t13fgp2q/ptoas/dynamic_csa_full.pto`。
精确输入绝对路径、大小、各核端点与同核区间见 `evidence.json`。

该任务包含 Attention tick 插桩，`eligible_for_performance=false`。插桩会
改变运行及下游时序，以下不是无插桩完整 L1 性能，不与其他任务数字相减。
notify_start/end 是阶段发布前后 Scalar 端点；某段含业务、等待、缓存维护
和发布。跨核最大值之差只是包络，不是某段纯计算耗时，也不是可达收益。

| 同轮端点 / μs | A0 | B1 | A2 |
| --- | ---: | ---: | ---: |
| GATES 最晚发布开始 | 42.22 | 40.26 | 38.26 |
| NORM 最晚发布开始 | 57.98 | 56.02 | 53.80 |
| COMB 最晚发布开始 | 72.28 | 71.42 | 68.50 |
| QA 最早发布开始 | 76.46 | 73.06 | 72.26 |
| QA 最晚发布开始 | 79.70 | 76.08 | 75.98 |
| COMB 最晚发布结束至 QA 最早发布开始 | 3.42 | 0.88 | 2.70 |
| OB 最晚发布开始至 OUT 最晚发布开始 | 27.32 | 27.58 | 26.94 |

## 前段：已有并行已隐藏 Sinkhorn，小临时量已在 UB

同作业 `vector_schedule.py:522–545` 先执行 mix/RMS 并发布 NORM，随后计算
Sinkhorn 并发布 COMB，最后等待 QA；`cube_schedule.py:244–259` 等有效 NORM
owner 后执行 QA。因此 NORM 完成后，两种数学已经在 AIC/AIV 间并行。
三轮里**全部 COMB 发布结束均早于最早 QA 发布开始**。这个记录不支持把继续
加速 Sinkhorn当作当前 QA→QR 关键链优化；不能凭 COMB 自身耗时较长就再测标量版。

GATES→NORM 包络约 15.5–15.8 μs，确有业务计算，但不能把它全部看作同步空洞。
该阶段需要保持四路混合加法树、BF16 舍入点、16 个 K256 平方和顺序及归一化
权重 dtype。`hc_stages.py:563–567` 表面上的 `create_tensor` 平方和/逆 RMS
暂存，在本次真实 PTO 中已经是 `loc=vec` 的 `[1,8] FP32` Tile：

- `norm_sq_sum`：Vec 地址 65536；其输入通过 UB `pto.tmov` 取得。
- `y_inv`：Vec 地址 91136；逆 RMS 的返回 Tile 通过 UB `pto.tmov` 回绑。

摘录见 `hc_inverse_local_tiles.pto.txt`。这两项没有 GM store/load，不能提出
“删除平方和/逆 RMS GM 中间物化”作为新候选。两处小 UB copy 也没有独立计时，
不据 32 字节的逻辑量换算完整性能收益。

QA 当前等待所有有效 NORM 行 owner。把等待进一步缩到每个 QA M 组的行集合，
理论上可能让部分 AIC 更早启动；但现有三轮只显示 NORM 端点散布较窄，不能说明
最晚 QA 组或完整输出会提前。已有 HC 行依赖与细化输入依赖实验，不据这一点
再发起同类小改动。这里没有宣称已证明“所有更细分组方案都无效”。

## 尾部：有工作，但没有新的可直接消除物化

OB→OUT 的包络约 27 μs，包含八组 INT32→FP32 反量化左折叠、权重 scale、
BF16 舍入、逐 token HCpost、等待/缓存维护/发布。当前已经把投影 BF16 Tile
直接传入 HCpost，**没有独立的 out_bf16 GM 往返**可再删。

既有 `parallel_20261018/ob_hcpost_stream_audit/README.md` 已区分：行外层提前
归约主要隐藏每名 AIV 第一行的一部分；组外层需要多行长期 UB accumulator，
或增加明确的 GM 往返。当前新增端点没有分解每组 OB 和归约/HCpost 成本，无法
补足那份审计所缺的重叠收益依据。因此不重复实现逐组消费，也不把全部 27 μs
当可隐藏空间。OA 量化重复读取虽已成功减少生成 TLOAD，但两个完整实测档无收益，
不能换名再次建议。

## 已排除的历史方向

已核对主记录 `vllm_ascend/ops/pypto/DSV4_CSA_SPMD_OPTIMIZATION_LOG.md` 及
以下原始专项。它们保留不同历史源、fixture 和时序，数值不能跨轮直接比较。

| 已有方向 | 原证据 | 本轮处理 |
| --- | --- | --- |
| 消费者直接归约 partial、删除 RAW | `parallel_20261018/../hc_consumer_reduce_v1`，MASTER 的 HC_CONSUMER_REDUCE 条目 | 已测；不重做 |
| reduce→gates→mix/norm 融合及 Tile 直传 | `csa_multishape_spmd_20261016/HC_FRONT_AND_INPUT_DEPENDENCIES.md`，MASTER 历史原文 051 | 已有完整实验；不重做 |
| mixed 两遍间保留 UB chunks，减少 GM 重读 | `csa_multishape_spmd_20261016/HC_MIX_CHUNK_BUFFER_EXPERIMENT.md`，历史原文 052 | 已测同类物化消除；不重做 |
| 48 AIV / 每成员 2 行、mix+QA 调度 | `csa_multishape_spmd_20261016/HC_MIX_QPROJ_EXPERIMENTS.md`，历史原文 053 | 已有失败/真机记录；不重做 |
| HC AIV→AIC 流式供数 | MASTER hc_stream_pre/GM FIFO 纠正条目 | A3 当前仍经 GM；不称零拷贝、不重开 |
| OB→dequant/HCpost 行/组流式消费 | `parallel_20261018/ob_hcpost_stream_audit/README.md` | 新端点未补足收益依据 |
| OA 量化复用原值、通知批量删除 | 主仓本轮归档与 MASTER；前者两档无完整收益，后者已审计 | 不重复提出 |

复现仅读取已有文本：

```bash
python read_existing_clocks.py \
  --canonical /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
```

脚本只生成本目录派生 JSON 和小段 PTO 摘录，不创建/运行内核、不复制二进制。
本轮到此收口，没有提交新的实现候选。
