# 真实 S=1 的动态 CSA 精度验证

日期：2026-10-18。权重：`/data/model/DeepSeek-V4-Flash-0731-w8a8`，第 4 层。

用户要求补充 S=1。Dspark 配置上限仍为 5；每请求本轮实际只有一个 query，
不是把输入补成 S6 再运行单 SPMD。正式动态源码取自提交 `2d68a00d`，
清单见 `source_manifest.json`，避免混入尚在验证的 HC 消费者归约候选。
此目录的源码只是版本快照，仍保留 `pl.dynamic` 和实际 device metadata。

## 场景与比较方法

| 历史长度 | B / T | 目的 |
| ---: | ---: | --- |
| 131072 | 4 / 4 | 小 token 尾块及长历史 |
| 131072 | 16 / 16 | 长历史的单 query 请求 |
| 8192 | 32 / 32 | 短历史的单 query 请求 |
| 131075 | 16 / 16 | 单 query 恰好完成 ratio4 压缩，检查新 KV 发布及 state 边界 |

前三例的本轮位置都不完成压缩；第四例覆盖相反分支。所有例子使用真实
Native builder 的 Q/L、position、compact metadata、slot 与页表。

比较分三个层次，分别记录结果：

1. 单 SPMD 对 Native：输出、Top-K、cache/state；保留零容差明细，同时离线
   统计既有 `atol=0.01, rtol=0.01` 输出门槛，不为通过而放宽。
2. 对原分 task PTO：旧实现有 S6 假设，不能直接作为 S1 oracle。因此真正
   执行 S6，令每请求第一行、位置、页表及全部初始缓存与 S1 一致，比较
   第一行输出、Top-K、score 和共同 cache/state 字节；额外五行分别用 X/-X，
   检查因果前缀独立性。只排除 S6 独有的合法写槽，包含全部 S1 合法写入。
3. 单 SPMD eager 对两次同地址 graph replay：重置初始 cache/state、填毒
   scratch，检查完整输出/state 一致、信号代际和输入/工作区保护区。

第 2 项通过不等于 Native 精度或整模型输出 token 通过。这里不测性能，
也不代表不等长混合请求、空 rank 或连续多步生命周期全部覆盖。

## 执行与证据

`jobs.json` 保存四个最终任务号及结果目录，全部完成。
`../summarize_query_precision.py` 离线生成 `result_summary.json`。
原始结果中的 `precision_exact` 指 Native 零容差，`causal_extension.passed`
指原分 task 的因果前缀精确对照，不能混用。

| 历史 / B | 对原分 task 因果前缀 | 图重放 / 保护区 / 代际 | 对 Native 输出 max_abs | 既有输出门槛超差元素 |
| --- | --- | --- | ---: | ---: |
| 131072 / 4 | 全部逐 bit 一致 | 通过 | 0.0234375 | 134 / 65536 |
| 131072 / 16 | 全部逐 bit 一致 | 通过 | 0.03125 | 1150 / 262144 |
| 8192 / 32 | 全部逐 bit 一致 | 通过 | 0.03125 | 235 / 524288 |
| 131075 / 16 | 全部逐 bit 一致 | 通过 | 0.03125 | 945 / 262144 |

因果前缀的“全部”包含输出、Top-K、score、五份 allocation 的共同字节，
两种额外 query 符号都通过，并包含全部 S1 合法写入。Native 对比没有非有限值。
在这四组输入中，没有发现本次动态融合相对既有 PTO 数学新增的精度问题；
**Native 既有输出门槛仍不通过，不能宣称 Native 精度或整网 token 验收完成。**
HC 消费者归约候选未混入此快照，其两档 S6 验证不能自动扩大为 S1 已覆盖。

## 首轮测试参照修正

前三例首次已完成 Native、SPMD eager 和两次 graph 验证，随后因果参照在
`state物理页映射必须完全一致` 断言处退出。问题是测试要求 S1/S6 页表
整个 shape 相同；S6 合法多出几个未来 token 的 state 逻辑尾列。

修正只核对 S1 所有可访问列与 S6 的共同前缀逐项一致，保留 S6 的合法尾列。
实际物理分配容量和初始字节仍相同，没有重排页表、放宽数值比较或忽略 S1 写入。
例如 H131075/B16 的 state 页表分别为 `[16, 65539]` 与 `[16, 65542]`，
共同前缀精确一致。前三例的最终任务为：

- `task_20261018_001338_26183106270`
- `task_20261018_001339_261935721422`
- `task_20261018_001341_262026610065`

压缩边界例 `task_20261018_000945_25299648729` 在导入参照 helper 前已包含
这项修正。首轮任务、失败断言、原始日志见 `jobs_initial_attempt.json` 和各
`native_*_s1_v1/`；没有把首轮原结果覆盖成 PASS。数值快照也改为先于可选
因果参照保存，避免参照失败时丢失已经取得的 Native 比较证据。

从仓库根目录复现一个场景（输出目录必须是新目录）：

```bash
case_dir="$PWD/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
task-submit --device auto --max-time 1200 \
  "bash '$case_dir/run_native_query_probe.sh' '$case_dir/s1_repro_new' --history 131075 --batch 16 --query-tokens 1 --causal-extension-reference --spmd-source '$case_dir/s1_precision_20261018/source/deepseek_v4_flash_csa_spmd'"
```

复现其余场景只修改历史长度和 batch。Native 不精确时驱动仍返回 1，
须结合逐项检查判断，不能把退出码直接解释成设备崩溃或融合错误。
大 `.pt`、二进制及构建缓存仅本地保留；文档和文本结果入 Git，不做 hash 校验。
