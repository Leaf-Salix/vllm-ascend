# WO-A搬运、流水及owner增量对拍证据

正式精度源48c58ce7c与旧后缀a31a78f8保留。候选只在独立实验文件中；没有切换默认路由。
结论见[报告](../../DSV4_FLASH_CSA_TND_WOA_PIPELINE_OWNER_AB_20261020.md)。

## 文件

| 文件 | 内容 |
| --- | --- |
| raw-results.tar.gz / RAW_SHA256.json | 12项AB原始JSON、DFX、proof、RUNS任务清单、离线复核结果；保留被测字节 |
| reproduction-tools.tar.gz | 各候选及父版、历史probe、CPU检查、运行/统计脚本、失败源码快照和固定nalinaly参考 |
| woa-pipeline-run-receipts-v1.tar.gz | 追加M256前15项任务最终状态、CPU v1..v9回执快照；3项失败保留 |
| woa-pipeline-cpu-v5-generated.tar.gz | nopipe/K128实际PTO、C++、编译报告 |
| woa-pipeline-cpu-v7-generated.tar.gz | 显式Mat控制/heads-tail实际生成物 |
| woa-pipeline-cpu-v9-generated.tar.gz | tail权重驻留及两项owner组合实际生成物 |
| woa-pipeline-cpu-v11-generated.tar.gz | M128/M256同K128单缓冲配对控制生成物 |
| woa-pipeline-run-receipts-v2.tar.gz | 追加M256任务后的16项最终状态、CPU v1..v11；v1为先前15项快照 |
| woa-dfx-k128narrow-t240-v3-evidence.tar.gz | 9个L1原始窗口及结果/log |
| woa-pmu-k128narrow-t240-v1-counters.tar.gz | PMU失败任务的原始result/log与9份CSV；设备计数独立核验，不改为PASS |
| independent-reviews.tar.gz | 设计、源/生成物、owner依赖、最终结果独立审查 |
| REPRODUCTION.md | 环境身份、CPU资格、版本脚本选择、设备门禁、离线复核 |
| CHECK_HISTORY.md | 编译、观测脚本及离线分析错误的修正历史 |
| SHA256.json | 本目录所有发布文件SHA，不包含自身 |

## 范围限制

真实L4权重，合成后缀输入；金值是冻结PTO数学。本轮不是新的Native整层或整模型验收。
12项正式AB五边界160项逐位检查均通过，仍没有候选稳定快于冻结后缀；不进入8K/128K。
PMU/L1可重叠，不能相加或相减成为计算、搬运、纯等待的唯一分解。
SDK原环境和正式基准均保留，不把局部实验收益误记为已交付整层收益。
