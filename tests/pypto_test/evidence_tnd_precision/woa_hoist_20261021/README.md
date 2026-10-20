# 纯DSL独立计算前移与依赖等待证据

完整结论见[报告](../../DSV4_FLASH_CSA_TND_DSL_WAIT_HOIST_20261021.md)。
正式数值源48c58ce7c、文档起点8fe011517均保留；只有独立实验与文档，无正式路由切换。

## 材料

| 文件 | 内容 |
| --- | --- |
| raw-results.tar.gz / RAW_SHA256.json | 8项原始NPU JSON、RUNS命令、候选SHA/父链、离线重算、独立审查JSON与回执复核 |
| reproduction-tools.tar.gz | 4个纯DSL候选、完整harness/检查/启动/复核脚本、owner2/phase1父源与SDK身份 |
| cpu-generated.tar.gz | 4个完整CPU编译的实际PTO/C++、MIX编排、配置、ABI和prebuilt SHA，生成后未做替换 |
| independent-reviews.tar.gz | 独立源码与实际生成物、依赖/UB/索引审查、最终结果复核；完整CSA集成为未实现设计 |
| woa-hoist-run-receipts-v1.tar.gz | 8项最终exit0、完整日志/原JSON，4项无设备CPU检查回执 |
| REPRODUCTION.md | 环境、精度/性能口径、停止门禁、复现方法 |
| CHECKLIST.md | 完成与未进入的验收阶段 |
| SHA256.json | 发布文件完整性，不包含自身 |

## 结论范围

8项短任务实际运行97～115秒，1280项后缀逐位指标通过。
仅T240的obowners两次比冻结后缀快17.52%/16.64%；T126/T144仍慢冻结，不能视作全形状性能交付。
没有新增完整CSA、真实整模型Native token或TND不等长metadata验收，没有新8K/128K整模型结果。
coefearly以obowners为父版，其余各自以owner2为父版；不跨任务相加收益，不改原始JSON字节。
源proof中的compiled/device_validated=false记录候选准备时状态，最终资格以实际CPU/NPU收据为准。
