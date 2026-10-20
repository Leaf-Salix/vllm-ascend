# WO-A轮询干扰和任务切点证据

正式精度运行源48c58ce7c不变，报告起点ae9d9cc47。
完整结论见[报告](../../DSV4_FLASH_CSA_TND_WOA_POLLING_CONTENTION_20261020.md)。

## 材料

| 文件 | 内容 |
| --- | --- |
| raw-results.tar.gz / RAW_SHA256.json | 8项NPU原始JSON、源/父链proof、RUNS命令与SHA、逐窗重算、L1/PMU复核 |
| reproduction-tools.tar.gz | 纯DSL候选/父版、诊断harness、CPU/运行/离线脚本、固定nalinaly源码与代码生成audit |
| independent-reviews.tar.gz | 设计、前缀实际PTO/C++、两任务依赖、AIV48资源/角色及最终结果独立审查 |
| woa-contention-cpu-v1-v2-generated.tar.gz | 纯等待控制实际PTO/C++与CPU成功回执；主24控制为wait24aligned |
| woa-phasecut-cpu-v1-v3-generated.tar.gz | 两MIX实际编排与内核，初稿缺acquire不占卡，合格为v3编译的acquire源码 |
| woa-quantcut-cpu-v1-generated.tar.gz | 未获设备资格的中段双AIV初稿，编译成功但实际注册未包含业务量化，留证不占卡 |
| woa-quantcut-cpu-v2-generated.tar.gz | 单vector_quant、48flat AIV实际配置/元数据、编排与内核 |
| woa-contention-instruments-v1.tar.gz | 两档18个原始L1窗口、9份PMU设备CSV及task结果/log |
| woa-contention-run-receipts-v1.tar.gz | 8项最终exit0/完整日志、7项CPU检查结果/日志及kernel_config/compiled_meta，保留命名检查失败 |
| REPRODUCTION.md | 具体前置条件、版本脚本选择、task-submit矩阵、源身份和停止门禁 |
| CHECK_HISTORY.md | CPU/编排错误在设备前拦截、下载审批超时唯一重试及本地校验问题 |
| SHA256.json | 所有发布文件完整性，不包含自身 |

## 范围

真实第4层权重、合成后缀输入；前缀gold为完整phase1有效FP32 accum，完整后缀gold为冻结PTO五边界。
8项NPU任务均结束、711项逐位通过；没有新增Native整层/整模型或非等长metadata资格。
轮询会拖慢同一WO-A，T126明显、T240弱；不把MTE2 busy当纯stall或DDR字节。
两段T126局部改善但T240回退；三段T126只有0.38%/9窗改善且仍慢冻结7.56%，不推广。
正式源码/SDK/默认路由保持，无8K/128K整模型复验，无遗留任务或自动化。

nalinaly参考audit为代码生成证据，device_validated=false，不是本轮其设备启用或收益证明；没有采用C++等待变换。
