# Attention优化与T95修正证据

对应[完整报告](../../DSV4_FLASH_CSA_TND_ATTENTION_OPTIMIZATION_AND_T95_QA_20261023.md)。原H3为`a74eb99b9b30e27df845d855aff71f0b6f4146d6`；nalinaly固定`cafcadec96d48790608e5b55d456150fcf00e794`。`FILES.json`封存本目录文件，不包含自身和本文。

## 证据分组

- `results/`：原始单层result，含所有窗口、实际schema和精度门禁；失败结果也保留。`vector-v5-failure`的1407 μs精度失败，不能作有效性能结果。
- `vector-qualified-8k`、`vector-qualified-8k-reverse`、`vector-qualified-128k`：最终Vector v7数值源，两次8K和一次128K均通过Native A/B八边界及ABA。L1与dep分开。
- `qa95-unit`：真实FULL输入及真实NZ29权重，旧QA helper有BF16差异，K512 packet候选逐位一致；另含两个随机输入和污染回放。
- `qa95-formal`、`qa96-formal`：正式无debug T95/T96混长FULL graph路由回归。
- `qa95-full-observed-fix`：保留原始继承的旧说明字段，实际门禁看`matches_original_native`及stage逐位结果；不能只读顶层旧`accuracy_executed`字段。
- `executed_sources.tar.gz`：按实际manifest封存的脚本、DSL候选及whole v3助手，不含权重、bank payload、SDK二进制或build目录。CPU收据只对该源文件封存有效，移植后需要重编译。
- `vector_whole8k_raw.tar.gz`/`vector_whole128k_raw.tar.gz`及对应`vector_whole*_summary.json`：16rank原始输出、FULL资格、时延及CPU复核结果。公开摘要两项轨迹digest字段只改名，原字段留在raw，转换记录见`PUBLIC_SUMMARY_DERIVATION.json`。
- `TASKS.json`：任务关系、超时和结果。whole v2的旧variant类型检查失败发生在生成前，已保留；v3严格检查新tuned类型并拒绝旧combined，未放宽数值门禁。

## 复现要求

227资产路径、SDK、bank与Native gold见[上轮说明](../nal_module_gap_20261023/README.md)。在自有新目录解包，校验manifest；所有路径更换后重新封存、无设备编译，不拿旧收据替代。单层先用v7的`compile.sh`再`run.sh`，32个6/history8192或16个6/history131072；反向捕获加`--reverse-capture`。

正式启用是显式`PTO_CSA_VARIANT=tnd_precision_tuned`。短T<96保留原路由，T>=96走Vector计划和双缓冲；原精度alias/default不切换。共享QA95修正仅新增95的K512 packet路由。H3父提交可直接恢复。

whole v3必须先8K成功再128K，命令为`task-submit --device auto --device-num 16 --max-time 780 "bash <独立目录>/whole/run.sh 8k"`及同目录`128k`。输出目录已存在会拒绝复用。内层750秒；warm192＋measured192；确定性、NZ2、FULL、TP1/DP=EP16、EPLB关闭。128K提交24条，每卡实际21条活跃，真实T126/图bucket144；这不是每请求补到S6。原Native16rank的token及入场事件都由audit逐项检查。

统计脚本`summarize_whole.py`在源码归档内：对齐20个forward步骤/19个周期后，每步取16rank最大再统计。whole耗时包含未替换的HCA/FFN；周期另含草稿和采样等。没有新成对Native/旧基准whole计时，不据此宣称受控整模型加速。

## 当前DFX与源码关系

`FORMAL_SOURCE_PROOF.json`记录正式文件与whole v3/Vector v7受测副本的AST关系；唯一入口import搬迁连接到计算函数一致的正式QA95修复。

`VECTOR_STAGE_ANALYSIS.json`及`vector_l1_and_dependency_raw.tar.gz`保留27个L1原始窗口、独立4个依赖窗口及重算脚本。重算脚本将`timing`、`dependency`目录放回相应collected位置后，先通过源manifest/派生SHA和所有task/core覆盖门禁，再做映射。它不测新性能、不把重叠跨度相加。编排CPU/JIT不是相同binary，具体alpha等价门禁和保留的失败记录见报告。
