# 完整PTO CSA TND：临时精度交付版（2026-10-18）

## 交付结论

当前完整精度版通过本轮短query及混合1～6行的单层FULL graph验收。
结合此前同一数学实现的真实整模型8K/B40、128K/B24的192-token验收，
冻结为**固定模型、硬件、SDK和配置范围内的临时精度交付版**。
本轮没有修改数学代码或SDK，没有新增Native2、整模型或128K任务，没有重新测性能。

| 验收项 | 精度结果 | 性能/限制 |
| --- | --- | --- |
| 本轮8K附近单层，短query及混合1～6行 | 四例、A/B共64项外部边界比较全部逐bit一致 | 不计时；固定query边界FULL图 |
| 前轮真实整模型8K/B40，192 token | 全16rank输出122880 token及接受事件一致 | target forward为Native的13.62倍耗时 |
| 前轮真实整模型128K/B24，192 token | 全16rank输出73728 token及接受事件一致 | 窗口实际并发21，forward为Native的9.51倍耗时 |

精度保证限定于下述已验收范围。不能把有限测试扩大为所有输入、所有平台、
所有图档位或所有内部tensor均有形式化逐bit等价证明。

## 冻结版本与替换范围

- 分支：`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
- 交付Git版本：以**包含本文件的交付提交SHA**为准，不以日后移动的分支HEAD代替。
- 数学基线：`a20bcdb76`，本轮测试时Git HEAD为`2fc05d81e2ff4f425223ab97eccf75bf19c56b60`。
- 完整精度包：`vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd_full_precision`。
- **必须选择`PTO_CSA_VARIANT=tnd_full_precision`**。`performance`、旧`tnd_precision`
  等其他变体不属于本次完整替换精度交付，默认值也没有改成精度版。
- HC-pre到HC-post完整PTO替换：HC-pre、输入Norm、Q/KV及后处理、Indexer、
  sparse attention、O-proj、HC-post；一个PTO顶层入口、内部七阶段、62个tensor参数。
- 选中包15份Python数学源码，无手写C++ helper；编译器正常生成的C++不是额外Native模型调用。
  Native metadata、cache所有权及connector仍保留；HCA与FFN保持Native。
- 真TND按原生query bounds读取真实请求，**不要求每请求补到S6**。
  kernel tile对齐、整模型graph桶补位与逐请求S6补位分开判断。

选中包及其复用精度包共38份Python文件，连同入口、选择器及复现依赖共43份关键文件，
均与实际运行快照逐字节相同，SHA列入[交付manifest](evidence_tnd_precision/temporary_delivery_20261018/delivery-manifest.json)。
完整运行overlay另有3362文件manifest，SHA为
`7b0650b926a557241b622eba4f032731e7b21d75d807c3b6baccfe689b29c960`。
它是既有冻结运行树加已验收数学，不把overlay目录名等同于整个Git提交。
原性能版、原黄金和备份分支`backup/full-pto-seven-call-cpp-20261018`均保留。

## 本轮单层FULL图结果

模型`DeepSeek-V4-Flash-0731-w8a8`，真实L4/L26权重；hidden及历史cache为种子1024的合成数据。
单卡TP1，不加载MoE。Native/TND使用相同输入、独立同值cache；不再运行BSH。

| 层/历史位置 | 每请求query行数 | 真实T/图容量 | 逐请求/图桶补位 | A/B对Native八类边界 | 实际运行 |
| --- | --- | ---: | --- | --- | --- |
| L4 / 8191 | 1、1、1、1 | 4 / 4 | 0 / 0 | 全部逐bit一致 | 2分41秒 |
| L4 / 8191 | 3、3、3、3 | 12 / 12 | 0 / 0 | 全部逐bit一致 | 2分37秒 |
| L4 / 8191 | 1、2、3、4、5、6 | 21 / 21 | 0 / 0 | 全部逐bit一致 | 2分38秒 |
| L26 / 8189 | 6、5、4、3、2、1 | 21 / 21 | 0 / 0 | 全部逐bit一致 | 2分36秒 |

每例两个不同hidden输入A/B；同地址A→B→A恢复。八类边界为：最终输出、TopK、
SWA/raw KV、compressed KV、main state、indexer key、indexer scale、indexer state。
四例共64项边界比较按原始字节核验，包含BF16/FP32的bit差异，而不是allclose阈值。
有效slot之外的cache字节及metadata保护检查均通过。

图证据不是仅设置FULL标签：使用`torch.compile(backend="npugraph_ex", fullgraph=True,
dynamic=False)`，实际获取ACLGraph并raw replay，禁止Host updated nodes。
output填NaN、TopK填`-12345`后必须由图恢复；compiled与raw结果一致，
TND入口不允许fallback。根schema、62参、实际调用和全部保护结果均保存。

Native/TND均确定性level1、Torch确定性、HCCL确定性；NZ2、静态kernel开启、
`force_eager=False`、`super_kernel_optimize=False`。这是SK关闭口径。
本轮每张图的query边界固定，A/B改变hidden；**未验收同一张图动态修改query边界**。
单层手动构造的混合FULL图通过，也不等于整模型调度器已支持任意ragged FULL分发。

## 整模型证据与性能限制

前轮同一数学实现使用真实75分片权重、TP1/DP=EP16、DSpark出5验6、每请求192 token、
greedy、确定性开启、EPLB关闭、HCA保持Native。完整输入/入场门禁通过，零抢占；
rank0目标hidden、aux和MTP的全部生成轮次位摘要也与Native一致。
这些检查不能扩大成全16rank所有内部临时tensor均已逐bit观测。

| 配置 | Native target forward | 本精度版target forward | PTO/Native耗时 |
| --- | ---: | ---: | ---: |
| 8K/B40 | 89.482 ms | 1218.433 ms | 13.62倍 |
| 128K/B24，计时窗口实际并发21 | 79.730 ms | 758.039 ms | 9.51倍 |

这是整模型target forward设备时间，不是单层CSA延迟，也不是服务吞吐。
本交付优先保留精度，**不作性能提升承诺**，不能用旧性能版速度代表它。
完整协议、重复轮数及分布见[性能报告](DSV4_FLASH_CSA_TND_SINGLE_ENTRY_DSL_REAL_MODEL_PERFORMANCE_20261018.md)，
整模型精度门禁与原始证据见[单入口纯DSL精度报告](DSV4_FLASH_CSA_TND_SINGLE_ENTRY_DSL_20261018.md)。
采用自有冻结Native bank，公开问题/顺序/chat后缀已对齐参考方案；未证明完整bank payload等同nalinaly原bank。

## 环境与启动约束

- 已验收SOC为Ascend910_9392；CANN/ATB9.2.0-beta.2、PTOAS0.66，Python3.12，
  Torch2.10.0+cpu、Torch-NPU2.10.0.post2、vLLM0.25.1、vLLM-Ascend0.25.1rc1 overlay。
- 原私有PyPTO基线为`3e87a843`的已验收运行包；不能用上游版本号替代其中实际文件。
  冻结206份私有文件、2个扩展、22份Simpler Python、1967份vLLM Python、
  13个静态包和13个所选工具/动态库的实际SHA。这是所选依赖清单，不是完整操作系统镜像。
- 原静态包metadata SHA为
  `863a1e5daffc54aec0e9e8afb9f809f1bc6f4b3a2487e440615f3c942f36c602`。
  每个任务保留13包/SDK/SOC/确定性/安装校验，真实reselect一次，随后验证后复用。
- 必须保留已验收静态资格加载流程。单层助手在实际安装及reselect后授权runtime；
  整模型使用原精度报告归档的资格bootstrap。**只pip安装分支、只设置变体开关，
  不足以声明进入了已验收完整PTO路径**，也不得伪造`reselected=True`绕过校验。
- 当前完整精度入口要求T≤256及原有dtype/layout/并行资格。本轮不保证任意T、任意batch、
  任意权重dtype、其他CANN/SOC、SK1、EPLB开启或其他模型自动维持同样精度。
  扩大范围须重新对照Native验收；本轮没有修改SDK/codegen或降低门禁。

## 复现

1. 固定包含本文件的Git提交，选择`tnd_full_precision`，保留原基准。
2. 解压[复现归档](evidence_tnd_precision/temporary_delivery_20261018/evidence.tar.gz)，
   按其中`frozen/assets-verified.json`逐文件核验，**缺失也必须失败**：
   `python verify_assets.py frozen/assets-verified.json`。
   原227目录复核结果为2228个去重依赖、3362个源文件通过。
   重新运行collector生成新清单不能替代与交付冻结清单比较。
3. 原227环境助手目录为
   `/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/tnd-precision-delivery-20261018`。
   执行`source env.sh`，在无设备条件下运行`preflight.py`，核验源码、助手、导入路径和工具链。
4. 使用下面命令提交一个全新标签；每例通过才继续下一例。发现TND首差立即停止并保存诊断。
5. 完成后下载原始`result.json`、两份`*-witness.pt`、日志，再用
   `qualify.py <结果目录> --output <不存在的新JSON>`进行CPU字节级独立复算。
6. 整模型复现严格沿用上述两份原报告的模型、固定bank、Native黄金、资格助手和192-token参数。
   模型、完整bank及SDK二进制不打包；迁移环境后需要重新建立等价依赖并验收，不能仅改路径宣称同一环境。

```bash
task-submit --device auto --device-num 1 --max-time 780 --ptoas 0.63 \
  --env PYTHONNOUSERSITE=1 \
  --env ASCEND_HOME_PATH=/data/pyptouser/yejia/vllm-cann92-main/env/Ascend/cann-9.2.0-beta.2 \
  --env PTOAS_ROOT=/data/pyptouser/yejia/vllm-cann901-csa-0251rc1/experiments/nalinaly-71153bb3/reference-env/bin \
  --env CCACHE_DIR=/data/pyptouser/yejia/vllm-cann901-csa-0251rc1/src/pypto/.ccache \
  --env CMAKE_BUILD_PARALLEL_LEVEL=4 \
  "bash /data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/tnd-precision-delivery-20261018/run-case.sh <新标签> 1,1,1,1 4 8191"
# 其余三项替换为：3,3,3,3 4 8191；1,2,3,4,5,6 4 8191；6,5,4,3,2,1 26 8189。
```

本轮四项任务均max780秒，实际最长2分41秒，已全部终态，无本轮残余任务或自动调度。
任务ID及原始文件SHA见[汇总与索引](evidence_tnd_precision/temporary_delivery_20261018/index.json)。
大体积tensor见证保留本地及227原目录，其SHA写入CPU资格报告；小归档不包含大模型或完整bank。
先前BSH首差记录仍独立保留在[上轮报告](DSV4_FLASH_CSA_TND_BSH_RAGGED_FULL_20261018.md)，
本轮按新授权只判断TND对Native，不修改BSH、不以其已知误差阻止交付。

## 提交检查

独立复核重新读取四例原始CPU tensor见证，确认64项边界逐字节一致；
同时复算归档及43份关键源码SHA，交付范围与复现说明审查通过。
全仓`bash format.sh ci`在隔离副本执行，受到历史文件lint/格式问题及本机缺少shellcheck阻塞；
没有为消除这些既有问题修改历史源码或原始证据。本次新增文件增量pre-commit及Markdown检查均通过。
