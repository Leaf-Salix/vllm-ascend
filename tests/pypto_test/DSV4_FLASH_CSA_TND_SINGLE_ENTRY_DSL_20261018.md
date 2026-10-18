# 单入口、全DSL的CSA TND精度对齐（2026-10-18）

## 结论

现有DSL方案已实现：HC-pre至HC-post只有一个PTO顶层入口，内部七阶段，选中算子包15个Python文件、零手写C++，未改编译器/SDK。保留真正TND、原Native metadata/cache/connector。

| 真实整模型配置 | 请求数 | 输出token | 生成轮次 | 全16rank token/事件/入场 | rank0 hidden/aux/MTP | 实际占卡 |
| --- | ---: | ---: | ---: | --- | --- | --- |
| 8K/B40 | 640 | 122,880 | 51 | 完全一致 | 全部逐bit一致 | 8分23秒 |
| 128K/B24，实际并发21 | 384 | 73,728 | 78 | 完全一致 | 全部逐bit一致 | 8分41秒 |

两档均192 token/请求、零抢占，Native黄金复用，未跑Native2。全21CSA层、全16rank两档位均有新单根真实warmup/capture证据。本轮未测正式性能；原七调用/C++版本的延迟不能直接套用。

## 目标与基准

工作分支为 `dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`，原已验收提交 `94feb6a2f2d5dad8fd480f90456920402644d3e0`，原数学实现 `425ae1f11`。备份分支 `backup/full-pto-seven-call-cpp-20261018` 保留原来的七次PTO注册调用和两个C++ helper，227原冻结源码、Native黄金结果和SDK均不变。

本次目标是HC-pre至HC-post只有一个注册的PTO顶层入口，内部保留七个阶段，算子数学全部由现有DSL表达。没有修改PyPTO编译器、后端、PTOAS、Simpler或CANN，没有重装框架。Native的metadata生产、cache所有权和connector继续复用；HCA、FFN不在CSA替换范围。保持真正TND，token维来自真实请求边界，不要求逐请求补到S6；物理tile对齐不等于请求补S6。

## 实现

- 注册入口 `dsv4_csa_tnd_full_precision::attention`，62个tensor参数，输入 `[T,4,4096]`，直接写调用方同形输出。
- 七阶段：HC-pre、输入RMSNorm、WKV投影、Indexer权重投影、CSA attention主体、O-proj、HC-post。各阶段之间使用父scope创建的tensor，禁止把子scope返回SSA别名带到兄弟scope。
- HC-pre用DSL实现HF32操作数舍入和每8乘积加旧acc的一次FP32 RNE。使用FP32误差展开保留求和低位，最后按sticky方向正确舍入，对齐Native HF32的操作数与累加边界。
- Norm用DSL显式半区平方和树，最后复制为合法32B输出；原已对齐的标量平方根/除法及BF16边界保留。
- WKV、Indexer、O-proj、HC-post的原数值计算和顺序保留，仅提供inline入口。
- 删除选中实现目录的 `hc_pre_linear_hf32.cpp`、`input_norm_sum.cpp`。编译器自动生成的编排/设备代码仍是工具链正常产物，未手写新的C++算子。

HF32等价范围先限定为本模型/硬件/既定静态配置和已测有限数据；不从有限样本推论所有IEEE特殊值均等价。

## 实现前后

| 项目 | 原完整精度版 | 本次实现 |
| --- | --- | --- |
| Host每层PTO数学调用 | 7次 | 单一root调用 |
| root内部阶段 | 分开注册执行 | 同root内7阶段，兄弟scope |
| HC-pre线性 | 手写C++ HF32 | 现有DSL显式等价算术 |
| Norm平方和 | 手写C++ | DSL半区归约树 |
| 其他数值边界 | 已验收顺序 | 保留原顺序 |
| 请求布局 | 真TND | 真TND，未引入S6请求补位 |
| metadata/cache/connector | Native | Native |
| 编译器/SDK | 已验收私有包 | 同一包，未修改 |

## 模块测试结果

| 测试 | 配置 | 检查组数 | 结果 |
| --- | --- | ---: | --- |
| RMSNorm | T=1/6/11/21/40/144/240/256 | 40 | sum、FP32 rstd、BF16输出均0bit差异 |
| 精确九项求和 | 24,000随机/抵消样本，1/17/750个32列block | 15 | 与独立整数oracle均0bit差异 |
| HC-pre小尺寸 | T=1/11 | 20 | 有效线性分片及Native三个输出均0bit差异 |
| HC-pre大尺寸 | T=6/40/144/240/256 | 50 | 有效线性分片及Native三个输出均0bit差异 |

每个尺寸先eager，再固定地址graph A/B/A-restored/zero；输出NaN poison，检查输入/权重不变、前后guard和B确实变化。Norm和HC-pre使用真实L4权重、合成BF16 hidden；三输出gold直接调用实际安装的Native算子。冻结旧extern只作sum/partial诊断参考，未作为新候选实现。模块结果不能代替整模型验收。

任务分别为：

- Norm：`task_20261018_103226_380191526611`，实际43秒。
- 九项求和：`task_20261018_103330_384639125926`，实际33秒。
- HC-pre小尺寸：`task_20261018_104054_40661542310`。
- HC-pre大尺寸：`task_20261018_104246_413257118963`。

## 生成与首次失败记录

- 本地PyPTO源码/扩展不匹配（缺失AccPhase），使用227原私有SDK进行生成/编译，未重建邻近checkout。
- 单根初稿缺少callee constexpr默认值，随后转置形状推导失败；显式ABI绑定并平铺attention主体及原bind_dynamic后生成通过。
- 兄弟scope最初使用子scope返回SSA别名，静态审查发现越域；改为只依赖父级Out描述符，完整编排CXX编译通过。
- HC-pre首轮任务 `task_20261018_103802_39990961242`：T1分片0bit，但Native comb `[T,4,4]` 与PTO `[T,16]` 的诊断shape校验失败。立即停止；v2保留原布局，只把连续Native gold reshape为同形后重新验收。
- DSL原语曾因混用tensor/tile比较接口、tile位运算的scalar参数而生成失败，均用现有tile API修复，没有改编译器。
- 精确求和CPU模型初稿在24,000样本中有10例sticky处理错误；修为跳过零展开分量后，整数oracle和设备测试均通过。

## 整模型8K测试（通过）

模型：`DeepSeek-V4-Flash-0731-w8a8`，真实75分片权重。沿用原固定8K、40种问题、DP16/TP1、DSpark出5验6、192 token、FULL decode、capture `[6,240]`，静态13包，确定性开，EPLB关。EPLB和接受步长等原始参考参数不作为本轮新验收变量。

只跑候选，复用原Native1。任务 `task_20261018_104823_15532728695`，16卡，max780秒。每步先检查实际入场、请求顺序、query边界、input IDs、positions、padding和DP向量，再检查rank0目标hidden/aux/MTP摘要；首次差异立即退出。最后核对16rank全640请求、122,880个输出token及接受事件、零抢占，并验证21CSA层两档单根的真实warmup/capture。没有新跑Native2。

候选manifest SHA：`7b0650b926a557241b622eba4f032731e7b21d75d807c3b6baccfe689b29c960`。
工具manifest SHA：`de60bda5a991ab764f5e6cfb7d5b3e1f94bf1dc591c3e691c24169ffa165e8b8`。

8K完整离线资格审计已通过：16rank共640请求、122,880输出token及接受事件一致，实际入场/输入全对齐；rank0全部51轮target hidden/aux/MTP逐bit一致。21CSA层每rank两档单根真实warmup/capture齐全。128K无设备预检通过后执行独立任务 `task_20261018_105937_68830117376`，max780秒，最终完整审计通过。

## 复现入口与精度判定

1. 还原原完整PTO精度记录中的模型、固定bank、Native1、官方13包static metadata及私有PyPTO运行包；验证冻结文件SHA。不得用另一个Native版本或bank代替已绑定gold。
2. 用本分支数学源码创建独立snapshot，校验manifest；选中变体 `PTO_CSA_VARIANT=tnd_full_precision`，`PYTHONNOUSERSITE=1`，环境导入路径以source preflight为准。
3. 先运行模块脚本：`norm-v1/run_norm.sh`、`sum9-v1/run_sum9.sh`、`hcpre-v2/run_hcpre.sh`、`hcpre-v2/run_hcpre_large.sh`。输出目录必须是全新目录，禁止覆写历史结果。
4. 模块全通过后，用 `compile_root.py` 做完整生成和编排编译，并执行 `model-v1/source_preflight.py`、`verify_single_assets.py`。单根真实dispatcher schema须为62tensor，第一项x_hc、末项x_out，namespace必须匹配。
5. 通过 `task-submit --device auto --device-num 16 --max-time 780` 提交 `model-v1/run-case.sh precision-witness <全新标签>`。task-submit显式传原CANN/PTOAS_ROOT、PYTHONNOUSERSITE、checkout ccache与四路编译参数。每次监控日志与STOP文件；不要将队列等待算作设备执行时长。
6. 完成后执行 `model-v1/qualify_single.py --native <绑定Native1> --candidate <新输出> --out <新报告>`，验证Native/derived/源码/helper/bank/static全部SHA、16rank入场和token/events、rank0 outer逐步bit摘要、21CSA层单根warmup/capture。冷构建统计与数值验收分开记录；bypass/invalid/storage/forced等错误必须为0。
7. 仅8K完整通过，才执行同样独立的128K/B24任务；沿用原128K Native1与固定bank。任务不得超过15分钟；需要拆分时按独立可验收的小任务拆，不延长占卡。

本轮不把“token一致”扩张为“所有rank、所有内部tensor逐bit均已观测”；模块逐位覆盖和整模型outer/token门禁分别报告。也不把数值通过扩张为性能提升。

## 环境

227实际沿用Python3.12、Torch2.10.0+cpu、Torch-NPU2.10.0.post2、vLLM0.25.1、vLLM-Ascend0.25.1rc1，CANN/ATB9.2.0-beta.2、PTOAS0.66。PyPTO为已验收的3e87a843基线私有运行包，Simpler为a54c0509552b01e13fb0960e23ce409a01b5024f；精确导入文件路径与版本随preflight保留。不要用本机失配的PyPTO扩展判定此分支。`task-submit --ptoas 0.63`是调度参数；runner显式设置的PTOAS_ROOT及实际工具链记录为0.66，不能只据调度选项称运行版本为0.63。

模块任务实际占卡时间：Norm43秒、sum9 33秒；HC-pre首轮观测shape失败45秒，修正后小尺寸46秒、大尺寸39秒。生成/CCEC的无设备阶段未占用NPU。

8K实际占卡8分23秒。实际请求query长度只覆盖1和6，首步40请求各1行、实际/物理T均40；这不能扩张为本轮已完整覆盖2/3/4/5行混合请求FULL图。实现仍按Native query bounds进行TND寻址，没有新增逐请求S6补位。

全量format gate在独立临时checkout运行：基线旧文件仍有格式/禁用import等失败，未把格式化后的历史脚本当原始证据。当前修改的生产Python增量门禁已过；保留全量日志。

全量format日志未出现本次修改的 `deepseek_v4_flash_csa_tnd_full_precision` 目录错误；失败项来自基线旧文件。正式源码未被全量格式化改写，只在临时checkout检查。

## 回退

本次改动作为单独commit保存。需要退回原实现时revert该commit即可恢复七注册调用和两C++ helper；另有原94feb6a2f的备份分支。不要把旧性能版TND与当前完整替换精度版的延迟混用。本轮没有测新的正式性能数字。

## 128K/B24完整结果

复用原 `retest192-native1-b24000` 和固定131072 bank，24条提交、实际峰值并发21，允许其余等待入场。全部16rank入场、GPU输入、192 token和完整接受事件与原Native一致；384请求、73,728输出token，零抢占。rank0全部78轮target/aux/MTP摘要逐bit一致。

原128K完整门禁保留：73次FULL144，首次真实FULL6为0基第74轮，完整覆盖后续FULL6尾部；raw trace79轮含1轮合法空尾。均按Native实际query bounds寻址，未新增逐请求S6补位。

平均accepted=3.8264033264033266，接受事件及计数与gold完全一致。每rank21层的6/144两档单根捕获/前序warmup均通过，namespace和62tensor ABI严格核验。生成/二进制构建计数为0、object复用有效、各错误计数为0；READY计数为0，因此严格READY复用指标为false，与本轮数值资格分开报告，不能将其写成全缓存门禁均通过。

## 冻结证据与复现细节

完整原始证据为[337份证据载荷及清单归档](evidence_tnd_precision/single_entry_dsl_20261018/evidence-complete-v1.tar.gz)，2,915,570字节，SHA256：
`38e2d37e54be04a73263beb9114c1209e9f920f6454b36114a732b668b96e936`。
归档逐文件SHA/字节数均已下载复核，原始脚本/JSON/日志不格式化；[索引](evidence_tnd_precision/single_entry_dsl_20261018/index.json)包含各文件SHA。

[8K完整资格报告](evidence_tnd_precision/single_entry_dsl_20261018/qualified-8k.json)、
[128K完整资格报告](evidence_tnd_precision/single_entry_dsl_20261018/qualified-128k.json)分别保存全部实际并发/身份/计数；归档包含15个实测数学源码、完整3362文件source manifest、原private bootstrap/env、runtime/static身份、全部任务实际起止时间及失败日志。

在现有227环境复现同一冻结快照：复制原未改的 `full-pto-8k-b40-20261018/step12/source`（旧manifest `56d688b2…`），用归档 `source/` 的15个Python文件覆盖对应包，并删该包两份旧C++ helper。不要直接将新增说明文档混入冻结source来伪造相同完整manifest。再按归档 `helpers/compile_root.py` 生成manifest并核对 `7b0650b9…`；原目录、模型、bank、Native黄金及SDK保持原样。重新使用全新job/output目录时统一更新实验路径、冻结helper清单；Native/derived/static与数学源码SHA不可放宽。

复现128K时使用 `helpers/model-128-v1/run-case.sh precision-witness <新标签>`；完成后执行同目录 `qualify_control.py <结果目录> --output <新报告>`。它保留原78轮、完整首FULL6/144、全16rank输入/token/events、rank0 outer和静态选择门禁。先8K通过再128K，始终max780秒，首差停止后定位。

本地待提交的15个Python文件已逐文件匹配实测manifest；本次代码和本记录同commit。备份基准分支为 `backup/full-pto-seven-call-cpp-20261018`，指向原94feb6a2f，旧性能版及所有旧黄金/实测源码未修改。
