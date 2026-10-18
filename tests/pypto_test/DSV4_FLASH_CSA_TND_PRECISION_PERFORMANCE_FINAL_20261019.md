# TND 单入口纯 DSL 精度版 V6：性能调优与真实模型复测

## 结论

本轮保留 Native 计算及舍入顺序，纳入四项有实测依据的工作划分、Cube 布局和搬运优化。完整 HC-pre 到 HC-post 仍通过一个 62-tensor PTO 顶层入口执行，算子使用现有 DSL，保留真实 TND，不要求每个请求补到 S6。

两档真实模型的 192-token 预热及三轮各 192-token 测量均与保存的 Native 黄金结果一致。性能有改善，但**尚未追平 Native 或 nalinaly 性能版**。

| 真实整模型配置 | 历史 Native forward ms | 历史 V4 forward ms | 当前 V6 forward ms | 比 V4 耗时降低 | 比 Native 耗时增加 | 精度：token / 接受事件 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 8K/B40，16 rank | 89.482 | 116.360 | **110.289** | 5.22% | 23.25% | **491,520 / 105,152 全一致** |
| 128K/B24，16 rank | 79.730 | 95.910 | **92.389** | 3.67% | 15.88% | **294,912 / 61,568 全一致** |

计数覆盖预热和全部三轮；两档均零抢占。V4、Native 是历史任务，V6 是新任务，以上改善是跨任务观察。没有新跑 Native2。Forward 为每个对齐 step 取 16 rank 最大值后均值，V6 每轮 20 步、三轮共 60 步，不是单层延迟或服务总吞吐。

完整 cycle（相邻 target forward 开始事件间隔，包含 draft、sampling、调度间隔等）为：8K V6 **125.910 ms** / Native 105.575 ms；128K V6 **104.724 ms** / Native 92.293 ms。每轮 19 个样本，共 57 个；rank 时钟未做全局同步，不将其直接当作服务 TPS。

## 与 nalinaly 性能版的同卡对照

| 单层配置 | 同轮 Native µs | 同轮 nalinaly BSH µs | 当前精度 TND µs | TND / BSH | TND 精度门禁 |
| --- | ---: | ---: | ---: | ---: | --- |
| 8K/B4，T=24 | 559.544 | 489.360 | **820.725** | 1.677 倍 | 八个输出/cache 边界逐 bit 一致 |
| 8K/B40，T=240 | 1393.111 | 1272.611 | **2391.330** | 1.879 倍 | 同上 |

nalinaly 固定提交 `a919f11942d93627441c8fa694e310dcbd1c671b`，分支 `dsv4-flash-pto-v0.25.1rc1`，38 个实际源码文件逐 SHA 核验。三方均覆盖 HC-pre 至 HC-post，使用相同真实 L4 权重、seed=1024 合成 hidden/history8191、独立 cache、FULL graph、SK0、NZ2 和确定性。同卡每轮正反序交错，保留全部 8 组、每组 20 次 replay；没有 DFX 插桩。Nal 的数值差异只记录，不阻断其性能测量；Native/TND 任一数值首差即停。

这里是单层受控结果，不是整模型三方吞吐。不能与公开 SK1 截图混算，也没有测量 nalinaly 整模型性能。V4 历史 TND B4/B40 为 938.189/2669.736 µs；本轮表观下降 12.52%/10.43%，但 Native/Nal 同样存在跨轮漂移，不能把这些比例全部归因于代码。尤其 B4 的 Native/Nal 也下降约 4.85%/5.30%。

## 纳入的四项优化

每项均先在独立候选中做 Native 位精度门禁，再组成正式快照。下列单因素收益来自不同实验，不能相加或代替整模型结果。

| 优化 | 保持的数值语义 | 单因素实测 |
| --- | --- | --- |
| checked-index 48 worker | 每行整数检查、交换和写入不变；退出 cache flush、planner 等待关系保留 | 完整 CSA：混合 T21 降约 12.9%，T240 降约 2.6% |
| KV 三槽 L1 复用 | Native K512 softmax、QK/PV 次序及 BF16/FP32 舍入不变；保留后两个 K128 面板供 PV 消费 | T240 两次独立复测均降约 1.12%；T144 降约 0.88%，小 T 无明确收益 |
| WO-A 分档 seed K256 | 每输出仍按原循环起点执行全部 256 个 K16；BF16、整行量化、WO-B 和反量化顺序不变 | T≥192 启用已验收新路径，T240 O-proj 降约 22.9%；T144 保留原路径 |
| HC linear M64 | 相邻 M16 行共享权重，K16 高低位与 split 归约顺序不变；末 panel 显式有效行 | T144/T240 完整 HC-pre 降约 12.4%/21.7%；T6 无性能退化 |

KV 实际 L1 为 464 KiB，KV/Q/probability 三段不重叠，并检查实际生成物的动态偏移及 event1/2/4 防覆盖。HC 在父作用域建立共同 partials；T240 最后一块只有 48 有效行，consumer stride 仍按原 padded16。O-proj 两个分支写同一父 GM，L0C 不跨分支。

未纳入全 T seedK256：它虽改善 T240，却让 T144 慢约 19.2%；最终仅作分档。HC 提前 BF16 布局/gather、WO-A M256 等收益小或不稳定，未合入。没有采用 nalinaly K128 分块 softmax 归并等改变 Native 舍入边界的路径。早期 KV slice→transpose 丢偏移、HC caller 的动态作用域失败均先定位修复，失败候选记录保留。

## 最终源码和单层门禁

- 分支：`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
- V4 回退点：`da3b2d9fc4f64aee5747300ecafb2a7367521efd`；原精度交付 `c87e278e55cea4becd627f174bd5e3a382e9d895` 和原性能变体的 Git 历史保留。
- 正式 V6 完整快照：3365 文件，SHA256 `6d9656de4580ba5f768881c417f3093a7744808ac6518c594e9db881971e7ab2`；当前 43 个运行源码逐项绑定。
- 数学清单：`a10789d99c63a65eb181ae054d2312cd70771cde0944db65fc95c11c4aafcb02`；整模型助手清单：`6e3d416475783fd7ceb77154c7a5fb69f17418c30f97563c970d23dead792f1f`。
- 相对 V4 仅修改 `hc_pre.py`、`hc_pre_panel.py`、`o_proj.py`、精度包 `decode_sparse_attn_csa.py` 四个运行文件。外围 metadata、connector、HCA/FFN 及 Native 源码未改；未修改 SDK/codegen、未新增手写 C++、未重装框架。编译器生成的 C++ 只作为实际编译证据。

| 最终源单层 FULL graph | 结果 |
| --- | --- |
| T=1，单短请求 | Native/PTO 输出、TopK、六类 cache/state 八边界 A/B 均逐 bit 一致 |
| 混合 query=[1,2,3,4,5,6] | 真实/物理 graph 行数均为21，每请求 padding=0；同上 |
| B24/T144、B40/T240、B4/T24 | 同上；两档 O-proj 分支、HC 尾行均覆盖 |
| replay 行为 | 三方实际 FULL 捕获、compiled/raw 自一致、live-input ABA、TopK poison 恢复、写入 guard；PTO 真路由、Native fallback=0 |

资格仍限定当前 A3、TP1、1≤T≤256、已验收静态参数及 HC 权重合同。共享 precision Sparse 的改动也会影响引用它的 legacy 精度入口；本次不能外推为所有 legacy 入口均已复测。有限输入下的边界逐 bit 与整模型 token 相等，不等于任意浮点输入或所有内部 tensor 的数学等价证明。

## 真实模型协议与实际入场

| 项目 | 8K/B40 | 128K/B24 |
| --- | --- | --- |
| 模型 | DeepSeek-V4-Flash-0731-w8a8，真实75 shard | 同左 |
| 并行/推测 | TP=1，DP=EP=16，DSpark 出5验6 | 同左 |
| 提交请求 | 每 rank 40 | 每 rank 24 |
| 计时窗实际请求/行数 | 40活跃，真实240，graph240 | 21活跃、3等待，真实126，graph144 |
| 运行 | 192-token 预热 + 3×192-token measured | 同左 |
| 确定性 | NPU level1、Torch、HCCL=true，CANN event mode1 | 同左 |
| 参数 | NZ2、SK0、静态13包；EPLB关闭 | 同左 |

未强制接受步长3.8，也没有新启用 EPLB；不是此前参考配置的完整性能验收。整模型全部21个CSA层使用完整PTO入口，20个HCA及FFN保留Native；正式性能窗口不开阶段hook、DFX或GPU输入复制。沿用公开40题的顺序、chat后缀和自身已冻结Native bank；未证明与 nalinaly 原始完整前缀/cache bank 字节相同。

运行时为 CANN/ATB 9.2.0-beta.2、PyTorch2.10.0+cpu、torch_npu2.10.0.post2、vLLM0.25.1、vLLM-Ascend0.25.1rc1、PyPTO3e87a843、Simplera54c0509、实际PTOAS0.66。提交器参数0.63不是实际编译器版本，命令显式覆盖PTOAS_ROOT。原环境、模型和静态缓存保留。

## 128K 观察审计首差及处理

最初审计要求每个候选测量轮都与 Native 历史第一轮 CPU 记录相同，在候选第2轮失败。首差后停止新增设备任务，保留原失败，再逐字段定位：

- 候选第1轮与Native历史第1轮：320行、全部CPU字段（除进程call号）逐字相同。
- 候选第2、3轮分别与Native历史第2轮：后两轮各320行逐字相同；连同首轮320行，三轮合计960行。**包含input_ids，没有归一化或丢弃该字段**。
- 跨阶段比较各有96行、416个0/-1元素差异，完整位置保留；这些位置在CPU观察切片的actual_tokens范围内，不将其称为GPU padding。

observer读取 `runner.input_ids.cpu`，不是 `_model_forward` 的实际 `input_ids` 参数。原生路径会仅在 GPU 上清理 placeholder，CPU缓冲不回写；还有异步输入更新。因此CPU观察值不能直接当作实际GPU参数字节证明，也未将差异唯一归因于初始化/capture。

经过独立源码和原始数据审查，修正为固定的 **first measured→Native第1轮、repeated measured→Native第2轮** 参照规则，不按数据挑最近窗口。选定pair仍仅忽略call号，其余全部CPU字段严格比较；跨阶段差异保留。原 token、接受事件、零抢占、source/cache/资格门禁未改变。七项CPU协议回归包含1正6负，连同阶段0↔-1改变也必须拒绝。原失败、原工具、新工具、review、policy及全部差异均归档。

## 最终 DFX 与下一步方向

最终V6 B40/T240另开一个单卡任务采集3个DFX窗口，level4、dep-gen关闭；采集前Native/PTO的A/B八边界及图门禁仍逐bit通过。使用本次live Worker artifact重建映射，实际67个token/core/执行数三窗均严格通过。

| 当前PTO阶段 | 3窗活动跨度 µs |
| --- | ---: |
| Sparse QK/PV | 686.52～699.74 |
| O-proj | 357.08～362.30 |
| HC-pre | 344.24～369.00 |
| checked-index | 107.48～109.36 |
| InputNorm | 55.36～61.98 |
| HC-post | 35.96～40.06 |

它们是插桩后的活动跨度，包含等待，互相可能重叠，**不能相加，也不能代替无插桩延迟或拆成纯计算/DMA等待**。这里只有PTO DFX，没有同轮Native/Nal模块泳道，因此不是三方逐模块差值。

HC内部widen为109.26～135.30µs，input-pack为119.18～156.08µs，M64 linear本体已仅28.64～32.22µs；O-proj的A为146.40～149.86µs，B64为109.30～113.40µs。下一步先在QK/PV内拆分计算、搬运和同步等待，再单因素验证保持Native K512/舍入树的流水；HC先检查输入准备，继续扩大HC linear并非当前首要方向。没有同轮Native/Nal DFX，不能断言这些就是三方差距最大的模块；以上是待验证方向，尚未实现或宣称收益。

实际编排SHA `3c595e65…` 与预编译版本不同，初次绑定被拒绝；独立复核确认607个局部变量仅双射改名，运算/控制流/数字/外部符号不变，末尾仅增加ABI描述。重新依据实际文件建立67-token映射并复核后才接受三窗。原reference和拒绝记录保留，不套用旧V4或预编译编号。

## 证据、复现与校验

当前证据见 [V6验收manifest](evidence_tnd_precision/current_full_precision/manifest.json)；原V4独立保存在 [V4 manifest](evidence_tnd_precision/current_full_precision/formal-v4-manifest.json)。原始包包含16rank结果、Native黄金token/事件、历史Native性能窗、实际43源码/完整快照清单、受哈希校验的助手；不含模型权重或`.pt`。

- [V6 mixed索引](evidence_tnd_precision/current_full_precision/formal-v6-mixed-index.json)、[8K索引](evidence_tnd_precision/current_full_precision/formal-v6-8k-index.json)、[128K索引](evidence_tnd_precision/current_full_precision/formal-v6-128k-index.json)。
- [单因素、实际编译和三方证据](evidence_tnd_precision/current_full_precision/formal-v6-tuning-index.json)。模块私有候选证据与正式V6源绑定验收分层保存，不将中间候选冒称最终源码测试。
- [fresh-label复现步骤与工具](DSV4_FLASH_CSA_TND_PRECISION_PERFORMANCE_REPRO_20261019.md)：明确 `--profile v6`、`PERF_REPEATS=3`、新标签/目录及助手SHA，先8K完整审计通过再128K。每个任务`--max-time 780`，没有定时任务；不重复安装框架或覆盖原环境。

在workspace `.venv311` 中完成 **52项CPU测试通过**，覆盖当前源/原始证据重放、HC权重合同、资格、WKV/Indexer/O-proj校准，以及实际DSL AST的行覆盖/K序/尾界/回退。另7项观察协议正负例通过。CPU测试不执行新的NPU数值计算。

```bash
source /Users/jiayetcs/Desktop/Project/PyPTO/.venv311/bin/activate
python -m pytest -q \
 tests/pypto_test/test_full_precision_current_evidence_cpu.py \
 tests/pypto_test/test_full_precision_work_partition_cpu.py \
 tests/pypto_test/test_full_precision_hc_weight_contract_cpu.py \
 tests/pypto_test/test_full_precision_eligibility_cpu.py \
 tests/pypto_test/test_full_precision_wkv_cpu.py \
 tests/pypto_test/test_full_precision_wkv_profile_cpu.py \
 tests/pypto_test/test_full_precision_indexer_projection_cpu.py \
 tests/pypto_test/test_full_precision_oproj_cpu.py
```

最终增量pre-commit及三份相关Markdown检查均通过。隔离全仓 `format.sh ci` 仍有与干净V4/旧基线相同的8类既有失败、575条相同Ruff记录，本轮文件无新增诊断；不将其报告为全仓CI通过。

本轮正式整模型任务为8K `task_20261019_064154_8716615272`、128K `task_20261019_065130_4934354508`，均正常退出，约9分钟；各小任务独立提交，均限780秒。性能变化仍需在其它batch、模型/硬件以及EPLB等配置下另测，不将这两档外推为完整服务性能结论。
