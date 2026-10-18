# TND单入口纯DSL精度版：性能调优与真实模型复测（2026-10-19）

## 当前结论

本报告记录正式V4检查点 `da3b2d9fc`；V4源码/48项CPU结果须在该提交复核。最新V6见[最终精度与性能报告](DSV4_FLASH_CSA_TND_PRECISION_PERFORMANCE_FINAL_20261019.md)，本页历史结果不重标为V6。原精度交付 `c87e278e55cea4becd627f174bd5e3a382e9d895`、本轮起点 `15a9c1a3ec97da9dd91340e675429b7a55589821` 及性能TND变体保留，可以按Git历史回退。本轮没有改变SDK/codegen或增加手写C++算子。

正式V4在真实8K/B40的16卡预热及测量中，共245,760个输出token与52,576个接受事件全部匹配Native黄金，零抢占。128K/B24也通过：147,456个输出token和30,784个接受事件全一致，零抢占。结论绑定正式43个runtime源和两档实际快照，未用中间候选的验收代替最终源文件。

| 真实整模型档位 | Native历史forward ms | 本轮起点/中间版 ms | 正式V4 forward ms | 正式V4精度 |
| --- | ---: | ---: | ---: | --- |
| 8K/B40，每卡40活跃 | 89.482 | 起点651.470；ComboV2 127.086 | 116.360 | 245760 token、52576事件一致 |
| 128K/B24，每卡21活跃、3等待 | 79.730 | ComboV2 100.706 | 95.910 | 147456 token、30784事件一致 |

8K相对本轮起点耗时降低82.14%，相对历史Native仍慢30.04%；128K相对历史Native慢20.29%。Native是同环境已保存的性能窗，本轮按要求未新跑整模型Native2；不能把它称作本轮交错Native控制。整模型主指标是每个对齐步骤的16rank最大耗时均值，非单层CSA延迟、非全局同步服务吞吐。

## 保留的实现变化

| 模块 | 变化 | 保留的数值边界 |
| --- | --- | --- |
| HC-pre投影 | 现有DSL的BF16 Cube单次K16消费8个高位、8个低位，复现A3 Native HF32权重与K8边界 | 原HF32舍入、每split原K顺序、FP32 partial/reduce；不拆成两次独立matmul相加 |
| HC输入和平方和 | 大T按4个独立工作项分组，短query直接使用已验收Cube路径 | 各工作项内部次序不变，跨split归约不变 |
| Input RMSNorm | sum/inverse/apply三段变成逐行单任务 | 原折叠树、标量sqrt纠正及倒数、两次乘法、BF16舍入不变 |
| WKV | M64共享加载；两组N16的全部K顺序一致时合成N32，否则分别计算 | 每列原K256次序、FP32累加和BF16发布；短T保持M16 |
| O-proj A | 同K起点的行共用M128 weight，K256搬运分包与stage2流水 | 实际K16顺序、BF16发布；异起点或短T回退原路径 |
| O-proj B | 大T使用M64 INT8×INT8，短T保持M16 | 原K512顺序、INT32精确累加；上界2^27，不溢出 |
| Attention计划 | full精度入口使用32个planner worker | 原TopK、slot、validity及Native pair交换规则 |
| Heads发布 | 每组32次小copy改成一次宽copy | 位拷贝、cache依赖和TaskId保持 |

仍为HC-pre到HC-post完整PTO替换、单一62tensor顶层入口，内部阶段全部用DSL；真正TND沿用Native请求边界，不要求每请求补齐S6。21层CSA被替换，20层HCA、FFN及外围metadata/cache/connector仍使用Native。支持范围仍为本分支已限定的A3、TP1、T≤256及已验收静态配置。

### HF32优化的资格与限制

加载阶段检查原FP32 ND HC权重：原值必须有限，HF32舍入后为零或绝对值在[2^-48,2^40]，高/低BF16可逐bit重建。资格不满足直接报错，不在捕图期间首次转换。资格摘要和原始/舍入后权重SHA可追踪；检查不修改Native权重。

宽指数合成测试曾与无限精确整数金值不一致，停止该候选后，通过既有Native HF32观察器复核A/B/zero银行196608个去重结果：Native与BF16实现逐bit一致。因此以真实Native硬件合同为准，不声称任意IEEE输入的无限精确求和；动态hidden无通用值域检查。本轮模型全部HC权重满足资格，完整模型token和独立模块门禁共同限定结论。

## 与nalinaly性能版的受控参照

参考分支 `nalinaly/vllm-ascend:dsv4-flash-pto-v0.25.1rc1`，固定提交 `a919f11942d93627441c8fa694e310dcbd1c671b`。本轮只将其性能实现作为速度参照，不把其已知精度差异作为TND验收基准。

使用真实L4权重、合成hidden与history8191，同卡Native/TND/BSH交错8轮raw FULL graph，统一SK0、NZ2、确定性。不是128K整模型指标，不能与公开SK1截图混算。旧起点的受控结果：

| 档位 | Native µs | nalinaly性能BSH µs | 原TND精度15a9 µs |
| --- | ---: | ---: | ---: |
| B4/T24 | 550.643 | 499.248 | 4651.325 |
| B40/T240 | 1339.779 | 1285.230 | 28055.433 |

正式V4重新同卡交错的结果如下；TND A/B八边界、FULL/ABA/guard逐bit通过，BSH精度只记录不作性能门禁。

| 档位 | 同轮Native µs | 同轮nalinaly性能BSH µs | 正式V4 TND µs | TND/BSH耗时 |
| --- | ---: | ---: | ---: | ---: |
| B4/T24 | 588.073 | 516.747 | 938.189 | 1.816倍 |
| B40/T240 | 1375.258 | 1290.203 | 2669.736 | 2.069倍 |

两档仍慢于BSH性能版，不能称为已经对齐其性能。原始8组数据全部保留，不剔除离群样本。

nalinaly性能版K128局部softmax后归并的边界与当前Native K512不同。为保留精度，没有直接搬用该归约流程；正在分离验证不改变归约和舍入的KV搬运/同步优化。

## 单因素记录与未保留项

| 单因素 | 已观测结果 | 决策 |
| --- | --- | --- |
| WKV M64N32 | T240 145.10→111.64µs，T144 125.22→105.27µs，Native逐bit | 保留；前者基准有离群，原样保存样本 |
| WO-B M64 | T240 681.14→496.41µs，T144 487.24→385.62µs，四阶段逐bit | 保留 |
| RMSNorm融合 | T240 244.19→128.81µs，T144 179.38→107.57µs，output/rstd逐bit | 保留 |
| Planner16→32 | 完整CSA T240 3141.52→2993.44µs，T144 2283.73→2146.96µs | 保留；小T没有明确收益 |
| HC平方和group4 | T240完整HC 490.18→427.29µs；T144 335.87→300.57µs | 保留；T<64维持旧分工 |
| HC短query Cube | T6完整HC 1988.65→160.79µs；T1…15全逐bit通过 | 保留 |
| WO-A seedK256全形状 | T240 503.70→379.72µs，但T144 378.69→451.41µs | 拒绝全量合入，另测分档候选 |
| WO-A M256 | T240约2.8% | 暂不保留 |
| HC提前BF16布局/gather | 完整HC收益均小于1% | 无稳定收益，不保留 |
| KV复用初版 | CCEC通过但实际slice→transpose丢偏移，静态审查发现 | 阻止设备执行，独立v2修正后另验 |

模块数字是各自同卡单因素，不累加成整模型收益。所有失败候选、编译器限制和首差记录保留，不修改SDK掩盖问题。

## 完整模型测试方案

参照nalinaly低接受率方案，恢复本环境已验收的Native固定bank，40/24道不同问题、greedy、ignore EOS、DSpark出5验6，每请求192 token；比较所有16rank逐请求完整token、接受事件及统计，检查零抢占。公开问题和后缀协议已核验，未证明与nalinaly原前缀/cache的所有payload逐字节相同。DP16为相同问题副本，不是640/384个独立问题。没有强制接受步长3.8。

| 参数 | 8K | 128K |
| --- | --- | --- |
| 模型 | DeepSeek-V4-Flash-0731-w8a8，真实75个shard | 同左 |
| TP / DP=EP | 1 / 16 | 1 / 16 |
| 每卡提交/计时窗口活跃请求 | 40 / 40 | 24 / 21，3条等待 |
| FULL图桶/实际token | 6、240 / 240 | 6、144 / 126 |
| KV块 / utilization | 22000 / 0.95 | 24000 / 0.97 |
| max-num-batched-tokens / NZ / event | 400 / 2 / 1 | 400 / 2 / 1 |
| 确定性 / HCCL / EPLB | level1 / true / 关闭 | 同左 |
| 预热/测量 | 192 token + 1轮192 token | 同左 |
| 计时 | 跳过8个满档step，20个forward、19个cycle | 同左 |

测量不启用DFX、profiler、逐层hook或GPU输入拷贝；静态13包实际reselect一次，计时窗口无编译/强制重建。批末检查预热和测量的全部输出，首次已观察差异停止后续配置，不声称每个forward均有在线浮点首差hook。全部源、helper、bank、SDK、静态包身份按SHA核验。

## 环境、源码与复现

Python3.12、Torch2.10.0+cpu、Torch-NPU2.10.0.post2、vLLM0.25.1、vLLM-Ascend0.25.1rc1；CANN/ATB9.2.0-beta.2、A3 Ascend910_9392、实际PTOAS0.66、原私有PyPTO3e87a843/Simpler a54c0509。提交参数 --ptoas0.63 不代表runner实际工具，显式PTOAS_ROOT及preflight为准。全程不重装vLLM、PyPTO或Simpler，使用已核验PYTHONPATH和源快照。

正式V4源快照3365文件SHA `6107e6ed747925793dcfe0e5188dee8486c3a8b91d0c1b7e073b62d38215bc28`，该V4检查点的43个runtime文件逐项匹配其正式源；完整快照含历史测试/文档，不冒称新增报告也曾在该快照内运行。

证据、CPU复核和227复现命令将在验收完成后附于本节。设备任务均以780秒上限提交，拆成单因素/单配置小任务；排队不计占卡，执行时间另以实际start/finish记录核实。没有自动化调度。

### 额外观测与精度结论边界

正式单层混合query `[1,2,3,4,5,6]`，真实/物理行均21，无每请求S6 padding；同输入A/B八个边界（output、TopK和六类cache/state）、ABA恢复、毒化回放、guard及实际PTO route均逐bit通过。T1另有独立门禁。模块原始实验仍保留其私有helper身份；只有数学AST/格式/import转换一致的证据，不能贴成正式模块已逐项重新导出内部浮点值。

真实整模型本轮复核完整token与事件，不声称所有内部浮点tensor逐bit已经观测。8K所有rank对历史Native三个窗口的完整CPU metadata（除进程call号）相同。128K第一个Native窗口也全部相同；第二窗口96条rank/step记录有CPU observer中的0↔-1差异，列表长126、差异发生在列表内部，不称为已证明的GPU padding差异，也不据此证明实际GPU输入逐bit相同。几何、位置、有效请求顺序均另核对，原始差异逐项保留。

### 证据与本地CPU复核

V4历史证据位于 [formal-v4-manifest.json](evidence_tnd_precision/current_full_precision/formal-v4-manifest.json)，schema v2，identity FormalV4。三份压缩原始包约4.27MB：实际mixed21 source/helper/数值报告、8K全部16rank结果/Native gold/历史Native性能窗、128K同类完整数据。共253个载荷SHA逐项核验；不含模型权重或`.pt`。

新增 [CPU证据测试](test_full_precision_current_evidence_cpu.py)直接重读raw token/events、实际PTO替换/图捕获资格、43个当前源SHA和完整helper身份，并重新计算maxrank时延。它是保存证据的CPU重放，不是一次新的NPU数值测试。WKV/Oproj/Indexer原校准表及历史raw源锁定仍保留，没有降低数值验收阈值。

```bash
source /Users/jiayetcs/Desktop/Project/PyPTO/.venv311/bin/activate
pytest -q tests/pypto_test/test_full_precision_hc_weight_contract_cpu.py \
  tests/pypto_test/test_full_precision_wkv_cpu.py \
  tests/pypto_test/test_full_precision_oproj_cpu.py \
  tests/pypto_test/test_full_precision_indexer_projection_cpu.py \
  tests/pypto_test/test_full_precision_eligibility_cpu.py \
  tests/pypto_test/test_full_precision_wkv_profile_cpu.py \
  tests/pypto_test/test_full_precision_current_evidence_cpu.py
```

结果48项通过。该命令在本机仓库根目录执行，使用已有CPU环境，无需本机vLLM/NPU。

### 227原位设备复现

原私有环境、两档Native bank/黄金、13包静态缓存和本轮冻结snapshot均保留。复现必须先核对这些资产的SHA，不重新生成bank，也不替换PyPTO/Simpler或重新编译vLLM。whole原始包 `helpers/` 内含实际 `env.sh`、`run-case.sh`、preflight、bank检查和性能observer源；同包 `source/source-manifest.json` 指向本次完整3365文件快照。

原job为 `/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/precision-overnight-20261019/whole-formal-v4`。旧输出目录有意保留，`run-case.sh`使用排他mkdir，不能原地直接重复覆盖。另建fresh-label复现目录，保持runtime源/manifest字节不变，仅重定位helper的job及output路径、重新计算helper manifest；使用 [fresh-label复现步骤与工具](DSV4_FLASH_CSA_TND_PRECISION_PERFORMANCE_REPRO_20261019.md)。复现仍分8K和128K两个task，每个 `--device-num 16 --max-time 780`；8K完整token/事件及零抢占审计通过才提交128K，不跑新Native2。

底层实际模型入口及参数为：

```bash
python "$SOURCE_ROOT/tests/pypto_test/offline_pd/run.py" decode \
  --output "$PERF_OUTPUT" --backend pto --pto-attention csa \
  --mixed-requests --batch 40 --decode-dp 16 --max-num-batched-tokens 400 \
  --weight-nz-mode 2 --deterministic --port 30981 \
  --gpu-memory-utilization 0.95 --event-work-mode 1 \
  --bank "$FIXED_8K_BANK" --decode-tokens 192 \
  --graph-mode full_decode_only --capture-sizes 6 240
```

必须通过已验收helper/env启动，不能只复制该底层命令漏掉runtime/sitecustomize/preflight。128K仅按参数表改batch24、utilization0.97、bucket144、KV块24000和固定128K bank。

### 格式检查与任务记录

增量pre-commit按改动文件执行；网安扫描hook按用户此前要求明确跳过，未下载扫描器。隔离副本运行 `bash format.sh ci` 与干净15a9基线存在相同8类既有hook失败、575条Ruff记录逐条一致，改动运行源码无新增记录；不做全仓格式清理、不把该结果报告成全仓CI通过。正式runtime源预格式化后再做硬件验收，源SHA未因最后文档或测试格式调整而变化。

V4整模型任务：8K `task_20261019_052827_104403218230`，128K `task_20261019_053718_143156420549`，均成功退出，分别约9分钟和8分钟。每个task提交上限780秒；单因素任务独立提交，不使用长占卡任务或自动调度。

后续优化应另建候选、先同卡单因素并检查实际生成代码，首次差异停止定位；只有稳定收益及当前正式source绑定的精度复测均通过，才追加新检查点，保留此版本和原始证据。

模块、HF32合成Native银行、单因素原始结果、实际DFX映射及正式三方数据见 [调优证据索引](evidence_tnd_precision/current_full_precision/tuning-index.json)。压缩包内部有中文README、逐文件SHA及标准库`verify_archive.py`；与三份正式验收raw包分层保存。
