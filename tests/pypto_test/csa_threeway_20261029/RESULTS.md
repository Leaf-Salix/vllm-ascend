# 8K/B40 三方整模型测试：前置检查与首差记录

日期：2026-10-29～31。本轮8K整模型输出token检查通过；接受事件严格一致性尚未通过，性能未形成新结论。

最终token结果：固定共同框架`41ba392f`、我方数学包`65bebf7f`，
TP1、DP=EP8，40个不同问题在每个rank重复，分8请求×5波、每请求192token；
**8rank共320个请求副本、61,440个输出token全部与原Native v5逐个一致**。
21CSA替换及实际FULL回放覆盖检查通过，HCA均Native，零抢占。
接受事件轨迹不同，原严格postflight仍失败；不能据token结果宣称全程逐bit一致或公平性能可比。
这是共同框架上当前数学包的限定结果，不是未测试软件栈或128K的结论。

此前诊断摘要：WO-A 修正接回后，T240 完整入口 35 个边界（至最终输出）均逐 bit 一致；
实际生产 55 参数 FULL 入口及 T6 回归也通过。新增 T40/B40、每请求 1 行的诊断入口
35 边界逐 bit 通过；10 月 30 日下午环境恢复后，正式 T40 服务55入口也已通过。
补测第 2 层发现主压缩 Norm 及 Indexer 权重投影首差；两处修正接回后，
**第 2、42 层 T240/B40 的 35 个 Native 边界均全部逐 bit 一致**；
第 4 层正式 55 参数服务 FULL 入口回归也通过，未回退 Native。
18 时继续补测：T40 单行及 C4 闭合、混合18＋dummy6的同图回放、物理T21的六种query长度，均逐 bit 通过。
用户新增 14 卡禁用后，已终止 16 卡 Native，本轮整模型仍未完成。
单卡只使用 8–13、15，原精度基准保留。

本日后续另获本轮八卡`0,8,9,10,11,12,13,15`授权：Native v4已生成全部61,440个token，
但有480次抢占，未通过且不能作gold。v5改为40题分8题×5波，保留零抢占门禁，
21:45用户要求后续避开前8卡，已取消新的Native排队任务并撤销0卡临时白名单；
其后用户临时批准本轮14卡；仅用8–15的五波Native已通过零抢占及完整生成验收。
同配置PTO首次在KV预算检查处退出，未进入token生成；仅减小PTO任务窗口后，
T6/T8/T240正式FULL回归均逐bit通过。整模型重试已在第36次草稿接受事件首差处停止：
`h8192_q04`的PTO接受5、Native接受4；尚不能判定该步输出token是否分叉，整模型验收未通过。
全程不使用前8卡。

最新补充：修复观测终态后，真实模型首步NONE/T8、8请求各1行的21CSA同输入Native逐bit通过。
后续独立token诊断已完成192token检查；接受事件首差仍保留，未改数学包。

## 目标与已授权范围

参考 nalinaly 的
[低接受率测试方案](https://github.com/nalinaly/vllm-ascend/blob/dsv4-flash-pto-v0.25.1rc1/tests/pypto_test/low_acceptance_20261001/RESULTS.md)，
比较 Native、我方精度 TND 与 nalinaly 最新 TND SPMD。
下表保留原16卡三方计划；10月30日后续八卡验证的配置和状态见文末，不混计两种结果。

| 项目 | 本轮设置 |
| --- | --- |
| 原小 T 基准 | `da6b14f0ac158b2b9675dfb79b605a3c46bd90fd` |
| 修正前正式 T240 基准 | `a3498431af3a9d8c9125a44c2a6cd8fcbea829e2`，模型 v4 快照保留 |
| nalinaly 初始 / 更新后的 CSA 提交 | `41ba392f` / `79ea6d3c8460338a172cb5570d14999d5a544262` |
| 当前分支 | `dev/pypto-dsv4-csa-nal-tnd-native-align-20261027-codex` |
| 共同框架计划 | 三方采用 `41ba392f`；我方和 nalinaly 分别覆盖完整 TND SPMD 包 |
| 模型 | `DeepSeek-V4-Flash-0731-w8a8`，75 个正式权重分片 |
| 整模型目标 | TP=1、DP=EP=16，8K/B40，每请求 192 token |
| 替换范围 | CSA；HCA 均保留 Native |
| 配置 | 确定性 level=1、HCCL=true、NZ2、atomic=0、EPLB 关闭 |
| FULL 档位 | 6、240；草稿 5 token；temperature=0 |
| 资源 | 本轮原获 16 卡批准；后新增 14 卡禁用，16 卡任务已终止，单卡限定 8–13、15 |
| 限时 | 单卡任务 780 秒，进程 750 秒加最多 15 秒清理；整模型计划 720 秒执行上限 |

`prepare.py` 的共同框架方案用于隔离 CSA 算术差异，因此今后的整模型结果应写成
“我方数学包在共同框架上的结果”，不能写成未经修改的 `da6b14f0` 完整软件栈结果。
原16卡阶段的 v4 快照已冻结；三方 preflight 及四项公共模块缓存预热通过，
本轮 Native 16 卡基线因后述共享临时目录故障，在获得设备前撤下；环境恢复后重新提交，
后因 14 卡禁用而终止，未完成有效整模型结果；该16卡阶段两份整模型 CSA 未提交。

## 固定输入与工具链

沿用自有只读 `low-acceptance-20261010/banks/fixed8192`：40 个不同问题，
`audit.json` 为 PASS。模型配置有 43 个 target 层，其中 21 个 CSA、20 个 HCA。
8K 指前缀长度；加上问题后各请求实际 history 不完全相同，例如第一个问题为 8393。
本轮不重新生成 bank，不声称缓存字节与 nalinaly 当年的原 bank 完全相同。

公共工具链 `csa-shared-20261024-group-dims`：

- PyPTO：`d44925c17d9ff7d7fa8e5067e41c4b4f80bf3063`。
- PTOAS：`914863ed3d0eb3e92a4433a4a706a86cabf4c91f`。
- Simpler：`57e499238fd4ee3548947ad860be456aaeeb16b5`。
- CANN 9.2.0-beta.2；Torch 2.10.0；Torch-NPU 2.10.0.post2；vLLM 0.25.1。

没有重新安装 vLLM、PyPTO 或 Simpler。已有已验收环境与基准均保留。

## 已完成的前置测试

以下均为真实第 4 层权重、合成 hidden/history 的单层检查，不能替代真实整模型生成。

| 测试 | 结果 | 设备占用 |
| --- | --- | --- |
| T6 生产 55 tensor 入口，无设备生成 | 通过，未初始化 NPU | 无 |
| T6 Native / 我方生产 FULL | 输出及六份 cache/state 逐 bit 一致，保护区通过 | 约 69 秒 |
| T240 生产入口，无设备生成 | 通过，未初始化 NPU | 无 |
| T240 默认 ring heap | AICPU 报临时内存耗尽，尚无精度结果 | 约 306 秒 |
| T240 ring 1 改为 512 MiB | 可以执行；Native 自重复、双方各自 FULL/eager、保护区通过；对 Native 首差在 SWA cache | 约 62 秒 |
| T240 debug 生产入口 | 已保存生产状态与后续观测控制，不能据此判精度通过 | 约 279 秒 |
| T240 分模块观测 | HC、输入 Norm 一致；最早差异为 Q-A 投影 | 约 281 秒 |
| Q-A 三臂微测试 v1 | Native 重放一致；诊断调用漏传 constexpr，未进入三臂数值验收 | 约 21 秒 |
| Q-A 三臂无设备生成 v2 | 三个 receipt 均 PASS，未初始化 NPU | 无 |
| Q-A 三臂 FULL v2 | 原臂复现 44 个差异；仅 N32 仍有 44 个；Native 按列次序为 0 | 约 58 秒 |
| 接回 Q-A 后完整生产入口 | 八状态 FULL/eager、保护区通过；不等于 Native 精度验收 | 约 289 秒 |
| 接回 Q-A 后逐阶段观察 | HC、Norm、Q-A、QR 量化及 scale 均逐 bit 一致；首差推进至 WKV | 约 279 秒 |

### T240 首差证据

生产 55 入口的首个外部差异：`swa.0` 有 24 个 BF16 元素不同，
最大绝对差 `0.001953125`。我方 FULL 与自身 eager 一致。

进一步分模块 FULL 观测先检查观测根与未经观测生产根的一致性，再按依赖顺序比较：

| Native 边界 | 当前结果 |
| --- | --- |
| HC mixed | 逐 bit 一致 |
| HC post gate | 逐 bit 一致 |
| HC combination | 逐 bit 一致 |
| 输入 Norm | 逐 bit 一致 |
| Q-A BF16 投影 | 44 个元素不同，最大绝对差 `0.00048828125` |
| QR、WKV、Indexer、attention、O-proj | 因 Q-A 首差，尚未继续验收 |

接回 Q-A 后的复核：`q_a_projection_bf16`、`q_r_quantized`、`q_r_scale`
均为 0 差异；下一首差 `wkv_projection_bf16` 为 26 个 BF16 元素，
最大绝对差 `0.00048828125`。该复核包含生产/观测等价、FULL/eager 与保护区门禁。
随后启动 WKV 同输入三臂检查；没有跳过该首差推进整模型。

WKV 三臂在物理 8 卡完成，实际占卡约 50 秒：原臂精确复现保存的 FP32 accumulator
及 26 个 BF16 差异；仅改 N32/K256 自然次序仍为 26 个；按 Native 列分组次序后为 0。
三臂 FULL/eager、输入和权重只读、保护区均通过。接回正式候选后仍须完整入口复核。
原微脚本 JSON 中的 `nal` 臂仅表示我方继承的旧投影 helper，
**不是 nalinaly 最新完整 CSA 的实测结果**，不能据此填写三方整模型表。

因此尚不能把 SWA 的全部差异归因于 WKV，也不能声称当前新分支在 B40 已完全对齐 Native。
已验收的小 T 结果仍有效，但不应跨形状推广。

### 内存故障与独立数值问题

默认 ring 1 容量为 `268435456` 字节；故障时使用 `255987712` 字节，
剩余 `12447744` 字节，下次申请 `33554432` 字节，设备日志明确报 heap exhausted。
使用现有公开参数：

```python
pypto.torch.init(ring_heap=(256 << 20, 512 << 20, 256 << 20, 256 << 20))
```

仅调整容量后即能执行并观测到数值首差；没有修改 SDK、生成代码或运算顺序。
后续双方 PTO 若测性能，必须共用相同容量，并记录固定 KV block 数及内存影响。

## 下一项受控验证

`probe_qa240.py` / `qa240_kernels.py` 使用本次保存的 Native Norm 输出和正式 WQ-A 权重，
三臂保持原 N128 生产者及 M64/M16 行归属：

1. 原 N128、K256 自然顺序，要求复现保存的 44 个 BF16 差异。
2. 同一生产者细分为 N32，仍使用自然顺序。
3. 同样 N32，按 Native 的 96 列分组恢复 K256 累加次序。

逐臂核对 FULL/eager、输入只读、前后保护区；第三臂必须与实际 Native CV 边界一致。
该项已通过：原臂精确复现保存的实际输出；Native CV 边界与 canonical linear 一致。
仅细分 N32 不改变误差，恢复按列累加次序后逐 bit 一致；三臂各自 FULL/eager、
输入与权重只读、前后保护区全部通过。正式候选仅为 `tokens==240` 接入已验证次序，
其余 T 保留原路径；正在重新验证完整生产及观测入口，尚不能称整层或整模型通过。

首轮微测试漏传 `q_a_stage(..., QA_M16_PARALLEL=False)`，这是诊断脚本错误，
已显式补齐；不得把该失败计为生产算子精度失败。

## 三方整模型精度和性能

| 实现 | 8K/B40、192 token | 整模型性能 |
| --- | --- | --- |
| Native，共同框架 `41ba392f` | 环境恢复后已重新提交，等待结果；此前未执行模型，不跑 Native2 | 尚未测量 |
| 我方精度 TND `a3498431` | 前置 Q-A、WKV、TopK、WO-A 首差已修复；完整 35 边界通过，整模型待测 | 尚未测量 |
| nalinaly `79ea6d3c` TND SPMD | 尚未运行；不能用历史误差代替本次结果 | 尚未测量 |

计划先核对 Native、我方、nalinaly 的真实入场与 KV 容量，再逐 token、逐请求接受事件比较。
只要发生首差就记录现场并终止后续生成。性能窗口要保留实际 T、位置、FULL 档位，
不能把 graph 的物理 240 行称为每步都有 240 个有效 token。
如果 token 分歧改变后续工作量，整模型耗时不能直接解释为 CSA 纯算子收益。

## 本地测试工具故障

本地清理测试 `test_launch.py` 的子进程参数最初缺少显式角色，导致测试脚本递归启动。
脚本已改为明确的 `parent` / `rank` 分支。
随后本机所有新进程创建持续返回 `Resource temporarily unavailable (os error 35)`，
连精确匹配本次测试的 `pkill` 也无法启动；无法据此声称已清理全部本地残留。
用户已授权清理本次残留。17:20 恢复后，精确匹配旧测试路径未发现残留。
按用户要求，后续测试与编译在 227 运行，本机只做编辑、读取、打包和传输。
修正后的进程清理、token 首差、事件尾部及计时测试在 227 共 6 项通过，耗时 0.86 秒；
进程树显式限定为一个 parent、两个 rank、每 rank 一个 worker，验证同组终止。

早前中断时最后一项单卡任务于 15:58:56 退出；恢复后的 Q-A 微任务也于 17:22:10
退出释放物理 15 卡。随后提交完整 T240 候选验证；本轮仍未提交 16 卡整模型任务。
17:37 通过 GitHub `ls-remote` 再次确认 nalinaly 分支仍为 `41ba392f`。
后续 `prepare.py` 必须用 `--ours <完整提交>` 显式冻结我方完整算术包，
不能把 QA 或后续 WKV 修正继续标成旧 `da6b14f0`。旧基准保留。
提交前 formatter 还规范了 `qkv_stages.py` 的既有换行。227 上比较格式化前后完整 AST
（不含位置属性）一致，证据见 `evidence/qkv-format-proof.json`；该静态证明不算新的 NPU 结果。

### 集中 graph padding 的 Host 合同修正

对照 nalinaly `6acb0ce`，仅删除 `tokens > requests * max_query_tokens` 的物理容量拒绝。
Native 可以让单个失活 dummy 承接全部尾部 padding；图容量与真实 query 上界是不同量。
真实 `actual_tokens`、query 上界 1～6、页表和历史范围，以及本分支 `T≤256` 限制均保留。

227 上修正前的定向 CPU 回归为 2 failed / 5 passed：
`capacity32 / actual9 / requests4 / query6` 与 `capacity240 / actual40 / requests40 / query1`
均被原乘积检查误拒绝。修正后连同清理、首差与计时检查共 13 项通过，0.78 秒。
这只是 Host 合同验证，不能替代新增 padding 场景的设备精度检查。

### WKV 排队记录

WKV 微任务第一项未获得设备，等待客户端的 600 秒期限到达后自动取消；
取消回执明确“超时前未分配到设备”，输出目录不存在，没有数值结果或占卡时长。
确认取消后重新提交物理 8 卡的同输入微任务，仍在允许的 8～15 范围内。
后续排队使用已知 ID 的 `--status` 观察，避免等待客户端再次自动取消排队任务。

### 卡外编译与共同 bootstrap

`precompile_eager.py` 使用本次已固定共享 SDK 中现有的 eager artifact 解析接口，
不更改 SDK 或生成代码。3 个 Q-A 小根在 227 CPU 完成编译，分别约 12 秒，
均确认 `Worker` 未创建、NPU 未初始化。完整候选改用同一 eager 持久缓存预热；
普通 program compile 使用不同缓存，不能用它的完成来推断 NPU 运行时会命中。
缓存命中及完整入口数值仍需各自的设备任务验证。

共同框架 `41ba392f` 内原有 bootstrap 不支持本轮 `--native-plugin-root` 参数，
已在无设备 preflight 中发现，未提交整模型。准备脚本随后显式为三方统一覆盖
我方已验证的测试 bootstrap，并单独记录该 helper 提交；模型框架仍为 `41ba392f`。
失败的 v2 源码快照留作记录，后续重新生成新快照，不能沿用旧 manifest。

## 复现入口

公共环境与物理卡检查由 `nal_native_alignment_20261027/run_rms.sh` 负责。
`ALIGN_ROOT` 设为自有工作区，先 source 该目录的 `src/tests/pypto_test/nal_native_alignment_20261027/env.sh`。
下列任务均显式选用物理 8 卡，外层 780 秒；CPU 编译与 pytest 不占 NPU。

```bash
test_dir="$ALIGN_ROOT/src/tests/pypto_test/nal_native_alignment_20261027"
lengths=6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6
task-submit --device 8 --max-time 780 \
  --env PYTHONNOUSERSITE=1 --env HCCL_DETERMINISTIC=true \
  --env PTOAS_ROOT --env ASCEND_HOME_PATH \
  "bash $test_dir/run_rms.sh fresh-t240-production spmd --lengths $lengths"
```

生产任务完成后，给第二个小任务增加
`--env SPMD_PRODUCTION_REFERENCE=$ALIGN_ROOT/results/fresh-t240-production/production.pt`，
标签换成 `fresh-t240-native`，其余参数相同。第二项先证明输入、源码和生产/观测输出相同，
再按依赖次序比较 Native；应读取 `result.json` 的 `pass`，不能只看任务退出码。
诊断脚本捕获首差后保存完整结果并正常退出，因此 `exit=0` 不代表数值通过。

Q-A 微测试使用 `probe_qa240.py`，增加
`--env QA240_PROBE_REFERENCE=$ALIGN_ROOT/results/t240-stage-native-v1/observation.pt`，
将 probe 参数从 `spmd --lengths ...` 换成 `qa240`。该保存输入的历史生成版本为
`da6b14f0`，其 first difference 必须为 Q-A 的 44 个元素；微根原臂显式保留修前
M64/N128 helper，避免正式 dispatcher 修复后丢失负对照。
第三臂通过仍必须继续完整入口验证，不能直接宣称整模型通过。

CPU 验证在 227 执行：

```bash
python -m pytest "$ALIGN_ROOT/src/tests/pypto_test/csa_threeway_20261029/" \
  -v --forked
```

整模型准备脚本必须在包含所选 Git 提交的源码仓库运行：
`prepare.py <新快照目录> --ours <已验证的新提交>`。
随后用 `link_runtime.py` 只在新快照绑定既有 Native 二进制，再按快照依次运行无设备 preflight。
这些步骤准备就绪并不等于已完成整模型测试；本轮当前仍被 WO-A 数值门禁阻挡。

## WKV 修正后：首差推进至 TopK

WKV T240 微测试在 227 物理 8 卡完成：原 helper 和仅缩窄 N32 的负对照均复现
26 个 BF16 差异；采用 Native 按输出列分组的 K256 累加顺序后，差异为 0。
原臂 FP32 累加器与完整入口保存值逐 bit 相同，FULL/eager、输入只读与保护区检查通过。
这仍是单层证据，不是整模型 token 验收。
微测试 JSON 中的 `nal` 是我方继承的旧 WKV helper 标签，不能解读为 nalinaly 最新整套实现。

接回 Q-A 与 WKV 的完整 T240 观测 `t240-qakv-native-v3`，先证明生产/观测根及输入相同，
随后 HC、输入 Norm、Q-A、Q 路径、WKV、六份 cache/state、Indexer query、FP16 scale
和 FP16 head weights 等前 31 个边界均逐 bit 一致。第一个差异为 TopK：
第 183 行的第 60、61 位（均从 0 计数），我方为 `[1967, 820]`，Native 为 `[820, 1967]`；
整张 TopK 表仅这两个 INT32 元素不同。

当时尚不能把它定性为“同分排序”：必须先核验两个候选实际 FP32 分数。
诊断脚本只导出现有 `topk_scores` 缓冲，并增加 eager/FULL 与生产/观测一致性控制；
不改变 kernel 算术、排序树或分配布局。Native QLI 当前接口不支持 `return_value=True`，
不能以未写入的输出或 CPU 求和代替 Native 分数参考。

后续 `t240-score-native-v4` 确认生产/观测 `topk_scores` 逐 bit 相同；
两候选的我方 FP32 分数同为 `2.7790469175670296e-05`，位值均为 `938024852`。
这没有直接导出 Native 分数，不能据此声称全部打分累加已与 Native 逐 bit 相同。

严格同分控制 `topktie-native-control-v2` 使用实际保存的 Native QLI metadata，
只让第 183 行的 head0 参与，Q/K 的 128 维均为 INT8 1、scale/weight 为 1，
两个原始逻辑位置 820/1967 因而严格得到 `128 / 1024 = 0.125`，无 head 求和歧义。
Native 自重复一致，顺序为 `[820, 1967]`；我方原 1152/897 半叶树为 `[1967, 820]`；
2048 前缀排序为 `[820, 1967]`，两种 DSL 微根的 FULL/eager 均一致，脚本耗时约 12 秒。
该试验只证明这两个正分数的同分优先级，不能外推为所有尾长或完整模型通过。
v1 因没有展开 Native `[240,1,512]` 输出维度，约 1.4 秒后退出；保留失败回执，
不计入数值验收。

修复候选保留原 QK/WS/FIX/scale 数学与 FIFO，短历史 G6 的两个 AIV 将分数写入
现有 288×8192 arena 的真实 query 行，最终发布者保持原两个 owner 等待，
再按连续 2048 段排序、后段优先归并。没有新增 GM 分配或改用槽 ACK 作为完成信号。
`topktie-native-control-v3` 已验证最终显式分段 helper：两个正分数的顺序与 Native 相同；
再令全部 2049 个有效候选严格同分，完整 512 个 TopK 索引也与 Native 相同，
前缀为 `[2048, 0, 1, 2, 3, 4, 5, 6]`，FULL/eager 一致，脚本约 20.7 秒。
这覆盖当前尾 1 的截断与后块优先级，不代表所有尾长、整模型或性能已经通过。
当前正在 227 CPU 检查完整根生成物；必须再经完整入口首差检查，不能用微控制替代验收。

### 完整 T240：TopK 与 attention 已对齐，首差在 WO-A

`t240-topk-production-v7` 完整生产控制通过，FULL/eager 的全部外部状态、
TopK 分数与保护区均通过，脚本约 33.7 秒。`t240-topk-native-v7` 先证明生产/观测、
源码与输入相同，再与 Native 比较：前 33 个边界全部逐 bit 一致，
包含整张 TopK 和 inverse RoPE 后的 attention heads。

首差为 `wo_a_bf16`：240×8192 个 BF16 元素中 607 个不同，
最大绝对差为 `0.0001220703125`，脚本约 69.1 秒。
最终输出尚未因这次结果取得验收资格；本轮没有启动任何 16 卡整模型任务。
已停止后续配置，转为使用相同 Native heads、正式 WO-A 权重的投影微测试。

微测试保留原 M128/N256 的 64 个父任务及 owner，依次验证原臂 FP32 复现、
仅拆 N128、以及 Native owner 对应的 8 个 K256 起点假设。
历史 T240 标定的起点由 24 核分工推导：前 48 个 N128 逻辑项每核 3 项，
后 16 项每核 2 项，`first_panel=(native_owner % 8)*2`。
历史数据只用来提出假设；当前 Native 自重复、微根正负对照及完整入口验证仍不可省略。

`woa240-native-order-v2` 已在物理 8 卡完成，约 30.9 秒：

| WO-A 微臂 | 与 Native 不同的 BF16 元素 | 控制结果 |
| --- | --- | --- |
| 原实现 M128/N256 | 607 | FP32 逐 bit 复现完整入口保存值 |
| 只拆成 N128，K 自然序 | 607 | 未解决误差 |
| 保留父 owner，N128 + Native 八起点 | 0 | Native BMM 重放、3 次 FULL/eager、保护区、只读均通过 |

这将误差缩小到 K 累加次序；尚未赋予新完整入口或整模型资格。
v1 在投影执行前因源码清单数量检查错误退出，约 0.44 秒；修正后仍逐文件核验完整清单，
没有放宽数值或 guard 检查。三个微根均先在 227 CPU 编译，分别约 12 秒，NPU 未初始化。

微测试复现：使用上述公共环境与物理 8 卡，给 `run_rms.sh` 的 probe 参数 `woa240`，
并传入 `--env WOA_PROBE_REFERENCE=$ALIGN_ROOT/results/t240-topk-native-v7/observation.pt`。
它要求保存输入为 T240、40 个 S6，且首差确为 WO-A、此前所有边界通过；
不能把别的版本保存件直接改名后使用。
原三臂微测试应使用[接回前检查点](https://github.com/Leaf-Salix/vllm-ascend/commit/2daa609bc2e6815892c41da360d2746dc1c9fa18)复现：接回后其 MODE0 公共入口已走新路径，
不能再把它称为原 607 差异的负对照。

### WO-A 正式接回与当前单层结论

正式实现只在 `T=240 / OA_ROWS=128 / OA_COLS=256` 使用新路径，保留原 64 个父任务及
每组发布/等待协议，父任务内部计算两份 N128；其余 T 与矩形路径保留。
八个起点显式为编译期常量 0/2/4/6/8/10/12/14，避免现有 SDK 无法证明动态 NZ K 偏移
非负的问题。该表达与已验证的 owner 公式等价；没有修改 SDK、codegen 或生成 C++。
v8/v9/v10 都在 CPU 生成检查时退出，没有占用 NPU。

| 检查 | 结果 | 脚本耗时 |
| --- | --- | --- |
| v11 完整 debug / probe 无设备编译 | 均通过；NPU 未初始化、Worker 未创建 | 234.5 / 223.4 秒 |
| `t240-woa-production-v11` | 全部外部状态、分数、FULL/eager 与保护区通过 | 48.9 秒 |
| `t240-woa-native-v11` | 35 个 Native 边界逐 bit 一致，含 WO-A 与最终输出 | 80.8 秒 |
| `t240-woa-production55-v12` | 实际 55 参数入口通过，无 Native 回退；物理 12 卡 | 62.6 秒 |
| `t6-woa-production55-v11` | 实际 55 参数小 T 回归通过，无 Native 回退；物理 8 卡 | 54.6 秒 |

生产 55 入口的 v11 尝试在执行投影前被旧 T18 源码清单挡住，约 0.004 秒退出。
v12 使用 `PADDING_STAGE_REFERENCE` 明确绑定已通过的同版本 35 边界记录，逐文件完全核验，
不继承旧两文件例外；数值、Native 自重复与保护区门禁均保留。

排队期间只跟踪本任务 ID，没有修改共享服务或他人任务。
后续单卡提交使用自有 `DEVICE_WHITELIST=8,9,10,11,12,13,14,15` 配置和同范围
`TASKQUEUE_DEVICE_POOL` 自动选 1 卡，禁止忽略白名单，并在结果中记录实际物理卡号。
曾准备取消固定 8 卡排队任务时，任务已完成，客户端未执行取消，也没有重复提交该观测。

本节仍是第 4 层真实权重、合成 hidden/history 的单层资格。
不能将 35 个边界或满 S6 控制通过写成整模型 192 token、所有尾长、混合请求或性能已验收。

### 整模型编译准备

`csa_threeway_20261029/precompile.py` 在公共 bootstrap 选定的冻结源码下，使用服务实际
`vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd` 模块名编译 55 参数入口，
不使用诊断私有 namespace；前后均检查没有 NPU/Worker 初始化。
nalinaly `41ba392f` 的 T240、T6 已通过此无设备检查。
最终整模型快照仍需按本次新提交重新冻结，再对准确源码路径做预热与 preflight，
不能将旧 v3 的我方数学包或这些 CPU 回执当作新整模型结果。

冻结前再次查询 GitHub，nalinaly 已更新至 `79ea6d3c`。新增短历史下界支持和设备选档
实验记录，8K 使用的 entry/Cube/Vector/算术文件没有变化；policy/service 的变动主要
扩大短历史准入。本轮保持共同框架 `41ba392f`，参考臂覆盖最新 `79ea6d3c` 的完整 CSA 包，
我方臂覆盖本次已验收数学包，以隔离框架变动。报告应称“共同框架下的两份实现”，
不能称 nalinaly 最新整仓、SDK、实验性 94 策略 C++ 派发器的完整复刻。

无设备 eager 缓存预热在 227 CPU 完成，debug/probe 两根分别约 242/246 秒，
均未初始化 NPU 或创建 Worker；命中后设备任务的脚本耗时约 34/64 秒。
本机没有运行 pytest、JIT 编译或硬件测试。三方新快照的 CPU preflight 已通过，
上述单层首差已消除，整模型轮次开始；尚无整模型 token 或性能结果。

### v4 整模型验收门禁

v4 的共同框架为 `41ba392f`，我方 CSA 为 `a3498431`，nalinaly CSA 为 `79ea6d3c`。
两份实现的公共模块 T6/T240 无设备预热分别约 190.6/219.0 秒和 154.1/214.7 秒，
全部在 227 CPU 执行，没有初始化 NPU 或创建 Worker。

随后静态复核确认，首步 T40/request capacity=40 可能走普通服务路径；固定 query 上界 6
在此选择 G2、OB_ROWS=96、关闭半叶 UB，和上述两档均不同。因此额外补齐两份实际公共
模块的 T40 缓存：我方 199.7 秒、nalinaly 161.1 秒，均通过无 NPU/Worker 断言。
`precompile.py --tokens 40` 使用 requests=40，其余复现参数与 T240 相同。
profile T400 超过服务 T≤256 范围，不进入 PTO；若实际发生其他非均匀 eager/图重派发，
仍可能产生额外策略，不能用这六份 CPU 回执宣称所有潜在档位都已预热。

独立复核发现，混合请求入口提前返回会绕过普通入口已有的覆盖断言。
因此父启动器增加 `postflight.py`：全部 rank 结果落盘后，再核验以下条件并保存
`postflight.json`；文件中精度、替换覆盖和计时失败分别记录。

- 16 个 rank，每个 40 个不同请求，各输出 192 token；低接受率协议计数一致、零抢占。
- 全部 21 个 CSA 实际安装目标 TND SPMD runtime，20 个 HCA 均为 Native。
- 21 个 CSA 的 T6/T240 准备期 eligible 选择记录存在；此累计计数本身不证明 T6 实际回放。
- 满 B40/FULL240 路由全部 CSA 图回放，无 Python 重提交；合法首步 T40 普通提交、
  完成请求后的空闲 EP 补齐不混入此判据。
- 8 步热身后存在连续 20 个目标 FULL240 计时样本，时间值有限且为正。
- 两份替换实现相对本轮 Native 的逐请求 token、草稿接受事件完全相同，
  并核对计时窗内实际请求数、真实 T、图容量和位置轨迹。

父启动器与验收脚本 SHA256 写入 `experiment.json`；v4 模型源码快照保持不变。
计时是整模型 FULL240 target forward 与相邻 target 周期，不是截图中的单层 L1 耗时；
周期包含草稿、采样、metadata 和 CPU 参考检查，解释性能时需保留此限制。
门禁和清理回归共 22 项在 227 CPU 通过；最终增量 10 项也通过。本机只运行静态检查。

### 调度提交参数校验

本机安装的 `task-submit` 只读取一个完整命令字符串，不能按普通 argv 形式传递
`bash script args`。首次 Native 提交错误地只保存了 `bash`；该任务始终 pending，
在获得设备前已取消，没有执行模型或占卡。无设备参数探针随后以单一字符串提交，
成功输出 `Native|40|192`，也证明 CPU 派发正常。

修正后的 Native 任务已逐字段核对自身队列元数据：完整脚本和参数、物理 0–15、
设备池、白名单以及 `MAX_TIME=780` 均正确。随后因共享临时目录故障，在获得设备前撤下，
不能当作整模型结果。
两份替换实现也必须在对应 Native 的 `postflight.json` 通过后，按同一方法顺序提交。

本轮已获明确 16 卡授权，提交形式如下；其他测试仍遵循物理 8–15 的默认范围。
`arm` 依次为 `native`、`ours`、`nalinaly`，替换实现的 `reference` 指向已验收的 Native
结果目录，Native 使用 `none`；每次使用新的 `label` 保留历史。

```bash
source "$align_root/src/tests/pypto_test/nal_native_alignment_20261027/env.sh"
cohort="$align_root/cohorts/threeway-source-v4"
job_command="bash $align_root/src/tests/pypto_test/csa_threeway_20261029/run_case.sh $cohort $arm $label $reference"
TASKQUEUE_DEVICE_CONF="$align_root/csa-approved-threeway16-20261029.conf" \
TASKQUEUE_DEVICE_POOL=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
TASKQUEUE_IGNORE_WHITELIST=0 TASKQUEUE_IGNORE_GROUP_AFFINITY=0 \
task-submit --device 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --max-time 780 \
  --env PYTHONNOUSERSITE=1 --env HCCL_DETERMINISTIC=true \
  --env PTOAS_ROOT --env ASCEND_HOME_PATH "$job_command"
```

## 10 月 30 日：首步 T40/B40 服务策略补测与环境阻塞

等待整模型资源期间，补测 40 个真实请求各 1 行的首步：T=40，history=8192，
实际 seq length=8193，没有逐请求补到 S6。使用第 4 层真实权重、合成 hidden/history，
沿用现有单层 Native 控制源；不能替代 21 个 CSA 层的整模型 token 验收。
正式数学包仍为 `a3498431`，本次只修改诊断选档和形状参数。

复核发现，旧诊断按 `max(request_lengths)` 选策略，服务却固定传
`service_config.MAX_QUERY_TOKENS=6`，真实 Q/L 仍由 metadata 决定。
T40/B40/query1 下，两者会选择不同的半叶 UB 路径。因此本次诊断及其 eager 预编译
统一使用服务上界 6；记录中 `service_query_bound=6`，实际 query 仍为 1。
历史报告保留原脚本版本，不能将旧实际 query 选档结果改称本次服务路径结果。

私有命名空间下 57/85/55 三个入口先在 227 CPU 编译并加载，分别约
191.7/193.8/191.1 秒；均确认没有初始化 NPU 或创建 Worker。

| 任务 | 实际物理卡 / 墙钟时间 | 结果 |
| --- | --- | --- |
| `t40-service-native-v1`，生产 debug 控制 | 10 / 58 秒 | FULL/eager、TopK 分数与保护区通过；仅保存生产快照，不是 Native 精度通过 |
| `t40-service-native-v2`，同快照分模块观测 | 10 / 76 秒 | 35 个 Native 边界全部逐 bit 一致、无首差；观测/生产控制也通过 |
| `t40-service-production55-v1`，正式服务入口 | 10 / 39 秒 | Native FULL/eager 及两次回放逐 bit 一致；Simpler 初始化失败，PTO 计算尚未执行 |

对应轻量 JSON 与任务起止日志位于本目录 `evidence/t40-service-*`。
前两项脚本分别耗时 41.5、59.9 秒；第三项故障脚本 21.2 秒。
所有单卡任务均退出并释放设备，没有超 15 分钟；本机只做编辑、同步和静态检查。

### 故障证据与当前限制

01:46 的错误来自 `LoadAicpuOp::GenerateAicpuOpJson`：无法创建
`/tmp/simpler_inner_7df8dbfd4cd92d81_273217.json`，返回 `-1000`。
01:52 只读快照确认：

| 文件系统 | 总 inode | 空闲 inode | 剩余字节 |
| --- | --- | --- | --- |
| `/tmp` | 1,048,576 | **0** | 1,045,429,604,352 |
| 本轮 `/data` 实验所在文件系统 | 738,197,504 | 705,980,004 | 6,182,209,052,672 |

这不是磁盘字节不足，也没有测出新的数值首差：失败发生在 PTO 初始化、正式计算之前。
现有公共 Simpler 的 `src/common/aicpu_loader/host/load_aicpu_op.cpp` 使用字面量
`/tmp/simpler_inner_%016lx_%d.json`，没有读取 `TMPDIR`；仅改环境变量不能绕开。
代码与文件系统快照见 `evidence/shared-tmp-inode-blocker-20261030.json`。

没有修改共享 SDK、全局设备策略或其他用户文件。修正命令后的 Native 任务
`task_20261029_233842_26648469997` 在 pending 状态取消，没有执行模型；
当前本轮任务已全部退出或取消，没有遗留排队/运行任务，也没有创建自动化。

环境恢复后，先复测这一项正式 T40 服务入口：
`run_rms.sh <新label> padding --full-batch 40 --query-rows 1`，
并显式传入 `PADDING_STAGE_REFERENCE=$ALIGN_ROOT/results/t40-service-native-v2/result.json`。
该参数默认仍为 6，不改变历史满 S6、混合或投影测试。此项通过后，再恢复上节已授权的
16 卡 Native → 我方 TND → nalinaly TND 顺序；仍不跑 Native2 或 128K，首差停止定位。

## 10 月 30 日下午：环境恢复，正式 T40 服务验证通过

14:52 复核 `/tmp` 已有 630,625 个空闲 inode。三方无设备 preflight 再次通过，
公共工具链仍为 `csa-shared-20261024-group-dims`、status=validated；导入路径、
40 个固定问题、75 个权重分片和冻结版本均保持。共享源码是归档目录，没有 Git 元数据，
本次按公共 manifest 与精确导入路径核对，没有安装或重新构建工具链。
摘要见 `evidence/resume-20261030-preflight-summary.json`。

正式入口复测 `t40-service-production55-v2`（任务 `task_20261030_145630_1700530485`）
在物理 8 卡通过，脚本耗时 54.161 秒，设备已释放：

- B40、每请求 query=1，物理 T40，没有补到 S6，服务选档上界仍为 6。
- 实际调用55参数生产入口，明确检查 eligible、实际提交计数和 capture contract，未回退 Native。
- FULL 两次回放、eager 与 Native 的七组外部结果全部逐 bit 一致，保护区通过。
- Native 自重复、Native FULL/eager 也逐 bit 一致；complete/pass=True，无首差或异常。

原失败 `v1` 保留为环境故障记录，不改写成数值首差。
这补齐的是单层真实权重/合成输入的服务入口资格，不能扩展为整模型 token 或性能已通过。

随后按同一轮既有授权恢复16卡8K/B40/192-token基线：
`task_20261030_145850_178273515760`，新输出目录 `threeway-v4-native-8k-b40-r2`。
自身提交元数据已确认完整命令、显式物理0–15、设备池及780秒上限；内部仍720秒。
此任务尚在等待调度，将作为本轮首次实际执行的 Native 基线，不是 Native2。
GitHub 再次查询的 nalinaly 分支 tip 仍为 `79ea6d3c`，本轮继续沿用既有冻结版本。

## 10 月 30 日：第 2 层补测与压缩 Norm 首差定位

用户要求优先补单卡精度。诊断新增 `--layer`，默认第 4 层不变；生产控制快照记录层号，
后续 Native 观测显式验证同层。先用真实第 2 层权重、8K 历史、B40/T240、各请求 6 行，
不把单层合成 hidden/history 对拍当成整模型 token 验收。

| 验证 | 任务 | 结果 / 脚本时间 |
| --- | --- | --- |
| 第 2 层生产 FULL/eager 控制 | `task_20261030_161802_413690422912` | 完成，35.677 秒；仅保存生产参考，尚未对拍 Native |
| 同输入 Native 分模块观测 | `task_20261030_162039_122281819` | 前 11 边界逐 bit 一致；主压缩 KV 首差，39.422 秒 |
| 同 pooled 的 Norm 单因素验证 | `task_20261030_165913_122872215557` | 物理 8，12.148 秒；直接除法臂通过，三次 FULL 回放与保护区通过 |

首差为 `main_compressed_cache` 的 `[2114,0,0,501]`，仅一个 BF16 元素：
我方为 1.8125，Native 为 1.8046875，差 0.0078125。第 42 层及混合请求补测随即停止。
同次观测/生产等价、Native 重复、压缩 state、此前 Q/KV/RoPE 边界均通过；
不能仅因任务 exit=0 就认为精度通过，本项 `complete=True, pass=False`。

227 离线定位把该点映射到 compact 第 32 行、闭合 token 195、RoPE 第 53 列。
我方 RoPE 后 FP32 为 1.8085938692092896，仅比相邻 BF16 中点 1.80859375 高一个 FP32 ULP。
Native `arch32/rms_norm.h` 的 `RowDivs` 是先直接除以 sqrt，再乘 gamma；
原 PTO 实现为先 recip，再乘 pooled。Native 没有在 Norm 与 RoPE 之间额外舍入 BF16。

### 单因素方案与结果

使用本次保存的全部 40 行有效 pooled、BF16 gamma、compact cos/sin，完整计算 512 列，
只改变 Norm 除法，保持平方和树、eps、gamma、RoPE 两次乘法再加法及末端 BF16 不变：

| 检查 | 不同元素数 |
| --- | --- |
| 原 recip-mul 微根对保存的生产 FP32，全部 512 列 | **0**，先确认观测有效 |
| 原微根 BF16 对 Native 融合 compressor | **1**，精确复现 `[32,501]` |
| 直接除法臂 BF16 对 Native 融合 compressor | **0** |
| 两臂各三次 FULL graph 对自身 eager | **0** |

微根先在 227 登录端 CPU 编译，明确没有 NPU 初始化或 Worker；设备任务只执行缓存产物。
诊断文件是本目录相邻的 `nal_native_alignment_20261027/compressor_rope_kernels.py`、
`precompile_compressor_rope.py`、`probe_comprope.py`。后者通过
`COMPRESSOR_ROPE_REFERENCE=<原观察目录>/observation.pt` 固定输入；命令为
`run_rms.sh <新label> comprope`。任务使用一个完整 shell 命令字符串提交，
外层 780 秒、内层 750 秒加 15 秒退出清理上限。

轻量证据见 `evidence/t240-layer2-native-v1.json`、`layer2-compressor-cpu-focus.json`、
`layer2-compressor-rope-cache-v2.json`、`layer2-compressor-rope-native-v1.json`。
独立复核确认正式补丁只改 Norm 的 NoPE/RoPE 两处为直接除法；原 `a3498431` 数学基准与
三方 v4 冻结快照保留。微根通过不能推广为整层或整模型通过。

修正后的完整 debug57/probe85 根无设备编译分别耗时 208.147/211.424 秒，无 NPU/Worker。
第 2 层生产控制 `task_20261030_170512_140116317675` 耗时 34.168 秒；
Native 观测 `task_20261030_170843_149141030816` 耗时 63.958 秒，均使用物理 8。
**主压缩 KV 首差消失，前 30 个精度边界逐 bit 一致；新的首差是 `qli_weights_fp16`，2 个元素。**
因此完整第 2 层仍为 `complete=True, pass=False`；未测第 42 层或恢复整模型。
修正后证据见 `evidence/t240-layer2-direct-native-v1.json`。

新的差异坐标为 `[73,55]`、`[188,48]`，我方/Native 分别为
`-1.8715858459472656e-5 / -1.8596649169921875e-5` 和
`-0.00015354156494140625 / -0.0001544952392578125`。
正式权重投影目前按四份 K1024 部分和（每份两段 K512）再左折叠；
已保留 Native 的投影 BF16、乘系数 BF16、consumer FP16 边界。
下一微测试固定保存的输入、真实第 2 层权重与舍入，只隔离 K 累加，不能先认定自然升序就是 Native 顺序。

### Indexer 权重投影单因素结果

`probe_weight240.py` 固定同次保存的 Native 输入与第 2 层正式 BF16 权重，
先用 Native `F.linear → scale → half` 重放三份实际 hook，全部逐 bit 通过。
MODE0 必须完整复现我方已保存的 FP32 系数；随后仅改投影累加，保持两处 BF16 边界。

| 臂 | 投影 BF16 不同元素 | scale 后 BF16 | consumer FP16 |
| --- | --- | --- | --- |
| 0：原四份 K1024 partial，各含两段 K512，再左折叠 | 5 | 5 | 2 |
| 1：M16/N64 不变，连续 K512 自然序 | 6 | 6 | 4 |
| 2：连续 K256 自然序 | 6 | 6 | 4 |
| 3：M16/N64/K256 不变，按行块调整 K 起点与方向 | **0** | **0** | **0** |

三臂负控任务 `task_20261030_172241_186394630064`（20.685 秒）排除“只要连续累加就会对齐”；
四臂任务 `task_20261030_172649_19539362045`（20.346 秒）确认 MODE3。
均使用物理 8，每臂三次 FULL graph 对自身 eager 逐 bit 一致，保护区及只读输入检查通过。
负结果没有用放宽阈值改成通过，也没有采用误差更大的自然序。

MODE3 参考旧精度基准对 T240 的标定，但以上结论来自本次 CANN 9.2 与真实第 2 层输入的验证：
`rb = row // 16`，`direction = 15-step`（rb 为奇数）否则为 step，
`block = (((rb % 14) // 2) * 2 + direction) % 16`，实际 K 偏移为 block×256。
规则不依赖 N 组号，故本次继续保留 N64；没有把旧环境的精度结论直接挪用。

四臂无设备编译凭证为 `evidence/layer2-weight240-*-cache-v3.json`，
结果见 `evidence/layer2-weight240-native-v1.json`、`layer2-weight240-native-v2.json`。
复现使用 `precompile_eager.py <新JSON> weight240-0/1/2/3`，再通过单卡启动器执行
`run_rms.sh <新label> weight240`，显式传
`WEIGHT240_PROBE_REFERENCE=<t240-layer2-direct-native-v1目录>/observation.pt`。

接回版本在 `indexer_native_query_stages.py` 中增加仅 T240 生效的 dispatcher，
完整投影写入原 partial 缓冲第一段，消费者在 T240 跳过其余三段；
其他 T 保持原实现。两个 Cube 调用点均切到 dispatcher，保留全部 24 核发布和保守等待，
不改公共 ABI、请求 metadata、舍入或量化边界。
接回完整根仍需无设备编译及 Native 对拍，尚不能称为整层或整模型已通过。

接回时的无设备编译曾被 NZ 非负偏移检查拒绝，未因此提交设备任务。
使用现有 `CompileOptions` 的 pass dump 发现：Simplify 移除 `max` 后，
方向是分支合并的 phi，后续 NZ 检查无法追踪其非负性。只增加 worker clamp 或改变 max
左右顺序均未解决，没有修改 SDK。最终 DSL 写为
`parity=rb%2; direction=step*(1+14*parity)+15*parity`，再计算原 block 模 16：
奇数行块的该式与 `15-step` 相差 `16*step`，故 K 顺序完全相同，且全部项非负。
诊断入口的 `--dump-passes` 强制新 CPU 编译并保留 IR，其收据不能充当 eager 缓存命中证明。

### 两处修正接回后的完整验证

等价正项索引版本的 57/85/55 根 CPU 预编译分别耗时 227.931/214.592/214.751 秒，
均确认未初始化 NPU、未创建 Worker。此前 NZ 失败的编译不计入设备精度结论。

| 验证 | 任务 | 物理卡 | 脚本时间 | 结论 |
| --- | --- | --- | --- | --- |
| 第 2 层 T240/B40 生产 FULL/eager 控制 | `task_20261030_174658_2571453129` | 10 | 34.902 秒 | 控制通过，不单独计 Native 精度 |
| 第 2 层同输入完整 Native 观测 | `task_20261030_175033_266102215806` | 10 | 75.978 秒 | **35/35 边界逐 bit 一致，complete/pass=True，无首差** |
| 第 42 层 T240/B40 生产 FULL/eager 控制 | `task_20261030_175333_279974929237` | 10 | 33.970 秒 | 控制通过，不单独计 Native 精度 |
| 第 42 层同输入完整 Native 观测 | `task_20261030_175530_28868617284` | 8 | 74.121 秒 | **35/35 边界逐 bit 一致，complete/pass=True，无首差** |
| 第 4 层 T240/B40 正式 55 参数服务 FULL 入口 | `task_20261030_175750_292702622155` | 8 | 50.808 秒 | **七组外部结果与 Native 逐 bit 一致，无回退** |

输入、观测与生产等价控制、Native hook 控制、保护区均通过；结论包括最终输出，
没有通过注入 Native 中间值替换我方计算来获得验收结果。
这证明本次第 2、42 层真实权重/合成输入配置通过，仍不能扩展为整模型 192-token 已通过。
轻量证据见 `evidence/t240-layer2-weight-order-native-v1.json` 与
`evidence/t240-layer42-weight-order-native-v1.json`、`evidence/t240-weight-order-*-cache-v4.json`；
原 `a3498431` 与模型 v4 冻结包未覆盖。本轮没有新增性能测量，任务脚本时间不是算子延迟。

服务入口固定 `PADDING_STAGE_REFERENCE=<第2层通过目录>/result.json`，严格核对全部数学源码 hash；
服务实际仍加载第 4 层，独立生成本层 Native 金标，不借用第 2 层的输出或中间值。
检查实际 55 tensor ABI、eligible/提交计数及 graph contract，禁止静默回退；
七组外部结果、两次 FULL 回放、FULL/eager、Native 自重复和保护区全部通过。
证据见 `evidence/t240-weight-order-production55-v1.json`。

本次接回的新版本只新增上述 T240 回归；小 T、混合 query、dummy 的此前通过记录属于各自历史版本，
不能无条件继承为本次修正后的新结果。下一步先补这些小配置，再考虑重新准备整模型快照。
完整任务起止与设备锁释放日志见 `evidence/layer2-layer42-task-lifecycle.log`；
本轮所有任务均已结束，单卡设备占用低于 2 分钟，没有遗留排队或运行任务。

完整链路复现：先 source 本实验 `env.sh`，由公共 bootstrap 的 `--shared-toolchain`
与 `--native-plugin-root "$SOURCE_ROOT"` 启动 `precompile_eager.py <新JSON> spmd-debug`、
`spmd-probe`，默认 `--lengths` 为 40 个 6，确认 57/85 参数 receipt 无设备编译通过。
随后在单卡任务执行 `run_rms.sh <生产label> spmd --lengths <40个6逗号分隔> --layer 2`，
控制通过后，用相同参数、新 label 和
`SPMD_PRODUCTION_REFERENCE=<生产结果目录>/production.pt` 执行 Native 观测。
必须验证 `complete/pass=True`、35 个精度边界无首差及输入/观测等价性，不能只检查退出码。

### 14 卡禁用后的任务处理

用户新增明确要求 **14 卡不能用**。此前本轮批准的 16 卡 Native 任务在 16:55 左右获得设备，
收到限制后立即请求终止，16:58:29 确认 `completed (exit=130)`；本次未完成整模型基准，
不记录 token 或性能通过。后续不重提包含 14 卡的任务。

单卡池和私有白名单统一改为 **8,9,10,11,12,13,15**，启动脚本再次检查实际卡号，
即使调度参数错误也拒绝 14 卡；没有修改全局或其他用户策略。

## 10 月 30 日 18 时：短 query、混合与闭合边界补测

用户要求继续检查精度。正式算术固定在 `05989cfb06cd4e3fc7597a3971b0903d5c12faf9`，
本轮没有更改算子或工具链；仅测试预检增加实际 Python/SDK 导入路径断言和记录，
服务对拍脚本增加 `--history 8191/8192`，默认仍为 8192。
每项均在 227 物理 8–13、15 白名单内单卡运行，Native 开启确定性，首差停止后续配置。

无设备预检通过，精确 PyPTO/Simpler 路径与公共 manifest 一致，PTOAS 和 CANN 存在；
见 `evidence/precision-recheck-1818-preflight-v2.json`。
T40、T24 正式 55 参数入口先在登录端 CPU 预编译，分别耗时 192.430、190.133 秒，
明确未初始化 NPU 或创建 Worker。测试没有重新安装依赖，也没有在本机运行 UT/JIT/模型。

| 配置 | 物理输入 / bucket | Native 对拍 | 脚本时间 |
| --- | --- | --- | --- |
| 第 4 层 B40，query 全 1，history=8192 | T40，无每请求 S6 padding | 正式 55 参数 FULL；输出及六份 cache/state 逐 bit 通过 | 48.102 秒 |
| 第 4 层 B4，满 S6→`[1,5,6,6]`→满 S6 | 同一 FULL 图；24→18 有效＋6 dummy→24 | 三阶段全部逐 bit 通过，metadata 地址不变，未重新提交 | 54.608 秒 |
| 第 4 层 B40，query 全 1，history=8191 | T40，单行恰好 C4 闭合 | 正式 FULL；压缩写回及其余外部结果逐 bit 通过 | 47.823 秒 |
| 第 4 层 B6，query=`[1,2,3,4,5,6]`，history=8192 | **物理 T21**，无每请求或全局输入 padding | 生产57控制＋观测85 FULL，对 Native **35/35 边界逐 bit 通过** | 控制33.022秒；观测39.282秒 |

三项任务分别为 `task_20261030_182456_362124329534`、
`task_20261030_182817_367228816585`、`task_20261030_183233_373650321465`，均使用物理 8。
每项 `complete/pass=True`，无 first_difference/error；实际调用 55 参数入口，
检查 eligible、提交计数和 replay contract，未回退 Native。
Native 自重复、初始 FULL/eager、PTO FULL/eager 与两次回放、只读输入和保护区均通过。

单行 history=8192 未触发压缩 KV/Indexer 写回，不能单凭它验证新 Norm。
所以额外用 history=8191 覆盖闭合：两边主压缩 cache 均修改 **37,414 字节**，
Indexer cache 均修改 **5,166 字节**，状态也实际更新；所有 `outside_slot_bytes=0`。
这项是实际闭合写回对拍，不能混同未触发该分支的空通过。

混合 case 明确记录 `actual_tokens=18`、`physical_tokens=24`、
`queries=[1,5,6,6]`、`mixed_replay_validated=True`；6 行是全局 bucket 的 dummy，
没有把每个请求分别补成 S6。回到原配置后，Native 与我方均恢复原结果。
比较范围是全部有效输出行及完整 cache/state；dummy 输出本身不作为数值合同，但不得污染有效结果或合法写槽之外的存储。

T21 任务 `task_20261030_183422_376564718483` 先保存生产控制，
`task_20261030_183657_38111779715` 随后完成同输入 Native 对拍，均使用物理 8。
实际记录 `tokens=21`、`lengths=[1,2,3,4,5,6]`、`request_padding_rows=0`，
fixture 直接分配21行输入，Native 请求累计边界为 `[0,1,3,6,10,15,21]`。
内部计算工作区仍可按 Cube tile 对齐，这不等于给每请求补 S6 或扩大外部输入。
`complete/pass=True`，35项全部逐 bit 一致，无首差/error，输入等价、观测/生产控制和保护区也通过。
57/85根分别先在227 CPU预编译185.868、190.835秒，均无设备/Worker。
这项检查的是生产调试根及分阶段 FULL 观测，不写成“T21正式55服务入口已独立验收”。

原始结果分别见 `evidence/precision-recheck-t40-service-v1.json`、
`precision-recheck-mixed18-bucket24-v1.json`、`precision-recheck-t40-close8191-v1.json`；
预编译凭证为 `precision-recheck-t40-spmd-cache-v1.json`、`precision-recheck-t24-spmd-cache-v1.json`。
复现仍通过公共 bootstrap、同一 env 和排除 14 卡的私有池，以单个完整命令字符串提交：

```bash
run_rms.sh <新label> padding --full-batch 40 --query-rows 1
run_rms.sh <新label> padding --mixed
run_rms.sh <新label> padding --full-batch 40 --query-rows 1 --history 8191
```

三个命令均显式设置
`PADDING_STAGE_REFERENCE=<实验根>/results/t240-layer2-weight-order-native-v1/result.json`，
严格核对相同正式源码 hash；实际金标仍由本次第 4 层 Native 独立计算。
任务 max780 秒、内部750秒加15秒清理，每次仅提交一项。

T21复现先用 `precompile_eager.py <新JSON> spmd-debug/spmd-probe --lengths 1,2,3,4,5,6` 预编译；
再执行 `run_rms.sh <生产label> spmd --lengths 1,2,3,4,5,6 --layer 4` 保存控制，
以新 label、相同参数和 `SPMD_PRODUCTION_REFERENCE=<生产目录>/production.pt` 完成 Native 观测。
证据为 `evidence/precision-recheck-t21-production-v1.json`、`precision-recheck-t21-native-v1.json` 及两份编译receipt。

本轮新增四类精度配置均通过，没有发现新的数值首差；正式数学仍保持05989cfb。
五个设备任务均已退出释放物理8卡，实际占卡50～71秒，没有超15分钟或遗留排队任务；
起止与锁释放见 `evidence/precision-recheck-task-lifecycle.log`。未使用14卡。
以上仍是单层真实权重/合成输入精度结论，不代表新版本整模型192-token或128K已验收；
本轮没有新增性能测量。

## 10 月 30 日 19 时：八卡 8K 整模型逐 token 验证

用户明确批准本轮临时使用物理 `0,8,9,10,11,12,13,15`，14继续禁用。
原先DP=EP16 Native已终止，不能用其未完成目录作为token基准；本轮另建同八卡Native，
不跑Native2、128K或nalinaly性能组。下列状态为本次八卡实验，不能混入原16卡报告。

| 项目 | 本轮设置 |
| --- | --- |
| 我方数学源码 | `05989cfb`；冻结自`65bebf7f6b2e4f3f6492fd6463f73106506f3496`，后者仅补测试记录 |
| 共同模型框架 | `41ba392f482bf88f7b6a797e2542443c81497338` |
| Decode拓扑 | TP1、DP=EP8，每rank32/256个专家 |
| 模型与输入 | 完整DSV4 Flash w8a8、原固定8K bank、40条不同请求、192输出token |
| 图与数值配置 | FULL_DECODE_ONLY、capture6/240、NZ2、atomic0、确定性level1/HCCLtrue、event1、EPLB关 |
| 替换范围 | 21个CSA的HC-pre至HC-post完整PTO TND；20个HCA保持Native |
| 硬件边界 | 精确八卡白名单和pool；task780秒、内部720秒及清理，首差停止 |

八卡适配仅发生在新冻结测试副本：放开CSA混合请求DP8的两处CLI限制，
完成屏障和结果汇总改为8个rank。生产CSA数学、模型专家数、SDK与旧快照不改。
新启动器在加载环境前检查实际物理设备集合，并兼容task包装器追加的device参数。
PTO运行前及最终汇总都核验Native基准的TP/DP/EP、卡序、bank、共同框架、图/容量等配置，
拒绝DP16或startup-only结果充当八卡gold。

227 CPU入口测试22项通过，耗时0.26秒；Native和我方冻结源码的bootstrap预检均通过，未初始化NPU。
只读75个safetensors头估算：每卡checkpoint权重49.7009GiB，单卡HBM64GiB；
这不是运行峰值，尚不包含权重转换、KV、图和工作区，必须实际验证容量。
复用已有vLLM二进制和公共SDK，无重装；当前PTO public T6/T40/T240根在227 CPU预热。

| 阶段 | 结果 |
| --- | --- |
| 八卡Native启动/容量预检 | `task_20261030_191040_59142222382`，500.63秒退出；权重载入48.2755GB，KV分配OOM，未生成token |
| 同拓扑Native 192 token基准 | v4有抢占不能作gold；含0卡v5已取消；现只用8–15重提五波Native，等待结果 |
| 当前完整PTO CSA 192 token | 待Native基准通过 |

复现先运行`prepare.py <新cohort> --ours 65bebf7f --decode-dp 8 --arms native ours`，
在227自有目录用`link_runtime.py`绑定已验收二进制，分别执行preflight；
再用public-module的`precompile.py --tokens 6/40/240`预热。八卡设备任务以完整命令字符串提交：

```bash
TASKQUEUE_DEVICE_CONF=<本轮私有白名单> \
TASKQUEUE_DEVICE_POOL=0,8,9,10,11,12,13,15 \
TASKQUEUE_IGNORE_WHITELIST=0 TASKQUEUE_IGNORE_GROUP_AFFINITY=0 \
task-submit --device auto --device-num 8 --max-time 780 \
  --env PYTHONNOUSERSITE=1 --env HCCL_DETERMINISTIC=true \
  --env PTOAS_ROOT --env ASCEND_HOME_PATH \
  'bash <测试目录>/run_eight_card.sh <cohort> native <新label> none --startup-only'
```

启动通过后去掉`--startup-only`生成Native基准，再以`ours`和Native结果目录作reference运行PTO。
目标验收8×40×192=61,440个输出token、逐请求接受事件和实际CSA图替换覆盖；
首差即停。当前尚无新增整模型token一致性结论。

### 八卡容量问题与受限入场方案

首个启动任务实际组网DP=EP8、每rank32个专家，模型载入成功。
各rank profile可用KV为5.61～5.62GiB，对应自然块数约7160；
旧16卡实验却显式覆盖为22000块，配置阶段将这个override当作有效容量，
最终在`_allocate_kv_cache_tensors → torch.zeros`申请690MiB时OOM。
该任务已退出并释放卡，不是数值首差，也没有token结果。

新`eightcard-source-v2`保持同一模型和数学包，只把双方共同块数改为4000，
并标记`precision_only=true`：40条不同请求仍各生成192token，但允许等待入场。
这不是同时活跃B40的性能测试。PTO额外公开ring容量共1.25GiB，
不能仅按Native预算选择6000块并假定PTO也能放下，因此选更保守的4000。
测试进程内包装现有`get_kv_cache_configs`及engine引用：
原函数完成CPU分配计划后，遍历`kv_cache_tensors`；本实验connector开启，
每个实际raw分配计入`size + 2MiB` backing，再按512字节对齐计总量，`shared_by`别名不重复计，
逐配置对照**原profile可用字节数**；超过预算在实际KV分配前直接报错。
不修改共享vLLM文件或容量算法；此门禁也不保证后续graph/workspace峰值必然可容纳。
Native与我方两臂均完成无设备实际导入，确认两个函数引用相同且门禁生效。

precision-only验收仍要求全部8rank的40条请求、192token、接受事件、零抢占，
21CSA/20Native HCA正确。所有正请求FULL6/FULL240步骤都必须21CSA全部回放且无Python重提交；
普通步骤必须21CSA全部实际提交，不以“任一层提交”代替完整覆盖。
真实入场人数、有效T、物理bucket、metadata请求容量与最大历史按Host描述符记录，
不读NPU tensor。若没有满B40的连续计时窗口，保留timing不足记录，
仅可通过精度验收，`performance_qualified=false`，不填满并发性能结论。

新增预算与受限入场负向测试后，227 CPU共24项通过，耗时0.28秒。
两臂三次public T6/T40/T240的首轮CPU预热均通过（187.23/194.86/217.74秒），无设备或Worker；
v2新快照继续检查缓存复用，不重装工具链。
v2复现的prepare参数增加`--gpu-blocks 4000 --precision-only`；不再执行旧22000块配置。
初始化检查并入新Native基准任务：只有预算门禁与模型初始化通过才开始生成，避免额外重复加载一次模型。
完整预检凭证和OOM原始rank日志无损归档于`evidence/eightcard/*.json.gz`及
`evidence/eightcard/eightcard-native-startup-v1/*.log.gz`，可用`gzip -cd`读取；
未改写CANN原始错误文本。入口测试、容量估算及首轮编译凭证亦保留在同一证据目录。

v2 public T6/T40/T240根也已在227 CPU全部通过，分别191.83/195.20/220.92秒，
`npu_initialized=false`、`worker_created=false`；对应`evidence/eightcard/eightcard-ours-t*-cache-v2.json`。
新冻结路径的artifact identity与v1不同，不能写成同一binary命中；SDK和vLLM仍未重装/重编。
测试入口、首轮容量失败与复现方案已在`3e1d3124`正常推送到本分支，原数学与精度基准保留。

### 20 时 48 分：修正测试门禁 API，尚无 token 结果

v2任务实际20:09:31开始、20:13:54结束，运行263.56秒，设备已释放。
自然KV预算仍约5.61GiB，统一4000块的CPU配置生成成功，报告94,456-token缓存容量；
但测试门禁错误读取`KVCacheConfig.tensors`，当前版本字段是`kv_cache_tensors`，
随后抛`AttributeError`。未进入生成，不计为CSA数值失败，也不能当作Native基准。
先前CPU测试用模拟对象重复了错误字段，因而没有发现这个API问题。

修正后CPU回归直接构造227实际vLLM的`KVCacheConfig`、`KVCacheTensor`，
覆盖真实字段、size字节单位、512对齐、超预算拒绝、配置数量不一致、connector每分配额外2MiB。
独立复核确认backing开销必须计入，未通过放宽门禁让任务继续。
最终v4快照的24项CPU检查通过，耗时8.83秒；两臂实际导入/别名检查也通过，均无NPU。
v3只做无设备预检，没有NPU任务；v4保留原数学与4000块预算，旧v1/v2/v3均保留。

重新提交的Native为`task_20261030_204818_311256912632`，20:48:18提交，仍使用批准的八卡，
max780/内部720秒；PTO仍未提交。公共T6/T40/T240的三个独立constexpr根在227 CPU并行预热，
每进程仍配置四路编译，不占NPU，不重建共享SDK或vLLM。
观察到19:59至20:40有一次长间隔，不能声称该区间满足每5分钟观察；
实际设备任务仍受硬时限且已在4分24秒退出。
后续在本回合前台SSH会话每45秒只读检查唯一任务，完成即退出，不创建scheduler/自动化。
目前尚无当前版本真实整模型192-token通过结论。

### 21 时 13 分：Native 完成生成，但拒绝使用有抢占的结果

v4进程正常退出，运行378.59秒；8个rank均输出40条完整请求，每条192token，
合计61,440个输出token。Native覆盖检查无错误，接受长度0～5全部覆盖，
加权平均接受数3.67236，Prometheus与逐请求计数一致。
但累计抢占480次，224份副本请求发生过抢占，因此postflight拒绝。
该结果不作为Native gold，未据此提交PTO，也未放宽零抢占要求。
满B40计时窗口未形成，不提供满并发性能结论。设备已在约6分19秒后释放。

v5仍从相同40题bank取全部不同请求，改为每波8题、共5波：
偏移0/8/16/24/32，每波完成后在客户端等齐8rank，再提交下一波。
5波共用一次begin/end observation，累计保存全部请求token、接受事件和路由；
模型`max_num_seqs=40`及capture6/240不变，只限制本轮测试实际入场数量。
Native与PTO采用同一分波，基准配置门禁同时比较`request_wave_size=8`，
不能复用v4的40题同时提交结果。

另确认观察中的FULL6、L6、正请求数1可能来自
`execute_dummy_batch → model_runner._dummy_run(6, uniform_decode=True)`，
这条idle EP路径本来就构造正token数，不能按attention metadata正数判为用户请求。
v5测试观察层只读包装真实`_dummy_run`入口，try/finally记录并恢复调用来源标记，
结束观察时恢复原方法；完整descriptor保留`is_dummy_batch`。
仅明确来自该调用路径的synthetic步骤不计用户CSA路由，
**不按history阈值或输出误差豁免**，真实用户步骤的21CSA全提交/FULL全回放要求不变。

227 CPU共25项检查通过，耗时6.59秒，包括40题5波、单观察窗口、dummy异常恢复，
两个冻结副本的相关文件AST及实际导入检查通过。新v5 T6/T8/T40/T240 public根均在227 CPU通过，
T8显式requests8，用于每波首步；均无NPU或Worker，不重建SDK/vLLM。
新基准`task_20261030_211344_38317151314`于21:13:44提交，继续使用精确批准八卡、
max780/内部720秒；随后因用户更新用卡范围，于21:45在pending状态取消，未占卡。
复现prepare参数在v4基础上增加`--request-wave-size 8`；数学仍05989cfb/65bebf7f。
Native v4的完整8rank结果、逐请求统计、配置和一份日志保存在
`evidence/eightcard/eightcard-native-v4-generated-with-preemptions.tar.gz`，仅作失败历史。
v5四个CPU根T6/T8/T40/T240分别191.41/194.30/196.69/212.68秒通过；
`evidence/eightcard/eightcard-ours-t*-cache-v5.json`与25项入口回归日志保留于同一目录。

### 21 时 45 分：撤销前八卡使用，整模型待满足允许卡数

用户最新要求尽量不占物理0～7。含0卡的v5任务取消成功，状态`not_found`，
未运行；前台只读monitor随后退出。自有临时白名单已改为`8,9,10,11,12,13,15`，
旧0卡授权不再用于后续提交。此前已执行的三个Native任务均发生在本轮0卡授权期间，
分别约8分21秒、4分24秒、6分19秒，均已释放设备，没有PTO或其他遗留任务。

14未获恢复确认，继续禁用；当前只有7张允许卡。该完整模型的现有DP=EP8/16入口
不能改为7卡，且不通过改变专家数量、丢层或换小模型代替整模型验收。
所有准备、代码、输入、编译凭证和失败结果保留；待有获准的完整八卡集合后，
沿用v5的8题×5波先取得零抢占Native，再做PTO逐token比较。
本文此前含0卡的命令仅记录历史授权实验，不代表当前可执行用卡权限。

### 22 时 13 分：临时批准14卡，仅最后八卡恢复验证

用户明确临时允许本轮使用14卡，0～7继续不使用。14卡只读health查询为OK，
专用wrapper与设备断言、私有白名单/pool统一为`8,9,10,11,12,13,14,15`。
227 CPU设备门禁1项通过0.04秒，两臂实际导入/预算门禁预检通过，无NPU。
只改外层设备选择，复用原`eightcard-source-v5`数学/测试副本与四份CPU编译凭证，
不重编vLLM、PyPTO、Simpler或重新创建数学快照。

Native `task_20261030_221539_154150020473`于22:15:39提交，
label为`last8-native-8k-waves8-v5`：仍为40题分8×5波、每题192token、
共同KV4000块、单次总观察、显式dummy来源、max780/内部720秒。
任务元数据确认pool/whitelist严格8–15，目前排队；含0卡的旧任务不恢复。
继续前台每45秒只读观察，不创建scheduler。Native零抢占及所有门禁通过后，
才按同一物理卡序与同配置比较当前完整PTO CSA。
14卡授权仅限本轮临时验证，不改变其他工作的默认禁用约定。

### 23 时 51 分：零抢占 Native 基准通过，提交 PTO 对拍

Native `task_20261030_221539_154150020473` 于23:40:45实际获得设备，
进程运行368.11秒，正常exit0、postflight通过，设备已释放。
实际设备顺序为`8,9,10,11,12,13,14,15`，TP1、DP=EP8；
40个不同问题分8题×5波，各rank均完整输出每题192token，合计61,440token。
本轮没有Native2或128K任务，不以DP8分波结果代替原DP16/B40满并发结果。

| 验收项 | Native v5 结果 |
| --- | --- |
| 8 rank × 40题 × 192 token | 全部完整、finished=true |
| CSA/HCA覆盖 | Native控制组无PTO runtime，coverage无错误 |
| 累计抢占 / 发生抢占请求数 | 0 / 0 |
| 接受长度0～5直方图 | 848 / 1224 / 1624 / 1048 / 1080 / 7344 |
| 平均接受步长 | 3.695018226 |
| 接受事件与Prometheus | 一致，acceptance qualified=true |
| 精度基准资格 | 通过，可作为同八卡、同分波PTO的gold |
| 满B40连续计时 | 未形成；performance_qualified=false，不提供性能结论 |

Native完整8rank输出、接受事件、实验配置、postflight、exit和rank0日志保存在
`evidence/eightcard/last8-native-v5-evidence.tar.gz`，不包含模型权重或KV tensor文件。
该有效gold与之前480次抢占的v4失败历史分开保留。

将全部8rank的实际`policy_descriptors`按我方当前`select_policy().constexprs()`去重，
只排除明确`is_dummy_batch=true`的708步及无有效token步骤，得到两组策略：
普通/小FULL入口（代表T8、请求容量8、history8467、table68，累计136步），
以及FULL240入口（请求容量40、history8473、table68，累计1896步）。
两组均有相同v5公共模块与共享SDK的CPU编译凭证，缺失数0；未再编译或重装。
策略规划凭证保留于227自有results的`last8-policy-cache-plan-v5.json/.py`。

PTO任务`task_20261030_235118_6532871582`于23:51:18提交，
label为`last8-ours-8k-waves8-v5`，引用上述Native gold。
数学仍为05989cfb/65bebf7f，冻结源仍`eightcard-source-v5/ours/source`，
共同框架41ba392f、共享SDK和bank均未改变。
私有白名单、pool与wrapper继续严格8–15，max780/内部720秒；
逐请求token或接受事件发现首差即STOP，21CSA必须实际执行/FULL完整回放，20HCA保持Native。
随后PTO获得设备，但在下述KV预算检查处退出，尚不能宣布整模型精度通过或恢复nalinaly性能。

本轮复现创建新cohort时完整参数为：

```bash
python tests/pypto_test/csa_threeway_20261029/prepare.py <新cohort目录> \
  --ours 65bebf7f6b2e4f3f6492fd6463f73106506f3496 \
  --decode-dp 8 --arms native ours --gpu-blocks 4000 \
  --precision-only --request-wave-size 8
```

公共bootstrap及四份CPU预热方法与前述相同，必须使用新cohort的public模块名；
复制旧路径的编译凭证不能证明新路径缓存可用。本轮没有新建cohort，直接复用v5。
实际提交方式如下，只有本轮14卡临时授权有效时使用该集合；label每次必须唯一：

```bash
align_root=/data/pyptouser/yejia/vllm-nal-tnd-native-align-20261027-codex
repo="$align_root/src"
source "$repo/tests/pypto_test/nal_native_alignment_20261027/env.sh"
cohort="$align_root/cohorts/eightcard-source-v5"
# 先arm=native、reference=none取得合格gold；通过后再按下列PTO配置提交。
arm=ours
label=last8-ours-8k-waves8-v5
reference="$align_root/results/last8-native-8k-waves8-v5"
command="bash $repo/tests/pypto_test/csa_threeway_20261029/run_eight_card.sh $cohort $arm $label $reference"
TASKQUEUE_DEVICE_CONF="$align_root/csa-approved-last8-20261030.conf" \
TASKQUEUE_DEVICE_POOL=8,9,10,11,12,13,14,15 \
TASKQUEUE_IGNORE_WHITELIST=0 TASKQUEUE_IGNORE_GROUP_AFFINITY=0 \
task-submit --device auto --device-num 8 --max-time 780 \
  --env PYTHONNOUSERSITE=1 --env HCCL_DETERMINISTIC=true \
  --env PTOAS_ROOT --env ASCEND_HOME_PATH "$command"
```

### 10 月 31 日：PTO 启动被显存预算门禁拦截

PTO `task_20261030_235118_6532871582` 实际23:56:19开始，
进程运行278.57秒，00:01左右已退出并释放全部设备，未进入token生成。
加载后21层CSA的SPMD服务已准备，但KV原始分配前的预算检查拒绝：

| 指标 | Native | PTO |
| --- | --- | --- |
| KV4000块及backing计划 | 3,741,446,144B，约3.484GiB | 相同 |
| profile可用KV预算（rank0） | 6,022,865,408B，约5.609GiB | 2,409,795,584B，约2.244GiB |
| 完整生成 | 61,440token、零抢占 | 未进入 |
| 对Native token结论 | 有效gold | 尚无结果，不能记为数值失败或精度通过 |

8个PTO rank均被同一门禁拒绝；预算范围2,409,795,584～2,424,899,993B。
没有绕过`EIGHT_CARD_KV_BUDGET_EXCEEDED`或放宽抢占要求。
原始8rank日志、命令、exit、实验配置、CPU策略规划及实际环境凭证在
`evidence/eightcard/last8-ours-v5-budget-failure.tar.gz`。

预算差包含torch peak及non-torch项，不能直接称为权重副本或永久占用。
初步怀疑WO-A的ND/NZ副本，但41ba的adapter已有recast告警，本次完整日志没有该告警，
因此没有改WO-A布局，更没有切换会丢失Native累加顺序的旧ND helper。

新增`runtime_memory.py`及`run_runtime_memory.sh`仅测公开运行时初始化，
不加载模型、不提交CSA kernel；单卡严格限定8–13、15，14仍排除。
227无设备preflight与CLI导入通过，共享工具链和原数学包均未改。
原配置空载任务`task_20261031_002114_156500132378`在物理8完成exit0：
heap仍为256/512/256/256MiB，task window及dep pool用默认值，
仅`pypto.torch.init`即使设备free减少**3,588,612,096B，约3.342GiB**，
Torch allocated增量0，init本身0.281秒。
这解释了约3.37GiB预算差的主要部分；仓库旧注释的1.343GiB不适用于当前共享SDK。
随后单因素任务`task_20261031_002341_160663718099`只改变公开`ring_task_window=2048`，
物理8、exit0，heap不变，dep pool仍未显式覆盖：
空载设备增量降为**1,657,851,904B，约1.544GiB**，
回收1,930,760,192B，约1.798GiB；init本身0.234秒。
这两项init耗时不代表完整任务占卡时长，任务均受240秒上限并已退出。

不改WO-A或其他数学，继续保持KV4000块、40题分8×5波、原Native gold及原heap。
公共T6/T8策略与T240策略在2048窗口配置下的CPU artifact identity均与v5原值相同：
`ed64ac914b804089baa81ccb78e7e6eb8eb5722a9c3e5bc2b2e5373e026e9d27`及
`f9568db9086a412422c72e5cd93af733676caa655b4e428e8d3f543d25d3c33c`。
仅加载已有artifact约20秒，没有重编算子或SDK。

| 正式55入口窗口回归 | 任务 | 结果 | 脚本耗时 |
| --- | --- | --- | --- |
| T240/B40，物理10 | `task_20261031_004805_216277131273` | 七外界对Native逐bit；FULL/eager、重复回放、保护区通过 | 50.23秒 |
| T6/B1，物理8 | `task_20261031_005452_234929419789` | 七外界对Native逐bit；FULL/eager、重复回放、保护区通过 | 45.68秒 |
| T8/B8，各请求1行，无S6 padding，物理8 | `task_20261031_010250_257785617652` | 七外界对Native逐bit；FULL/eager、重复回放、保护区通过 | 54.57秒 |

三项实际`KernelConfig.ring_task_window`均为`[2048,2048,2048,2048]`，
未发生窗口allocator failure或精度首差。全部单卡任务只从8–13、15选择，14不参与。
T240首次小任务`task_20261031_003818_194938016445`在历史来源校验处退出，尚未运行kernel：
新wrapper漏传当前`PADDING_STAGE_REFERENCE`，误用了旧默认检查点。
修正为显式绑定已通过的`t240-layer2-weight-order-native-v1/result.json`，
227 CPU确认30个数学源码hash全部匹配，再重提上述v2；没有放宽来源门禁。

为整模型保留现有v5及编译缓存路径，测试launch只新增带内容hash的runner/observer副本，
原runner和observer先对照原manifest校验、保持字节不变，数学包及SDK不改。
副本只更换Worker观测扩展并增加模型就绪后的CPU配置RPC，
直接读取Worker已绑定的`KernelConfig`，记录实际window、原heap和dep未显式覆盖状态。
不修改forward、五波入场、token、接受事件、dummy判定或清理逻辑。
postflight增加逐rank实际配置校验，其余容量/精度/路由/零抢占门禁不变。
Native无PTO runtime，窗口仅作为PTO配置记录，无须仅因此重跑Native或改写gold。

227 CPU共36项回归通过5.14秒；加强冻结manifest保护后三项相关回归通过4.32秒。
实际派生入口AST、原manifest及导入检查通过，均无NPU初始化。
独立只读review没有阻断。首次CPU收集缺冻结测试helper路径，补齐`PYTHONPATH`后通过，
原失败日志同样保留。轻量原始证据为`evidence/eightcard/window2048-qualification-v1.tar.gz`，
不含大型tensor快照；后续短T及整模型结果另行追加。

复现窗口回归使用`run_window_check.sh <新label> 40/1/8`：分别为T240、T6、T8，
仍经公共bootstrap、指定v5 Native插件及当前数学包运行，单任务max780秒。
整模型PTO复用上节原命令，仅另加`--env PTO_CSA_RING_TASK_WINDOW=2048`；
outer wrapper自动启用`--pto-task-window 2048`及实际Worker审计，
完整模型21CSA覆盖、逐token/接受事件一致、零抢占均通过后才可验收。

### 01 时 10 分：窗口资格通过后重试八卡 PTO

三项正式FULL资格通过后，提交`task_20261031_011000_28035386797`，
label为`last8-ours-8k-waves8-window2048-v1`。
复用原`last8-native-8k-waves8-v5`有效gold，不跑Native2；
模型、SDK、数学65bebf7f、KV4000、8题×5波、192token、FULL6/240和所有数值配置不变，
唯一候选变化是公开`PTO_CSA_RING_TASK_WINDOW=2048`，heap仍为256/512/256/256MiB，
dep未显式覆盖。Native没有该运行时，原gold不受这项PTO容量配置影响。
新实际配置RPC及postflight必须验证所有rank读取到2048，不能只凭环境变量判生效。

仍仅使用本轮临时批准的物理8–15，严格whitelist/pool、max780/内部720秒。
01:12:28获得设备，执行349.68秒后因首差停止，全部设备已释放。
本次KV预算门禁通过：rank0可用4,343,304,192B，约4.05GiB，
大于原4000块计划3,741,446,144B。实际权重48.2791GB，与Native一致；
FULL图捕获及生成前实际窗口配置RPC均已通过，未修改数学包或原gold。

八个rank均记录同一首差：请求`h8192_q04`，接受事件zero-based index35，
我方`[5,5]`，Native`[5,4]`；任务以`reason=first_difference`退出。
这里的两个数分别为本次草稿数与接受数，不是token ID。
接受事件回调早于该步`_update_request_with_output`，因此当前步token尚未核验，
不能把“尚未记录token首差”解释为完整192token一致。
本次没有完成mixed输出或最终postflight，也不构成性能结论。

原始首差、八rank日志、已完成请求统计、实验命令和退出记录存于
`evidence/eightcard/last8-ours-window2048-first-difference.tar.gz`。
当前先做真实模型首步同输入逐层CSA诊断，不继续后续大配置。
T6、T8及原文件manifest/审计导入证据补存于
`evidence/eightcard/window2048-qualification-v2.tar.gz`；此前原始失败和T240资格仍保留v1归档。

### 02 时 11 分：真实首步同输入逐层诊断

提交`task_20261031_021130_16373599506`，label为`last8-ours-first-step-v1`。
仍为原v5模型、bank、KV4000、窗口2048及原Native gold，仅新增测试观测。
只在rank0首个真实、非dummy、非draft、非capture的普通`NONE` forward，
逐个CSA执行生产55参数PTO，再恢复同一初态执行Native的HC-pre至HC-post半层。
首差保存该层输入及前后状态再停；21个CSA首步全部一致也立即诊断完成退出。
这不是新的Native整模型基线，也不是192token、FULL replay或性能验收。

观测保留五个完整physical-page view：SWA、压缩KV、主压缩state、Indexer state、
Indexer key与scale合并页；该范围不包含底层allocation对齐前后的存储。
快照在CPU，完全重合的view去重；Native使用独立输出，影子执行前后及恢复后同步。
还保存/恢复共享TopK、request顶层字段及其直接Tensor内容，
检查实际PTO输入、输出和cache保持原结果；嵌套对象及factory闭包缓存未做全面快照。
硬拒IndexCache/skip_topk、TP>1、CP、OTP、olora，避免影子调用产生额外集体通信。
已有offline connector的逐层load/save为空实现，本诊断不会改写只读bank。

Worker实际window/heap/dep审计随首步报告持久化并再次验收；
缺少任意CSA层、任意五carrier、状态恢复失败、共享TopK差异都不能通过。
`FIRST_STEP_DIAGNOSTIC.json`与真正的`FIRST_DIFFERENCE.json`分开，已有首差优先，
不以诊断主动终止的非零子进程退出码冒充完整生成成功。
原冻结runner/observer/数学包保持字节不变，只新增内容hash命名的测试helper和runner。

复现沿用前述八卡命令，PTO额外传`--first-step-probe`给`run_eight_card.sh`，
同时显式`--env PTO_CSA_RING_TASK_WINDOW=2048`；只能使用本轮临时批准的8–15，
outer780秒、内部720秒。Native gold仍是`last8-native-8k-waves8-v5`。
227 CPU最初10项检查通过5.36秒；补强审计及首差门禁后6项回归通过4.43秒，
实际冻结入口派生和导入通过且未初始化NPU，独立只读review无剩余阻断。

等待期间仅用CPU复核已有Native gold：八个rank的40题输出token及接受事件均完全一致，
每rank为7,680token；q04 event35均为`[5,4]`。
这是同一轮Native的跨rank检查，不是额外Native2运行，也不证明全部内部Tensor逐bit一致。

补充227 CPU异常恢复回归：两项通过4.39秒，含Native已写cache/TopK/metadata后抛异常的路径；
确认finally恢复实际PTO状态，不初始化NPU。CPU资格、派生入口来源及Native跨rank审计凭证存于
`evidence/eightcard/first-step-cpu-qualification-v1.tar.gz`。
02:44八卡诊断仍pending，尚未占卡；只读核对本任务候选集为8–15，14未被策略排除；
按本实验精确cache路径及设备任务环境筛选，也未发现本实验遗留的设备进程。
这些CPU/只读检查不能代替尚未执行的真实首步或192token精度验收。

### 首步 v1 原始结果与观测终态修正

`task_20261031_021130_16373599506`运行381.24秒后已退出；原harness记录
`reason=first_step_diagnostic_complete`、`diagnostic_passed=true`，设备释放。
21个实际CSA行均记录`tokens=8`、输出`[8,4,4096]`，输出、五carrier、共享TopK及恢复均逐bit通过；
所有层的IndexCache/skip_topk为false，实际PTO窗口2048、原heap和默认dep审计通过。
但报告外层`first_forward`变成`NONE/tokens240/actual48/requests8`，与逐层记录不相符。
**这份raw只保留为21层T8同输入阳性证据，不作为唯一首步完整验收，更不是192token通过。**
原始JSON、命令、退出和八rank日志保持不改，归档为
`evidence/eightcard/first-step-model-v1-evidence.tar.gz`。

独立review发现确定的状态闭包漏洞：成功时只写STOP并抛异常，未永久claim/冻结。
若引擎在launcher清理前再次进入forward，会覆盖外层标签，却保留旧21行及成功receipt，
使旧结果与新标签混合；该漏洞足以解释现象，但不能仅凭原raw证明实际的再入时序。
因此不从外层240推断实际层T，也不把这个观测问题称为CSA数值首差。

修正仅在测试代码：Lock保护单次`armed→running→terminal`状态；
第一次selected执行先claim，重入或后续调用在origin前拒绝，终态后报告/receipt不能改写。
外层、每层行和receipt携带同一个`forward_id`并闭合验收；
每层另记实际Service context的模式、hidden形状及五组metadata的Q/L，硬验NONE且非capture。
CPU回归专门模拟成功后下一次240进入，要求拒绝执行并保持所有报告文件逐字节不变。
227共9项CPU回归通过6.37秒，新hash入口实际生成/导入通过、无NPU初始化，独立复核无阻断。
冻结原runner/observer/数学包及SDK保持不变。

随后同配置重提`task_20261031_031035_2107052519`，label为`last8-ours-first-step-v2`。
仅8–15、max780/内部720，原Native gold及数学不变，不跑Native2/128K或后续192生成。
新helper为`first_step_bad9a51d655c.py`，完整SHA与派生runner凭证保存在prepare-v3记录。

### 单次归属修复后的真实首步通过

`task_20261031_031035_2107052519`已completed0，程序374.22秒，物理8–15，设备释放。
header、21行、receipt均为`forward_id=1`，外层和每层实际均`NONE/T8/B8`，
Q前缀`[0,1,2,3,4,5,6,7,8]`，实际seq lens为
`[8394,8411,8394,8467,8411,8225,8394,8411]`。
21个输出及105个物理cache carrier比较无不同byte；
共享TopK、输入、输出及metadata恢复全部通过，实际生产55参数PTO入口未回退Native。
这次首步标签一致、终态不再覆盖；原始证据为
`evidence/eightcard/first-step-model-v2-evidence.tar.gz`。
仍不等于FULL后续或192token通过；数学及原gold未改。

### FULL240桶实际8请求：Native参考资格未过

新增`run_window_check.sh <label> 40 --active-requests 8`，先满40捕获，
再同图改8个真实S6请求、48有效行、192dummy行，最终拟返回满40。
沿用全部guard，初始及fresh builder后显式确保Native SAS/QLI生产，旧mixed分支保留。
CPU首次帮助入口包装漏目标helper目录而失败；补齐导入路径后CLI及7项policy检查通过0.07秒，未初始化NPU。

独立review事先指出：Native图内compressor metadata固定捕获`num_reqs_actual=40`，
B阶段fresh builder的8及SAS/QLI刷新不会改变这个标量。dummy的Q跨度仍为6，
start=0、table=0，可能生成额外压缩slot；因此先把此case视为Native合同资格诊断。
不以缩短dummy Q跨度或放宽写域规避，不把失败归为PTO数学差异。

`task_20261031_034706_158045324099`在物理8运行49.86秒后退出1：
满40控制七外界逐bit通过；B阶段Native第一次回放即guard失败：

| Native B写域 | 超出fresh合法slot的byte | 首个allocation偏移 | 其余状态 |
| --- | ---: | ---: | --- |
| compressed cache | 936 | 128 | Native写域资格失败 |
| Indexer key/scale页 | 129 | 128 | Native写域资格失败 |
| SWA、主state、Indexer state | 0 | — | 通过 |

该状态的PTO回放和返回A均未执行。不能将其当作可靠Native金值，
也不能仅据原始NPUGraph的捕获标量合同认定真实整模型Native有bug。
小原始证据为`evidence/eightcard/bucket240-active8-native-contract-v1-evidence.tar.gz`；
大型tensor快照仅留227，未下载。

### 将接受事件与输出token分开诊断

原严格event35失败仍保留。新增独立`--inspect-token-sequence`：
只把接受事件首差写`first_acceptance_difference_rankN.json`，不发STOP；
token及输出长度依旧调用原Checker，首差立即停止。
完整生成后仍执行原postflight，接受事件不同仍是`passed=false`及task exit1；
另外`token_diagnostic.json`逐请求复核全部61440token，单列coverage、抢占和事件结果。
因此token相同也不能冒充严格验收或性能同口径通过。
原冻结scheduler/runner先验manifest，只新增hash helper；不修改数学、SDK或gold。

227 CPU共25项回归通过5.37秒；实际派生runner导入通过、未初始化NPU，独立review无阻断。
凭证为`evidence/eightcard/token-diagnostic-cpu-v1-evidence.tar.gz`。
新runner为`run_token_diagnostic_b50e1e4a4382.py`，helper完整SHA及原scheduler SHA在prepare记录。
`task_20261031_040432_255043622088`已提交，label为`last8-ours-8k-token-diagnostic-v1`；
仍只本轮临时8–15、max780/内720，原40题分8×5波、192token、原Native v5不变，
不跑Native2和128K；发现token首差即停。本项不用于性能比较。

### 最终8K逐token结果：61440/61440一致

`task_20261031_040432_255043622088`已结束；生成子进程exit0，执行418.63秒，约6分59秒，
物理8–15。调度任务exit1来自保留的严格postflight失败，不是模型生成崩溃。
本实验设备进程的精确路径/任务环境审计为空，没有遗留占卡。

| 项目 | 本轮结果 |
| --- | --- |
| 真实模型 | DeepSeek-V4-Flash-0731-w8a8，75个正式分片 |
| 共同框架 / 我方数学包 | `41ba392f` / `65bebf7f`，无新数学改动 |
| 请求口径 | 每rank40题；8请求×5波，非40条同时活跃 |
| 输出 | 8rank×40×192＝61,440token，逐个与原Native一致，错误0 |
| 替换覆盖 | 21CSA均PTO；真实FULL回放、窗口/heap审计及HCA Native检查通过 |
| 抢占 | 0 |
| DSpark接受均值 | 3.6950182260，与Native均值相同；0..5长度均覆盖 |
| 接受事件逐条一致 | 否，8rank均有差异；首差仍为q04 event35的`[5,5]`对`[5,4]` |
| 原严格postflight | 失败，precision errors仅为上述8条接受事件差异，coverage errors为空 |
| 本轮性能结论 | 无；418.63秒含加载、profile、图捕获和生成，不能算CSA延迟或相对性能 |

因此，先前“接受事件首差”并未在这40题的192token输出中形成token首差。
不能将接受率均值相同推成接受轨迹相同，也不能将token一致推成所有中间浮点值逐bit一致。
本轮达到了用户要求的最低逐token口径；原更严格门禁继续保留为未通过。
单层原始FULL桶子集的Native金值合同问题也继续保留，不与整模型token结论混计。

原始8rank输出token、逐请求接受事件、全部覆盖/运行时凭证、首事件差异及日志在
`evidence/eightcard/last8-token-diagnostic-v1-evidence.tar.gz`。
复现沿用上述`run_eight_card.sh`，额外传`--inspect-token-sequence`，
显式窗口2048、同一Native gold及相同模型/bank/参数；需重新取得当轮八卡授权。
默认单卡任务仍仅使用8–13、15；本轮14卡临时许可不能作为后续默认。
