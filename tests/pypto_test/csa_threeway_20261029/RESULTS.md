# 8K/B40 三方整模型测试：前置检查与首差记录

日期：2026-10-29。本记录尚未完成整模型验收，不能作为新分支的 token 一致性或性能结论。

最新状态：WO-A 修正接回后，T240 完整入口 35 个边界（至最终输出）均逐 bit 一致；
实际生产 55 参数 FULL 入口及 T6 回归也通过。本轮真实整模型仍待完成，原精度基准保留。

## 目标与已授权范围

参考 nalinaly 的
[低接受率测试方案](https://github.com/nalinaly/vllm-ascend/blob/dsv4-flash-pto-v0.25.1rc1/tests/pypto_test/low_acceptance_20261001/RESULTS.md)，
比较 Native、我方精度 TND 与 nalinaly 最新 TND SPMD。

| 项目 | 本轮设置 |
| --- | --- |
| 我方基准 | `da6b14f0ac158b2b9675dfb79b605a3c46bd90fd` |
| nalinaly 初始 / 更新后的 CSA 提交 | `41ba392f` / `79ea6d3c8460338a172cb5570d14999d5a544262` |
| 当前分支 | `dev/pypto-dsv4-csa-nal-tnd-native-align-20261027-codex` |
| 共同框架计划 | 三方采用 `41ba392f`；我方和 nalinaly 分别覆盖完整 TND SPMD 包 |
| 模型 | `DeepSeek-V4-Flash-0731-w8a8`，75 个正式权重分片 |
| 整模型目标 | TP=1、DP=EP=16，8K/B40，每请求 192 token |
| 替换范围 | CSA；HCA 均保留 Native |
| 配置 | 确定性 level=1、HCCL=true、NZ2、atomic=0、EPLB 关闭 |
| FULL 档位 | 6、240；草稿 5 token；temperature=0 |
| 资源 | 用户已明确批准本轮三方 16 卡 8K 测试；前置单卡显式使用物理 15 |
| 限时 | 单卡任务 780 秒，进程 750 秒加最多 15 秒清理；整模型计划 720 秒执行上限 |

`prepare.py` 的共同框架方案用于隔离 CSA 算术差异，因此今后的整模型结果应写成
“我方数学包在共同框架上的结果”，不能写成未经修改的 `da6b14f0` 完整软件栈结果。
当前 v4 快照已冻结；三方 preflight 及四项公共模块缓存预热通过，
已提交本轮唯一一次 Native 16 卡基线，等待完成后再顺序提交两份 CSA 实现。

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
| Native，共同框架 `41ba392f` | 本轮一次基准已提交，等待结果；不跑 Native2 | 尚未测量 |
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
设备池、白名单以及 `MAX_TIME=780` 均正确。当前仍在等待调度，不能当作整模型结果。
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
