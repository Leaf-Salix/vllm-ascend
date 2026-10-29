# 8K/B40 三方整模型测试：前置检查与首差记录

日期：2026-10-29。本记录尚未完成整模型验收，不能作为新分支的 token 一致性或性能结论。

## 目标与已授权范围

参考 nalinaly 的
[低接受率测试方案](https://github.com/nalinaly/vllm-ascend/blob/dsv4-flash-pto-v0.25.1rc1/tests/pypto_test/low_acceptance_20261001/RESULTS.md)，
比较 Native、我方精度 TND 与 nalinaly 最新 TND SPMD。

| 项目 | 本轮设置 |
| --- | --- |
| 我方基准 | `da6b14f0ac158b2b9675dfb79b605a3c46bd90fd` |
| nalinaly 冻结提交 | `41ba392f482bf88f7b6a797e2542443c81497338` |
| 当前分支 | `dev/pypto-dsv4-csa-nal-tnd-native-align-20261027-codex` |
| 共同框架计划 | 三方采用 `41ba392f`；我方覆盖完整 TND SPMD 算术包 |
| 模型 | `DeepSeek-V4-Flash-0731-w8a8`，75 个正式权重分片 |
| 整模型目标 | TP=1、DP=EP=16，8K/B40，每请求 192 token |
| 替换范围 | CSA；HCA 均保留 Native |
| 配置 | 确定性 level=1、HCCL=true、NZ2、atomic=0、EPLB 关闭 |
| FULL 档位 | 6、240；草稿 5 token；temperature=0 |
| 资源 | 用户已明确批准本轮三方 16 卡 8K 测试；前置单卡显式使用物理 15 |
| 限时 | 单卡任务 780 秒，进程 750 秒加最多 15 秒清理；整模型计划 720 秒执行上限 |

`prepare.py` 的共同框架方案用于隔离 CSA 算术差异，因此今后的整模型结果应写成
“我方数学包在共同框架上的结果”，不能写成未经修改的 `da6b14f0` 完整软件栈结果。
目前冻结目录仍是准备状态，尚未提交任何本轮 16 卡任务。

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
| Native，共同框架 | 尚未运行本轮基准；不跑 Native2 | 尚未测量 |
| 我方精度 TND | T240 前置发现 Q-A 首差，暂停推进 | 尚未测量 |
| nalinaly `41ba392f` TND SPMD | 尚未运行；不能用历史误差代替本次结果 | 尚未测量 |

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
这些步骤准备就绪并不等于已完成整模型测试；本轮当前仍被 WKV 数值门禁阻挡。
