# TND 精度版：Vector/MTE2 拆分与 SWA staging 单因素优化

本轮基于已验收 `7078bc644d618663c21f558e86432a8cac1bf5dc`，原基准及 Native 金值保留。只改变 SWA 中没有消费者的 KV staging，不改变数学与舍入顺序。分支为 `dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。

## 优化及精度边界

上一轮已经跳过 SWA 无效的 Cube 计算，但其 512 行 KV staging 仍完整填零、写 GM、读入 L1。本轮在原有效片条件 `qk_sb > 0 or qk_part == 0` 内执行这些搬运。SWA 只传 part0；compressed 仍传全部四片。

| 每个有效 SWA query 的逻辑工作 | 基准 | 本轮候选 |
| --- | ---: | ---: |
| 两个 AIV 合计 BF16 零初始化元素 | 262,144 | 65,536 |
| KV staging GM 写 | 512 KiB | 128 KiB |
| QK 对 staging 的 GM 读 | 512 KiB | 128 KiB |

保留 part0 完整零初始化，短窗口和非法页仍为正零；保留 PV part0 重载、512 列概率布局、Native 实际 K 归约、CAST_ROUND、FP32 online 更新次序、四个跨核 event、L1 槽和 fence。43 份运行源只修改 Sparse 一份，62 Tensor 单入口、纯 DSL 和真正 TND 不变。逻辑字节数不是实际 HBM 带宽测量。

## Vector、MTE2 和等待的测量范围

| 引擎 | 当前工作 | 本轮证据与限制 |
| --- | --- | --- |
| AIV Vector | staging 填零、scale/bias/max/exp、实际 K 求和、概率舍入、online 更新 | 去掉无用零尾不改变 softmax。余下 Vector 工作仍多，不能直接换成性能版的数值路径 |
| AIV MTE2 | paged KV 采集、score/bias、旧统计量、alpha、PV 和 compact64 回读 | compressed 存在逐行 Scalar 寻址、小 DMA；尚未测出逐事件等待周期 |
| AIC MTE2 | Q、QK KV、PV probability 和需要重载的 KV | AIC-only 已证明少读，但总周期未改善；busy 与逻辑字节量并非一一对应 |
| 同步等待 | 核间通知、复用槽和依赖等待 | 现有 kernel-mode 不支持本次 PMU 接口，DSL 无逐事件 cycle marker；没有用 total−busy 冒充等待时间 |

PMU 在独立 program-mode 进程采集，每进程先两次无 PMU，再一次 event2；与无 DFX Graph 延迟分开。AIC 24 核、AIV 48 核，均按实际 func_id 映射及正周期核统计。

| 同卡 9，平均 cycles/core | 基准 | 仅 AIC 读清理 | 完整 staging 清理 |
| --- | ---: | ---: | ---: |
| AIC total | 120100.83 | 129344.21 | 127422.79 |
| AIC MTE2 busy | 23752.88 | 19610.83 | 25559.58 |
| AIV total | 123763.67 | 134147.12 | 131607.90 |
| AIV Vector busy | 35283.10 | 35323.58 | 34499.81 |
| AIV MTE2 busy | 22193.81 | 23062.88 | 23301.31 |
| AIV MTE3 busy | 16970.69 | 17953.31 | 14481.23 |

仅 AIC 清理的 MTE2 busy 下降 17.44%，但总周期增加；完整候选 MTE2 busy 没有下降。完整候选 Vector busy 约降 2.22%、MTE3 busy 约降 14.67%。这是单个 PMU 窗的方向信息，计数会重叠，不能相加拆分总时间，也不能据此推导稳定等待收益。第一次完整候选 PMU 在卡 8 的数据也保留，不据跨卡差值作归因。

## 对照 Native 和 nalinaly

Native 按 actualSingleProcessSInnerSize 限制 L1 循环；固定参照 nalinaly `a919f119…` 使用 K128 面板，二者都没有我们原 SWA 的无消费 384 行 staging。其 softmax/online/舍入顺序不能直接作为当前精度版的替代。

上轮同 bank 的参考计数如下，沿用原始记录，未伪称本轮重测。PTO 为每 core 的平均千 cycles，Native 的 busy 只有 CSV 原始比例，不以假定频率换算；Native total 由原始整数按核数均摊。

| 指标 | Native CANN 历史 | nal BSH 历史 | 本轮候选同卡 9 |
| --- | ---: | ---: | ---: |
| AIC total，千 cycles | 90.05 | 78.44 | 127.42 |
| AIC MTE2 busy | 53.23% | 17.34k cycles | 25.56k cycles |
| AIV total，千 cycles | 96.28 | 87.89 | 131.61 |
| AIV Vector busy | 8.70% | 21.11k cycles | 34.50k cycles |
| AIV MTE2 busy | 37.30% | 24.09k cycles | 23.30k cycles |
| AIV MTE3 busy | 15.37% | 10.53k cycles | 14.48k cycles |

范围必须区分：nal qk_pv 还包含 inverse RoPE 与发布，实验 cos=1/sin=0；TND qk_pv 不含这两项，完整 sparse root 才包含。Native 是 CANN 静态混合核、PTO 是 program-mode task。因此它是路径诊断表，不是受控整层排名。nal 该固定 bank 的 heads relative L2 为 0.2544%，未通过我们的逐bit门禁，不能以其低延迟替代精度验收。

静态看，当前两个 K512 softmax 块的主 exp 每 query 为 65,536 个元素，nal 旧参照五个 K128 为 40,960 个；但还有不同的 online/归并、inverse RoPE 等工作，不能按元素比推导周期。nal 旧参照 Vector 约 21.11k cycles/core，当前清理后约 34.50k，主要差距仍没有被解释完。

另已核对 nal 最新 `289cc072…` 的新 SPMD 路径：在前置 plan 生成已验证物理行，一次载入 64 个索引后 MGATHER，减少重复请求解析与 Scalar 读地址。其工具链和数值路径与当前基准不同，本轮没有混入。下一轮可在保留 Native compressed pair 排序/回退与全部数学的前提下，单独检验地址生成/载入是否有效。

## 受控 Graph A/B

以下是独立 sparse attention 诊断 root 的 ACLGraph replay，包含 planner、统计和 heads 发布，不是完整 CSA，更不是整模型 Forward。全部物理卡 9；每项 20 对，每臂每轮 100 replay，原始样本不剔除。

| 档位 | 基准 µs | 候选 µs | 延迟下降 | 正收益对数 | 配对差值均值 95% t 下界 µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| B4 第一次 | 219.472 | 216.441 | 1.38% | 16/20 | 1.620 |
| B4 第二次 | 228.245 | 220.859 | 3.24% | 20/20 | 5.605 |
| B40 第一次 | 867.470 | 838.166 | 3.38% | 20/20 | 24.411 |
| B40 第二次 | 883.235 | 868.197 | 1.70% | 20/20 | 12.665 |

B4 pilot 后、B40 和重复测试前确定门槛：各档两次任务、各次至少 75% 配对为正、均值正且双侧 95% t 下界为正。四项全过，合计 76/80 对为正；没有把上轮 80/80 重标成本轮。

每项 71 个数值/保护门禁全过，包括 Native BF16 heads、原 PTO raw FP32 mi/li/oi、A/B/ABA、Q0、NaN poison、guard 及设备只读输入。PMU 每项 59 门禁全过。原始 FP32 统计金值没有重新生成。

## 完整 CSA 及最终源身份

先行 v1 完整 CSA T1、混合 T21、B40/T240 的八类边界对拍全部逐bit通过。v1 在 reports 下误用默认 88 列格式，实际计算 AST 不变；最终按仓库 120 列配置重新冻结为 v2。v1 原记录保留，不重标为 v2。

最终 v2 真源另测混合 1～6 行 T21，八边界 A/B/ABA、poison/guard、Native 全逐bit通过，确认真正 TND、不逐请求补 S6、实际 PTO dispatch、无 fallback。

- 最终 Sparse SHA256：`dafd595cacf0bad5978776b07594ea9f8ae25f6fb22ea15d2545180ced47c434`。
- 完整 source manifest：`11219447daf4f0c0024e02cb983d08885d9870777fbe10d55ed38837a9aee01b`。
- candidate math：`75de0eadea0c8855e1e374a1a29ad18ecd0bf8f6c7af8809604a92d029cec901`。
- 整模型 helper manifest：`59f9c9c460036579c6f42b6ea4fb0b3720ad8ca18a179eedaeb70d9524cc7258`。

## 整模型结果

最终v2两档均完成：16rank、192token预热+3轮各192token；21个CSA全为PTO、20个HCA为Native、62Tensor单入口，零抢占。8K为40活跃；128K提交24、实际21活跃/3等待，126有效token进入144图桶。

| 档位 | Native历史Forward ms | 上轮SWA128 Forward ms | 本轮Forward ms | 相对上轮观察到的下降 | token / 接受事件对Native |
| --- | ---: | ---: | ---: | ---: | --- |
| 8K/B40 | 89.482 | 108.707 | 108.405 | 0.28% | 491,520 / 105,152，全部一致 |
| 128K/B24，21活跃 | 79.730 | 91.659 | 91.434 | 0.24% | 294,912 / 61,568，全部一致 |

| 档位 | Native历史cycle ms | 上轮SWA128 cycle ms | 本轮cycle ms | 相对上轮观察到的下降 |
| --- | ---: | ---: | ---: | ---: |
| 8K/B40 | 105.575 | 124.367 | 124.058 | 0.25% |
| 128K/B24，21活跃 | 92.293 | 105.617 | 103.952 | 1.58% |

cycle为每步16rank最大值，包含Forward之外的循环开销；两项都保留全部样本。这些是跨时段历史窗比较，不能把0.2%级Forward变化认定为稳定整模型收益。当前仍慢于Native：Forward约慢21.15%/14.68%，cycle约慢17.51%/12.63%。受控稳定方向的依据仍是上面的同卡局部Graph A/B。

8K三轮Forward为108.316/108.443/108.457ms；cycle为123.969/124.105/124.100ms。128K三轮Forward为90.925/91.940/91.437ms；cycle为103.321/104.400/104.135ms。本轮128K cycle最大108.984ms、总体标准差1.256ms；上轮含191.342ms长尾、总体标准差11.483ms，原始样本均保留。此处两组均以57点的pstdev重算；上轮原报告的11.585ms采用样本标准差口径。因此1.58%的cycle均值下降也不能单独归因于本轮小改动，未删除旧慢样本或改用median。

两档CPU metadata均严格阶段匹配：8K共960行对Native三窗2,880对一致；128K第1轮对应Native第1窗，第2/3轮对应Native第2窗。仅排除call，保留input_ids_cpu，不做0/-1归一化。原跨阶段差异、政策及7项正负fixture原字节保留；政策自测不冒充新候选硬件执行。CPU观测不等同于证明实际GPU输入字节或全部内部tensor逐bit，整模型验收是保存的token、接受事件、资格和调度轨迹一致。

8K实际运行15:18:29～15:27:29，共540秒，排队1317秒；128K实际运行15:34:45～15:44:38，共593秒，排队330秒。调度Done列表时长包含排队，不能当占卡时间。未新增Native2整模型任务。

原始任务为8K `task_20261019_145632_162707714602`、128K `task_20261019_152915_388649013465`，均exit0并已释放设备。完整记录以相应whole归档、source/helper身份及严格audit为准。

## 复现与限制

以下为已绑定实际两档结果的命令、归档及审计身份。每任务运行上限 780 秒，内部 750 秒；排队不占卡，排队时间与实际运行时间分别记录。模型、bank、SDK 留在 227，GitHub 只归档 UTF-8 JSON/CSV/源码/日志，不含权重或 .pt。

环境沿用 Python3.12、Torch2.10.0+cpu、Torch-NPU2.10.0.post2、vLLM0.25.1、Ascend0.25.1rc1、CANN/ATB9.2.0-beta.2、PyPTO3e87a843、Simplera54c0509、实际 PTOAS0.66。模型 DeepSeek-V4-Flash-0731-w8a8，75分片。未安装、升级或重编框架；没有 SDK/codegen/C++ 修改。

真实模型口径为 TP1、DP=EP16、NZ2/SK0、FULL graph、Native deterministic level1/HCCL true、EPLB关，接受步长3.8未强制。8K/B40实际40请求；128K/B24提交24、允许等待入场，实际并发以审计结果为准。192 token 预热、3轮各192 token，严格比较所有rank的token/接受事件/零抢占、21个CSA PTO与20个HCA Native资格以及CPU metadata。沿用公开40题、相同顺序/chat suffix和自行Native生成的prefix/cache bank，未证明与nal原bank字节相同。只有CSA替换，HCA/FFN仍Native。

### 新目录复现

将当前仓库 `prepare_tnd_precision_performance_repro.py` 用 rsync 上传到227自己的新目录；然后在227执行。工具只准备目录，不提交设备任务；label、destination、输出目录必须未使用，不执行归档的 prepare_snapshot.py。

```bash
BASE=/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex
EXP="$BASE/vector-mte2-20261019-codex"
REPRO_LABEL="repro-live-$(date +%Y%m%d-%H%M%S)"
NEW="$EXP/whole-$REPRO_LABEL"
python3 "$EXP/repro-tools-v2/prepare_tnd_precision_performance_repro.py" \
 --profile swa_live --reference-job "$EXP/whole-live-staging-v2" \
 --destination "$NEW" --label "$REPRO_LABEL"
export PERF_REPEATS=3
bash "$NEW/preflight.sh" 8k
bash "$NEW/preflight.sh" 128k
sha256sum "$NEW/helper-manifest.json" "$NEW/snapshot/manifest.json"
```

历史见证标签repro-live-20261019-a、目录whole-live-repro-a的prepare及8K无设备preflight已实际通过：source仍11219447…，其helper为eef938e2ac8faa56d31b1e823957dde3386d3e7d2f110c663b33ead64a1e3f0e。该目录已存在，复现示例使用时间戳生成新标签。原最终任务两档无设备preflight均通过。新路径的helper SHA必须重新计算，不借用原59f9…或历史eef938…；需要227原模型、bank、SDK、Native金值及静态13包，这不是任意机器的离线安装包。

先提交8K，一档一个任务：

```bash
task-submit --device auto --device-num 16 --max-time 780 --ptoas 0.63 \
 --env PYTHONNOUSERSITE=1 --env PERF_REPEATS=3 \
 --env ASCEND_HOME_PATH=/data/pyptouser/yejia/vllm-cann92-main/env/Ascend/cann-9.2.0-beta.2 \
 --env PTOAS_ROOT=/data/pyptouser/yejia/vllm-cann901-csa-0251rc1/experiments/nalinaly-71153bb3/reference-env/bin \
 --env CCACHE_DIR=/data/pyptouser/yejia/vllm-cann901-csa-0251rc1/src/pypto/.ccache \
 --env CMAKE_BUILD_PARALLEL_LEVEL=4 \
 "bash $NEW/run-case.sh 8k pto"
```

其中task-submit的0.63是既有调度标签，实际PTOAS由显式PTOAS_ROOT指定为0.66，须复核真实预检输出。任务exit0后进行严格CPU审计：

```bash
AUDIT="$BASE/precision-overnight-20261019/whole-formal-v6-audit"
OUT="/data/pyptouser/yejia/lc128/perf-$REPRO_LABEL"
python3 "$AUDIT/independent8k.py" \
 --root "$OUT-8k-pto" --identity Live-repro-a --repeats 3 \
 --expected-source-manifest 11219447daf4f0c0024e02cb983d08885d9870777fbe10d55ed38837a9aee01b \
 --candidate-math "$NEW/candidate-math.json" > "$NEW/independent8k.json"
```

只有8K审计退出0且pass=true才以相同task-submit选项提交 `bash $NEW/run-case.sh 128k pto`，仍max780。完成后将审计器改为independent128k_phase_matched.py、root改为$OUT-128k-pto。审计器旁原phase-policy不可替换：第1轮对Native1、第2/3轮对Native2；CPU metadata仅排除call，保留input_ids_cpu，禁止0/-1归一化。原严格跨阶段失败及7项政策正负fixture保留，政策自测不冒充新候选硬件结果。

每轮取20步Forward与19步cycle的逐步16rank最大值，保留所有样本和长尾。局部sparse A/B是同卡交错；整模型对旧Native或上轮PTO是跨时段比较，不能凭小幅差值单独宣称稳定端到端改善。

单层最终源入口为 `$EXP/full-csa-live-staging-v2/run-csa.sh mixed21 1,2,3,4,5,6 4 8191`，也必须经task-submit单卡max780。私有微测量助手、实际生成源码/PMU CSV/数值门禁/配对样本/原统计金值收据均保存于下列micro归档。CPU回放只验证记录完整性，不执行NPU数值。

## 已冻结的文本归档

| 归档 | SHA256 |
| --- | --- |
| swa-live-staging-v2-micro-raw.tar.gz | 3db6fa0cb183b7ad0afb1e84d9be20f5bfed281e27530993d9fe19f69ca701c2 |
| swa-live-staging-v2-mixed21-raw.tar.gz | cdd2cc33f3991bb28e3b1074e1d336657c44698c9cee0e38ba16ac75a332751d |
| swa-live-staging-v2-8k-whole-raw.tar.gz | 76ad42efaace421801ada8e430d9b755b9b4337bfa013ef4140988fdf9b15b3b |
| swa-live-staging-v2-128k-whole-raw.tar.gz | bc46250f873f411e9a24d0876c2760ea22e8002a89a1942f271b0cf92d49705d |
| swa-live-staging-v2-reference-phase-raw.tar.gz | 8180f5d2ff0a1ee9c37a988238accd02c4d12d4b966b3d753cab5a5d7648581a |

micro包含105份UTF-8载荷，继承原v1的97份原字节并追加最终格式桥和v2准备工具。早期清单中的pending只代表其冻结时状态；完整模型的最终状态以独立whole归档和实际审计为准。v1/私有源没有重标成v2硬件运行。原SWA128及V4/V6所有归档和manifest保持历史身份。

技术失败保留：初版preflight过度过滤运行hook导致无设备import失败，已恢复已验收的runtime/package选择并限制未知whole hook；一次device参数不合法尚未产生NPU任务；SSH连接关闭重试、PTY长行中断后恢复rsync；v1默认格式导致多余换行，最终v2正确冻结后另跑真源T21。没有将环境/工具错误当作数值失败，也没有删掉有效慢样本。

## 当前验收与后续方向

[当前源和完整模型证据manifest](evidence_tnd_precision/current_full_precision/manifest.json)绑定最终dafd/11219447源、实际59f9助手和两档三轮结果；[上轮SWA128 manifest](evidence_tnd_precision/current_full_precision/swa-cube128-v2-manifest.json)保留7078bc644的独立历史身份。原V4/V6、私有micro、v1与v2不相互重标。

本轮只保留有稳定局部收益的SWA无消费staging清理，精度门禁保持。Vector差距仍明显，MTE2 busy未下降，也没有精确拆出同步等待。下一候选是保留Native pair排序及全部舍入的物理行计划/批量索引载入，或减少SWA无效softmax列工作；均须独立测量，不能按静态工作量认定瓶颈、照搬性能版数学或在本轮混入第二因素。

nal当前源码参考：[289cc072的SPMD采集优化](https://github.com/nalinaly/vllm-ascend/commit/289cc072fd1063c886961a2763994e1fbdd00800)。其提交自报性能使用另一工具链，不作为本轮受控结果。本轮表中nal PMU仍固定旧a919参照。

## 提交前检查

8个相关CPU文件共56项通过，包含原54项和两项新证据/复现绑定检查。当前source manifest对应43份运行源，CPU回放直接复算全部rank的token、接受事件、metadata和maxrank timing，并检查micro的105份SHA、实际生成物、59/71门禁及80个配对样本；不以CPU回放代替NPU执行。

```bash
python -m pytest -v --forked \
 tests/pypto_test/test_full_precision_current_evidence_cpu.py \
 tests/pypto_test/test_full_precision_eligibility_cpu.py \
 tests/pypto_test/test_full_precision_hc_weight_contract_cpu.py \
 tests/pypto_test/test_full_precision_indexer_projection_cpu.py \
 tests/pypto_test/test_full_precision_oproj_cpu.py \
 tests/pypto_test/test_full_precision_wkv_cpu.py \
 tests/pypto_test/test_full_precision_wkv_profile_cpu.py \
 tests/pypto_test/test_full_precision_work_partition_cpu.py
```

增量pre-commit通过（按要求跳过扫描类hook），未宣称仓库全量CI通过。第一次hook运行因未激活venv而找不到python，激活既有环境后通过，未改依赖。独立审查复算原始证据并验证：逆转两个guard后完整Sparse AST等于旧基准，旧manifest原字节保留，五份新归档均为UTF-8文本，不含模型或.pt。两项整模型任务全部结束，无本轮残余设备任务。
