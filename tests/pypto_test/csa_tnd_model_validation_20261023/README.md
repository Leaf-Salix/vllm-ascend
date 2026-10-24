# CSA TND 真实模型功能与图重放验证

本目录记录当前 TND-MPMD 与 TND-SPMD 接入真实模型的验证。
PyTorch 单卡复现手册保持原有范围，不把本目录的 16 卡流程加入该手册。

## 当前结论（2026-10-24，v27 图重放修复）

两种 TND 均通过 16 卡 128K/B16、B24 的满批、尾批和 padding 模型图重放，
实际 21 层 CSA 在这些长历史 FULL 步骤中全部重放，未发生 Python 重新提交。
两种实现各自与 v25 eager 参考的全部输出 token、DSpark 接受事件相同；
MPMD 与 SPMD 之间也完全相同。零抢占，PTO 外部临时工作区为 0。

下表是**每卡、每种实现**的真实模型步骤数；16 个 rank 的计数相同。

| 档位 | 满批 S6 | 尾批 | 其中大图 padding | 其中单请求 6 行小图 | 真实 FULL 重放总步数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B16 | 33 | 15 | 15 | 0 | 48 |
| 128K/B24 | 33 | 24 | 20 | 4 | 57 |

尾批包含 padding 和单请求小图，两列不能再与尾批重复相加。
每请求完整生成 192 token，B16 每侧 256 请求/49152 token，B24 每侧 384 请求/73728 token。
每 rank 首个 S1 步由框架选择 NONE，执行一次 eager；短历史空闲补齐调用
`max_seq_len=6` 仍在不支持范围内。这些调用单独记录，不计入真实长历史 FULL 的通过结论。
本轮没有采集性能，不从这些功能任务推导性能收益。

集中证据：[graph_replay_model_validation.json](graph_replay_model_validation.json)。
它保存四个任务的 exit=0、逐 rank 21 层图检查、真实请求数、重提交计数、显存及配置来源。

## 单卡指定场景

使用实际 custom-op 与 ACLGraphWrapper，先通过单卡，再开始 16 卡。

- **128K/B16/S6/T96（C04）**：16 个请求，本轮各 6 token，共 96 token。
- **128K/B16/固定混长 T56（C08）**：query 长度和请求顺序固定为
  `[1,2,3,4,5,6,6,5,4,3,2,1,1,3,4,6]`，输入容量也为 56，不改成 96。
  “混长”指 query 长度，历史均为 131072。
- 另测 128K/B24/S6/T144、最后单请求 T6，以及各大档的尾批、padding、恢复满批；
  混长场景额外覆盖顺序反转。派生功能场景不替换性能规范中的原始场景。

每种实现 15 次、共 30 次真实图重放检查通过。输出/cache/state 与独立 eager TND
参考逐 bit 一致，显式诊断入口的 Top-K/score 一致，保护区通过，PTO 外部 scratch 为 0。
这里的 bit 对齐对象是相同 TND 算术的 eager 参考，不是 Native 算术。
证据：[graph_replay_service_validation.json](graph_replay_service_validation.json)。

## Native 对照与完整问答

| 档位 | Native 平均接受草稿数 | MPMD | SPMD | PTO 与 Native 完整 token 一致请求 | 简单请求一致 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B16 | 3.80650 | 3.79567 | 3.79567 | 64/256 | 32/32 |
| 128K/B24 | 3.62189 | 3.73754 | 3.73754 | 144/384 | 64/64 |

复杂问题仍有差异：B16 的位置 token 差异为 22416/49152，B24 为 39136/73728；
接受事件完全一致分别为 32/256、80/384 请求。差异与 v25 eager 阶段相同，
不能据此声称 Native/PTO 浮点 bit 一致或复杂问题输出一致。

完整对照见 [B16](graph_replay_b16_comparison.json)、[B24](graph_replay_b24_comparison.json)，
对应 `.texts.md` 保留三侧的完整问题后缀和全部已生成回答。
工作区下载目录：`downloads/CSA_TND_GRAPH_REPLAY_QA_20261024/`。
01 为 B16、02 为 B24，HTML/Markdown 独立展示 Native、MPMD、SPMD；
`完整模型输入/` 保留 24 个不同的完整 128K 输入，JSON 保留原始输出 token 和来源。
192 token 是实际生成上限，回答可能未结束；导出不截取、不补写。
仅在已经证明 16 个 DP 副本逐 token 相同时才去重。

Native 沿用 `v02_native_b16`、`v12_native_b24` 的历史正确性参考，
经过 [源码审计](graph_replay_native_reference_audit.json) 确认其实际 Native 路径未变，
同时严格核对 CANN、权重、KV、问题、推理配置及 Native 扩展来源。
该复用仅用于正确性，不用于跨版本性能比较。

## 修复内容与内存边界

旧 v25 模型 dummy capture 的短历史使 TND 服务不满足最低 8192 长度要求，
捕获 Native 后，真实 decode 被图合同拒绝，再 eager 重派发 PTO。
旧结果的 FULL 标签不能证明重放，旧审计保留在 `graph_execution_audit.json`。

本次只给 TND 模型登记合法的捕获历史长度，dummy 长度和 positions 同时匹配；
真实调度以及 Native/BSH 默认路径保持原样。继续使用 Native compressor_metadata，
将图内请求和输出容量固定，由 device Q/L/start_pos 决定有效行。
失活请求通过图内 device L 将传给 Native 生产者的 start_pos 映射为 -1；
不原地修改原 metadata，不新增 D2H 或 Host 逐请求计划。
图合同保留 Q/L、页表、KV 根地址与策略检查，对 slots/RoPE 使用初始化容量视图。

普通 TND 入口仍只接收 55 个外部 Tensor；scratch、普通 Top-K 和 SPMD 信号均由
Simpler 内部管理。MPMD 保留 AICPU 多 task 调度，不增加软同步。
单卡脚本按真实模型保留所有图档及输入/binding，解决测试脚本的第二档图池生命周期断言；
该断言的原始证据保留在 `graph_replay_v26_summary.json`，不能误报成 PyPTO 算子缺陷。

显存比较是**同一 TND-MPMD 16 卡 128K/B24**的 v25 eager 与 v27 graph，
两侧早已采用内部工作区；在每请求生成完 192 token、模型尚未销毁时测量 rank0：

| 指标 | v25 eager | v27 graph |
| --- | ---: | ---: |
| 空闲显存 MiB | 260.27 | 2292.88 |
| PyTorch allocated MiB | 57609.25 | 55342.25 |
| PyTorch reserved MiB | 58292 | 56258 |
| 每层 binding 数 | 15 | 3 |
| Python 侧 PTO 临时工作区字节 | 0 | 0 |

相关配置相同，空闲增加 2032.61 MiB。尚未逐张量归因，不能把全部变化独立归因于
只开启 replay，也不外推其他档位或更长生成。
原始字节数及两侧来源见 [显存对照](graph_replay_mpmd_b24_memory_comparison.json)。

## 固定配置与版本

- 权重 `/data/model/DeepSeek-V4-Flash-0731-w8a8`；TP1、DP=EP16，DSpark 出 5 验 6。
- 21 层 PTO CSA、20 层 Native HCA；NZ2，EPLB 关闭。
- CANN 9.2.0-beta.2，npugraph_ex、FULL_DECODE_ONLY；static kernel 与 superkernel 关闭。
- 确定性 level=1，HCCL_DETERMINISTIC=true，AIV，event_work_mode=1。
- B16：显存预算 0.92，HCCL1800；B24：预算 0.935，HCCL768。
- 两种实现均为 `heap_mb=[16,256,16,16] MiB`、`task_window=4096`。
- 公共工具链 `csa-shared-20261024-group-dims`：PyPTO `d44925c1`、
  PTOAS `914863ed`、Simpler `57e49923`；源码与构建从公共 manifest 绑定。
- 冻结源码 `experiments/v27_graph_lifetime/source`，基础提交 `15b64665a`，
  四个生产文件 overlay 与该提交相同；单卡脚本额外保留图生命周期。
  正式运行时代码已核对与测试版本一致，不做重复逐文件 hash 扫描。
- 只读使用本地 `csa-repro-20261021-clean/kv-repro/mixed128k_fixed` 的前 16/24 个问题。
  历史实际长度 131104..131346，来自 128K 背景加问题后缀。

## 本地验证入口与证据

先 source 工作区 `env-dsv4-0251rc1.sh`，再 source 公共工具链 `env.sh`。
`prepare.py <实验名> --graph-replay-fix` 冻结框架、指定运行时文件、测试入口及 SDK 清单；
不复制 SDK、权重或 KV。测试期间不修改冻结包或在原位重建公共 SDK。

单卡经 task-submit 调用 `model_graph_probe.sh <冻结源码> <输出目录> <mpmd_tnd|spmd_tnd>`。
两种实现全部通过的合并证据作为 `run.sh` 第六个参数，才能进入 16 卡。
完整已执行命令和任务号在 `jobs.json` 的 `graph_replay_model_v27`；
`run.sh` 检查源码/SDK、门禁、batch/ring 后运行真实模型，随后执行严格 `check.py`。
先 B16 两侧及输出对照，通过后再 B24；每次使用新输出目录。

`compare.py --reuse-native-reference --native-source-audit <审计JSON>` 生成三方对照，
错误版本审计会被拒绝。`export_qa.py --comparisons <B16报告> <B24报告>` 导出全文。
单卡输出、Native 精度、整模型图重放分别报告，不相互替代。
详细中文过程见主 `DSV4_FLASH_CSA_VALIDATION_LOG.md` 的 735.115 起各节。

旧外部工作区和 v25 eager 的结果保留原始记录，不能当作本次 graph 结果：
`b16_comparison.json`、`internal_b16_comparison.json`、`internal_b24_comparison.json`、
`graph_execution_audit.json`、`production_source_audit.json` 及相应任务目录均为历史证据。
相关 PyPTO 编译问题的复现和状态见 `../CSA_THREE_SESSIONS_ISSUE_STATUS.md`；
本次 replay 修复没有新增 SDK 修改。
