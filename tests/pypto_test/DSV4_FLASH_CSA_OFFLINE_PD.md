# DSV4 Flash 离线 P/D 验证

当前合同、阶段状态及测试顺序见 [任务清单](DSV4_FLASH_CSA_TASK_CHECKLIST.md)。
先用单卡 case 完成定位与受影响验证，再进入正式权重 16 卡对照。
后续所有功能与性能测试均关闭 EPLB。测试入口固定关闭动态重平衡和专家热度采集，
不继承外部开启状态，已移除 `--eplb` / `--eplb-interval`；EP 并行配置保留。

## 数据与执行入口

`offline_pd/run.py` 管理 P TP4×DP4/EP16 缓存和 D TP1×DP16/EP16 测试。
仅使用 `/data/model/DeepSeek-V4-Flash-0731-w8a8`、当前固定的 release 环境。
P 使用 Native；D 的 `--backend native/pto` 选择实现。
H 表示 P 已计算的 token 数；D 加载 h(H)，再计算 H+1 token prompt 中的最后一个 token。

`offline_pd/connector.py` 按 Native cache group 保存/恢复缓存，
`prefix.py` 定义有效前缀，`observer.py` 提供真实模型的 profile、逐层对照和快照。
bank 的 `plan.json`、manifest、token 输入及 tensor payload 是复用所需数据，不能只保留报告。

当前保留的 bank 均位于 `results/release_offline_pd_20260923/`：

| 历史长度 | 目录 |
| --- | --- |
| 255 | `smoke_bank/` |
| 4095 | `h4095_det/`，Native 确定性配置 |
| 8192 | `h8192_bank/` |
| 32767 | `h32767_bank/` |
| 131071 / 131072 / 131073 | 对应 `h131071_bank/`、`h131072_bank/`、`h131073_bank/` |

bank 的 audit 只表示该份输入的既有检查范围，不表示当前 PTO 已完成整模型验收。
同一场景两侧复用同一 bank；不重复生成全部缓存，不做 hash 扫描。

## 命令与配置

```bash
source ../env-dsv4-0251rc1.sh
python tests/pypto_test/offline_pd/run.py --help
```

| 子命令 | 作用 |
| --- | --- |
| `plan` / `audit` | CPU 创建输入计划、核对 bank；已有有效 bank 按需复用 |
| `prefill` | 16 卡生成 Native P 缓存 |
| `decode` | 16 卡生成输出，记录 token、DSpark、配置与入口命中 |
| `steady` | 预热后采稳态 decode 指标 |
| `performance` | 同一次加载先采无 profiler 的完整 decode 周期、execute 和采样/草稿完成点，再独立采 Level0 各层区间；保存两轮 token 与 DSpark 统计 |
| `bitcompare` | 从相同状态比较真实层输出；诊断不作为性能测量 |
| `argdump` | 在 PTO 调用前保存 schema=2 输入/存储，调用后参考另存 |
| `profile` / `profile-export` / `profile-compare` | 采设备窗口、CPU 解析及对照 |
| `swimlane` / `swimlane-export` | 采指定 rank/层的 DFX，独立于正式性能统计 |
| `padding-capture` | 有针对性地检查实际 padding/graph 档位 |

需要上卡的命令通过 `task-submit --device auto --device-num 16 --max-time 7200` 提交，
在任务内切换到本仓库并加载 `../env-dsv4-0251rc1.sh`；只使用 `$TASK_DEVICE` 的分配。
用 `--status`、`--log` 查看任务，不用 `--wait`。两侧顺序运行、使用独立结果目录。

主场景参数为 `--batch 16 --graph-mode full_decode_only`，对应 H8192 bank。
两侧显式传相同 `--weight-nz-mode`，分别评估 1/2，保留另一档结果与 ND 路径。
精度诊断使用 `--deterministic`；性能按部署配置，完整记录两侧开关。
确定性由 `offline_pd.worker.OfflineNPUWorker` 在真实 worker 构造时设置，图捕获前生效；
结果中的 `worker_runtime_config` 保存实际读取的级别和 EPLB 状态。
旧日志里的父进程 `OFFLINE_DETERMINISTIC level=1` 不能单独证明 spawn worker 已开启。
通过 `PTO_CSA_VARIANT=precision/performance` 选择 PTO 入口，不能混用结果。

两侧完成 `decode` 后，在 CPU 比较全部 rank 的逐 token 和 DSpark 计数：

```bash
comparison_root=tests/pypto_test/results/csa_baseline_20260926/model_b16h8192_nz1_fixed
python tests/pypto_test/offline_pd/compare.py \
  --native "$comparison_root/native" --pto "$comparison_root/pto" \
  --bank tests/pypto_test/results/release_offline_pd_20260923/h8192_bank \
  --batch 16 --decode-tokens 96 --ranks 16 --output "$comparison_root/comparison.json"
```

比较器要求 bank 声明的 case、全部 rank、请求数和 token 数完整；同时比较草稿数、
草稿 token 数、接受总数及逐位置接受计数。缺少统计或任何差异均返回非零退出码。
此处 PASS 仅指 token/DSpark 对照通过，层级误差、状态、保护区和性能仍独立验收。

`--capture-sizes`、`--rank-batches` 和 `--rank-decode-tokens` 按待验证场景显式指定；
当前仅声明 S=6 合法档位。新泛化对比为 TP1/DP=EP16、131072 tokens、
单卡 B4/8/16/24/32/40；两侧 EPLB 均关闭。
本轮扫描固定 `max_num_seqs=40`，用
`--batch 40 --sweep-batches 4 8 16 24 32 40 --capture-sizes 24 48 96 144 192 240`：
每侧只加载一次正式权重，各档从同一 bank 恢复并独立预热、采样及采 trace。
结果分别落在 `--output` 的父目录下 `b{B}/{backend}`；加载日志保留在原 `--output`。
DSpark 按每档开始前快照作累计计数差分，报告 `batch` 和 `max_num_seqs`；
对照器显式要求 `--max-num-seqs 40`。旧 B4/容量 4 的首轮方法验证单列，不混入容量 40 矩阵。

## 结果判读

- 无 profile 的稳态 decode 指标用于整模型性能；`elapsed_including_io_seconds` 不作加速比。
- `steady` / `performance` 的设备事件在图外记录，收尾统一读回，不逐步加同步；
  schema=2 同时记录 execute 首尾、采样/草稿完成点及连续满档步骤起点。
  主机时间单列，完整周期使用 NPU Event 的 `elapsed_time`，不猜测原始计数单位。
  默认 `--steady-cycles 20`，预热后取前 21 个满档起点形成 20 个周期；中途变档、
  采样未完成或实际采样输出计数缺失时拒绝放行，不跨越不完整步拼接周期。
  `--decode-tokens 192 --warmup-steps 8` 为窗口结束留余量，避开请求结束时的输出裁剪。
  全局吞吐按共同样本序号取各 rank 最慢周期，是同步周期吞吐的保守估计；每 rank 原始周期和吞吐另列。
  历史 schema=1 只有 execute_model 时间，不能追认为完整 decode 周期。
- `performance` 的 Level0 trace 与无 profiler 窗口独立；层区间应从首末设备任务取差，
  按 rank/层报告并保留首次 metadata 生产成本，不能累加并发 kernel 时间。
  采集窗口中途发生档位变化时会保留原始观测并拒绝作为验收结果。
- profiler 使用同窗口、同实际配置；完整 HC_pre→norm→CSA→HC_post 区间包含内部间隙与适配。
- DFX 用于定位，不能与正式性能数据混算，也不能重复累计 scheduler/worker 时间。
- 诊断必须有有效样本；逐 token、DSpark、层级误差与状态按清单合同独立检查。
- 未关闭的失败保留最小复现和最新报告；已过时的试错、重复日志和快照及时删除。
