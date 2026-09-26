# DSV4 Flash 离线 P/D 验证

当前合同、阶段状态及测试顺序见 [任务清单](DSV4_FLASH_CSA_TASK_CHECKLIST.md)。
先用单卡 case 完成定位与受影响验证，再进入正式权重 16 卡对照。

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
通过 `PTO_CSA_VARIANT=precision/performance` 选择 PTO 入口，不能混用结果。

`--capture-sizes`、`--rank-batches` 和 `--rank-decode-tokens` 按待验证场景显式指定；
当前仅声明 S=6 合法档位。EPLB 等暂停范围以清单末尾为准。

## 结果判读

- 无 profile 的稳态 decode 指标用于整模型性能；`elapsed_including_io_seconds` 不作加速比。
- profiler 使用同窗口、同实际配置；完整 HC_pre→norm→CSA→HC_post 区间包含内部间隙与适配。
- DFX 用于定位，不能与正式性能数据混算，也不能重复累计 scheduler/worker 时间。
- 诊断必须有有效样本；逐 token、DSpark、层级误差与状态按清单合同独立检查。
- 未关闭的失败保留最小复现和最新报告；已过时的试错、重复日志和快照及时删除。
