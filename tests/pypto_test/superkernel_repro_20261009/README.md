# Native SuperKernel 最小复现

目的：在当前 CANN 9.2 安装包上走真实的 `aclmdlRICapture* → aclskOptimize → aclmdlRIExecuteAsync`，
验证两个 AIV 子算子在同一物理核上的连续调用及 Early-Start 合同。
这是人为构造的机制用例，不能用它的百分比预估 CSA 收益。生产算子和公共环境均不修改。

2026-10-09 已在物理卡 2 完成四组对照，任务 `task_20261009_165431_362034525498`，exit=0。
四组输出精确一致，SK 两个入口同核且同步配置确实生效。
[全部结果、版本和原始样本](results.json)；[结果解释和后续步骤](../DSV4_FLASH_CSA_SUPERKERNEL_ANALYSIS.md#72-当前-native-最小复现)。

## 用例与对照

A 读取 16384 个 FP32 元素，连续加 1 共 32 次，输出到 GM；B 做 256 次有校验值的整数递推，
随后读取 A 的输出并加 2。输入是 0～30 的整数，全部输出可以精确比较。
两阶段各用一个 AIV block，UB 地址复用只发生在 B 等待之后。
计时结束后输入全部加 7，中间和最终输出填成 NaN，再重放一次校验，避免旧输出掩盖依赖错误。

| 文件 | 图方式 | Early-Start | B 等待位置 |
| --- | --- | --- | --- |
| `01_graph.jsonl` | 两个普通 kernel 的捕获图 | 关闭 | 无跨 SK 等待 |
| `02_sk_early_off.jsonl` | SK | 关闭 | SK 外部同步 |
| `03_sk_wait_first.jsonl` | SK | 开启 | 独立 Scalar 前 |
| `04_sk_wait_after_scalar.jsonl` | SK | 开启 | 独立 Scalar 后，首次输入读取前 |

四组算术和工作量相同，分别 5 次预热、20 次计时、1 次独立校验，保留全部计时样本和 min/mean/max。
01→02 包含融合派发的变化；03→04 才是移动等待的对照。输入输出复制不在计时区间。
计时包含探针的固定插桩和图重放开销，不是纯 Vector 硬件时间。

`SK_BIND` 的第二项是能力位：A 为 2（Set），B 为 1（Wait），不关闭默认 DCCI。
两个 SK 入口读取 `SkGetTaskSyncCfg()`；使用公开接口 `SetNextTaskStart<PIPE_MTE3, PIPE_FIX, true>()`
和 `WaitPreTaskEnd<true>()`。普通入口另有最终 `PIPE_ALL`，SK 入口由外层完成协议收尾。
B 的 `InitSocState`、UB 使用和输入 DMA 均在 Wait 之后，提前区只有不可变参数和整数运算。

## 复现

从当前仓根目录执行，先 CPU 编译，成功后才排队占卡：

```bash
bash tests/pypto_test/superkernel_repro_20261009/build.sh
/data/server-toolkits/pto-task/app/task-submit --device auto --max-time 180 \
  'bash /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/superkernel_repro_20261009/run.sh'
```

任务结束后：

```bash
python tests/pypto_test/superkernel_repro_20261009/summarize.py \
  tests/pypto_test/results/native_superkernel_repro_20261009
```

默认产物在上述 `results/native_superkernel_repro_20261009/`；两个 shell 脚本接受另一个输出目录作为第一参数。
运行脚本同时接受并核对队列自动追加的 `--device N`，实际设备仍以 `TASK_DEVICE` 为准。
复跑使用新目录，避免覆盖本轮证据。源码不得在编译、排队和执行之间变更；需要修改时重新编译并重新提交。
直接使用安装的 `libascendsk.so`，不经过当前尚不接受 `early_start` 参数的 torch_npu Python 白名单。

## 观测字段与证据边界

- `trace[0:4]`：A 的物理核号、实际 Early-Start 配置、入口时间、提交最后 Set 后的时间。
- `trace[16:23]`：B 的物理核号、配置、入口时间、Scalar 起止时间、Wait 后时间、整数校验值。
- 时间为 `get_sys_cnt()` 原始 tick；没有假定 tick 频率。`samples_us` 是 ACL event 的微秒时间。
- 同类 AIV 的预期配置为 A 含 `0x10`（Set）、B 含 `0x20`（Wait）。SK=0/early=0 的配置应为零。
- `tasks_before/after` 是捕获图任务总数，可能包含模型控制任务；不能单独以数量推断流水重叠。
- A 的 `issued` 是 Scalar 提交时间，不是 Vector/MTE3 完成时间；B 的 Wait 耗时也包含接口开销。
  同核号、非零配置、输出一致只能先证明实际协议执行。判定重叠还要结合等待前置/后移的对照，
  不能把函数返回时间当作计算完成时间。

本机 `llvm-objdump` 对设备指令返回 `<not available>`，Bisheng 拒绝设备 `-S` 和 `-mllvm -print-after`。
因此没有把符号表或预处理代码冒充设备汇编证明。编译中间文件保存在 `build/`。
当前只针对 AIV 单 block；AIC、不同核数、多个前驱、真实 CSA 和 Simpler 动态 task 都不在本用例的证明范围。
