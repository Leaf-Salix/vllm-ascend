# gated后继参数准备回到AICore：128K/B16验证

日期：2026-10-11，实施过程054。保留该清理；单轮测得小幅收益，尚未证明稳定加速。

## 改动

Simpler `3f86962a7` 去掉 gated pending handoff 在 AICPU 上额外构造参数向量的工作。
所有 gated task 继续沿用普通提前派发的路径，由接收任务的 AICore 在调用后继前准备
张量描述符指针和 Scalar 参数；该工作可以与本核前驱计算流水重叠，不读取输入张量数据。
ready task 仍在 AICPU 准备参数，完整 fanin、sync_start 门铃、原 ACK/FIN、选核和
用户 allow_early_resolve 均未改变。PyPTO `58ccfcc72` 仅同步 SDK pin/gitlink；
PTOAS `52b8eaf1` 和上一过程的 CSA 六条 MIX 局部语句保持不变。

## 同SDK三路性能

| 路径 | min（μs） | mean（μs） | max（μs） |
| --- | ---: | ---: | ---: |
| 普通 | 960.10 | 982.46 | 1004.46 |
| 原 early_start 标记范围 | 959.00 | 982.22 | 1001.40 |
| early_start + MIX | 959.30 | 979.16 | 1002.42 |

单卡0，任务 `task_20261011_183915_2833602685`，退出0且设备已释放。
CANN9.2/TMR/NZ2/atomic=0/deterministic=0，诊断关闭，第2层真实权重+合成128K历史，
B16×验6；区间含HC_pre+norm+CSA+HC_post。每路径warmup5次后轮换30次单次图重放，
缓存恢复在计时外。完整90条样本见 [csa_results.json](csa_results.json)。

相对本轮普通路径，MIX候选均值减少3.31μs（0.34%），max减少2.04μs，18/30配对更快；
相对本轮原标记范围，均值减少3.07μs，但max增加1.02μs。当前只支持“小幅候选收益”结论。
上一过程在另一张卡上普通/原标记/MIX均值为989.03/981.37/992.01μs；
不能把992.01→979.16μs的跨轮绝对变化全部归因于本改动，也不能声称已稳定消除回退。

9项输出/缓存/状态以及改变输入后的图重放全部与本轮普通路径逐bit一致，
每条路径31项保护区/metadata检查通过。本轮没有测试Native、8K、七档或16卡整模型。

## 功能与重叠证据

- 11项调度CPU测试通过，覆盖ready参数、idle gated参数、pending及MIX分成员参数路径。
- 单卡任务 `task_20261011_183716_274484923248`，卡0，9项通过，设备已释放。
- idle/pending × 普通/sync_start 四组合，两轮均48/48前缀先于远端依赖完成；
  pending两组合的两轮均48/48与无数据依赖的本核前驱形成重叠，数值、等待边界及保护区通过。
- ready纯AIV两轮均48/48重叠；ready纯AIC分别15/24、24/24重叠。
- MIX两轮24/24 AIC与48/48 AIV均在前驱整组完成前进入Scalar；AIV均48/48同核交接。
  本轮这些MIX AIV在局部Wait处没有观测到大于5tick的剩余尾等待。
  这既不能证明CSA是Scalar bound，也不能证明参数准备期间没有重叠；本测试不是CSA。
- 原始时序在 [device_traces](device_traces)，完整pytest结果在
  [aicore_args_device.xml](aicore_args_device.xml)。没有用静态标记数代替实际CSA命中数。

## 剩余探测窗口与约束

用户确认：当前只手工标识局部 `wait_pre_task_end()`，不做自动识别；开启chip swimlane、
PMU或参数dump时仍回退。这两项是接受的行为，不列为缺陷或待补齐目标。

运行时仍只在前驱Scalar返回时探测一次，随后若进入阻塞PIPE_ALL，后到的已派发后继
不能再插入前缀。检查本地CANN9.2头文件发现 `try_wait` 声明，但头文件存在不代表A3可用。
CPU编译探针最终确认其两参数签名在 `dav-c220-vec` 上被拒为
“does not support the given target feature”；未接入运行时，也未占卡试运行。
最初 `-S` 不支持和零/一/三参数试探的错误记录一并保留在 [late_arrival_probe](late_arrival_probe)。
该证据只否定此目标上直接使用该builtin，不能断言所有非阻塞完成查询都不可行。
后续先审查其他可用的完成信号与实际CSA命中情况，避免盲目增加轮询或改动调度策略。

## 复现与版本

完整版本见 [versions.json](versions.json)，Simpler/PyPTO补丁随目录保存。
私有环境：`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009/env.sh`。
构建脚本 `build_aicore_args_runtime.sh` 对齐binding/runtime/torch适配的三处SDK版本；
先运行单卡功能脚本，然后用前一过程的CPU编译驱动预编译 baseline/all/mixed，
最后经 `task-submit --device auto --device-num 1` 运行本目录的CSA脚本。
未修改其他会话的公共HC_PRE文件，未push。
