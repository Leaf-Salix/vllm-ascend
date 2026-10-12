# 过程087：gated 参数准备四路展开，功能通过但性能回退

2026-10-12。候选撤下，没有新保留的生产代码；正常运行时仍为 Simpler `7043f8c1b`。

## 候选与边界

提前派发的 task 在 AICore 上遍历 Tensor/Scalar 参数并填充原 args 数组。
本轮只对这两个原有循环添加 `#pragma unroll 4`，试图降低循环控制开销。
不改参数布局、原读写语义、dispatch、ACK、核选择或 early-resolve，
也不提前操作前驱仍占用的计算流水。手工局部 Wait 和诊断串行回退保持。

候选只在独立 RuntimeBuilder 目录编译；编译结束立即恢复源码，不覆盖正常运行时或 SDK。
两个执行器对象的可执行代码确实变化：AIC 6132→6444 字节，AIV 6148→6460 字节，各增加 312 字节。
这仅排除了 pragma 完全未生效，不证明指令级成本或缓存影响。
现有 73 个 CSA 成员的源码参数槽数量从 1 到 21 不等；只作为场景说明，不当作运行时成本。

## 单卡性能

唯一设备任务 `task_20261012_081625_16558975322`，设备 0，CANN 9.2，TMR。
128K/B16、第 2 层固定真实权重，HC_pre＋norm＋CSA＋HC_post，
NZ=2、atomic_add=0、deterministic=0，关闭 profiler。
原→候选→原三个进程在一次卡锁内运行；每进程两条路径各 5 次预热、
30 次轮换单次图重放，reset 在 Event 外。同一套 SDK 7043 的冻结 CSA 编译产物。

| 运行时 / 路径 | min μs | mean μs | max μs |
| --- | ---: | ---: | ---: |
| 前测原版，普通 PTO | 975.40 | 993.85 | 1034.70 |
| 前测原版，early-start | 960.84 | 989.91 | 1015.08 |
| 循环展开，普通 PTO | 987.16 | 1004.53 | 1031.90 |
| 循环展开，early-start | 965.96 | 994.28 | 1028.60 |
| 后测原版，普通 PTO | 977.90 | 995.83 | 1022.04 |
| 后测原版，early-start | 970.36 | 991.46 | 1011.42 |

候选 early-start 对前/后原版 mean 多 4.37/2.82 μs，max 多 13.52/17.18 μs；
min 对前测多 5.12 μs、对后测少 4.40 μs。相对两次控制均值中点多 3.59 μs，
原 early-start 前后 mean 自身漂移 1.55 μs。普通 PTO mean 也多 10.68/8.70 μs。

同进程 early-start−普通 PTO 的配对均值前/候选/后为 −3.94/−10.25/−4.36 μs，
配对中位数 −4.73/−7.73/−1.34 μs，18/24/16 个轮次 early-start 更快。
候选的“相对普通领先更多”包含普通路径变慢，不能据此认定优化有效。
因此撤下，不追加采样、遍历展开因子或扩展其他档位。
没有指令级归因证据，不将回退定性为 Scalar-bound 或缓存容量问题。

## 功能及收尾

候选先通过 AIC/AIV/MIX 三项检查，两次输入 AIC 各 24/24、AIV 各 48/48 同核完整尾部交接；
MIX 各 24 AIC/48 AIV 在前驱组完成前进入前缀。数值、Scalar、保护区和等待检查通过。
完整尾部/组级插桩不能替代逐成员原始计算重叠证据。

三个性能进程中，两路径各 31 项保护区全部通过，9 项输出/状态及改变 hidden 输入后
图重放逐 bit 一致。任务退出 0，设备已释放，无失败重跑。
Simpler、PyPTO、PTOAS、PTO-ISA 和公共 CSA 默认均未因本候选改变。
临时实验入口归档后从私有 vLLM 测试目录移除，没有新增 Native、七档、token 或 16 卡测试。

## 证据与复现

- [180 条原始采样及统计入口](analyze.py)、[比较结果](comparison.json)；
  各进程的 `results.json` 和 `run.log` 保留全部采样及检查。
- [候选两行补丁](runtime/candidate.patch)、[代码段变化](code_sections.json)、
  [参数槽静态分布](arg_slots.json)、[环境版本](versions.json)。
- `diagnostics/` 保留三项功能日志与原始 trace。
- [构建/加载隔离候选入口](repro/tmr_csa_gated_args_demo.py)、
  [同卡完整执行命令](repro/run.sh)。复现时把入口放回私有 checkout 的
  `tests/pypto_test/`，先用 `--build --runtime-root <独立目录>` 构建，
  之后经 task-submit 运行；不覆盖默认 RuntimeBuilder 缓存。
