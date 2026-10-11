# 过程068：直接通知 metadata 化的单卡验证

2026-10-12。**功能通过，完整 CSA 性能未改善，不切换默认编译回调。**
本次仅验证 TMR、单卡 128K/B16；无新的 Native、七档或整模型数据。

## 修改与不变量

- Simpler `6174ec5db`：在原 TaskPayload 的最后 4 字节 padding 中编码 AIC/AIV scratch，
  地址以 64B 为单位加一编码，零表示保留回调。原派发填写 GlobalContext，payload 大小不变。
  两个 AIV lane 使用相同的编译函数，但仍逐成员检查 tail/callback 能力。
- PyPTO `44b4ffe76`：支持内部 `task_boundary_direct_scratch` metadata。
  私有 hook 在 PTOAS 后检查实际通知事件，全部为 7 且地址可编码才启用直接路径；
  非 7 的分配继续走原编译回调，不降低 early_start 覆盖。
- 私有 vLLM `15d44c45`：四路对照驱动。PTOAS/PTO-ISA 未改。

实验协议为 `RuntimeDirectPollClean=9`，默认仍为 `RuntimeCompiledPollClean=8`。
局部 Set 无 GM 写入；未插桩的普通收尾不再刷新未修改的控制缓存行。
继续按真实 worker 类型调用过程067的 AIC/AIV 强符号函数。
原队列、派发、ACK 触发、核选择、用户 early_resolve、完整 fanin 与本核流水等待不变。
使用手工局部 Wait；chip swimlane/PMU/参数 dump 保持串行回退，未新增自动识别或算子参数。

## 编译与功能证据

CPU 调度测试 14 项、原 orchestration 相关测试 3 项通过。
SDK binding、torch adapter、Python ABI 常量统一到上述 Simpler revision。
生成的 73 个 kernel C++ 在去掉整行注释、归一化进程内递增的 inline ID 后一致；
orchestration 只增加 66 处资源 metadata，见 [源码对照](timing/codegen_comparison.json)。
原始源码直接比较会因 inline ID 不同而不同，不能称原始文件逐字一致。

显式动态事件 0/7 的 AIV 编译用例继续使用 MTE2→MTE3 event 6，且没有直接通知 metadata，
见 [事件回退检查](fallback/result.json)。这里只补编译契约检查，没有重复上卡验证旧回调。
首次证据读取脚本误用 `kernels/*.cpp`，实际文件位于 `kernels/aiv/`；改为递归查找，
复用已成功编译产物完成检查，没有重编译或重跑 NPU。

设备任务 `task_20261012_005911_200591512733`，卡 0，退出 0，已释放设备。

| 小例子 | 两组输入的后到交接 | 重放后到交接 | 通知核型核对 |
| --- | ---: | ---: | ---: |
| AIV | 2 / 9 | 21 | 137 / 137 |
| AIC | 8 / 18 | 18 | 72 / 72 |

loops=4，额外 Scalar 循环=0；输出、真实 fanin、保护区、完成顺序及换输入重放通过。
后到计数随派发时序变化，不直接用来计算净性能；完整尾等待包含通知，不代表独立原计算时长。
MIX 两项通过，含跨核 FIFO 与 sync_start；两轮各 24/48 个 AIC/AIV 成员进入安全前缀。
每轮通知计数 producer AIC/AIV=24/0、consumer AIC/AIV=24/48，已发通知全部核型匹配。
producer AIV 无通知只表示其该次路径未发起通知，不等于 MIX AIV 不支持交接。
详见 [小例子](small/results.json)、[核型检查](small/engine_route_counts.json)、
[MIX](mix/mixed_chain_results.json) 和 [MIX 核型检查](mix/engine_route_counts.json)。

## 同 SDK 的 128K/B16 完整耗时

第 2 层真实权重、合成 128K 历史、B16×verify6，范围是 HC_pre+norm+CSA+HC_post。
CANN 9.2、TMR、NZ2、atomic_add=0、deterministic_level=0，关闭 profiler/PMU/参数 dump。
四路全部重新编译；预热 5 次后各 30 次轮换，NPU Event 计量单次图重放，reset 在计时外。
单位 μs；这是 PTO 内部对照，表中没有 Native。

| 路径 | min | mean | max |
| --- | ---: | ---: | ---: |
| 普通 PTO | 966.72 | 987.82 | 1013.12 |
| 现行编译回调 | 972.16 | 990.22 | 1007.46 |
| 直接通知，Set 写入资源 | 960.24 | 988.55 | 1012.04 |
| 直接通知，metadata 下发资源 | 968.30 | 994.75 | 1018.26 |

metadata 候选相对现行回调 mean **+4.53 μs**、max **+10.80 μs**，9/30 配对更快，
配对中位差 +4.12 μs。相对带 Set 的直接路径 mean **+6.20 μs**、max **+6.22 μs**，
8/30 配对更快，中位差 +7.92 μs。
9 项状态与换输入重放均逐 bit 一致，每路 31 项保护区通过。
[完整结果及版本](timing/results.json)、[120 条样本](timing/samples.csv)、
[配对比较](timing/paired_comparison.json)、[任务日志](timing/task.log)。

结论：去掉 Set 写入与普通收尾 flush 并不足以改善完整 CSA。
本次没有把通知成本、协议分支成本与调度时序逐项分离，不能将回退定性为某一项的独立开销，
也不能据此证明局部 early_start 无效。保留私有实验和功能证据，不切默认、不追加重复参数搜索。
完整 CSA 的稳定净收益仍未成立；后续继续定位通知成本与实际可交接窗口。

## 复现与归档范围

本目录是过程067之后的**增量归档**，包含三个仓的零上下文补丁、版本、驱动及全部本轮样本。
依赖已有私有环境和冻结 CSA 包；不是从任意主分支全新 clone 即可运行的独立包。
应用补丁使用 `git apply --unidiff-zero`，基准与目标见 [versions.json](versions.json)。
PyPTO runtime gitlink 与 `_kernel_abi.py` 必须同时匹配 Simpler；不复制其他 checkout 的 `.so`。

将 `repro/` 中脚本放回私有环境的 `artifacts/`，依次执行构建、编译；单卡执行必须排队：

```bash
bash "$private_root/artifacts/build_clean_direct_20261012.sh"
bash "$private_root/artifacts/compile_clean_direct_20261012.sh"
task-submit --timeout 900 --max-time 900 --device auto --device-num 1 \
  --run "bash $private_root/artifacts/run_clean_direct_20261012.sh"
```

生成产物位于私有环境 `artifacts/csa_clean_direct_128k_b16_20261012/compiled/`。
编译日志位于 `build/`；保留现有诊断及测试中的弃用告警，不把告警计为功能失败。
公共算子默认不变，其他会话的 HC_PRE 改动未纳入本次提交。
