# idle / sync_start Scalar 前缀覆盖与 128K/B16 验证

2026-10-11。私有 Simpler 版本为 `8171b78c5`，PyPTO `19f2c091a`，
PTOAS `4a5ce22a`。公共 SDK 与算子默认路径未切换。

## 修改与功能证据

- idle 核已提前派发的标记任务，完成原参数准备后进入安全 Scalar；局部 Wait 再等完整输入 doorbell。
- sync_start 后继也可先运行 Scalar；本核前驱的真实 FIN 在整组等待前发布，
  原 pending 晋升和 cohort doorbell 仍决定计算及 ACK 何时允许执行。
- 原核选择、派发顺序、ACK 后流程、allow_early_resolve 和整组同步语义不变。
- 在旧实现上，idle 能力与 sync_start pending 回归失败；修改后10项 CPU 调度检查通过。
- 任务 `task_20261011_173025_10211416583` 使用单卡0，7项功能测试通过，退出0并释放卡。
  功能测试使用5c8418a02加本轮补丁；随后以相同实现锁定为8171b78c5，供精确SDK计时构建。

| 场景 | 两轮提前于远端完成的前缀 | 两轮与本核前驱流水重叠 |
| --- | --- | --- |
| idle，普通 | 48/48、48/48 | 无本核前驱 |
| idle，sync_start | 48/48、48/48 | 无本核前驱 |
| pending，普通 | 48/48、48/48 | 48/48、48/48 |
| pending，sync_start | 48/48、48/48 | 48/48、48/48 |

所有消费者计算均在远端输入真实完成之后；sync_start 计算还在全部本核前驱完成之后。
改变输入的第二轮、输出和保护区通过。上述48/48是构造用例，不能当作CSA的实际交接率。
[汇总与逐场景文件名](summary.json)、[设备报告](functional.xml)、[设备日志](device_task.txt)。

## CSA 性能

任务 `task_20261011_173903_4357828989`，单卡2，退出0并释放卡。
两版均为同一新SDK：普通版无局部Wait/Set，early_start版保留所有已有标记。
第2层真实权重、seed1024合成历史；CANN9.2/TMR/NZ2/atomic=0/deterministic=0，profiler关闭。
区间含 **HC_pre + norm + CSA + HC_post** 及调度，缓存恢复不计时；每版warmup5次，交错各30次。

| 128K/B16，单位μs | min | mean | max |
| --- | ---: | ---: | ---: |
| 普通 | 971.10 | 987.81 | 1012.80 |
| early_start | 955.02 | 979.30 | 1017.04 |

mean少8.51μs（0.8619%），max多4.24μs。输出/缓存/状态9项及改输入重放逐bit一致，
两版各31项保护区通过。单次对照有均值收益，最大值未改善；未证明所有档位或整模型获益。
上轮8171之前的同SDK七档测试，128K/B16为983.60/984.32μs；两轮普通基线不同，
不能用这两轮直接相减声明idle/sync_start的精确因果收益。
[完整60次样本与功能结果](csa_128k_b16.json)、[执行日志](csa_device_task.txt)。

首次任务 `task_20261011_173201_16262712771` 在加载预编译产物时，被精确clean SDK检查拒绝，
未执行CSA。随后提交并同步SDK pin，重建binding/runtime/torch adapter、在CPU重新预编译后，
完成上面的成功任务。[首次失败日志](csa_failed_preflight.txt)。

## 未完成范围与下一步

MIX最小用例已在CPU复现前端统一拒绝：`local task boundaries do not support MIX kernel groups`。
当前仍有两个编译层缺口：Group提交未传递成员边界能力；PTOAS异步尾部检查尚不接受跨核pipe。
需要按AIC/AIV0/AIV1分别传递Wait/Set能力，保留每个成员的真实FIN和整个任务的完成计数，
并确保pipe初始化留在Wait之后。不能简单放开整组布尔标记。
[编译失败证据](mix_compile_failure.txt)。MIX尚未上卡，也没有宣称已支持。

只在前驱返回时单次探测后继、诊断模式关闭交接等缺口继续保留为待办。
当前goal继续有效，本阶段结果不缩小“所有已early_dispatch且有安全Scalar前缀都应有机会”的目标。

## 复现

本目录保存运行与构建脚本、原始样本及两仓补丁。
私有根目录为工作区`.worktrees/scalar-overlap-demo-20261009`，使用其`env.sh`。
先在CPU按`build_runtime.sh`重建匹配SDK与预编译CSA，再通过单卡`task-submit`执行`run_functional.sh`或`run_csa.sh`。
编译驱动沿用上一轮七档归档中的`tmr_csa_full_128k_b16_compile.py`，仅编译baseline和all两版。
