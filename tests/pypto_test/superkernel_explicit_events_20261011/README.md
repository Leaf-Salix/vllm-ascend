# 过程062：放行不与后到通知冲突的显式流水同步

日期：2026-10-11。PTOAS `7df7652b`、PyPTO测试驱动 `555d0fffe`。
Simpler `5085a6ce0`、私有CSA及生产PyPTO生成器均未改，精确版本见 [versions.json](versions.json)。
本阶段补编译覆盖，未重新测试CSA性能，也没有改变原派发、ACK、核选择或early-resolve。

## 去掉了哪些额外限制

此前有两层旧原型限制：InsertSync一遇到显式pipe-event便拒绝后到交接；
`runtime_barrier_all`的异步尾部白名单也拒绝普通pipe-event/barrier。
现在允许作者已有的显式同步保留原顺序，InsertSync不向其中再叠加自动事件。
通知资源检查与自动事件分配共用同一份pipe-pair规则，自动事件池没有扩大或缩小。

原完整通知占用以下pipe pair的event7：MTE2/MTE1/M/FIX/V→MTE3。
本阶段的行为如下：

| 输入 | 当前行为 |
| --- | --- |
| 通知pipe pair上的静态event0～6 | 允许，保留原Set/Wait |
| 其他pipe pair上的event7 | 允许，不把event7全局禁用 |
| 高层record/wait | 映射实际pipe pair后按相同规则检查 |
| 常量动态event ID | 取常量后按相同规则检查 |
| 非通知pipe pair上的未知动态ID | 允许，不额外推导与通知无关的ID |
| 通知pipe pair上的event7 | 明确报资源冲突，尚未自动重分配 |
| 通知pipe pair上无法确定的动态ID | 明确报尚不能证明避开event7，仍属实现覆盖缺口 |
| Wait之前的pipe操作 | 仍由独立Scalar前缀验证拒绝，不能越过局部Wait |

局部`wait_pre_task_end()`仍由作者标识，不新增自动识别；chip swimlane/PMU/参数dump串行回退保持。
旧joined-tail协议没有放宽为任意显式同步，本阶段针对当前使用的运行时全流水收尾。
没有要求算子作者重写整个InCore task，也没有隐式开启`allow_early_resolve`。

## 验证

CPU检查覆盖六个正向函数：真实TLOAD→手工Set/Wait→TSTORE、无关pipe pair的event7、
高层事件、常量动态ID、无关pipe pair的未知ID、未标记函数的既有event7行为。
三个通知冲突/未知ID诊断，以及一个Wait前事件安全诊断通过；
原自动通知尾与普通record/wait两个相邻回归也通过。见 `logs/explicit_event_checks.log`。

设备测试复用原两任务小例子的实际同步结果：先序列化正常InsertSync输出，再把已有事件作为
显式PTO IR输入编译。只移除需要前端重新生成的keep-prefix与尾部包装，不删load/compute/store
依赖，也不人为插入长Scalar等待。输入、已同步IR、显式IR和最终wrapper都保存在`generated/`。

任务 `task_20261011_223620_61477522673`，单卡0，退出0并释放卡。
AIC/AIV、loops=4、两组输入、原单次探测与完整后到通知各一路；本轮repeats=0，不做性能计时。

| 核/路径 | 输入0同核交接/其中后到 | 输入1同核交接/其中后到 | 重放同核交接/其中后到 |
| --- | ---: | ---: | ---: |
| AIV，单次探测 | 0/0 | 0/0 | 14/0 |
| AIV，完整通知 | 4/0 | 0/0 | 0/0 |
| AIC，单次探测 | 0/0 | 0/0 | 0/0 |
| AIC，完整通知 | 15/15 | 16/14 | 21/19 |

输出、完整依赖读取、保护区、无后继退出及改变输入后的重放检查通过。
AIC有明确后到交接；AIV本轮有普通同核交接，没有后到命中，不能声称本轮证明了AIV后到收益。
计数与尾部trace也不能替代原计算重叠长度或CSA净性能证据。

## 实现过程中发现的问题

第一份CPU用例误写高层枚举`TSTORE`，应为`TSTORE_VEC`，在解析阶段修正。
随后正向用例揭示第二层异步尾部白名单，故连同InsertSync资源检查一起修复。
序列化IR直接再次送入前端时，已有尾部包装被认为是Set之后的工作；移除需重生成的尾部后，
又发现已有keep-prefix私有声明会重复。测试驱动现在移除这两类自动包装及keep-prefix调用，
由前端重新生成；原显式流水事件始终保留。失败日志归档，均发生在CPU，没有失败的设备任务。

## 复现与后续

以`versions.json`的PTOAS源码重建私有编译器；在私有PyPTO目录source `../env.sh`，运行：

```bash
export PYTHONPATH="$SCALAR_OVERLAP_ROOT/pypto/python:$SCALAR_OVERLAP_ROOT/simpler/python:$SCALAR_OVERLAP_ROOT/simpler"
python tests/st/runtime/scheduling/task_boundary_explicit_event_probe.py \
  --compile --work 4 \
  --compiled "$SCALAR_OVERLAP_ROOT/artifacts/explicit_event_probe_compiled_20261011"
```

经task-submit单卡执行`repro/run_explicit_event_probe_20261011.sh`。
`run/`是完整输出和trace，`source/`保留用例，`*-incremental.patch`为零上下文增量，
用`git apply --unidiff-zero`应用。没有新CSA七档、16卡或性能结论。

仍需处理通知event7冲突时的编译器资源分配及动态ID范围证明；不能把它们一概解释成理论禁止。
`runtime_barrier_all`还有旧算子白名单，其他内存规划器的通知scratch能力也尚未全面验证。
后续先审查这些实际编译覆盖缺口；稳定CSA净收益仍未成立，继续区分功能覆盖和性能验收。
