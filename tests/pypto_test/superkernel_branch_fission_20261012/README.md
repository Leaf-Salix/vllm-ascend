# 分支内局部 Wait 后继续计算：保留 Scalar 前缀并共享等待边界

2026-10-12，过程080。私有PyPTO提交`f6edfc1fd`保留；公共CSA默认路径不变。
这是显式边界的功能扩展，没有新增CSA性能结果。

## 支持的写法和编译处理

过程078只覆盖各分支末尾的相同Wait。本轮接通顶层双分支中Wait后仍有计算的形式：

```python
if condition:
    value = scalar_prepare_a()
    pl.system.wait_pre_task_end()
    compute_a(value)
else:
    value = scalar_prepare_b()
    pl.system.wait_pre_task_end()
    compute_b(value)
```

PyPTO在tensor-to-tile转换、隐式加载和MIX拆分前，将它变成
“按原条件执行Scalar前缀 → 一个共享Wait → 按同一条件执行原后半段”。
复杂条件只求值一次；不执行未选分支，不把原后半段的输入读取提前。
跨Wait使用的Scalar通过SSA结果传递，原分支返回值和yield语义保持。
隐式参数加载和MIX管道初始化在共享Wait之后，PTOAS继续从这个边界分析同步和检查前缀。

两侧互斥分支的同类型Scalar共用传递槽，各类型的槽数取两侧数量的最大值，避免按两侧数量之和增加活跃值。
只有一侧需要的槽在另一侧填同类型常量0，不虚造Tensor或设备缓冲。
此项只说明生成结构的精简，没有单独的性能收益结论。

准入范围明确限定为：顶层if/else、每侧恰好一个直接的无参数/属性Wait、跨边界值为Scalar。
形状或stride表达式中使用的Scalar也计入依赖。已有的分支末尾Wait继续交给过程078处理。
更深的非末尾嵌套Wait、循环内Wait、跨边界Tensor/tile描述符仍未覆盖；缺失分支、重复Wait或不安全前缀不会因此获准。

作者仍手工写局部`wait_pre_task_end()`，本轮不做边界自动识别。
chip swimlane、PMU、参数dump启用时仍串行回退；这两项是用户接受的范围。
原派发、ACK、核选择、`allow_early_resolve`、完整本核尾部与fan-in/cohort等待保持不变。

## 验证结果

最终同类型槽复用版本：卡0，CANN9.2/TMR，任务`task_20261012_060137_81638327947`，退出0并释放卡。
三个用例各运行两次并改变输入，保留原数学计算、真实MIX FIFO和`sync_start`：

| 检查 | 第一次 | 第二次 | 证据范围 |
| --- | ---: | ---: | --- |
| 纯AIC同核Scalar/前驱流水重叠 | 24/24 | 24/24 | 同核时序及前驱完整尾部等待断言通过 |
| 纯AIV同核Scalar/前驱流水重叠 | 48/48 | 48/48 | 同上 |
| MIX早于前驱整组完成进入前缀的AIC成员 | 24/24 | 24/24 | 组级入口时序 |
| MIX早于前驱整组完成进入前缀的AIV成员 | 48/48 | 48/48 | 组级入口时序 |

纯AIC另有22/24、23/24个后继在最后一个前驱成员完成之前结束；纯AIV为46/48、42/48。
MIX两次通知计数均为生产者AIC/AIV=24/0、消费者AIC/AIV=24/48。
MIX组级提前入口不等同于逐成员的同核计算重叠证明。
真实输出、Scalar结果、Wait后读取、trace/输出保护区和完整尾部等待断言通过。
本轮未新增图重放检查，也不将带诊断的小用例当作性能计时。

- [最终设备报告](device/results.xml)、[设备日志](device/run.log)、[摘要](device_summary.json)。
- 原始结果：[AIC](device/local_aic_branch/ready_results.json)、[AIV](device/local_aiv_branch/ready_results.json)、[MIX](device/mixed_chain/mixed_chain_results.json)。
- [3项结构对照测试](preload_paired.xml)：既有顶层/分支末尾Wait及新增分支后半段，检查Scalar传递和隐式加载位置。
- [1项完整编译负例](unsafe_paired.xml)：一侧Wait前读取未知输入，仍以不安全前缀拒绝编译。
- [最终8个生成成员检查](prefix_audit_final.json)：各函数体恰好一个有效Wait，前面没有流水操作；路径相对原始kernels目录，对应文件归档于`compiler/`。

初版未复用Scalar槽时也完成三个设备用例，任务`task_20261012_055723_65261715183`，退出0并释放卡。
[初版摘要](unpaired_device_summary.json)仅保留功能历史，最终判据使用上面的槽复用版本。

## 失败与修正

首次结构测试读到了安装目录里的旧C++扩展，增量构建后补做私有安装。
随后暴露Expected IR遗漏原分支第三个SSA返回值；补齐预期yield，没有放宽结构比较。
负例最初遗漏block使用、使用了不支持的index-to-float cast，先修正用例使它到达目标验证器；
最终异常类型不变，仅让报错匹配兼容诊断表格换行，实际仍由不安全输入读取触发拒绝。
[初始失败摘录](initial_test_adjustments.json)、[最终构建](build_paired.log)、[安装](install_paired.log)、
[结构检查日志](preload_paired.log)、[负例日志](unsafe_paired.log)。

## 版本、复现与性能边界

PyPTO `f6edfc1fd`，Simpler `1d09ecd34`，PTOAS `4df4627e`，私有vLLM驱动`7f9f8213`。
本轮只改PyPTO，运行时ABI和PTO-ISA不变；测试源码与提交内容相同。
[完整版本](versions.json)、[PyPTO提交补丁](pypto.patch)、[纯核用例](repro/test_local_task_boundary_overlap.py)、
[MIX用例](repro/test_local_task_boundary_mix.py)。
在含`env.sh`的私有环境中，从PyPTO仓目录运行：

```bash
source ../env.sh
cmake --build build --target pypto_core --parallel "$PYPTO_BUILD_JOBS"
cmake --install build --prefix "$SCALAR_OVERLAP_ROOT/.venv/lib/python3.10/site-packages/pypto"
python -m pytest tests/ut/ir/transforms/test_convert_tensor_to_tile_ops.py \
  -k 'local_wait or branch_wait_with_work' -n "$PYPTO_TEST_JOBS" -q
python -m pytest tests/st/runtime/scheduling/test_local_task_boundary_overlap.py \
  -k branch_wait_rejects_pre_wait_input_read --platform a2a3 --codegen-only -q
```

将[单卡脚本](repro/run_branch_fission_20261012.sh)放在私有环境`artifacts/`目录，
通过`task-submit --device auto --device-num 1 --run 'bash <script>'`运行，使用队列给定的`TASK_DEVICE`。
重新编译安装时须避开正在使用该私有扩展的任务。

现有冻结CSA的[73个成员](csa_existing_boundaries_from_stage078.json)都使用顶层Wait，没有命中新规则，
因此不重复CSA计时。最新128K/B16 min/mean/max仍见[过程079](../superkernel_optional_trace_20261012/README.md)，
不能把那轮通知回调的收益归给本次分支扩展。本轮没有新Native、七档、token或16卡结果。
后续仍需处理一般只读GM前缀的写者/实际别名依据、更一般的控制流/描述符跨边界，以及通知资源全部不足时的后到观察窗口。
