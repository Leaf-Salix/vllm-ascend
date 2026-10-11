# 未插桩通知回调省去诊断判断

2026-10-12，过程079。保留为私有TMR编译器优化；公共CSA默认路径未切换。
继续使用手工局部`wait_pre_task_end()`/`set_next_task_start()`，不做边界自动识别。
chip swimlane、PMU、参数dump串行回退是用户接受的行为，不列为待消除限制。

## 改动和边界

此前每次编译通知回调都会读取`ctx->scalar_overlap_trace`，再判断是否更新诊断标志。
正常CSA没有trace缓冲，原派发也会将该指针复位为0，这次只省去上述读取、判断及条件更新代码。
32位代次通知、ID回绕回退、片上填充、64B DMA、全部流水汇合、轮询与完整尾部等待保持。
没有改变派发、ACK、核选择、用户early-resolve或算子算术。

PyPTO `dfe35bc31`默认不生成这两行；已有`async_return_trace`诊断属性仍生成原计数。
手工插入trace指针的AIC/AIV探针与CSA覆盖插桩改为显式调用内部生成器恢复计数。
这不是新增算子作者必须声明的整函数属性，也没有把无通知的阻塞回退伪装成通知路径。
Simpler `1d09ecd34`、PTOAS `4df4627e`、PTO-ISA保持不变；私有vLLM驱动为`7f9f8213`。

先在私有编译对照中删除两行，测出两轮收益后才接入默认生成器。
控制产物复用过程077，候选用过程078后的编译器；三路运行同一Simpler SDK。
程序IR相同，73个生成成员在一一规范内联变量编号后，唯一变化是71个回调的两行诊断代码。
首个字面源码比较因内联实例编号不同而失败，修正比较方法后通过；没有放宽流水或功能检查。
见[源码比较清单](compiler/source_comparison.json)、[Wait前流水检查](compiler/untraced_prefix_audit.json)、
[首次比较失败](compiler/initial_source_comparison_failure.log)和[最终检查](compiler/source_audit.log)。

## 单卡128K/B16性能

CANN9.2/TMR，第二层HC_pre+norm+CSA+HC_post，NZ2、atomic0、deterministic0，profiler关闭。
每轮三路各5次预热、30次轮换单次NPUGraph重放，reset在NPU Event计时区间外。

| 轮次/路径 | min（μs） | mean（μs） | max（μs） |
| --- | ---: | ---: | ---: |
| 首轮普通PTO | 970.10 | 987.24 | 1010.04 |
| 首轮原early-start | 969.84 | 989.91 | 1005.06 |
| 首轮去诊断判断 | 970.70 | 986.21 | 1004.36 |
| 确认普通PTO | 967.90 | 993.37 | 1024.82 |
| 确认原early-start | 969.30 | 990.11 | 1011.22 |
| 确认去诊断判断 | 969.52 | 986.52 | 1009.02 |

候选对原early-start两轮mean少3.71/3.59μs，配对中位差−2.99/−7.17μs，17/30及18/30配对更快。
max少0.70/2.20μs，min仍多0.86/0.22μs，不能说每项指标都改善。
对普通PTO mean少1.03/6.85μs，配对中位差−3.19/−7.63μs，16/30及20/30配对更快。
仅记录该档小幅改善，不据两轮样本证明所有负载稳定净收益或消除长尾。

[首轮90条样本](round1/results.json)、[确认轮90条样本](confirm/results.json)、[配对计算](paired_comparison.json)。
两轮三路9项输出/缓存状态和改变hidden输入后的图重放逐bit一致，各31项保护区通过。
没有新增Native、七档、16卡、token或DSpark结果。

## 诊断保真验证

[26项后端单测](backend.xml)通过，覆盖正常/诊断回调、事件分配、中转和资源不足回退。
保留实现另外重编译AIC/AIV小用例，在单卡上检查两次输入、图重放、Scalar依赖、输出与保护区。
两次输入的真实通知计数分别为AIV 24/47、AIC 22/24；通知计数没有因默认优化而归零。
后到交接计数分别为AIV 2/1、AIC 8/5；这里的尾部包含通知DMA，不等同于纯计算重叠时长。
小用例设置`repeats=0`，只作功能检查，不混入性能表。
见[诊断结果](diagnostics/results.json)、[计数摘要](diagnostics/summary.json)和同目录原始trace。

三次任务均卡0退出0并释放：

- 性能首轮：`task_20261012_052704_37062195876`。
- 性能确认：`task_20261012_052905_379308931782`。
- 诊断检查：`task_20261012_053359_39851744886`。

首轮输出目录一度因环境脚本改变位置参数而带`_--device`后缀，之后原样归位。
运行脚本现于source环境前保存显式的confirm选择；计时内容没有重写。

## 复现与后续

[版本和产物来源](versions.json)、[PyPTO提交](pypto.patch)、[私有vLLM提交](vllm-ascend.patch)。
在私有vLLM目录source父目录`env.sh`，设置NZ2/atomic0后运行：

```bash
python tests/pypto_test/tmr_csa_untraced_notice_compile.py \
  --output ../artifacts/csa_untraced_notice_128k_b16_20261012/compiled \
  --controls ../artifacts/csa_notification_id_128k_b16_20261012/compiled
```

通过`task-submit --device auto --device-num 1 --run 'bash <script> [confirm]'`
运行[单卡脚本](repro/run_untraced_notice_20261012.sh)。脚本放在私有环境的`artifacts/`目录。
诊断在私有PyPTO目录用`task_boundary_runtime_poll_probe.py --compile --compiled <dir>`编译，
再通过队列运行[诊断脚本](repro/run_untraced_diagnostics_20261012.sh)。驱动副本见`repro/`。

本次只处理回调诊断开销，没有解决一般只读GM前缀的写者/别名证明、非末尾分支/循环Wait，
或通知资源全部不足时的后到观察窗口。两项用户已接受的接口/诊断约束保持，不扩张为自动识别项目。
