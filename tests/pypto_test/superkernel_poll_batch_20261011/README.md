# 过程064：每次完成标志读取前检查四次寄存器，不采用

2026-10-11。私有PyPTO实验驱动`51d6f450d`、vLLM驱动`c42aaa19`；
生产生成器仍为过程063 `21a3fbcfe`，Simpler `5085a6ce0`、PTOAS `0c21f984`未改。
[完整版本](versions.json)。功能通过，性能复核回退，候选不接入生产。

## 候选及控制变量

原通知回调每轮先读取一次DATA_MAIN_BASE，再失效并读取HBM完成标志。
候选在两次HBM读取之间连续检查四次dispatch寄存器，希望减少缓存行读取并更快接住后到后继。
每次检查沿用原退出条件；通知DMA、事件汇合、真正的完成等待、runtime交接资格检查均保持。
完成标志可能晚几个Scalar检查才被观察到，因此需要实测，不能预设无成本。

只在私有包装器中替换循环；不改AICPU队列、派发、ACK、核/block分配或early-resolve。
局部Wait标识及chip swimlane/PMU/参数dump回退保持。已有普通/完整通知CSA产物复用过程063，
仅重编候选；没有重建SDK或重新生成参考路径。

## 单卡功能

任务`task_20261011_232542_251649521946`，卡0，先执行小例子再执行CSA。
24个AIC或48个AIV block，loops=4、Scalar额外循环0、两组输入及图重放。
输出、完整fanin读取、保护区和无后继退出通过，小例子不计时。

| 核/路径 | 输入0同核/后到 | 输入1同核/后到 | 重放同核/后到 |
| --- | ---: | ---: | ---: |
| AIV单次探测 | 0/0 | 0/0 | 8/0 |
| AIV四次寄存器检查 | 13/5 | 14/0 | 13/0 |
| AIC单次探测 | 0/0 | 0/0 | 0/0 |
| AIC四次寄存器检查 | 8/8 | 15/15 | 9/9 |

计数不能与其他阶段的不同负载直接比较。尾等待包含通知DMA，不能量化原计算重叠。
原始结果和trace见[small/](small/)，实际生成的循环见[generated/](generated/)。

## 128K/B16性能

两轮均为CANN9.2、TMR、NZ2、atomic0、诊断关闭，第2层真实权重HC_pre+norm+CSA+HC_post，
预热5次后每路30次单次NPUGraph重放、轮换次序，reset在外部NPU事件计时区间之外。

首轮同上述任务，卡0，单位μs：

| 路径 | min | mean | max |
| --- | ---: | ---: | ---: |
| 普通PTO | 976.22 | 989.50 | 1004.16 |
| 原完整通知 | 958.94 | 994.60 | 1096.74 |
| 四次寄存器检查 | 965.92 | 991.30 | 1009.24 |

候选较原通知mean少3.30μs，17/30配对更快，中位差−1.38μs；原版1096.74μs长尾影响明显。
没有据此判定稳定收益，仅补一次原通知与候选的两路复核，复用全部编译产物。

复核任务`task_20261011_232846_264541729662`，卡0：

| 路径 | min | mean | max |
| --- | ---: | ---: | ---: |
| 原完整通知 | 970.44 | 990.06 | 1009.24 |
| 四次寄存器检查 | 987.18 | 1000.87 | 1022.30 |

候选mean多10.81μs（1.09%）、max多13.06μs，仅4/30更快，中位差+8.65μs。
min/mean/max均回退，首轮最大值降低未重复，不将单次长尾差异当作稳定改善。
两轮9项输出/缓存/状态及换输入重放逐bit一致，每路31项保护区通过。
两项设备任务均退出0并释放卡，无失败设备任务；静态检查及格式检查通过。

[首轮结果](first/results.json)、[90条样本](first/samples.csv)、[配对差值](first/paired_comparison.json)；
[复核结果](confirm/results.json)、[60条样本](confirm/samples.csv)、[配对差值](confirm/paired_comparison.json)。

## 结论及复现

不采用四次轮询，不继续搜索次数。保留驱动和证据，正常路径仍使用原单次寄存器检查循环。
本轮不改变公共默认，也没有新Native、七档或16卡测试。
这不是通知开销的指令级归因；后续需要把128K/B16真实任务的后到命中与完整通知次数分开观察，
避免继续只凭小图命中数调循环。残余算子白名单和其他规划器scratch覆盖仍待处理。

按[versions.json](versions.json)恢复私有环境，先在私有PyPTO仓source `../env.sh`：

```bash
export PYTHONPATH="$SCALAR_OVERLAP_ROOT/pypto/python:$SCALAR_OVERLAP_ROOT/simpler/python:$SCALAR_OVERLAP_ROOT/simpler"
python tests/st/runtime/scheduling/task_boundary_poll_batch_probe.py \
  --compile --work 4 --batch 4 \
  --compiled "$SCALAR_OVERLAP_ROOT/artifacts/poll_batch4_probe_compiled_20261011"
```

再切到私有vLLM仓重新source `../env.sh`，用`tmr_csa_poll_batch_compile.py --batch 4`
生成候选；`--reuse`指向过程063 compiled目录，`--output`指向本轮compiled目录。
将[repro/](repro/)脚本复制回私有根目录的`artifacts/`再通过task-submit执行，脚本依赖该相对位置。
patch以`git apply --unidiff-zero`应用。测试驱动与生成示例保留，未复制权重或大型编译二进制。
