# 过程085：复用本核交接状态未形成可靠收益，候选撤下

2026-10-12，单卡0、CANN9.2、TMR、128K/B16，第2层HC_pre+norm+CSA+HC_post。
本轮运行时优化未保留；保留PyPTO诊断夹具修复 `5c0a87310`。
Simpler仍为 `7043f8c1b`，PTOAS/PTO-ISA和公共CSA默认不变。
[版本与任务](versions.json)、[计算汇总](comparison.json)。

## 候选及判断

执行器已用 `prepared_prefix_task_id` 记录成功交接的下一代次，进入下一task时却又从GM读取
`predecessor_context` 判断是否交接。源代码中只有交接函数写入非零前驱指针，
AICPU每次发布新代次前都将其清零，因此尝试复用本核代次比较结果。

[候选补丁](rejected_runtime.patch)只调整这一判断。保留原双缓冲代次、指针写入、DCCI、
完整尾部/fan-in/cohort等待及原ACK、派发和early-resolve。没有新增配置开关或自动寻找Wait位置。
候选在独立运行时目录构建，构建后立即恢复源码；整个夹测未切换正常安装。

## 128K/B16完整CSA对照

三进程原→候选→原共用一次卡锁，复用同SDK的同一组算子二进制。
每进程两路各5次预热、30次轮换单次NPUGraph重放；NPU Event计时，reset在区间外。
NZ2、atomic0、deterministic0，不开profiler。

| 运行时/路径 | min（μs） | mean（μs） | max（μs） |
| --- | ---: | ---: | ---: |
| 前测原运行时，普通PTO | 969.94 | 989.82 | 1020.26 |
| 前测原运行时，early-start | 963.94 | 990.63 | 1018.48 |
| 候选运行时，普通PTO | 965.34 | 998.28 | 1029.92 |
| 候选运行时，early-start | 968.00 | 985.08 | 1003.92 |
| 后测原运行时，普通PTO | 978.72 | 994.20 | 1030.98 |
| 后测原运行时，early-start | 956.92 | 984.41 | 999.84 |

候选early-start对前/后原实现mean分别−5.55/+0.66μs，max分别−14.56/+4.08μs，
min分别+4.06/+11.08μs；普通PTO mean反而分别+8.47/+4.08μs。
候选early-start比前后均值中点少2.44μs，但原实现自身mean漂移6.21μs。
同进程early-start−普通的配对均值前/候选/后为+0.81/−13.21/−9.79μs，
候选配对改善也包含普通路径变慢，不能仅挑该差值认定净收益。
没有可靠证据表明这个改动降低完整CSA耗时，因此撤下，不再补采筛选有利结果。
没有指令级根因证据，不把现象定性为某条Scalar指令或Scalar-bound。

180条样本完整保留：[前测](timing/control_before/results.json)、[候选](timing/candidate/results.json)、
[后测](timing/control_after/results.json)。三次各9项输出/cache状态及换hidden输入图重放逐bit一致，
每条路径各31项保护区通过。没有新增Native、七档、token或16卡测试。

## 功能检查及测试修复

候选运行时AIC/AIV/MIX三项设备检查通过，每项两次输入：纯AIC各24/24、AIV各48/48个
同核完整尾部交接通过；MIX每次24 AIC、48 AIV在前驱组完成前进入前缀。
输出、Scalar结果、GM读取值、保护区与等待顺序正确。
完整尾部含通知，MIX是组完成前入口证据，不单独证明每个成员的原始计算仍在重叠。
[最终功能日志](diagnostics/run.log)、[AIC](diagnostics/local_aic_branch.json)、
[AIV](diagnostics/local_aiv_branch.json)、[MIX](diagnostics/mixed_chain.json)。

首轮任务 `task_20261012_073448_9994925829` 退出1并释放卡，性能阶段未执行。
AIC在数值/保护区检查后遇到trace值263（完成7＋通知标记256），旧断言错误要求整个字段等于7。
运行时本来就把通知路径写在高位；MIX原有检查也只检查完成低字节。
[首次日志](diagnostics_first/run.log)保留该错误，不能把它当成kernel数值失败或通过。

[PyPTO修复](pypto_trace_fix.patch)统一低8位口径，其余所有等待顺序、数据和保护区断言不变。
[CPU定向检查](trace_flags_check.log)证明合法通知位可以通过，缺失完成位及后继提前结束等待仍被拒绝。
首次用包名导入夹具时遇到`tests`命名空间冲突，改为按文件加载后通过；这不是设备失败。
最终 `task_20261012_073842_24452811790` 三项功能和三进程计时全部完成，退出0并释放卡。

## 复现与当前状态

使用 `versions.json` 所列私有环境及现有冻结算子包；把
[驱动](repro/tmr_csa_handoff_state_demo.py)放回私有vLLM的 `tests/pypto_test/`，
把[设备脚本](repro/run_handoff_state_20261012.sh)放回私有环境 `artifacts/`。
source私有 `env.sh` 后，先运行驱动的 `--runtime-root <目录> --build` 做CPU构建；
随后通过 `task-submit --device auto` 运行脚本，只使用队列分配的 `TASK_DEVICE`。
驱动复用本地同SDK CSA产物，原版本与候选的运行时根目录分离。

`python analyze.py` 从归档的180条样本重新计算 `comparison.json`，不筛选样本。
本轮没有保留运行时代码，正常SDK和已有CSA二进制无需重建；实验驱动仅归档。
当前保留版本的本轮均值为前测990.63、后测984.41μs，不能将过程083的较低历史值当成这轮固定基线。
后续从剩余交接/通知开销和安全前缀覆盖继续推进，不重复这项无可靠收益的候选。
