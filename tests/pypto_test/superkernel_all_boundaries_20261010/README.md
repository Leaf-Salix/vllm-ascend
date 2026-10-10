# CSA 全部可接入单核 task 的局部 Set/Wait 对照

本次在私有环境扩展原 task 内局部标记，保持原计算和流水。
完整说明见[实施日志过程037](../DSV4_FLASH_CSA_SUPERKERNEL_IMPLEMENTATION_LOG.md#39-过程-037扩展到-csa-全部可接入单核-task再做同卡三组对照)。
MIX未接入是当前原型的实现限制，正常MIX组完成统计已经存在；不表示MIX从原理上不能使用局部Set/Wait。
具体缺口见[实施日志过程038](../DSV4_FLASH_CSA_SUPERKERNEL_IMPLEMENTATION_LOG.md#40-过程-038澄清-mix-未接入的原因和实际缺口)。

- [源码插入点清单](source_inventory.json)：含ND/其他静态分支，不能把源码条目数当成运行task数。
- [本次128K/B16编译清单](compiled_inventory.json)：73个kernel，59个Set/Wait、2个仅Wait、12个MIX成员。
- [五项编译合同检查](compiler_checks.json)：包含不允许将GM Scalar读取提前到Wait之前的反例。
- [算子增量补丁](csa_boundaries.patch)：基于私有vLLM提交46597692，十文件仅增加104行。
  补丁省略上下文，应用时使用git apply --unidiff-zero；不直接应用到公共运行环境。

三组为冻结的普通CSA、只接入NZ O-A、全部可接入单核task。全部使用CANN9.2、TMR、NZ2、atomic关闭、
128K/B16及第二层真实权重，无profiler。计时是单次完整HC_pre+norm+CSA+HC_post图重放，
每组三十次，循环改变执行顺序，状态恢复在计时区间外。

任务`task_20261010_192303_125266920291`已在0卡完成，exit=0。
普通版和两种候选的eager输出、更换输入的图重放，共9项结果/缓存状态逐bit一致，
三组各31项保护区/metadata检查通过。[精简结果及全部90个计时样本](results.json)。

| 路径 | min（μs） | mean（μs） | max（μs） | mean相对普通版 |
| --- | ---: | ---: | ---: | ---: |
| 普通PTO | 954.34 | 974.90 | 1004.82 | 基线 |
| 仅NZ O-A加入Set/Wait | 955.92 | 970.49 | 1011.26 | -0.45% |
| 全部可接入单核task | 975.56 | 989.89 | 1007.46 | +1.54% |

扩展版均值增加14.99 μs，26/30配对轮次比普通版慢，min增加21.22 μs；不是一个最大值样本拉高均值。
P95为996.17/981.22/1006.24 μs，扩展版比普通版高约1.01%，没有本轮显著孤立长尾的证据。
仅O-A版本轮均值较低，但此前同口径轮次也曾略慢，不据此声称稳定收益。

功能通过，扩展标记后的整体性能回退；本轮结果不支持将全量标记作为默认性能优化。
没有新采集机制trace，未将编译覆盖数等同于实际重叠次数，也未断言哪个task导致回退。
修改保留在私有实验环境；没有继续重复占卡。
源码提交：私有vLLM `2cd33a9a`，PTOAS `f556407f`；PyPTO/Simpler沿用`b32ee10ff`/`a1a25dc33`。

私有根目录：/data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009。
队列入口artifacts/run_csa_all_boundaries_v2.sh；预编译目录artifacts/csa_all_boundaries_compile_v2；
原始设备报告artifacts/csa_all_boundaries_device_v2/results.json。
通过task-submit --device auto --device-num 1调用入口；测试脚本使用队列分配的TASK_DEVICE。
