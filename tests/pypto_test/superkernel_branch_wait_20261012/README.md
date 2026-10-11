# 分支末尾局部 Wait：编译合并与 MIX 初始化修复

2026-10-12，过程078。保留在私有TMR原型中，公共CSA默认路径未切换。
本轮是手工边界的功能扩展，不是新的CSA性能结果。

## 接通的写法

作者在两个互斥分支中分别完成Scalar准备，并在各自末尾显式调用同一个Wait：

```python
if condition:
    offset = scalar_prepare_a()
    pl.system.wait_pre_task_end()
else:
    offset = scalar_prepare_b()
    pl.system.wait_pre_task_end()
# 此后的计算才可以使用前驱数据和本核流水资源。
```

PTOAS将两个相同的无返回值Wait合并到汇合点，支持嵌套分支。
只有参数和属性完全相同、调用后仅剩SSA yield的情况才合并；每条执行路径仍等待一次。
Scalar计算留在原分支，活跃的分支结果仍由`pypto_keep_prefix`保留在等待之前。
合并后继续使用已验证的顶层Wait同步起点，没有自动发现边界或放宽前缀安全校验。

PyPTO相应识别“每个分支出口都已等待”，将隐式输入加载和MIX通信管道初始化放在汇合之后。
实际第一次MIX编译失败正是因为旧插入工具只认顶层Wait，把FIFO初始化放到了分支前：
[错误报告](initial_failure/mixed_errors.txt)、[失败时的PTO输入](initial_failure/mixed_consumer.pto)。
此失败在编译阶段被挡住，没有把不安全版本提交到设备。

## 验证结果

卡0，CANN9.2/TMR；[设备任务](versions.json)为`task_20261012_050811_30087022386`，退出0并释放卡。
三个用例各执行两次并改变输入，分支由block或Scalar结果选择，两种分支均有覆盖：

| 检查 | 第一次 | 第二次 | 判据 |
| --- | ---: | ---: | --- |
| 独立AIC后继的同核重叠 | 24/24 | 24/24 | 相同物理核上，Scalar结束早于前驱计算尾部完成 |
| 独立AIV后继的同核重叠 | 48/48 | 48/48 | 同上 |
| MIX提前进入前缀的AIC成员 | 24/24 | 24/24 | 前缀入口早于整个前驱组完成 |
| MIX提前进入前缀的AIV成员 | 48/48 | 48/48 | 同上 |

独立AIC/AIV还检查Wait不早于本核前驱完整尾部完成；MIX保留真实FIFO和`sync_start`。
输出、分支Scalar结果和保护区全部通过。MIX组级入口计数不冒充逐个成员的同核流水重叠证明。
[AIC原始trace](device/aic_results.json)、[AIV原始trace](device/aiv_results.json)、
[MIX原始trace](device/mixed_results.json)、[汇总](device_summary.json)、[设备测试报告](device/results.xml)。

9项[编译检查](compiler/results.json)通过：嵌套合并、单臂缺失、参数不一致、不安全前缀，
以及既有普通Wait、Scalar控制流同步、重复Wait、GM读取和显式流水操作拒绝。
2项[结构对照测试](cpu/preload.xml)通过，检查顶层及分支Wait之后才加载隐式输入。
新增对照测试首次遗漏DSL空return，造成6/5语句不一致；补齐预期终结语句后通过，未放宽断言。
最终[8个生成成员检查](prefix_audit.json)均只有一次有效Wait，其前没有流水指令。
检查按单个函数体取范围，避免将MIX文件内的另一个成员算为重复Wait。

## 性能与范围

现有CSA的[73个成员](csa_existing_boundaries.json)全部使用顶层Wait，没有命中本轮的分支合并规则。
本轮没有CSA、七档或16卡计时；最新性能仍见[过程077](../superkernel_dispatch_notification_20261012/README.md)。
小用例有诊断trace，只用于功能与时序证明，不当作无插桩性能。

仍未覆盖：分支内Wait后还有计算的一般形式、循环内Wait、一般只读GM前缀的写者/别名证明，
以及通知事件完全不足时的后到任务观察窗口。手工标识和诊断串行回退是已接受的范围，不列为缺陷。
原派发、ACK、核选择、`allow_early_resolve`及诊断回退保持；没有新增运行时分支。

## 修改与复现

保留版本：PyPTO `0bb9ae1ea`、PTOAS `4df4627e`；Simpler仍为`1d09ecd34`，ABI及PTO-ISA不变。
[完整版本](versions.json)、[PyPTO补丁](pypto.patch)、[PTOAS补丁](ptoas-src.patch)。
PyPTO和PTOAS都在私有环境增量构建，测试时源码与提交内容一致，没有复制其他checkout的库。

PTOAS新增和既有检查的命令在各lit文件的`RUN`行中；结果与编译输出保留在`compiler/`。
PyPTO结构检查使用`pytest tests/ut/ir/transforms/test_convert_tensor_to_tile_ops.py -k local_wait`。
将[设备脚本](repro/run_branch_wait_20261012.sh)放到含`env.sh`的私有环境`artifacts/`下，
通过`task-submit --device auto --device-num 1 --run 'bash <script>'`执行，脚本使用队列分配的`TASK_DEVICE`。
