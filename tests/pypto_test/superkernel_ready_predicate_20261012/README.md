# 已判真的就绪task允许同核Scalar交接

2026-10-12，过程076。此项保留在私有TMR原型，未修改公共CSA默认路径。
作者继续用局部`wait_pre_task_end()`标识安全边界；不做边界自动识别。
chip swimlane、PMU及参数dump仍串行回退。

## 修改与语义

Simpler的`prepare_subtask_to_core()`原先在任务已有安全前缀能力时，仍用`!has_predicate()`禁止交接。
这会额外挡住条件已判真、正常排到忙核后面的task。现在删除这层排除。

- predicate由原就绪路由求值；为假时进入dummy完成队列，不发布到AICore。
- 尚未就绪的predicate任务仍不early_dispatch，原资格、派发/ACK触发、队列和核选择均不改。
- 只允许已有安全前缀能力的成员交接，局部Wait仍收尾旧计算，并等待完整fanin及原同步条件。
- 用户的`allow_early_resolve`保持；此项没有自动打开它。

此前文档只记录“原调度器不提前派发predicate”，漏掉了准备阶段的这层额外限制，已修正。
未把安全边界手工标识或已接受的诊断回退当成待消除限制。

私有Simpler `52d0ac5dc`，PyPTO `ef76af347`同步gitlink、ABI和设备回归。
PTOAS仍`6dee5a69`，PTO-ISA不改。[完整版本](versions.json)、[Simpler修改](simpler.patch)、[PyPTO修改](pypto.patch)。
私有绑定、TMR运行库与Torch适配器重建并确认版本一致，没有复制其他工作区二进制。

## 验证

16项生产调度器CPU测试通过；新增用例覆盖真条件的AIC/AIV/MIX逐成员能力、假条件及sync_start的dummy路由，
并确认原未就绪predicate排除保持。[CPU结果](predicate_ready_cpu_20261012.xml)。

新增设备用例第一次在CPU编译时共用Acc/Vec变量名，被类型检查拒绝；分开变量、各自store后通过。
这次失败未占卡。[初次错误](predicate_ready_codegen_20261012.log)、[编译通过](predicate_ready_codegen_retry_20261012.xml)。

CANN9.2、TMR、单卡0，任务`task_20261012_041357_98065922947`，退出0并释放卡。
同一个已注册入口依次执行真→假→真，输入和Scalar seed变化，两个设备测试通过：

| 核型 | 第一次真条件本核重叠 | 再次真条件本核重叠 | 组完成前已结束局部Wait（两次） |
| --- | ---: | ---: | ---: |
| AIC | 24/24 | 24/24 | 22/24、21/24 |
| AIV | 48/48 | 48/48 | 47/48、45/48 |

逐元素输出、Scalar结果、保护区及本核完整尾部等待均通过。假条件下全部后继输出和trace保持未写。
重叠判据匹配同一物理核，要求后继Scalar已结束、前驱计算尚未完成，并确认后继Wait后时间不早于旧尾部完成。
仅“后继早于整组完成”不被单独当作本核重叠证据。

[摘要](functional_summary.json)、[AIC原始时钟](predicate_aic_results.json)、
[AIV原始时钟](predicate_aiv_results.json)、[设备测试结果](device_results.xml)、[设备日志](device.log)。

设备测试使用尾部trace诊断回调验证机制，没有计时收益结论；MIX本轮只补CPU能力验证，未新测MIX设备交接。
当前CSA生成的orchestration中无predicate，本轮不重复CSA、七档、Native或16卡测试。
最新CSA性能仍引用过程075，不能把此项功能通过替换成新的性能数据。

## 复现

在私有开发环境按[版本](versions.json)检出代码、构建SDK；资源限制、虚拟环境及CANN配置沿用私有env。
[构建命令](build_predicate_ready_sdk_20261012.sh)与[设备命令](run_predicate_ready_20261012.sh)中的父目录为私有开发环境根目录。
设备命令必须经`task-submit --device auto --run 'bash <script>'`，使用队列分配的`TASK_DEVICE`。
测试入口为PyPTO的`tests/st/runtime/scheduling/test_local_task_boundary_predicate.py`。

一般只读GM Scalar前缀的写者/实际别名证明，以及通知资源全不足时的观察窗口仍未完整解决；
本次没有放宽对未就绪数据读取或流水资源复用的检查。
