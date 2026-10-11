# 过程063：编译期分配通知事件，保留原任务同步

日期：2026-10-11。私有PyPTO `21a3fbcfe`、PTOAS `0c21f984`、vLLM驱动`b78fac7e`；
Simpler仍为`5085a6ce0`。完整版本见[versions.json](versions.json)。
本轮扩大显式同步任务的功能覆盖，没有证明新的CSA性能收益，公共默认未切换。
手工局部`wait_pre_task_end()`及chip swimlane/PMU/参数dump的串行回退按用户确认保留。

## 修改及限制

过程062遇到通知pipe pair上的event7占用便拒绝。现在保留作者的事件号与Set/Wait顺序，
按每条通知边统计显式及已建模的宏内部事件，在编译期选择未占用ID，优先选7。
PTOAS生成事件合同，PyPTO同时修改通知Set/Wait两端；旧包装器遇到新合同会编译失败，
不会静默继续使用event7。没有新增运行时重映射、Scalar判断、派发/ACK路径或early-resolve修改。
自动同步路径仍使用原事件预留规则。

动态ID范围通过表达式保守证明，支持常量、掩码、无符号取余、选择、索引转换、有限集合运算、
条件分支；两条分支都要纳入。未知范围及集合占满0–7仍拒绝。
这里的占满是函数全局保守集合，尚未实现按事件生命周期复用，不能解释为理论上无法交接。
安全Scalar前缀的标识与验证没有改成自动识别，也没有放行Wait之前的流水操作。

## 定向验证

PTOAS构建成功，八项编译检查通过：静态低层与高层event7分别分配6；已有安全显式同步、
自动尾部及动态范围例子通过；未知动态ID、全部八个ID、Wait之前的pipe操作保持拒绝。
动态上半区4–7分配3，条件分支0/7分配6。PyPTO包装器六项单测通过。
编译日志及生成代码见[compile/](compile/)，测试源码见[source/](source/)。

单卡任务`task_20261011_230255_164235024429`在卡3退出0，设备已释放。
作者任务按block奇偶动态使用event0/event7，原数据依赖同步保持；新增通知使用event6。
AIC/AIV各两组输入，输出、完整fanin读取、保护区、无后继退出及换输入图重放通过。
本轮小例子repeats=0，不作为性能测试。

| 核/路径 | 输入0同核/后到 | 输入1同核/后到 | 重放同核/后到 |
| --- | ---: | ---: | ---: |
| AIV单次探测 | 0/0 | 0/0 | 0/0 |
| AIV完整通知 | 4/1 | 0/0 | 7/0 |
| AIC单次探测 | 0/0 | 0/0 | 0/0 |
| AIC完整通知 | 8/8 | 12/12 | 18/17 |

后到计数表明首次探测后的交接仍有效；尾等待包含新增通知DMA，不能据此量化原计算重叠长度。
静态event7路径有编译证据，本轮设备实际使用动态0/7路径，不另称静态设备测试已做。
原始结果与trace见[small/](small/)，两个核的实际包装器见[generated/](generated/)。

## 128K/B16三路同卡CSA

任务`task_20261011_231354_205726727494`，卡0，退出0并释放卡。
当前编译器重新生成普通PTO及MIX标记路径；上一版通知复用过程059同Simpler SDK产物，
没有复用过程061额外S→MTE3事件预留版本。
CANN9.2/TMR/NZ2/atomic0、诊断关闭，第2层真实权重HC_pre+norm+CSA+HC_post，
预热5次后各30次单次NPUGraph重放、轮换顺序，reset在外部NPU事件计时区间外。

| 路径 | min（μs） | mean（μs） | max（μs） |
| --- | ---: | ---: | ---: |
| 普通PTO，无局部交接标记 | 972.06 | 989.69 | 1005.20 |
| 上一版完整通知产物 | 958.16 | 988.37 | 1009.24 |
| 当前编译器，MIX完整通知 | 970.76 | 991.72 | 1019.80 |

新旧mean相差+3.35μs（+0.34%）、max相差+10.56μs；17/30配对更快，差值中位数−3.56μs。
较普通PTO mean多2.03μs，14/30配对更快，差值中位数+1.03μs。
不把本轮称为性能提升，也不从一次均值推断事件合同引入运行时成本。
本CSA没有使用自适应事件合同；73份kernel生成C++归一化编译批次的`_inline数字`标识后相同，
这一项只是源码检查，并非二进制等价或零性能影响的证明。

9项输出/缓存/状态逐bit一致，换输入图重放也一致，每路31项保护区通过。
本表是PTO内部对照，没有新测Native、七档或16卡。
[原始结果](csa/results.json)、[90条样本](csa/samples.csv)、
[配对差值](csa/paired_comparison.json)、[源码对照范围](csa/generated_code_comparison.json)。

## 复现与后续

在私有PyPTO仓内source `../env.sh`，设置私有PyPTO/Simpler的PYTHONPATH后执行：

```bash
python tests/st/runtime/scheduling/task_boundary_allocated_event_probe.py \
  --compile --work 4 --event-id dynamic \
  --compiled "$SCALAR_OVERLAP_ROOT/artifacts/allocated_event_probe_compiled_20261011"
```

CSA编译先切到私有vLLM仓并重新source `../env.sh`，执行本轮`tmr_csa_event_contract_compile.py`，
`--previous`指向过程059的mixed产物，`--output`指向本轮compiled目录。
将两份[repro/](repro/)脚本复制回私有根目录的`artifacts/`，再经task-submit单卡执行；
脚本依赖该相对位置。增量patch使用`git apply --unidiff-zero`；
PTOAS需按版本重建，Simpler SDK不变。

本轮先修正Ruff行长问题，设备任务没有失败。保留事件分配能力；后续回到128K/B16中
实际交接机会与完整通知成本，继续核查残余尾部算子白名单和其他规划器的scratch覆盖。
稳定CSA净收益仍未成立，不因小例子通过而宣称已覆盖全部已early_dispatch任务。
