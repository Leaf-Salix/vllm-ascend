# 尾部通知减少重复流水等待：两轮无稳定收益，候选撤下

2026-10-12，过程075。**仅保留本归档，不启用该候选。** 私有默认继续使用 PyPTO `847e341cc`、
PTOAS `6dee5a69`、Simpler `9fddae626`；算子与运行时没有新增默认改动。
PTOAS 源码恢复后重新构建，不重复测试已经验证过的相同默认代码。

用户确认的范围继续保持：用局部 `wait_pre_task_end()` 标识安全 Scalar 前缀，当前不做自动识别；
chip swimlane、PMU、参数 dump 开启时串行回退。原 dispatch、ACK、核选择和用户 early-resolve 不变。

## 尝试的改动

完整通知固定汇合 AIC 的 MTE2/MTE1/M/FIX，或 AIV 的 MTE2/V。
一些 task 的原同步已使某条流水的完成包含另一条流水的完成，例如 O-B 最后的 FIX→M 等待。
候选只在编译期证明完整覆盖时省去重复的通知 Set/Wait，原 kernel 的事件与计算不改：

- 只分析自动生成的同步，追踪源流水最后的常量 Set 及无条件尾部 Wait；新源流水工作会使证明失效。
- 两个分支都必须成立，循环包括零次执行；不根据一次实测路径或特定 kernel 名称推断。
- 不省去通知填充本身需要的流水：AIC 保留 MTE2，AIV 保留 V；MTE3 原顺序、barrier 和 64B 写回保留。
- 显式同步及包含跨核 FIFO/通信宏的成员保留原通知，不猜测宏内部事件。
- PTOAS 逐函数发出 join mask，PyPTO 校验核型、填充流水和合同冲突，旧 wrapper 编译失败。

实际命中 **15 个 AIC、2 个 AIV 核函数**，合计静态省去18组 Set/Wait；这是函数数量，不是一次CSA的通知次数。
O-A 省去 M，O-B/Q投影等省去 FIX，一个 KV Score 成员省去 MTE1/FIX，两个 AIV 省去 MTE2。
[完整名单](compiler/join_summary.json)、[原/候选通知回调及原task尾部](compiler/generated_callbacks.json)。

普通对照采用过程073的同算术包，只移除局部 Wait/Set。
原 early-start 与候选共用同一包和同一新编译器，原版只在实验编译驱动中保留完整通知。
去掉IR注释、对编译生成的 inline ID 作一一重命名后，两版计算及wrapper正文相同，
差异仅为回调中的 Set/Wait。未插入新的运行时开关，也未改变早派发策略。

## 单卡128K/B16

CANN9.2、TMR、卡0、第二层正式权重和Native缓存夹具；计时范围为 HC_pre+norm+CSA+HC_post。
NZ=2、atomic_add=0、deterministic=0，关闭 profiler/PMU/参数dump。
每轮三路分别预热5次，之后每路30次单次图重放，轮换顺序；NPU Event计时，reset不计时。
两轮共180条样本全部保留，没有删掉长尾点。

| 轮次 | 路径 | min（μs） | mean（μs） | max（μs） |
| --- | --- | ---: | ---: | ---: |
| 1 | 同算术普通 PTO | 960.20 | 992.76 | 1016.00 |
| 1 | 原 early-start | 964.30 | 993.39 | 1190.96 |
| 1 | 精简通知候选 | 976.30 | 993.73 | 1011.24 |
| 2 | 同算术普通 PTO | 974.52 | 995.45 | 1013.96 |
| 2 | 原 early-start | 968.98 | 993.72 | 1008.18 |
| 2 | 精简通知候选 | 972.32 | 993.63 | 1015.98 |

候选减原 early-start：

| 轮次 | mean变化（μs） | 配对中位差（μs） | 候选更快 | max变化（μs） |
| --- | ---: | ---: | ---: | ---: |
| 首轮 | +0.35 | +8.01 | 10/30 | −179.72 |
| 确认轮 | −0.09 | +0.65 | 14/30 | +7.80 |

首轮原版的1190.96μs出现在repeat=4；其次两个最大值为1006.74/1005.78μs。
这个异常点仍计入均值和最大值，没有证据把它归因于原完整通知。
确认轮长尾改善没有复现，均值也没有实质收益，因此撤下候选，不把指令减少当作整体加速。
没有新增时序trace，不能据此认定任何task是Scalar-bound，也不能给出此次小差异的精确因果解释。

原 early-start 相对同算术普通 PTO，两轮 mean 为 +0.62/−1.73μs；这仍不是稳定净收益的证明。
本次没有 Native、七档、16卡、输出token或DSpark的新结果。

两轮三路的9项输出/缓存状态以及改变输入后的图重放逐bit一致，各31项保护区全部通过。
数值对照是三条PTO路径之间，Native仅提供正式权重/缓存夹具，不能写成与Native逐bit一致。

## 验证与过程记录

- 后端28项UT、7项FileCheck通过。编译检查包含Cube零次/多次循环、后续条件写回使证明失效、AIV路径，
  以及原显式事件/中转合同、Wait前GM和显式流水拒绝；两版各73个CSA成员通过Wait前流水审查。
- 初版新编译用例的tile地址64不满足L0的512B对齐，改为512后通过，没有把错误用例提交设备。
- 首次CSA代码对照被两次独立编译产生的 inline ID 差异挡住；修正为一一重命名后重新审查通过。
  三路设备代码已成功编译，失败发生在CPU上的文本审查，非设备功能失败；失败摘要保留。
- 最终分析按一次循环体求保守交集，避免嵌套循环重复遍历；PTOAS重新构建，候选重编译并再次审查后才占卡。
- `task_20261012_034748_418837416534` 首轮、`task_20261012_035015_999202869` 确认轮均退出0，释放卡0。

[首轮完整结果](round1/results.json)、[确认轮完整结果](round2/results.json)，
配对差分别见[首轮](round1/paired_comparison.json)、[确认轮](round2/paired_comparison.json)。
[版本及决策](versions.json)、[编译与保护边界检查](compiler/filecheck/results.json)。
编译日志仅归档完成行和失败摘要；原完整日志仍在私有环境 artifacts/tail_join_*_20261012.log。

## 复现条件

此目录保存已撤下的候选，不能直接当作当前默认代码使用。两份candidate.patch以以上PyPTO/PTOAS提交为基线，
不含私有runtime符号链接改动；使用 `git apply --unidiff-zero`，`source/`另保留编译用例和后端测试。
在隔离的既有环境中应用补丁、重建PTOAS，将`repro/`中的两个Python驱动放入私有vLLM的`tests/pypto_test/`，
两个shell脚本放入私有环境的`artifacts/`。该环境仍需既有冻结CSA包、正式权重、CANN9.2及匹配Simpler SDK；
不是新克隆仓库的独立复现包。编译命令如下，设备执行再通过task-submit：

```bash
python tests/pypto_test/tmr_csa_tail_join_compile.py \
  --output ../artifacts/csa_tail_join_128k_b16_20261012/compiled
```

后续继续研究已派发任务的安全前缀覆盖和交接成本；一般只读GM前缀及所有现有通知事件都不足的场景仍有实现缺口。
手工Wait及诊断串行回退属于用户已确认的范围，不列为待消除限制。
