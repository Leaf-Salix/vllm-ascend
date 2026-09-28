# 均衡leaf之后：减少2560/3072排序padding

独立源码`.cache/csa-score-balanced-sort-8e176285`，基于8e176285叠加已测均衡leaf。
[本候选补丁](candidate.patch)只表示均衡leaf→局部排序变化，基础分片补丁见
[前一实验](../csa_score_balanced_leaves_20260928/candidate.patch)。生产尚未采用两项组合。

## 来源、收益假设和保留差异

最新ops-transformer 28f40354的QLI V2 A3 `AlignS2/SortAll`按实际长度对齐排序，
最后一级使用二/三/四路有效输入，不一律补成较大的2次幂。
前一实验长B16已证实均衡leaf使AIC最慢核321.480→281.750μs，
但AIV均值287.238→306.229μs；2560/3072个有效候选仍用4096排序，新增排序/根合并抵消收益。

本项只新增2560、3072候选两种局部排序宽度，保留分片、QK/WS算术、Key预取、任务数与调度标志。
先sort32与两级四路归并得到512元素有序段，再完整归并前2048个元素；
最后将一个/两个512尾段与前段Top-512做二/三路归并，只发布Top-512。
后2048候选段仍优先于前段，后段内部的512组顺序保持，以维持原PTO相等分数规则。
数学上的截断等价不代替设备tie验证。

PTO-ISA A3单输入mrgsort要求输入有效列数整除`4×block_len`，故不能直接对5120/6144对元素
执行最后一级1024归并。初版前缀slice因src/dst物理列数不同被PTOAS拒绝；
改为保留5120/6144物理形状，只把前缀有效列数设为4096，不新增前缀UB搬运，也不修改工具链。
生成C++确认`SetValidShape(1,4096)`后仅对该前缀做最后一级，后面的512/1024尾段保留原缓冲视图。
三路最终归并比原二路多一个512输入及输出搬运，不能仅凭少padding推断性能。

与pypto-lib的区别：仍直接读Native分页cache，并使用当前3/S6 query分组和均衡leaf，
本项吸收最新AscendC的尾排序策略，未采用入口复制或更换上游整套任务组织。

## 必要验证与当前状态

- [CPU入口](compile_all.sh)：先编译两个小排序探针，再编译完整CSA；lowering/PTOAS/CCE/链接/load通过。
  首轮前缀物理形状失败见compile_prefix_shape_failed.log，修正版见[compile.log](compile.log)。
  [CPU启动记录](compile_guard.jsonl)确认开始前均衡leaf设备任务已结束。
- [独立探针](sort_case.py)：只覆盖2048/2049、2560/2561、3072/3073边界和全相等、多重复、左右占优。
  检查CPU排序值、索引范围/唯一性/回读分数、保护行，并逐bit比较原4096排序输出；不依据两个PTO实现互相比对代替独立参考。
- [单卡入口](run_layer.sh)：探针通过后才对照128K/B16与8K/B16，基线是均衡leaf源码，
  每侧20次无profiler图计时、四个DFX窗口、八类状态及A→B→A。
  本轮结果不能直接与最初8e176285相减当成同轮累积收益；必要时最终B8用明确基线补测。
- 21:12提交auto单卡`task_20260928_211241_354047120592`，最大2400秒，已分配card6运行。
  十组排序探针已通过，候选value/index与原4096排序逐bit一致；
  [基线探针](probe/baseline/report.json)、[候选探针](probe/candidate/report.json)。
  随后的完整CSA对照仍运行；不重复七档或扩展EP16，先看长档最慢AIV/包络、merge、完整CSA与短档P95。

[汇总入口](summarize.sh)只在任务结束后于主机执行；独立DFX与正式计时分开，不归一化Native控制值。
