# 请求几何与页检查前移首版：六档改善，一档退步，尚不整体采用

2026-10-18。基线为已采用的动态行依赖版，正式权重第4层；本目录保留首版
全部尝试。新排程在 `../attention_preplan_overlap_evidence`，不覆盖这里的源。

## 动机、布局与生命周期

静态版没有动态版的逐query唯一请求查找及完整压缩页范围保护。它们并不
依赖Top-K，首版将同一规则提前到QR之后、ROPE发布之前，与AIC投影重叠。
Top-K与晚期plan复用已验证owner、begin、position、pages_ok和visible，
后置候选检查、空洞回退、向量数学、softmax及FFTS保持。

暂借 `attention_indices[T,512]` 每行前八列，保存五项INT32和三项对齐零；
不新增GM根、工作区、Host数值metadata或外部适配调用。它在前段尚不是
最终索引，Top-K在已有ROPE获取之后读取；计划等本行Top-K完成后才覆盖
完整512列。每行和pad保护保持，失活行每轮明确写owner=-1与其它零值。
早晚阶段仍遵守原发布与缓存获取，不能用这个别名省略A3的缓存维护。

首次CPU因同一GM根混用Scalar写及MTE3 store被拒绝，属于算子表达不符合
现有约束。改为先在UB写，再统一通过MTE3发布八列，完整CPU通过。没有修改
或绕过检查。`cpu_v1`失败源、traceback和`CPU_ATTEMPTS.json`完整保留。

`check_preplan_control.py`提取实际控制函数，以独立请求/页oracle核对320个
场景、6919条query，包含S1/S5/S6、混合长度、dummy、缺失/重叠Q、陈旧位置、
页空洞/短表、INT32边界和尾部保护。页检查函数AST、晚期plan向量数学调用
保持；该检查不是设备数学或同步证明。CPU仍是58动态轴、130根、一个MIX。

## 完整设备终态

十个任务均exit0。七档精确输出、Top-K/score、cache/state、A→B→A、图、
保护与代际通过；B1/S1、B5/S5对前版动态精确/图/状态通过；六次metadata、
四次padding同图回放通过。没有据此扩大为全mixed-query、空rank或连续轨迹。

完整五窗同进程三方数据见 `SEVEN_RESULTS_V1.md/json`。六档改善，但128K/B8
为646.70→653.38μs，回退1.03%，4/5配对窗口更慢。因此不整体采用，不把
B8点名排除，也不无修改重复同一个性能实验。相对原多task降低30%仍0/7。

下一候选只改变放置：在主压缩消费者之后、Indexer query Hadamard等待
之前准备几何，不阻塞更早的ROPE集合。所有AIV在Score入口本来就获取全体
IDX_SCALE，而几何先于每核IDX_SCALE发布，因此零Score工作也有完整获取链。
这一排程是否改善B8及其它档，由新目录的独立完整验证决定。

## 复现

`run_cpu.sh 全新绝对目录 --controls-json controls_long.json的绝对路径`；设备
按两个submission文件的完整命令经队列执行。`source_reference`与
`source_candidate`为实际被测源码，后续不得覆盖。大编译物/二进制本地
保留，只登记路径和大小；模型权重不入Git，不做hash。
