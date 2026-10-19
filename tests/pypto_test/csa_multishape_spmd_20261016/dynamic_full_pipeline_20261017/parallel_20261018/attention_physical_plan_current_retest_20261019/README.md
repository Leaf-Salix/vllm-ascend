# 当前流水重验物理行计划，合并压缩地址验证职责

基线ef608731，未混入未采用的提前KV通知。旧physical_plan_evidence仅迁移
逐行页表推导，消费者仍反复查询Q/L；当时只有约0.81%收益。当前已有前置
几何，又存在全页证明与最终选中页的重复检查。候选迁入旧Vector地址数学，
将完整合法性核对集中在既有plan，压缩消费者读取物理行并保留最终边界保护。

C<=2048、物理页数在1..2**24时用相同device容量谓词，内部indices改为
INT32物理行；地址不经FP32，最终行小于2**29。其余容量生产消费共同
回到原逻辑索引。外部Top-K不改，bias/mask数学不改；无效项-1，dummy每轮
覆盖。前置几何继续验证唯一owner/位置/visible，fast模式只取消未被其它
消费者使用的整页证明列；真正选中页在plan逐项校验。Top-K未读取pages列
已由源码核对。compressed gather跳过重复几何的前提是同一plan和同代
metadata，原phase/cache获取保持。SWA及fallback保留原验证与地址公式。

仅两份正式动态算子文件改动，无新Tensor根/phase/Host回读，无冻结形状。
先完整CPU与实际helper的地址正反例，之后同一v3三方完整L1和冻结门槛。
所有新收益待真机，不把旧probe或旧精度当作本版已经通过。

完整CPU通过58动态轴/单MIX，无样例shape；实际helper源码的132轮CPU
地址oracle通过，包括前/后置几何、容量回退、负页/OOB、dummy/mixed、
INT32/FP32边界、A→B→A及poison。该oracle不模拟同步或硬件指令。

先提交B16作业：同一卡先跑私有地址probe的30项（前/后置各15项、含
C2049回退），通过后才进入完整CSA精确/metadata/padding/图及五窗。
probe仅用外部已就绪Top-K，信号全1，模拟同核前缀发布/获取所需fence，
不作为跨核协议证明；真实跨核协议仍由后续整层门槛验证。正式源已
逐文件核对候选后恢复，所有设备任务只读冻结快照，未预记PASS。

第一作业在任何PTO调用前因probe漏传signal_word_count必填phase_count/
core_groups退出；已保留v1脚本与错误，补与真实ABI一致的60/24后重提
v2。不作为算子或组件失败，不预记设备通过。

私有真机地址probe的30项均通过，包含前/后置几何、C1/66/1026/2048
物理表示和C2049逻辑回退、混合query/dummy/陈旧位置、负页/OOB页以及
A→B→A地址/保护检查。随后已进入完整B16流程；这不是完整CSA已通过。
大T的LOCAL_STATS关闭、scale连续读取开启组合另做一份CPU，避免忽略
其联合资源路径，再决定补B28完整代表。

大T组合CPU同样通过58动态轴/单MIX。私有地址probe已完成全部30项，
B16已到参考源码编译（之前的精确及metadata/padding门槛未报错），
据此提交B28同v3三方五窗。不重复已通过的30项probe；新组合的整层
输出/状态/保护/协议/图仍由各任务独立检查。尚无本轮性能结论。

## 同轮七档完整结果（全开候选）

全部任务终态 exit0。原多task、参考单SPMD、候选单SPMD均使用隔离v3工具链；
每项同进程、相同输入/地址/正式权重，五个交错L1窗口。输出、Top-K、score、
cache/state、保护区、图重放及代际门槛全部通过；长B16另有metadata/padding。
真实S1/S5相对参考完整回归通过，不附加未测性能。

|档位|原多task μs|参考SPMD μs|全开候选 μs|相对参考耗时降低|
|---|---:|---:|---:|---:|
|128K/B4|504.44|493.72|509.10|-3.115%|
|128K/B8|644.02|629.80|628.82|0.156%|
|128K/B16|869.20|812.30|810.12|0.268%|
|128K/B24|1149.56|1082.86|1074.68|0.755%|
|128K/B28|1269.78|1233.48|1217.00|1.336%|
|8K/B16|687.54|664.00|636.24|4.181%|
|8K/B32|980.02|942.54|936.42|0.649%|

这是各自同轮对照，不能把跨轮基线变化叠加成收益。全开版本B4回退3.115%，
B8仅改善0.98μs，不整体采用。已有T>=64前置几何区间内五个目标点中位下降，
故新增PLAN_PHYSICAL_ROWS区间开关；小T保留原路径。该阈值沿既有M32行块
策略，不是按B/history列白名单；也不证明所有输入的瓶颈或收益。

## 区间接入与编译裁剪

`source_interval`是第一次区间接线，`source_interval_v2`为显式constexpr裁剪，
`source_interval_final`补全中文总览，三者独立冻结。运行中的作业不读可变正式源。
正式policy、entry、vector及两份Attention helper共五文件接线，所有Tensor参数
位置/方向、cache policy、GM容量与phase均保持。

首次关闭分支CPU成功但仍保留两份额外64×512 BF16 Tile状态，不能据开关False
宣称关闭路径无代价。将采集分支显式写成`if PLAN_PHYSICAL_ROWS and ...`，
让constexpr在Tile状态构建前裁剪；v2关闭生成代码与基线相同。开启的最终版
与全开已测候选同样相同。这里只去掉整行`//`注释，没有规范化变量、地址或事件；
见ON_INTERVAL_CODEGEN.json、OFF_INTERVAL_V2_CODEGEN.json、ON_FINAL_CODEGEN.json。
v1差异保留在OFF_INTERVAL_*diff；这项修正属于源码提前表明常量分支，未改编译器。

完整CPU均无样例shape，保留58轴、单MIX和sync_start。边界B10/T60关闭、
B11/T66开启已提交，v1关闭结果单独保留，最终关闭版重新验证；结果见
BOUNDARY_SUBMISSIONS.json/BOUNDARY_RESULTS.json。边界结束前不预记采用。

## 复现入口与结果范围

`run_cpu.sh <绝对输出目录> --spmd-source <source_interval_final绝对目录>
--controls-json <interval_on.json或interval_off.json绝对路径>` 只编译CPU。
`run_device_interval_final.sh <绝对输出目录> --batch B --history H --l1-windows 5`
须由`task-submit --device auto --max-time 2400`提交；加`--metadata-replay
--padding-replay`覆盖对应同址图变体。历史全开复现用`run_device.sh`。
`python collect_results.py`仅读取现存结果。

132轮CPU oracle解释实际helper但不模拟硬件/同步；30项设备probe使用已就绪
的伪信号，不能充当完整跨核协议验证。真实协议证据来自完整CSA任务。所有
尝试、失败、修正、编译器调用和逐窗数据保留；大产物路径/大小另列本地清单，
不做hash校验。当前七档相对多task降低30%的目标仍0/7。

## 最终区间采用

所有边界任务终态exit0；B11含metadata/padding图。数据仍为同进程三方五窗：

|边界/版本|原多task μs|参考SPMD μs|候选 μs|候选更快窗|
|---|---:|---:|---:|---:|
|B10/T60，v1|752.14|751.30|752.32|3/5|
|B11/T66，v1|755.58|722.00|715.70|3/5|
|B10/T60，final constexpr early pruning|742.90|744.36|744.26|2/5|

B11开启改善0.873%；最终B10关闭仅相差0.10μs，不归因物理行优化。
v1关闭时额外Tile的记录仍保留，当前采用显式constexpr裁剪的最终版。
正式五文件与source_interval_final逐文件一致；3072个纯policy组合确认只
新增T>=64开关，其它constexpr与ef608731逐项相同。全部已完成设备任务
的实际编译器调用日志均指向v3。对应证据在ADOPTION_AUDIT.json。

七档最终选择为：B4/B8读取同轮原参考结果，其余五档读取同轮物理plan
候选结果。没有把这些数字描述为重新运行最终接线七档；开关编译等价和
边界验证负责接线证据。正式源码保留pl.dynamic与device metadata，仍
不是所有历史/不等长query/长期轨迹均已验证，30%目标仍0/7。
