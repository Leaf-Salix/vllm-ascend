# 完整动态 CSA 的 72 核 incore 流水采集

2026-10-18开始，2026-10-19继续。用户明确要求采集所有核：当前完整CSA虽为
一次MIX kernel启动，内部按AIC/AIV及worker分工，具有MPMD特征。单核或独立
Score的轨迹无法代表24个核组的负载、跨核等待和整体长尾。

本次采用用户指定的 [pypto-lib incore-profiling](https://github.com/hw-native-sys/pypto-lib/tree/main/.claude/skills/incore-profiling)。
远端main的skill、两个脚本和规范已与本地内容核对一致；快照位于
`skill_snapshot/`。没有修改pypto-lib、共享安装或生产数学代码。

## 数据与代码来源

- 正式动态算子：`ops/pypto/deepseek_v4_flash_csa_spmd`，行为对应2db59be4。
- 用例：128K/B16/S6，正式W8A8第4层，沿用现有完整CSA fixture的历史数据。
- 捕获任务：`task_20261018_235240_21310317727`，completed、exit=0；原完整
  CSA驱动的精确对照通过，`result.json` 为complete且passed=true。
- 130个根Tensor、124份唯一物理storage，输入共960685464字节。直接D2H
  捕获底层物理内存，保留NZ格式、原storage别名与保护区；没有用逻辑`.cpu()`
  数据冒充NZ物理权重。输入和所有可写storage的真机after快照在
  `build_output/capture_v2/replay/`，只在本地保留。
- `prepare_standalone.py` 从实际wrapper恢复132个指针、56个动态shape标量
  和core参数。额外两个指针是idx_cache的0/64字节视图，不另分配存储。
  详细绑定及逐函数数学token核对见 `WORKLOAD_BINDINGS.json`。
- 独立入口只补FFTS硬件地址、block/lane编号和MIX起始会合。移除原PTOAS
  中四个GM假FFTS设置的方式与实际运行时wrapper一致；AIC/AIV数学体分别
  与实际已验证wrapper核对相同。保留45处实际使用的DCCI、volatile读及
  100tick退避；不把跨核等待删掉后画出理想化流水。

## 启动与验收

`run_profile.sh` 调用原版skill，再只修改生成的独立harness：按真实storage
分配一次并重建别名，单次启动24 blocks，覆盖24 AIC与48 AIV。CANN使用
现有9.2.0-beta.2及其配套worker/injection；目标a2a3、Ascend910_9392、
dav-c220，camodel为dav_2201。无需安装或跨版本拼接worker。

模拟器经CPU队列运行，不申请NPU。采集后必须核验72核轨迹、CUBE/VECTOR
真实计算、完整回放输出及所有已保存可写storage。导出成功不等于验收通过。
`build_output/`下原始trace、二进制与物理权重只本地保留，不入Git。

这是CANN模拟器的逐核指令流水，不能当成真机L1耗时或直接据此宣称性能收益。
保留整次launch的共同时间轴；不同核与流水的周期不能相加当作完整耗时。
后续依据轨迹修改后，仍按原目标同轮真机完整L1及精度门槛验收。

## 已记录的辅助脚本失败

首次捕获任务 `task_20261018_235132_209014720308` 在导入driver时失败，
未执行模型或算子。原因是本地runpy wrapper漏加driver所在目录；修正后
第二次捕获通过。日志和两个提交记录均保留，不将其上报为组件缺陷。
profiler runner运行前发现本地父目录索引少一级，已修正，无算子影响。

首轮独立构建`full_csa_all72_20261019_000527`在Host解析时报错：本地实际
wait-backoff helper签名使用设备专有的`pto::comm::WaitCmp`。真机wrapper
只做设备编译，独立`-xcce`额外解析Host版本。本次仅在生成harness中给
该helper补`__CCE_AICORE__`保护，保留设备端正文与所有调用，重新构建。
这是本地回放包装问题，尚无证据指向上游组件bug。

## 当前状态

第二轮独立构建`full_csa_all72_20261019_000900`通过，CPU模拟任务
`task_20261019_001135_28286531625`已启动。日志确认AIC0..23分别带两个AIV，
正在完整执行，尚无终态和完整轨迹。

为先排除独立包装/ABI错误，复用**同一份kernel.so**另建真实runtime的Host
入口，只将设备号取自队列TASK_DEVICE；模拟与真机输出使用不同目录。
任务`task_20261019_002239_326921630131`在设备0完成、exit=0：77份可写
storage、756895848字节与原完整CSA调用after快照全部逐字节相同，包含
输出、Top-K/score、cache/state、工作区和完整保护区。见
`DEVICE_REPLAY_VALIDATION.json`和`DEVICE_REPLAY_TASK.json`，未采性能。
这验证了独立二进制及真实物理输入绑定；不能代替模拟器完成和轨迹覆盖检查。

`CAPTURED_SCORE_ROLES.json`解读实际device计划：groups16、leaves6，24个
AIC各执行22个Score步骤。因此不能先把此档的差距归因为Score步骤数量不均；
仍需看供数、消费、每步实际代价及前后不同角色衔接。

准备的后处理顺序为`validate_replay.py`、`analyze_incore.py`、
`clean_incore.py`。原始Scalar/同步事件完整保留，清理图只重建原版工具
支持的SET/WAIT_FLAG连线，不伪造普通GM notify/wait的跨核箭头。采集结果、
72核完整性和逐角色瓶颈结论仍须等待实际导出，不预记PASS。
