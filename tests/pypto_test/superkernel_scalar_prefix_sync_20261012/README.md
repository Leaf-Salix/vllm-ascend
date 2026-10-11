# 过程070：修复Scalar前缀之前的自动Set；扩大前缀未形成收益

日期：2026-10-12。保留PTOAS正确性修复，不采用本轮三个CSA前缀位置调整。
仍使用手工`wait_pre_task_end()`，没有自动识别前缀；chip swimlane、PMU、参数dump的串行回退不变。
不改变队列、派发、ACK触发、核选择或用户`allow_early_resolve`。

## 编译器问题与修复

检查实际生成代码时发现：旧版`qproj_dequant_rms_nope_rope`的安全Scalar分支之前，
已有12条`set_flag`，均位于局部Wait之前。它们由PTOAS自动同步插入，并非算子手写。
这些Set操作前驱仍可能使用的流水事件，不能算作安全Scalar前缀。旧版数值检查通过，
不能据此宣称所有CSA前缀都没有流水操作；本轮撤回这一泛化结论，旧early-start产物不再作为后续对照。
这也不证明过去的性能波动全部由此问题造成，本轮没有再次上卡执行旧问题产物。

根因是事件分配把`syncIR_[0]`作为全函数初始Set锚点；前缀里的Scalar `if/loop`也进入SyncIR，
所以锚点可能落在Wait前。PTOAS `3b352f70`在经过校验的、顶层无条件Wait处重启流水分析节点序列，
保留此前收集的分配/视图别名及全部原始Scalar IR。算子计算不移动，无局部边界的函数不受影响。
MIX的成员Wait、FFTS/FIFO初始化和跨核汇合仍保持原合同。

- [最小回归](source/ptoas-src/test/lit/pto/local_task_wait_scalar_control_sync.pto)包含Scalar分支、循环和前缀描述符。
  [旧编译器检查失败](compiler/before_check.log)，[修复后生成代码](compiler/after.cpp)通过同一检查。
- [7项定向编译检查](compiler/checks.log)覆盖局部Wait、MIX、显式同步分配和不安全前缀拒绝。
- [完整生成代码检查](compiler/csa_generated_audit.json)：旧版73成员中只有上述成员有12条越界Set；
  修复后原边界与新前缀两条路径各73成员均无检查到的Wait前流水初始化。
- Q反量化的[旧边界](generated/qproj_before_boundary.cpp)、[修复后原边界](generated/qproj_fixed_current_boundary.cpp)、
  [修复后新前缀](generated/qproj_fixed_prefix_layout_boundary.cpp)可直接对照。

## 独立评估的前缀调整

[算子增量](compiler/operator_changes.diff)仅涉及冻结私有包中的三个文件：

| 位置 | 提前执行的计算 | 生成代码的跨Wait保留值数 |
| --- | --- | ---: |
| NZ O-B投影 | 权重K组偏移、输出N组偏移 | 1→3 |
| Q反量化 | head分组数、循环上限 | 2→4 |
| QK/PV MIX | 本核query数及循环次数 | AIC、AIV均1→2 |

这些计算只依赖Scalar任务参数/核号，没有提前GM访问或流水操作；算术、tile、Cache、依赖和调度参数保持。
保留值数只说明编译器为这些值保留了前缀位置，不等于机器指令数或新增重叠时间。
Score的规划包含GM长度读取，当前前缀验证器拒绝此类读取；本轮没有放开未经证明的内存访问。

## 单卡功能与性能

两轮均在单卡0，退出0并释放设备：

- 初测：`task_20261012_014327_369543324116`。
- 确认：`task_20261012_014535_378407213378`。

第2层真实权重、synthetic 128K历史、B16、DSpark验证6行，区间为HC_pre+norm+CSA+HC_post。
CANN9.2、TMR、NZ2、atomic0、deterministic0；没有chip swimlane/PMU/参数dump。
每路预热5次，之后三路各30次轮换单次NPUGraph重放，用NPU Event计时，状态reset在计时外。
普通PTO复用过程068同Simpler版本的产物；两条early-start路径全部用修复后的PTOAS重编译，
都使用过程069片上填充通知。确认轮复用同一批产物，无重编译或参数变化。

两轮中，两条early-start路径相对普通PTO的9项输出/状态、改变hidden输入后的重放均逐bit一致；
三条路径每路31项保护区检查均通过。测试是单层状态验收，没有新Native、七档、16卡token/DSpark结论。

| 轮次/路径 | min（μs） | mean（μs） | max（μs） |
| --- | ---: | ---: | ---: |
| 初测普通PTO | 956.00 | 984.66 | 1273.62 |
| 初测修复后原边界 | 962.40 | 985.79 | 1010.32 |
| 初测扩大前缀 | 964.46 | 986.05 | 1004.66 |
| 确认普通PTO | 965.88 | 988.18 | 1012.82 |
| 确认修复后原边界 | 961.96 | 986.97 | 1002.30 |
| 确认扩大前缀 | 967.90 | 988.12 | 1003.78 |

扩大前缀相对修复后原边界，两轮mean分别+0.26/+1.15μs，配对中位差+1.20/+4.34μs，
均为13/30对更快。初测max减少5.66μs，确认轮却增加1.48μs，没有稳定的最大值收益，故不采用。
初测普通PTO在repeat=29出现1273.62μs；保留该样本，不能把对它的max差归因于early-start。
修复后原边界相对普通PTO两轮mean为+1.13/−1.21μs，确认轮配对中位差+0.13μs、15/30更快，
尚不能声称完整CSA取得稳定净收益。

[初测原始结果](timing/results.json)、[90条样本](timing/samples.csv)、[配对统计](timing/paired_comparison.json)；
[确认原始结果](confirmation/results.json)、[90条样本](confirmation/samples.csv)、[配对统计](confirmation/paired_comparison.json)。

## 版本、复现与后续

[精确版本](versions.json)：PTOAS `3b352f70`、私有vLLM驱动`1e2c6525`；
PyPTO `9e2865861`、Simpler `6174ec5db`、PTO-ISA `327cd5869`不变。
PTOAS修复作为私有默认保留，CSA实验包仅保留复现入口，公共算子及公共环境没有切换。

目录中的两份`*-incremental.patch`相对versions.json记录的base，按`git apply --unidiff-zero`应用。
这不是从空仓开始的独立环境：依赖既有私有环境、冻结CSA包、同SDK普通对照产物和第2层权重。
将[编译脚本](repro/compile_scalar_prefix_layout_20261012.sh)、[初测脚本](repro/run_scalar_prefix_layout_20261012.sh)、
[确认脚本](repro/run_scalar_prefix_layout_confirm_20261012.sh)放回私有根目录`artifacts/`再运行；
先构建当前私有PTOAS并编译两条路径，再执行生成边界检查，设备脚本经`task-submit --device-num 1`提交。
编译前的`prefix_layout_pre_fix`只用于定位问题，没有执行设备测试。

下一步先区分“原编译器已提前完成的参数准备”和真正可增加的安全Scalar工作；
若要提前只读元数据，需要建立不依赖其他task写入的明确依据。继续以局部Wait表达边界，
不以放宽派发语义或跳过完整fanin来制造命中，不沿本轮三个微小前移做参数搜索。
