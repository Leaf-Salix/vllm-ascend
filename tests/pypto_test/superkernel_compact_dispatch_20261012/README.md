# 过程114：压缩延迟完成引用，在派发热行内保存交接描述

2026-10-12。仅固定5μs链，目标普通mean<295μs、标记mean<250μs不变；尚无本候选设备结果。
不叠加110—113的预取或读钟候选。原工作区仍保留Simpler b5a358b88 / PyPTO 4a2651638。

## 改动与语义

旧LocalContext为48B，其中AsyncCtx保存指向同一DeferredCompletionSlab的计数、错误和条目地址，
以及常量容量。派发内部改存slab基址；get_async_ctx按需展开，公开AsyncCtx仍为40B，
计数/错误/条目/容量/任务ID仍可使用，未删除延迟完成能力。

| 布局项 | 正常版本 | 候选 |
| --- | --- | --- |
| 派发内LocalContext | 48B | 32B |
| kernel参数数组起点 | 64B | 64B |
| 不可变交接描述 | 额外的GlobalContext冷行 | 原派发头48/56字节 |
| AICore可变控制行 | 独占、偏移不变 | 独占、偏移不变 |
| DispatchPayload总长 | 576B | 576B |

AICPU不再逐派发写额外描述行，AICore从已读取的派发头取得描述。GlobalContext旧位置留作保留空间，
保持冷工作区指针和本核可变控制行位置。派发选择/顺序/发布批次、early-resolve、
完整输入门铃、ACK后续流程、晚到观察及尾部流水等待均保持。

## 构建与验证边界

Simpler独立实验提交05220c71e；PyPTO独立实验提交845909ca3同步精确版本和runtime gitlink。
20项CPU检查通过，包括延迟完成引用展开后写回原slab及重绑定不串写、原有槽复用与交接合同检查。
最初未提交源码的核编译被exact-clean SDK guard拒绝，未绕过检查。
新建sdk-compact-dispatch-20261012私有venv，绑定、runtime及torch_npu适配器按相同提交重建；
PyPTO core编译器源码未变，复用原二进制。所有候选任务必须重新用候选头文件编译，
不加载正常SDK冻结任务，不修改旧manifest。

新venv首次未继承原环境third_party_readonly.pth指向的第三方包，误用了系统PyTorch，
适配器导入缺符号；已显式指向相同torch/torch_npu 2.10依赖并重配重编，版本一致检查通过。
共享安装和正常对照SDK未改。CPU构建和环境问题均未占用NPU。

后续运行脚本按正常SDK前→候选SDK→正常SDK后在同一卡测试原5μs链，
保留逐级输出、换输入和保护区检查；因内部布局变化，追加原有3项AIC/AIV/MIX局部功能检查。
50μs性能不采集。此阶段未作性能收益或功能验收完成的结论。

[布局与运行时补丁](simpler.patch)、[版本绑定](pypto.patch)、[CPU检查](cpu.log)、
[复现脚本](repro/run.sh)、[构建脚本](repro/build_sdk.sh)、[版本核对](sdk_revision.log)。

## 设备前检查已完成

私有SDK完整构建结束，绑定/runtime/torch_npu适配器均对应05220c71e。
普通和标记两路任务用匹配头文件重编成功；各路Python与生成的计算C++与正常SDK逐字相同，
只有内部头文件布局和SDK版本不同，没有改变负载或篡改旧产物。

单卡队列任务`task_20261012_194040_366826810628`已提交，当前pending，16卡被其他作业占用。
保留该句柄等待，不重提，不中断他人任务。没有候选设备成绩前，不以CPU检查或已编译代替验收。
[计算源比较](source_comparison.json)、[编译来源](build_source_check.json)、[编译日志](compile.log)。

复现需先建立两个匹配实验工作区及私有venv，应用本目录两个提交补丁并保持版本一致；
repro/env.sh展示实际依赖来源。原版前后对照始终使用原env.sh和原冻结任务。
