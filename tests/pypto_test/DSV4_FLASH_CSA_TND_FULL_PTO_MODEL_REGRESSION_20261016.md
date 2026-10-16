# 完整PTO分阶段候选：8K/B40整模型回归（2026-10-16）

## 结论与范围

测试代码提交 `259cf9a5f`，分支 `dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
用户要求先复测此前整模型8K/B40，再恢复WKV投影；本轮只跑一次Native，不跑Native2。

**Native基线完成；CSA候选在FULL图初始化时AIV UB越界，尚未生成token。**
所以当前不能认定候选整模型与Native一致，也没有观测到可归因于数值计算的token首差。
WKV恢复暂停，先用单卡小任务定位运行时故障，不继续扩大模型配置。
原`tnd_precision`混合基准保留，本次候选为`tnd_full_precision`。
已恢复PTO HC-pre、输入Norm和HC-post；WKV、Indexer权重投影及O-proj仍Native。
本次不是HC-pre至HC-post全部PTO替换的验收。

## 完整测试口径

| 项目 | 配置 |
| --- | --- |
| 模型 | DeepSeek-V4-Flash-0731-w8a8，真实完整75个权重shard |
| 输入 | 已固定的8192-token Native bank，每DP副本40条不同公开问题及聊天后缀 |
| 并行 | TP=1，DP=EP=16 |
| 请求 | 每卡提交40条；总640条，每条192个输出token |
| 生成 | greedy、ignore EOS、DSpark出5验6；记录实际接受事件，不固定接受步长 |
| 图 | FULL_DECODE_ONLY，capture sizes=[6,240]，max batched tokens=400 |
| KV | blocks=22000，memory utilization=0.95，预留检查2GiB |
| 开关 | Native确定性level1、HCCL确定性；atomic=0，event工作模式1，NZ=2，EPLB关闭 |
| 对齐 | 所有16rank实际GPU输入、请求顺序、positions、query bounds、DP实际token向量 |
| 精度 | 完整token与接受事件；rank0每轮outer hidden/aux/MTP位摘要；首差即停 |
| 不包含 | Native2、内部逐层stage witness、性能结论、128K |
| 时间 | 每项任务max-time=780秒，低于15分钟 |

沿用此前[整模型精度复现方案](DSV4_FLASH_CSA_TND_PRECISION_REPRO_20261016.md)的bank与环境，
没有重新随机生成前缀状态。仍未核验与nalinaly原bank的全部状态一致，不能声称完全相同输入资产。
本次两臂都使用同一3357文件step05源码快照，仅backend不同。

实际运行环境：Python3.12，Torch2.10.0+cpu，Torch-NPU2.10.0.post2，vLLM0.25.1；
CANN9.2.0-beta.2，匹配ATB cxx_abi_1，PTOAS0.66，A3/TMR。
使用既有PyPTO私有runtime-package及Simpler `a54c0509552b01e13fb0960e23ce409a01b5024f`；
实际import/toolchain路径、完整源码SHA与依赖清单由归档的source_preflight及参考文件固定。
没有重装vLLM、PyPTO、Simpler或修改系统环境。

## Native有效性

任务 `task_20261016_192621_7066592194` exit0。

- 640条请求均完成192个token，合计122880个输出token，零抢占。
- 51轮正式生成实际入场记录；各rank峰值40条请求。
- 16个DP副本的完整token和接受事件相同；这不是第二次Native重复测试。
- 接受0～5个草稿的事件计数：`[1696,2432,2816,2592,2240,14512]`。
- raw记录各rank55轮，rank4为52轮；末尾分别4轮/1轮均严格验为非活跃空尾，不参与正式生成窗口。
- 派生参考manifest SHA256：`be6ba496d189c91f9c40fcea5d8d73c68577d646fb34ea203add5612297c2834`。

实际日志覆盖约7分42秒，仅是日志时间范围，不等同调度器精确设备占用时长。
该Native参考保留复用，不因CSA初始化失败重跑Native2。

## CSA失败记录

任务 `task_20261016_204420_68175017924` exit1。

FULL capture第一档完成后，在下一档初始化同步处报错：
`AclrtSynchronizeDeviceWithTimeout`、507015、AIV VEC UB读写越界。
尚无正式生成input-trace、完整token结果或数值STOP记录。
错误由异步任务在同步位置暴露，栈中同步调用不能直接认定为具体故障模块。

设备日志外层kernel名为`aicore_kernel_mode_0`，program id78，hash11209719686180165410。
这是Simpler运行外壳；故障PC不在外壳符号范围内，尚未映射到具体PTO子核。
不能据此认定是Norm、HC-pre或原有CSA主体的某一核。
固定Tile的静态UB边界检查没有发现明确超预算项，仍需动态执行定位。

## 已完成的单卡缩小测试

| 测试 | 任务 | 结果与限定 |
| --- | --- | --- |
| L4 Norm，T6零输入A/变化B/恢复A | task_20261016_211316_18357253688 | exit0，独立模块eager及图重放逐bit一致 |
| L4 HC-pre，依次T6/T240零输入A/变化B/恢复A | task_20261016_211723_202443029939 | exit0，两形状独立模块eager及图重放逐bit一致 |

这些测试没有复现整模型越界，只排除了所测独立模块/输入组合；
不能外推多层、多program组合或完整模型精度通过。
后续组合测试结果另行记录，不改变本次CSA整模型未通过的状态。

## 复现与证据

[证据索引](evidence_tnd_precision/full_pto_20261016/model-8k-regression-index.json)及
[精简归档](evidence_tnd_precision/full_pto_20261016/model-8k-regression-evidence.tar.gz)。
归档SHA256：`4d52d22b1a656bfe47595ad6d14c1573ab600377a9a929405b3ade21ca584a22`，
52项文件均有独立SHA，包含源码快照清单、运行/预检/入场观测/两臂验收脚本、
Native参考绑定和实际结果、首错日志摘录、原始文件身份，以及上述单卡测试脚本与结果。
不含模型权重、venv、完整大张量或失败设备的大型dump。
归档仅保存所有rank原始文件身份和代表性结果，不含全部16rank原件；
完整历史重审仍依赖227保留的原始目录，不能仅凭精简包重新计算全部跨rank门禁。

```bash
# 227独立目录，先按归档恢复原路径资产并核验SHA；不能把旧实验目录覆盖使用。
source env.sh
python source_preflight.py
python check_bank.py
# 经task-submit、16卡、max-time780运行：
bash run-case.sh native-control <新的native输出名>
# 先在新副本中将prepare_generation_reference.py的raw指向上一步新Native输出。
python prepare_generation_reference.py
# Native派生参考必须绑定此次Native，不能借用未核验的另一轮结果。
bash run-case.sh precision-witness <新的csa输出名>
python qualify_two_arm.py <native输出目录> <csa输出目录> --output <验收json>
```

`run-case.sh`包含固定实验根路径；迁移环境时应在新目录中显式修改并重新冻结helpers，
同时更新验收身份；prepare_generation_reference.py没有CLI参数，raw路径须显式更新。
不能直接覆盖历史脚本或把另一轮结果记成本轮。
本次CSA没有完成，最终两臂qualifier没有产生PASS；不得用Native有效性文件替代它。
