# Score 转置与 NZ 写回：完整通过但退步，不采用

2026-10-18。本实验承接 key-major Score 的 FIX 写回疑点，仅核对直接 NZ
写回能否改善完整 CSA。测试源在本目录，正式算子未采用此路径，也未修改
PyPTO/Simpler/PTOAS 的共享安装。它不是单独衡量一条 TSTORE 指令的实验。

## 实现及复现边界

`source_reference/` 为协同前段已采用的35模块版本；`source_candidate/`
保持其它33模块，仅修改 binding 和 Indexer Score。候选保持原转置矩阵
数学，将每个64×16 FP32交接panel写成NZ，AIV读取其中前8列再恢复原布局。
实际CSA根入参130个、58个动态轴、一个MIX，仍从device metadata计算有效量。

PyPTO主线目前明确拒绝把NZ Tensor作为store目的，因此这里用测试侧
`nz_store_codegen.py` 做有限代码生成替换：只识别Score交接根及64×16 FP32
Acc写回，改为A3 ISA已有的NZ描述符，精确替换16个调用。参考侧替换0个。
移除新helper并还原调用后，生成C++与输入原文逐字一致；原数学、事件及等待
没有一并修改。完整审查见 `CODEGEN_AUDIT.json`、`OFFLINE_HOOK_CHECK.json`。
该hook是硬件能力实验，不声称候选是无需补丁即可独立运行的正式PyPTO实现。

第一版离线hook仅匹配到一个带名称的view，按相同GetData根追踪后修正，
失败记录未删除。CPU首版把GM字节别名写成普通reshape，得到Tile而无法供
gather_row读取，改用正确的 `pl.tensor.view` 后完整编译通过。这是算子表达
修正，不作为另一个编译器bug上报。全部过程见 `CPU_ATTEMPTS.json`。

## 真机结果

任务 `task_20261018_165718_32204744106` exit0。128K/B16/S6，正式W8A8第4层
权重、同进程同输入，完整精确输出、Top-K/score、cache/state、A→B→A、
图重放、代际与保护检查通过。五窗完整L1 incore首尾：

| 版本 | 五窗耗时 / μs | 中位 / μs |
|---|---|---:|
| 原多task PTO | 872.94 / 868.08 / 901.38 / 884.40 / 888.88 | 884.40 |
| 当前正式动态SPMD | 848.66 / 860.58 / 843.26 / 857.74 / 855.58 | 855.58 |
| 转置Score加NZ写回 | 1412.62 / 1415.88 / 1419.12 / 1370.70 / 1400.78 | 1412.62 |

候选比正式版慢65.11%，不采用、不扩测七档。该候选还改变了Score数据流和
AIV布局读取，不能将整层退步全归因于NZ store。摘要 `FULL_L1_RESULT.json`，
完整检查与逐窗数据 `h131072_b16_v1/result.json`、`source_matrix/result.json`。

CPU复现：`bash run_cpu.sh 全新绝对输出目录 --controls-json controls_long.json的绝对路径`。
设备必须经队列，完整命令见 `SUBMISSION_V1.json`。保留实际冻结源，不重用
已有输出目录；大编译物及二进制只登记路径和大小，不入Git，不做hash。

## 上游功能需求

[PyPTO #3036](https://github.com/hw-native-sys/pypto/issues/3036) 已创建，无标签、
无修复PR。正文包含无设备复现、A3硬件依据、建议实现范围和验收边界。
最小API复现见 `nz_store_api_repro.py`，官方只读限制已在 `api_repro_v1` 复现。
主线源和版本依据保存在 `UPSTREAM_NZ_GUARD.json`、`upstream_main_nz_pass.cpp`。

随后已[补充完整负收益结果](https://github.com/hw-native-sys/pypto/issues/3036#issuecomment-6248136037)，
明确这是后续能力补齐，不是当前默认CSA必须修复的bug，也没有本场景已证实
的性能收益。正文及评论远端回读见 `ISSUE_VERIFIED.json`、`ISSUE_FOLLOWUP_VERIFIED.json`。
