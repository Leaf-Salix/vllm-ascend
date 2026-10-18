# 本轮组件问题上游登记（2026-10-18）

用户最新要求：算子编写以外的组件问题及时登记 GitHub 主仓 issue；可附建议代码，不直接提交 PR，不添加 bug label。**Simpler 非 bug 能力增强继续本地保留**。

本轮新增 4 份 PyPTO issue，并向 3 份已有 issue 追加证据；全部远端回读核对正文，无 PR、无依赖分支 push。

| 范围 | 上游登记 | 当前证据/处理 |
| --- | --- | --- |
| 常量 Scalar slice 起点误保留为动态偏移 | [PyPTO #3032](https://github.com/hw-native-sys/pypto/issues/3032) | 无设备最小复现、建议补丁；65 项相关回归及完整动态 CSA CPU 编译通过；本地修正 commit d46a813ca |
| 动态 descriptor 算术 reshape 丢 metadata | [PyPTO #3033](https://github.com/hw-native-sys/pypto/issues/3033) | 主线派生环境失败、独立修正环境通过；未知 device scalar 负例保留 |
| inline Tile 注解丢失及同 helper constexpr 边界 | [PyPTO #3034](https://github.com/hw-native-sys/pypto/issues/3034) | 固定 Tile 修正六行；constexpr 名字边界尚未修复，正文明确区分 |
| 分支局部别名家族/线性访问顺序假干涉 | [PyPTO #3035](https://github.com/hw-native-sys/pypto/issues/3035) | 两份无设备复现、建议代码；真实旧值后续读取/回边反例保留 |
| 无单个最大 root 的跨阶段 48+12 → 32+32 KiB arena | [PyPTO #1908 补充](https://github.com/hw-native-sys/pypto/issues/1908#issuecomment-6246686747) | 既有问题追加 81920→65536 B 复现、增长门槛与 allocator 配套 diff，不重复建 issue |
| 旧 P3/P9 nested phi/identity alias 记录 | [PyPTO #3027 补充](https://github.com/hw-native-sys/pypto/issues/3027#issuecomment-6246687127) | 当前主线派生环境仅有既有 view/While 修正时两项历史回归已过；避免机械叠加旧放宽逻辑 |
| Torch schema 64 个参数上限 | [PyTorch #162870 补充](https://github.com/pytorch/pytorch/issues/162870#issuecomment-6246687515) | 追加 torch 2.10 的 64/65/132 边界、Tensor[] 与 functionalization 限制；已有 native 直调替代 |

继承的修正继续对应 PyPTO #3014/#3015/#3016/#3017（Group 与 PTOAS 备用 codegen）、#3026（FloorMod）、#3027（view）、#3028（While）、#3020（GM notify/wait），以及 PTOAS #1622（CFG 回边并行赋值）。本次没有把同一问题换名称重复提交。

Simpler CORE 32→256 Tensor、16 Scalar、dump 多字掩码/派发布局扩展是能力增强，按用户最新要求不提 issue，保留本地 commit 57e499238。构建依赖问题已有 Simpler #2331 / 已合入 PR #2340，采用主线修复时替换本地补丁。

历史 unroll 外层 IterArg 深克隆问题：最新主线源码已将外部 IterArg 按普通外部 Var 保留（`deep_clone_utils.cpp`）；本次只读确认，不重复建问题，也未重跑旧整层测试。旧源码 bridge 和失败过程保留，后续依赖更新时再收敛。

## 验证边界

`main_dependency_regressions.log` 为 11 项定向旧环境检查，5 失败/6 通过；`fixed_dependency_regressions.log` 为分支/JIT 的 9 项修正检查，全部通过。arena 两例已包含在常量修正的 65 项中，不重复整套。

`jit_direct_results.json`：两种最小 JIT 入口原环境失败，修正环境 specialize 通过。
`memory_direct_results.json`：arena/gather/reset 三个独立脚本原环境失败、修正环境通过。
所有 CPU 对照导入实际独立 worktree 扩展，未复制 .so、未更改共享安装、未追加 NPU。
这里的通过仅对应编译/类型/容量，不代表 CSA 新性能收益。
