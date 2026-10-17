# CSA 单 SPMD：问题、issue、PR 与本地修复登记

2026-10-16 核对。登记只覆盖本次把 CSA 动态排程固化成单个 SPMD 的工作。
“已发 issue”“已发 PR”“仅本地修复”分别列出，不能互相替代。

2026-10-16 用户另行明确授权提交 InCore 内点对点通知/等待功能需求，已创建
[PyPTO #3020](https://github.com/hw-native-sys/pypto/issues/3020)，作者 nalinaly，
状态 OPEN。与下面四份编译正确性 bug 分开登记，不改变本地组件补丁的提交范围。
正文、主线源码依据和回读结果见 `upstream_issues/incore_p2p_notify_wait_20261016/`。

## 此前用户收窄后的 bug 提交范围

只提交确定的编译正确性 bug issue。下面四项已经创建并回读核对标题、正文和状态；
代码补丁仍在本地，没有 push PyPTO/Simpler 分支或创建修复 PR。

| 本地编号 | 已提交 issue | 必要性与范围 |
| --- | --- | --- |
| P1 | [PyPTO #3014](https://github.com/hw-native-sys/pypto/issues/3014) | Group 共享 InOut 的 SSA 别名未绑定，直接阻塞当前 JIT 路径 |
| P4 | [PyPTO #3015](https://github.com/hw-native-sys/pypto/issues/3015) | PTOAS 分支后 view 使用错误的 SSA 句柄；备用路径正确性 |
| P5 | [PyPTO #3016](https://github.com/hw-native-sys/pypto/issues/3016) | 条件 gather 同存储 phi 生成非法 Mat→Mat；备用路径正确性 |
| P7 | [PyPTO #3017](https://github.com/hw-native-sys/pypto/issues/3017) | 行缓冲/列 carry 的 TMOV 形状不符；备用路径正确性 |

P2/P3/P6/P8/P9 的内存规划和状态搬运改进先留在本地；Simpler S1/S2 的
CORE 32→256 Tensor、64 位 dump 掩码扩展、派发及相关 ABI 也按用户指示留在本地。
`runtime` 子模块和 `_kernel_abi.py` 的修改只是独立环境与扩容 ABI 配套，不是另一个 bug。

Simpler S3 已有 [issue #2331](https://github.com/hw-native-sys/simpler/issues/2331)
与已合入的 [PR #2340](https://github.com/hw-native-sys/simpler/pull/2340)。主线使用
CCE depfile 跟踪头文件；本地调试分支缺少此改动，显式依赖仅作当前环境修复，
后续应优先采用主线实现，不新建重复 issue。

四份完整复现、正文、发布记录和回读结果保存在 `upstream_issues/`。
五个独立 CPU 用例在本地修复版全部通过；没有重跑完整设备或七档性能。
相同复现在未打本地补丁的 PR #3013 精确 head 中全部失败；该对照已补入四份 issue。
创建请求指定 bug 标签，但 GitHub 回读没有应用标签；标题和完整正文均一致，
没有为此重复创建 issue。一次 GraphQL 回读 EOF 后改用 REST 回读确认。

## 已提交的上游 issue

| 项目 | 链接与状态 | 问题、处理方式、证据 |
| --- | --- | --- |
| PyPTO #3009 | [issue](https://github.com/hw-native-sys/pypto/issues/3009)，OPEN，作者 nalinaly，2026-10-15 创建 | 自动 MIX 展开删除仅用于 while condition 的 iter_arg，产生 FREE_VAR。最小复现、原/预期 IR、命令见 PYPTO_MIXED_WHILE_LOOP_STATE_REPRO.md。当前显式 Group 两半路径绕开自动拆分步骤，不代表这个 issue 已修复 |
| PyPTO #3020 | [功能需求](https://github.com/hw-native-sys/pypto/issues/3020)，OPEN，作者 nalinaly，2026-10-16 创建 | 单个 InCore/SPMD 内，基于普通 GM 信号的点对点 notify/wait；明确动态阈值、单写者多消费者、代际、信号与 payload 内存语义、MIX lane 副作用及验收范围。尚未实现 API |

创建请求、创建结果、首次回读分别保留在 `issue_payload.json`、
`issue_creation_result.json`、`issue_verified.json`。
本次 GitHub 回读保留在 `upstream_records/csa-pypto-issue-3009-current.json`。
此前登记时尚未观测到关联 PR；现已按用户指示拉取
[PR #3013](https://github.com/hw-native-sys/pypto/pull/3013)，head
`72b7ffb98ea3cee12bf7bbf533ab89b848a8980c`，在独立 detached worktree 核对。
该 PR 修复 #3009 的 while 条件活跃性，和本轮四个新 issue 区分登记。
独立构建通过，七个定向回归通过；本会话原始 scalar/tile/control 三种复现
均完成代码生成且无 ready FREE_VAR，见 [PR 核对记录](PR3013_REVIEW_20261016.md)。

## PR 登记

本轮尚未提交 PyPTO 或 Simpler PR，也未 push 这两个独立修复分支。
分别按下面两个 head 查询全部状态 PR，返回均为空，原始结果在
`upstream_records/csa-pypto-pr-current.json` 与
`upstream_records/csa-simpler-pr-current.json`：

- PyPTO：`feat/csa-static-spmd-fixes-20261016`。
- Simpler：`feat/csa-core-args-256-20261016`。

后续创建 PR 时在此追加真实 URL、head/base、提交、验证范围和合并状态；
用户本次要求记录 PR，不等于已经存在 PR。

## 独立组件的本地修复

| 编号 | 问题 | 修改与状态 | 证据 |
| --- | --- | --- | --- |
| S1 | CORE 只有 32 Tensor，当前需要 132/133 | 已扩 256 Tensor + 16 scalar；上下文、派发、参数池、图缓存、依赖记录同步 | SIMPLER_CORE_EXPANSION_20261016.md；本地 commit `57e499238fd4ee3548947ad860be456aaeeb16b5` |
| S2 | 参数 dump 单 uint64，派发结构固定 512 B | 272 位分 5 字掩码；派发 2304 B，关联布局重新计算 | 同一 S1 commit；高位 dump 和四类 runtime/device 验证通过 |
| S3 | 修改 ABI 头文件后旧 CCE kernel 可能被缓存复用 | 增量编译依赖加入 ABI 头文件 | 同一 S1 commit |
| P1 | Group 内调用返回别名未映射到统一形参 | 使用 return_lineage 追踪完整 InOut 别名；同步 tuple/Submit/no_dep | 独立 PyPTO 本地修改；CPU 回归及最小共享 InOut JIT 真机 PASS |
| P2 | 跨阶段 48+12 / 32+32 KiB 无共同最大 root，错误占用 80 KiB | 超限时有界扩展复用区；最终分配按最大 offset+size 保留 | 三参数与真实 Indexer+O-proj 13 参数代码生成 PASS；三参数真机四阶段 max_abs=0 |
| P3 | 内层 phi 更新后纯别名仍保留旧缓冲 | 同遍历中即时跟随，防止外层选回旧 storage | 嵌套 phi 回归；完整 B4 Vec 减少 64 KiB |
| P4 | PTOAS view 仅按类型省略，引入不支配使用的 SSA | 要求实际源句柄一致，否则生成 reshape view | 完整 B4 两处 SSA 错误消除 |
| P5 | 同一物理窗口的条件 gather 仍引入 Mat→Mat 拷贝 | phi 保持同一窗口；多类型 identity 只复用已声明且类型完全匹配的 handle | Mat→Mat 错误消除；含 transpose+matmul 消费者的回归通过 |
| P6 | 分支原地循环的别名未加入对应 phi family | 只标记分支内同 allocation 的别名；保留外部独立读取约束 | gather 正/反例通过，两块 64 KiB 合并 |
| P7 | loop-carry 列向量向行 handle 搬运，shape 不匹配 | 先生成同存储的目标列 view，再搬运 | 列 carry 回归；完整 PTOAS 路径进入实际内存分配 |
| P8 | then→else 线性顺序造成假干涉 | 仅超限时加一次反向分支遍历；同分支/循环后续读取仍禁止复用 | 完整 Vec 197632 B，仍超 9216 B；相关三文件 378 项通过 |
| P9 | 旧 MemRef 的纯别名阻断循环 carry 目标传播，生成冗余状态搬运 | 只在循环 carry 追踪中允许越过旧 identity 窗口；累加器合并保持原门槛 | 最终相关两文件 169 项通过；QK/PV 恢复历史 181248 B；完整 B4 v21 代码生成 PASS |
| T1 | 当前 PyTorch build 的 schema 最多 64 项，完整入口为 132 项 | 用户已选择现有 JIT native 直调路线，62 项候选取消 | DIRECT_NATIVE_ENTRY_20261016.md；没有修改 PyTorch，没有为此提 issue/PR |
| P10 | 普通 GM 信号缺少单 SPMD 核间 notify/wait 接口，属于功能需求 | 独立 PyPTO 临时 Set/Ge API，复用 TNOTIFY/TWAIT；本地保留，未提交 PR | #3020；LOCAL_SIGNAL_NOTIFY_WAIT_20261016.md；与前九项修复分开 |

PyPTO 修改尚未 commit。补丁快照与基线/分支/状态见
`component_changes/pypto/`；Simpler 提交全文见
`component_changes/simpler/`。修改没有覆盖共享安装，没有改 PTOAS、PTO-ISA 或 pypto-lib。

## 仍待解决

- 完整 B4 默认路径代码生成和 program 二进制组装已通过；完整 B16 kernel 模式
  native JIT 直调现已三次通过，对冻结动态 PTO 输出/Top-K/cache/state 精确一致。
- PTOAS 正式路径在 Mat 分配超 16 KiB；DSA_RP 正式路径在 AIC 无容量内布局。
  没有把它们认定为硬件不可实现，也没有放宽硬件容量。
- 临时通知/等待及本核流水等待已在 B16 验证进度、状态与保护区；动态模板、
  ACLGraph、连续 decode 轨迹及性能仍在本轮验证范围之外。
- #3009 自动 MIX 的 while 活跃性问题未在本分支修改，不标为修复。
