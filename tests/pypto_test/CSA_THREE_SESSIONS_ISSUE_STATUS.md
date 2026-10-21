# CSA 三个会话创建的 issue 状态总表

最后在线核对：**2026-10-21 20:58:26 CST**。时间均为北京时间（UTC+8）。

覆盖 `CSA-SPMD`、`CSA-MPMD-TND`（原 CSA-多task）、`CSA-SPMD-TND`（原 CSA-TND）三个 Codex 会话。
本文件逐条记录已确认创建的 issue，并将对别人既有 issue 的补充、SDMA PR 评论和仅引用的记录分开，避免重复计数。

## 1. 总体结果与归属口径

**确认新建 41 条：10 条已关闭、31 条仍开放；开放项中 6 条已有明确修复 PR，均尚未合并。**

| 会话 | 确认新建 | 已关闭 | 仍开放 | 归属说明 |
| --- | ---: | ---: | ---: | --- |
| CSA-SPMD | 40 | 9 | 31 | 包含该会话推进的组件复现与 issue 反馈 |
| CSA-MPMD-TND | 1 | 1 | 0 | PyPTO #2886；另有 stride/确定性能力澄清评论 |
| CSA-SPMD-TND | 0 | 0 | 0 | 未发现新建回执；引用/复用现有 issue，不重复认领 |

| 仓库 | 新建 | 已关闭 | 仍开放 |
| --- | ---: | ---: | ---: |
| hw-native-sys/pypto | 28 | 9 | 19 |
| hw-native-sys/PTOAS | 9 | 1 | 8 |
| hw-native-sys/pto-isa | 2 | 0 | 2 |
| hw-native-sys/pypto-lib | 2 | 0 | 2 |
| hw-native-sys/simpler | 0 | 0 | 0 |

核对方法：先核对三个会话的提交/创建回执和项目记录，再与 GitHub 的 author、创建时间、状态、评论、关闭事件及关联 PR 交叉核对。没有把“会话里出现过链接”直接当成“该会话创建”。
同账号期间公开创建的 issue 共 43 条，另两条 #2948/#2949 仅查到引用、未找到这三个会话的创建回执，见第 7 节。GitHub 查询没有截断或遗漏分页。

上游状态、修复代码合入、本地组件更新和真机回归是四件不同的事情。本轮是只读状态核对，没有升级依赖或重新运行设备测试；不能根据 CLOSED 自动删除本地修复，也不能根据 OPEN 断言上游没有任何相关工作。

## 2. 已关闭的 10 条

| Issue | 问题 | 来源会话 | 关闭时间 | 修复关联与边界 |
| --- | --- | --- | --- | --- |
| [PTOAS #1622](https://github.com/hw-native-sys/PTOAS/issues/1622) | CFG 回边并行赋值被串行覆盖 | CSA-SPMD | 10-20 15:15 | 由提交 [7efff5cc](https://github.com/hw-native-sys/PTOAS/commit/7efff5ccde929a3cfe59fae3a5b45a1b9e09f8aa) 关闭；关联 [PR #1642](https://github.com/hw-native-sys/PTOAS/pull/1642) 本身仍 OPEN，不能将 PR 状态与 issue 状态混为一谈 |
| [pypto #2886](https://github.com/hw-native-sys/pypto/issues/2886) | A3 FP32 tensor 高精度除法的能力说明和诊断 | CSA-MPMD-TND | 09-28 11:14 | 维护者关闭，关闭事件没有关联 PR/commit；不能据此宣称 A3 HIGH_PRECISION 已实现。原请求主要是能力说明与诊断。 |
| [pypto #3009](https://github.com/hw-native-sys/pypto/issues/3009) | 自动 MIX 展开删除 while 条件使用的 iter_arg | CSA-SPMD | 10-19 08:27 | 修复 [PR #3013](https://github.com/hw-native-sys/pypto/pull/3013) 已合并（10-19 08:27） |
| [pypto #3014](https://github.com/hw-native-sys/pypto/issues/3014) | Group 成员 InOut 返回别名没有重绑定 | CSA-SPMD | 10-21 11:25 | 修复 [PR #3018](https://github.com/hw-native-sys/pypto/pull/3018) 已合并（10-21 11:25） |
| [pypto #3015](https://github.com/hw-native-sys/pypto/issues/3015) | 分支后 reshape 使用不支配使用点的 SSA | CSA-SPMD | 10-19 08:26 | 修复 [PR #3019](https://github.com/hw-native-sys/pypto/pull/3019) 已合并（10-19 08:26） |
| [pypto #3016](https://github.com/hw-native-sys/pypto/issues/3016) | 条件 gather 的同存储 phi 生成非法 Mat→Mat | CSA-SPMD | 10-19 18:40 | 修复 [PR #3029](https://github.com/hw-native-sys/pypto/pull/3029) 已合并（10-19 18:40） |
| [pypto #3017](https://github.com/hw-native-sys/pypto/issues/3017) | 行缓冲承载列向量 loop-carry，TMOV 形状不匹配 | CSA-SPMD | 10-20 19:08 | 修复 [PR #3053](https://github.com/hw-native-sys/pypto/pull/3053) 已合并（10-20 19:08） |
| [pypto #3026](https://github.com/hw-native-sys/pypto/issues/3026) | 负数 FloorMod 与 IR 简化器语义不一致 | CSA-SPMD | 10-19 09:49 | 修复 [PR #3030](https://github.com/hw-native-sys/pypto/pull/3030) 已合并（10-19 09:49） |
| [pypto #3027](https://github.com/hw-native-sys/pypto/issues/3027) | 原地重绑定后下游 view 的 MemRef 没有传播 | CSA-SPMD | 10-19 14:22 | 修复 [PR #3031](https://github.com/hw-native-sys/pypto/pull/3031) 已合并（10-19 14:22） |
| [pypto #3041](https://github.com/hw-native-sys/pypto/issues/3041) | 嵌套 inline 临时变量被改成非法自动名称 | CSA-SPMD | 10-21 15:51 | 修复 [PR #3092](https://github.com/hw-native-sys/pypto/pull/3092) 已合并（10-21 15:51） |

本次当天新增收口的两个重要修复是 PyPTO **#3014 → PR #3018** 和 **#3041 → PR #3092**。以上八个 PyPTO 修复 PR 均已合并；#2886 属于不同的能力澄清关闭情形。

## 3. 仍开放、已有修复 PR 的 6 条

以下六条均由 `CSA-SPMD` 创建。PR 的审查/CI 是当前 head 的 GitHub 汇总快照，不是本会话重新执行测试的结果。

| Issue | 问题 | 修复 PR | PR 状态 | 审查汇总 | CI 汇总 | PR head |
| --- | --- | --- | --- | --- | --- | --- |
| [pypto #3028](https://github.com/hw-native-sys/pypto/issues/3028) | 内存别名遍历遗漏 WhileStmt，嵌套循环复用失效 | [#3054](https://github.com/hw-native-sys/pypto/pull/3054) | OPEN，未合并 | 等待审查 | 汇总失败 | `fff2f535ff` |
| [pypto #3032](https://github.com/hw-native-sys/pypto/issues/3032) | 常量 SSA 切片起点不折叠，阻断合法内存复用 | [#3055](https://github.com/hw-native-sys/pypto/pull/3055) | OPEN，未合并 | 等待审查 | 汇总失败 | `2c3cc11fe3` |
| [pypto #3033](https://github.com/hw-native-sys/pypto/issues/3033) | 动态维度算术导致 reshape metadata 丢失 | [#3061](https://github.com/hw-native-sys/pypto/pull/3061) | OPEN，未合并 | 等待审查 | 通过 | `63b30cdf0a` |
| [pypto #3034](https://github.com/hw-native-sys/pypto/issues/3034) | JIT inline 特化丢弃显式 Tile 参数注解 | [#3070](https://github.com/hw-native-sys/pypto/pull/3070) | OPEN，未合并 | 已批准 | 通过 | `4c3446ea0e` |
| [pypto #3035](https://github.com/hw-native-sys/pypto/issues/3035) | 互斥分支别名与访问顺序造成内存伪干涉 | [#3071](https://github.com/hw-native-sys/pypto/pull/3071) | OPEN，未合并 | 已批准 | 汇总失败 | `51b27788ef` |
| [pypto #3046](https://github.com/hw-native-sys/pypto/issues/3046) | FlattenTileNdTo2D 未同步重写动态类型引用 | [#3090](https://github.com/hw-native-sys/pypto/pull/3090) | OPEN，未合并 | 已批准 | 通过 | `5d902a63b8` |

- **#3034 / PR #3070、#3046 / PR #3090：已批准且 CI 汇总通过，但尚未合并。**
- #3033 / PR #3061：CI 通过，仍等待审查。
- #3028 / PR #3054、#3032 / PR #3055：仍等待审查，CI 汇总失败。
- #3035 / PR #3071：已批准，CI 汇总失败；作者已补充最近一轮修改和验证说明。
- CI 汇总失败不等于编译器断言或设备数值失败。例如 [PR #3054 的作者说明](https://github.com/hw-native-sys/pypto/pull/3054#issuecomment-6284884578) 记录了 SSL/GnuTLS、artifact 上传和任务取消等问题；未将该解释机械套用到其它 PR。
- #3042 的时间线虽然引用了 PR #3090，但 #3090 明确关闭的是 **#3046**，不能把它列成 #3042 的完整 GM subview 功能修复。

## 4. 仍开放、尚无明确关闭关联修复 PR 的 25 条

以下全部由 `CSA-SPMD` 创建。这里的“尚无明确修复 PR”依据 issue 时间线和 PR 的 closing-issue 关联，不排除没有关联 issue 的其它开发分支。

### 4.1. pypto（13 条）

| Issue | 问题或需求 | 进展、已有证据与限制 | 最近更新时间 |
| --- | --- | --- | --- |
| [pypto #3020](https://github.com/hw-native-sys/pypto/issues/3020) | 同一 InCore/SPMD 内普通 GM Tensor 的点对点 notify/wait | 已提供普通 GM 信号的接口合同与 A3 缓存发布/读取要求；当前 CSA 的本地通知方案不能等同上游正式 API 已完成。 负责人：未分配。 | 10-19 14:29 |
| [pypto #3036](https://github.com/hw-native-sys/pypto/issues/3036) | 支持 AIC Acc 写回 NZ GM | 已补完整 CSA 负性能结果；属于功能需求，不能称为默认 CSA 正确运行的必修 bug。 [最新补充](https://github.com/hw-native-sys/pypto/issues/3036#issuecomment-6248136037)。 负责人：Little-oil。 | 10-19 14:19 |
| [pypto #3037](https://github.com/hw-native-sys/pypto/issues/3037) | 死 Tile 经 loop-carry/if-phi 构成闭环，仍占片上内存 | 已补本地 DCE 方案：从语义消费者求可达性，保留零次循环等边界；当前 issue 尚无明确关闭关联。 [最新补充](https://github.com/hw-native-sys/pypto/issues/3037#issuecomment-6249851423)。 负责人：未分配。 | 10-21 11:41 |
| [pypto #3038](https://github.com/hw-native-sys/pypto/issues/3038) | 显式 strided GM view，用于多段 ND2NZ 加载 | 已补显式 stride 的隔离原型、地址合同及验证结果；没有放宽非连续根 Tensor 的入口保护。 [最新补充](https://github.com/hw-native-sys/pypto/issues/3038#issuecomment-6282009526)。 负责人：Little-oil。 | 10-20 10:35 |
| [pypto #3039](https://github.com/hw-native-sys/pypto/issues/3039) | current/next 预取轮换生成不支持的 Mat 复制 | 属于预取缓冲轮换的编译能力/正确性问题；与允许真正 Mat 复制是不同诉求。 负责人：Little-oil。 | 10-19 14:21 |
| [pypto #3042](https://github.com/hw-native-sys/pypto/issues/3042) | InCore GM 子视图后显式 load，支持动态 pitch 的两行 DMA | slice 在 Tensor→Tile 阶段提前变成 Tile，后续 GM view 因此拒绝。已补 gm_view=True 原型、52 项 CPU 和 26 组 A3 地址验证；不是主线接口已合入。 [最新补充](https://github.com/hw-native-sys/pypto/issues/3042#issuecomment-6257775143)。 负责人：未分配。 | 10-21 11:40 |
| [pypto #3049](https://github.com/hw-native-sys/pypto/issues/3049) | 多个存活 Mat 的条件选择生成非法 Mat→Mat | 多份仍存活 Mat 的选择不能套用同存储 phi 修复；不能因为 #3016 已关闭就认为本项已覆盖。 负责人：Little-oil。 | 10-19 14:21 |
| [pypto #3062](https://github.com/hw-native-sys/pypto/issues/3062) | 合法 NZ 展平跨 InCore helper 丢失 TensorMeta | 已提交独立 JIT 元数据复现及建议，当前未见明确关联修复 PR。 负责人：未分配。 | 10-19 21:44 |
| [pypto #3063](https://github.com/hw-native-sys/pypto/issues/3063) | Scalar BOOL→INT32 符号扩展，true 得到 -1 | 已补 literal/constexpr/device 的 A3 数值对照；INDEX 与 FP32 的后端差异另外报到 PTOAS #1652。 [最新补充](https://github.com/hw-native-sys/pypto/issues/3063#issuecomment-6301892254)。 负责人：未分配。 | 10-21 05:29 |
| [pypto #3079](https://github.com/hw-native-sys/pypto/issues/3079) | gather_compare 单行计数输出未对齐，形成 4B UB Tile | 计数输出的逻辑形状与物理 UB 对齐需分别处理；已提供独立复现。 负责人：未分配。 | 10-20 17:42 |
| [pypto #3083](https://github.com/hw-native-sys/pypto/issues/3083) | libbacktrace 子构建裸 make -j，忽略外层并行预算 | 构建资源预算问题，与 SDMA 设备功能无关；本地复现脚本显式限定子构建并行数。 负责人：未分配。 | 10-20 20:48 |
| [pypto #3085](https://github.com/hw-native-sys/pypto/issues/3085) | 保序、有界的 value/index/count 三输出 Tile 压缩接口 | 与 PTOAS #1650 配套；需要保持 value/index 位置配对、真实 count、容量及同分顺序。 负责人：未分配。 | 10-20 23:13 |
| [pypto #3087](https://github.com/hw-native-sys/pypto/issues/3087) | ConvertTensorToTileOps 未重绑定 set_validshape 动态类型 | set_validshape 中动态类型引用未同步，不能仅以 FlattenTileNdTo2D 的 #3046 修复宣称本项已覆盖。 负责人：未分配。 | 10-21 00:27 |

### 4.2. PTOAS（8 条）

| Issue | 问题或需求 | 进展、已有证据与限制 | 最近更新时间 |
| --- | --- | --- | --- |
| [PTOAS #1623](https://github.com/hw-native-sys/PTOAS/issues/1623) | 不相交 L0C 列子视图被按整个父池处理，产生多余 FIX→Cube 等待 | 已补本地同步检查与完整设备正反结果；静态少一条依赖不代表所有档位加速。 [最新补充](https://github.com/hw-native-sys/PTOAS/issues/1623#issuecomment-6253756417)。 负责人：未分配。 | 10-19 02:50 |
| [PTOAS #1624](https://github.com/hw-native-sys/PTOAS/issues/1624) | GM partition_view 丢失固定列范围，产生多余 FIX barrier | 已补同源编译器 A/B 真机结果，长档改善、短档回退；未将同步减少直接等同普遍收益。 [最新补充](https://github.com/hw-native-sys/PTOAS/issues/1624#issuecomment-6254086363)。 负责人：未分配。 | 10-19 03:21 |
| [PTOAS #1625](https://github.com/hw-native-sys/PTOAS/issues/1625) | TEXTRACT 读取范围过宽，将 L1 双缓冲串行化 | 已附独立范围/循环依赖复现；需识别真实不相交槽，同时保留重叠访问保护。 负责人：未分配。 | 10-19 06:51 |
| [PTOAS #1647](https://github.com/hw-native-sys/PTOAS/issues/1647) | A3 compare-form TGATHER 的模板/参数与 ISA ABI 不匹配 | 属于编译器生成调用与当前 ISA ABI 的不匹配，不能仅改算子层参数掩盖。 负责人：未分配。 | 10-20 17:48 |
| [PTOAS #1648](https://github.com/hw-native-sys/PTOAS/issues/1648) | LLVM 19 EmitC 丢失嵌套位运算括号，改变条件语义 | 布尔/位运算组合的括号丢失可能改变合法条件的含义，属于代码生成正确性问题。 负责人：未分配。 | 10-20 21:39 |
| [PTOAS #1649](https://github.com/hw-native-sys/PTOAS/issues/1649) | TGATHER 子视图被 ResolveBufferSelect 扩宽；后续还补充 TMOV 写侧变体 | 最新补充包括 TMOV 写侧列子窗口；应同时关注读侧 gather 和写侧 assemble 的物理形状。 [最新补充](https://github.com/hw-native-sys/PTOAS/issues/1649#issuecomment-6299434307)。 负责人：未分配。 | 10-21 03:27 |
| [PTOAS #1650](https://github.com/hw-native-sys/PTOAS/issues/1650) | 共享掩码的 value/index 成对压缩及真实 count | 已有本地成对压缩、满容量直写及索引倍增原型和设备验证；整层复核收益不稳定，未作为 CSA 正式优化采用。 [最新补充](https://github.com/hw-native-sys/PTOAS/issues/1650#issuecomment-6298459118)。 负责人：未分配。 | 10-21 02:42 |
| [PTOAS #1652](https://github.com/hw-native-sys/PTOAS/issues/1652) | Scalar i1 常量/运行时转换不一致，A3 BOOL→FP32 编译失败 | 已给出常量与运行时 i1 转换对照、CPU 生成算术和 A3 编译失败证据；未声称失败 FP32 路径已完成设备执行。 负责人：未分配。 | 10-21 05:28 |

### 4.3. pto-isa（2 条）

| Issue | 问题或需求 | 进展、已有证据与限制 | 最近更新时间 |
| --- | --- | --- | --- |
| [pto-isa #352](https://github.com/hw-native-sys/pto-isa/issues/352) | A3 MGATHER ZERO 清零与 ND2NZ 写缺少顺序约束 | 涉及清零与后续写入的顺序正确性，当前没有关联修复提交/PR 可证明上游已收口。 负责人：未分配。 | 10-19 11:08 |
| [pto-isa #364](https://github.com/hw-native-sys/pto-isa/issues/364) | CPU-SIM TSORT32 同分顺序与 A3 不同 | 模拟器按 payload index 处理同分，而 A3 保留输入位置；模拟与真机语义对齐仍待上游确认。 负责人：未分配。 | 10-20 18:21 |

### 4.4. pypto-lib（2 条）

| Issue | 问题或需求 | 进展、已有证据与限制 | 最近更新时间 |
| --- | --- | --- | --- |
| [pypto-lib #1486](https://github.com/hw-native-sys/pypto-lib/issues/1486) | incore profiling 的 timeout 分钟/秒单位及外层时限不一致 | 脚本的秒级描述与 msprof 分钟级参数不同，可能导致外层超时误杀；当前未见关联修复。 负责人：未分配。 | 10-19 00:40 |
| [pypto-lib #1487](https://github.com/hw-native-sys/pypto-lib/issues/1487) | A3 模拟器 SYS_CNT 恒零，计时退避无法退出 | 同二进制真机通过、模型 SYS_CNT 不推进；模拟专用退出上限不能当作真机同步正确性的证据。 负责人：未分配。 | 10-19 01:38 |

41 条新建 issue 的评论区，截至核对时均没有非作者的文本回复；上游处理主要体现在分配负责人、关联修复 PR 和关闭事件。**这不等于无人处理**，已合并的修复和正在审查的 PR 已在前两节单列。

## 5. 向既有 issue 补充的复现与建议（不计入新建 41 条）

| 既有 Issue | 当前状态 | 本会话补充与当前结论 |
| --- | --- | --- |
| [PTOAS #1587](https://github.com/hw-native-sys/PTOAS/issues/1587) | OPEN | 原作者 Hzfengsy；维护者评论指向 [GitCode PR #290](https://gitcode.com/cann/pto-as/pull/290)。CSA-SPMD 后来补充了 [循环回边不相交行访问复现](https://github.com/hw-native-sys/PTOAS/issues/1587#issuecomment-6260959874)，该变体是否被覆盖尚未确认；没有把 stock 0.66/0.67 的结果当成对 GitCode PR 的验证。 |
| [PTOAS #1620](https://github.com/hw-native-sys/PTOAS/issues/1620) | CLOSED | 原作者 lyfne123；CSA-SPMD 补充了 [MTE2→MTE3](https://github.com/hw-native-sys/PTOAS/issues/1620#issuecomment-6249288578) 和 [动态 L1 基址](https://github.com/hw-native-sys/PTOAS/issues/1620#issuecomment-6255880887) 变体。上游由提交 [6f5e16af](https://github.com/hw-native-sys/PTOAS/commit/6f5e16afaaacd2ac1324e737235b96f33f12e08c) 关闭，提交同时将版本升至 0.69；本轮未重新验证设备覆盖。 |
| [PyPTO #2951](https://github.com/hw-native-sys/pypto/issues/2951) | CLOSED | 原问题随 [PR #3004](https://github.com/hw-native-sys/pypto/pull/3004) 合并关闭。CSA-SPMD 后来补充的 [两个动态维度 view 变体](https://github.com/hw-native-sys/pypto/issues/2951#issuecomment-6281892586) 来自隔离本地树，尚未确认干净最新 main 是否覆盖；不能说原修复无效，也不能说新变体已经解决。 |

仅引用的 Simpler [#2331](https://github.com/hw-native-sys/simpler/issues/2331) 不是这三个会话创建：原作者 ChaoWao，已于 09-19 随 [PR #2340](https://github.com/hw-native-sys/simpler/pull/2340) 合并关闭，内容是 AICore 编译的头文件依赖追踪。Simpler 参数容量扩展按用户要求留在本地处理，没有在本统计中虚列成新增 issue。

## 6. SDMA 的 PR 评论跟进（不是新增 issue）

CSA-SPMD 在两个既有 PR 下补充了通道绑定、隐藏 UB scratch 覆盖和公开复现。当前状态：

| PR | 当前状态 | 当前 head | CI 汇总 | 与本地吸收修复的关系 |
| --- | --- | --- | --- | --- |
| [pypto #3001](https://github.com/hw-native-sys/pypto/pull/3001) | OPEN，未合并 | `ed612fbdb7` | 通过 | PR head 仍是 ed612fbd；评论中的配套修复 0093d617 尚未进入该 head。 |
| [simpler #2583](https://github.com/hw-native-sys/simpler/pull/2583) | OPEN，未合并 | `97343062d0` | 汇总失败 | head 已包含驻留 AIV 独立通道修复；两 PR 目标都是 kernel-mode 集成分支，不能说 main 已支持全部链路。 |

- 原始问题反馈：[通道绑定](https://github.com/hw-native-sys/pypto/pull/3001#issuecomment-6269202617)、[公开复现与显式通道对照](https://github.com/hw-native-sys/pypto/pull/3001#issuecomment-6269309892)、[隐藏 UB scratch 与存活 Tile 重叠](https://github.com/hw-native-sys/pypto/pull/3001#issuecomment-6269721600)。
- 上游配套方案见 [最新收敛说明](https://github.com/hw-native-sys/pypto/pull/3001#issuecomment-6288341715)：Simpler `97343062`、PyPTO 配套提交 `0093d617`，保持原 PTO-ISA pin；PTO-ISA PR #363 已不再是该方案依赖。
- 当前本地历史功能回归通过，不能替代“PyPTO PR head 已吸收配套修复”的事实核对，也不能推出 SDMA 在完整 CSA 必然有性能收益。
- 这两处是既有 PR 下的 comment 产出，故不算进 issue 新建数量，也没有把它们虚记为本会话新建 PR。

## 7. 同账号但未确认属于三个会话创建的两条

| Issue | 当前状态 | 处理方式 |
| --- | --- | --- |
| [PyPTO #2948](https://github.com/hw-native-sys/pypto/issues/2948) | OPEN | 跨 incore task 的 Scalar/Vector-Cube 并发与 early_start 需求；同账号创建，但三个会话只找到后续引用，没有创建回执，不计入 41 条。 |
| [PyPTO #2949](https://github.com/hw-native-sys/pypto/issues/2949) | OPEN | pipeline 深度一律降至 1 和 MemoryReuse 诊断不足；同上，单列状态，不擅自归属。 |

## 8. 使用与更新说明

- 这是上述核对时间的快照，不是自动同步看板。后续查询应重新读取 GitHub 当前 issue/PR 状态。
- 上游已有明确修复的事项，需按实际合入提交核对本地依赖，再决定替换/删除隔离补丁；本文件不宣称已经完成这一步。
- 待合并项应同时记录 issue、修复 PR、PR head、审查/CI 状态；GitHub CLOSED 不等于对应旧 PR 必然 MERGED，PTOAS #1622 就是实例。
- 功能需求与正确性 bug 分开评估。本地原型可以证明可行性，性能负例也必须保留，不能因编译通过就认领整层收益。
- 本次只生成状态文档；未发布 issue 评论、修改标签、关闭 issue、修改依赖或新增设备测试。提交范围只有本 Markdown。
