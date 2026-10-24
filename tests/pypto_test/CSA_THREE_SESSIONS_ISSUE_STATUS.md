# CSA 三个会话与 HCA 的 issue 状态总表

最后在线核对：**2026-10-24 11:15:06 CST**。时间均为北京时间（UTC+8）。

沿用原文件名，覆盖 `CSA-SPMD`、`CSA-MPMD-TND`（原 CSA-多task）、`CSA-SPMD-TND`（原 CSA-TND），并纳入 `HCA-TND-MPMD` 及 HCA 历史引用。
新建、向既有 issue 补充、仅引用、PR 评论分别统计；跨会话跟进同一 issue 不重复计数。

## 1. 总览

**确认新建55条：16条CLOSED、39条OPEN。另跟进/引用11条既有issue，2条同账号issue的创建会话尚未确认；共列出68个不同issue。**

| 来源会话 | 确认新建 | CLOSED | OPEN |
| --- | ---: | ---: | ---: |
| CSA-SPMD | 40 | 15 | 25 |
| CSA-MPMD-TND | 3 | 1 | 2 |
| CSA-SPMD-TND | 6 | 0 | 6 |
| HCA-TND-MPMD | 6 | 0 | 6 |

| 仓库 | 确认新建 | CLOSED | OPEN |
| --- | ---: | ---: | ---: |
| hw-native-sys/pypto | 40 | 13 | 27 |
| hw-native-sys/PTOAS | 11 | 2 | 9 |
| hw-native-sys/pto-isa | 2 | 1 | 1 |
| hw-native-sys/pypto-lib | 2 | 0 | 2 |
| hw-native-sys/simpler | 0 | 0 | 0 |

相对10月21日旧表的主要变化：

- 新增14条：CSA-SPMD-TND 6条、CSA-MPMD-TND 2条、HCA 6条；原CSA-SPMD的40条全部保留。
- 原开放项新增关闭6条：PyPTO #3032、#3046、#3063、#3083，PTOAS #1652，PTO-ISA #364。
- 当前开放项有9条对应仍OPEN的GitHub修复PR；另有PTOAS #1623迁往GitCode、PyPTO #3087对应回归PR关闭未合并，单独说明。
- 撤回旧表“评论均无非作者回复”的过时结论：#1623、#1649、#364等已有维护者回复。

核对覆盖GitHub作者查询的全部57条结果（`author:nalinaly is:issue created:>=2026-09-01`，未截断），结合三个CSA日志、HCA日志和旧创建回执确认归属：55条确认新建，另2条仅能确认同账号。既有issue另外查询，不能把所有同账号活动都直接归属到某一个会话。
上游issue关闭、修复PR合并、本地SDK吸收和实际场景验证分别记录。本次更新状态文档，没有升级SDK、修改issue或新增设备测试。

## 2. 已关闭的16条新建issue

| Issue | 问题 | 来源 | 关闭时间 | 修复关联与边界 |
| --- | --- | --- | --- | --- |
| [PTOAS #1622](https://github.com/hw-native-sys/PTOAS/issues/1622) | CFG回边并行赋值被串行覆盖 | CSA-SPMD | 10-20 15:15 | 由提交[7efff5cc](https://github.com/hw-native-sys/PTOAS/commit/7efff5ccde929a3cfe59fae3a5b45a1b9e09f8aa)关闭；[PTOAS #1642](https://github.com/hw-native-sys/PTOAS/pull/1642)仍OPEN、未合并。 |
| [PTOAS #1652](https://github.com/hw-native-sys/PTOAS/issues/1652) | Scalar i1常量/运行时转换不一致，BOOL→FP32编译失败 | CSA-SPMD | 10-22 17:55 | 本次新增关闭，由提交[eea8f91a](https://github.com/hw-native-sys/PTOAS/commit/eea8f91ac341597a38389154aec5d33ee4a10977)关闭；[PTOAS #1666](https://github.com/hw-native-sys/PTOAS/pull/1666)仍OPEN、未合并。 |
| [pto-isa #364](https://github.com/hw-native-sys/pto-isa/issues/364) | CPU-SIM TSORT32同分顺序与A3不同 | CSA-SPMD | 10-23 08:59 | 本次新增关闭；维护者指向[GitCode PR #1946](https://gitcode.com/cann/pto-isa/pull/1946)，直连API确认10-21 20:59已MERGED；不能据此宣称本地旧ISA pin已更新。 |
| [pypto #2886](https://github.com/hw-native-sys/pypto/issues/2886) | A3 FP32高精度除法能力说明与诊断 | CSA-MPMD-TND | 09-28 11:14 | 能力澄清关闭，无关闭关联PR/commit；不代表A3 HIGH_PRECISION已实现。stride和确定性评论也保留在该issue。 |
| [pypto #3009](https://github.com/hw-native-sys/pypto/issues/3009) | 自动MIX展开删除while条件使用的iter_arg | CSA-SPMD | 10-19 08:27 | [pypto #3013](https://github.com/hw-native-sys/pypto/pull/3013) 已合并（10-19 08:27）。 |
| [pypto #3014](https://github.com/hw-native-sys/pypto/issues/3014) | Group成员InOut返回别名没有重绑定 | CSA-SPMD | 10-21 11:25 | [pypto #3018](https://github.com/hw-native-sys/pypto/pull/3018) 已合并（10-21 11:25）。 |
| [pypto #3015](https://github.com/hw-native-sys/pypto/issues/3015) | 分支后reshape使用不支配使用点的SSA | CSA-SPMD | 10-19 08:26 | [pypto #3019](https://github.com/hw-native-sys/pypto/pull/3019) 已合并（10-19 08:26）。 |
| [pypto #3016](https://github.com/hw-native-sys/pypto/issues/3016) | 条件gather同存储phi生成非法Mat→Mat | CSA-SPMD | 10-19 18:40 | [pypto #3029](https://github.com/hw-native-sys/pypto/pull/3029) 已合并（10-19 18:40）。 |
| [pypto #3017](https://github.com/hw-native-sys/pypto/issues/3017) | 行缓冲承载列向量loop-carry，TMOV形状不匹配 | CSA-SPMD | 10-20 19:08 | [pypto #3053](https://github.com/hw-native-sys/pypto/pull/3053) 已合并（10-20 19:08）。 |
| [pypto #3026](https://github.com/hw-native-sys/pypto/issues/3026) | 负数FloorMod与IR简化器语义不一致 | CSA-SPMD | 10-19 09:49 | [pypto #3030](https://github.com/hw-native-sys/pypto/pull/3030) 已合并（10-19 09:49）。 |
| [pypto #3027](https://github.com/hw-native-sys/pypto/issues/3027) | 原地重绑定后下游view的MemRef没有传播 | CSA-SPMD | 10-19 14:22 | [pypto #3031](https://github.com/hw-native-sys/pypto/pull/3031) 已合并（10-19 14:22）。 |
| [pypto #3032](https://github.com/hw-native-sys/pypto/issues/3032) | 常量SSA切片起点不折叠，阻断合法内存复用 | CSA-SPMD | 10-23 08:42 | [pypto #3055](https://github.com/hw-native-sys/pypto/pull/3055) 已合并（10-23 08:42）。本次新增关闭。 |
| [pypto #3041](https://github.com/hw-native-sys/pypto/issues/3041) | 嵌套inline临时变量被改成非法自动名称 | CSA-SPMD | 10-21 15:51 | [pypto #3092](https://github.com/hw-native-sys/pypto/pull/3092) 已合并（10-21 15:51）。 |
| [pypto #3046](https://github.com/hw-native-sys/pypto/issues/3046) | FlattenTileNdTo2D未同步重写动态类型引用 | CSA-SPMD | 10-23 12:36 | [pypto #3090](https://github.com/hw-native-sys/pypto/pull/3090) 已合并（10-23 12:36）。本次新增关闭。 |
| [pypto #3063](https://github.com/hw-native-sys/pypto/issues/3063) | Scalar BOOL→INT32符号扩展导致true为-1 | CSA-SPMD | 10-22 10:28 | [pypto #3095](https://github.com/hw-native-sys/pypto/pull/3095) 已合并（10-22 10:28）。本次新增关闭。 |
| [pypto #3083](https://github.com/hw-native-sys/pypto/issues/3083) | libbacktrace裸make -j忽略外层并行预算 | CSA-SPMD | 10-23 09:30 | [pypto #3103](https://github.com/hw-native-sys/pypto/pull/3103) 已合并（10-23 09:30）。本次新增关闭。 |

## 3. OPEN且有在审GitHub修复PR的9条

以下issue全部仍OPEN，PR全部未合并。审查列明确区分机器人批准与人工批准。CI按当前head、同一App的同名check取最新一次，再结合commit status；跳过项单列为通过中的跳过，不将旧失败重跑记录混入当前汇总。该汇总不代替分支保护规则或本地设备验收。

| Issue | 问题 | 修复PR | PR head | 当前head审查 | CI |
| --- | --- | --- | --- | --- | --- |
| [PTOAS #1624](https://github.com/hw-native-sys/PTOAS/issues/1624) | GM partition_view丢失列范围，产生多余FIX barrier | [PTOAS #1670](https://github.com/hw-native-sys/PTOAS/pull/1670) | `394c9bcc16` | 无当前head批准 | [有失败/取消（1项）](https://github.com/hw-native-sys/PTOAS/actions/runs/39779087006/job/119825412571) |
| [pypto #3028](https://github.com/hw-native-sys/pypto/issues/3028) | 内存别名遍历遗漏WhileStmt | [pypto #3054](https://github.com/hw-native-sys/pypto/pull/3054) | `1af4148070` | 机器人批准 | 通过（含跳过项） |
| [pypto #3033](https://github.com/hw-native-sys/pypto/issues/3033) | 动态维度算术导致reshape metadata丢失 | [pypto #3061](https://github.com/hw-native-sys/pypto/pull/3061) | `63b30cdf0a` | 无当前head批准 | 通过（含跳过项） |
| [pypto #3034](https://github.com/hw-native-sys/pypto/issues/3034) | JIT inline特化丢弃显式Tile参数注解 | [pypto #3070](https://github.com/hw-native-sys/pypto/pull/3070) | `4c3446ea0e` | 机器人批准 | 通过（含跳过项） |
| [pypto #3035](https://github.com/hw-native-sys/pypto/issues/3035) | 互斥分支别名与访问顺序造成内存伪干涉 | [pypto #3071](https://github.com/hw-native-sys/pypto/pull/3071) | `51b27788ef` | 机器人批准 | [有失败/取消（3项）](https://github.com/hw-native-sys/pypto/actions/runs/39530372180/job/118996170513) |
| [pypto #3036](https://github.com/hw-native-sys/pypto/issues/3036) | 支持AIC Acc直接写回NZ GM | [pypto #3131](https://github.com/hw-native-sys/pypto/pull/3131) | `d4d6f64331` | 机器人批准 | [有失败/取消（1项）](https://github.com/hw-native-sys/pypto/actions/runs/39937322215/job/120359988322) |
| [pypto #3039](https://github.com/hw-native-sys/pypto/issues/3039) | current/next预取轮换生成不支持的Mat复制 | [pypto #3133](https://github.com/hw-native-sys/pypto/pull/3133) | `ac26c4eddd` | 无当前head批准 | 通过（含跳过项） |
| [pypto #3062](https://github.com/hw-native-sys/pypto/issues/3062) | 合法NZ展平跨InCore helper丢失TensorMeta | [pypto #3099](https://github.com/hw-native-sys/pypto/pull/3099) | `8fe2038dd3` | 机器人批准 | 通过（含跳过项） |
| [pypto #3079](https://github.com/hw-native-sys/pypto/issues/3079) | gather_compare单行count形成未对齐的4B UB Tile | [pypto #3126](https://github.com/hw-native-sys/pypto/pull/3126) | `236f4006fb` | 机器人批准 | [有失败/取消（1项）](https://github.com/hw-native-sys/pypto/actions/runs/40052967942/job/120750540061) |

- #3054的head已更新为`1af4148070`，当前检查通过；旧表中的失败和等待审查已过时。
- #3071仍有3项失败，#3126有unit-tests失败，#3131有clang-tidy失败。这里未重新跑CI，也不把失败原因统一归为设备数值问题。
- #3033新增CSA-MPMD-TND的[动态reshape别名复现](https://github.com/hw-native-sys/pypto/issues/3033#issuecomment-6369484280)；仍按原CSA-SPMD创建计1条。#3061也被#3062引用，但#3062明确对应的修复PR是#3099，不能重复算成两个已完成修复。
- #3036的Acc→NZ写回是能力扩展；原完整CSA负性能结果仍有效，PR存在或设备功能通过都不代表必然加速。

### 3.1. OPEN issue的特殊关联

| Issue | 当前进展 | 本次判断 |
| --- | --- | --- |
| [PTOAS #1623](https://github.com/hw-native-sys/PTOAS/issues/1623) | [PTOAS #1664](https://github.com/hw-native-sys/PTOAS/pull/1664) CLOSED、未合并；作者明确迁往[GitCode !333](https://gitcode.com/cann/pto-as/merge_requests/333) | GitCode无代理直连API和pull页面返回403，未确认MR当前合并状态。不能沿用“无修复PR”，也不能把GitHub关闭当成代码已合入。 |
| [pypto #3087](https://github.com/hw-native-sys/pypto/issues/3087) | [pypto #3125](https://github.com/hw-native-sys/pypto/pull/3125) CLOSED、未合并；[作者说明](https://github.com/hw-native-sys/pypto/pull/3125#issuecomment-6359865161)称干净main `54f01da47`已能编译原例，PR只补回归覆盖 | Issue仍OPEN。作者的CPU验证是新的证据；本地定制旧版本与干净main不同，不能继续一概说main缺少该能力，也未在本次补跑设备验证。 |

## 4. 其余OPEN的新建issue：CSA 22条、HCA 6条

以下未找到明确仍在审的关联修复PR；不排除未关联issue的开发分支。本地绕行或补丁不改写GitHub OPEN状态。

### 4.1. CSA：PyPTO 12条

| Issue | 来源 | 问题 | 当前进展与限制 |
| --- | --- | --- | --- |
| [pypto #3020](https://github.com/hw-native-sys/pypto/issues/3020) | CSA-SPMD | 同一InCore/SPMD内普通GM信号的点对点notify/wait | 能力需求仍开放。本地信号/缓存协议不能等同上游正式API已发布。 |
| [pypto #3037](https://github.com/hw-native-sys/pypto/issues/3037) | CSA-SPMD | 死Tile经loop-carry/if-phi闭环仍占片上内存 | 已有本地DCE方案与验证建议；HCA #3136是更小的无循环/phi反例，不假定二者根因相同。 |
| [pypto #3038](https://github.com/hw-native-sys/pypto/issues/3038) | CSA-SPMD | 显式strided GM view，支持多段ND2NZ加载 | 已有隔离stride原型和地址合同；非紧密GM视图与放宽任意非连续根Tensor入口是两回事。 |
| [pypto #3042](https://github.com/hw-native-sys/pypto/issues/3042) | CSA-SPMD | InCore GM子视图后显式load及动态pitch两行DMA | 已有gm_view原型及CPU/A3地址验证。PR #3090修的是#3046，不代表本功能已完整实现。 |
| [pypto #3049](https://github.com/hw-native-sys/pypto/issues/3049) | CSA-SPMD | 多个存活Mat条件选择生成非法Mat→Mat | 多份仍存活Mat的条件选择不能直接套用#3016的同存储phi修复。 |
| [pypto #3085](https://github.com/hw-native-sys/pypto/issues/3085) | CSA-SPMD | 保序有界value/index/count三输出Tile压缩 | 与PTOAS #1650配套；必须保留value/index配对、真实count、容量和同分顺序。 |
| [pypto #3114](https://github.com/hw-native-sys/pypto/issues/3114) | CSA-SPMD-TND | Simplify在裁剪不可达分支前折叠整数除法，触发除零 | 新增；来自uniform-TND几何探测的纯CPU最小复现。不可达width=0分支仍被提前求值；未归因于设备数据。 |
| [pypto #3116](https://github.com/hw-native-sys/pypto/issues/3116) | CSA-SPMD-TND | CachePolicy.BYPASS未接通PTOAS runtime L2 offset | 新增；default/BYPASS生成完整C++相同。旧PTOAS #1537关闭不代表PyPTO端offset已接通。 |
| [pypto #3132](https://github.com/hw-native-sys/pypto/issues/3132) | CSA-SPMD-TND | GEMV写Acc列池后assemble生成不支持的Acc→Acc TMOV | 新增；已补本地别名/物理布局修复及编译、设备验证。完整B16池化GEMV性能未受益，不能把可编译写成优化成功。 |
| [pypto #3134](https://github.com/hw-native-sys/pypto/issues/3134) | CSA-SPMD-TND | Matmul/MatmulAcc缺少与FIX配对的AccPhase接口 | 新增能力需求；GEMV已有AccPhase，Matmul/MatmulAcc缺少对应前端入口；没有声称默认Matmul算错。 |
| [pypto #3137](https://github.com/hw-native-sys/pypto/issues/3137) | CSA-MPMD-TND | 自动SPMD包装无法编译复合尺寸内部Tensor | 新增；显式incore参数边界可绕过自动wrapper限制，仍由Simpler申请内部Tensor；没有退回外部工作区或新增MPMD软同步。 |
| [pypto #3138](https://github.com/hw-native-sys/pypto/issues/3138) | CSA-MPMD-TND | Group ABI规范化遗漏动态维度重绑定 | 新增；公共PyPTO本地d44925c1已修复，相关CPU回归及两种TND单卡通过。GitHub仍OPEN，尚无修复PR。 |

### 4.2. CSA：PTOAS 7条

| Issue | 来源 | 问题 | 当前进展与限制 |
| --- | --- | --- | --- |
| [PTOAS #1625](https://github.com/hw-native-sys/PTOAS/issues/1625) | CSA-SPMD | TEXTRACT读取范围过宽，L1双缓冲被串行化 | TEXTRACT需识别真正不相交槽，同时保留重叠访问保护；当前无明确关联修复PR。 |
| [PTOAS #1647](https://github.com/hw-native-sys/PTOAS/issues/1647) | CSA-SPMD | A3 compare-form TGATHER模板/参数与ISA ABI不匹配 | 生成调用与ISA ABI不匹配；前端#3079的count对齐修复不能替代后端端到端验证。 |
| [PTOAS #1648](https://github.com/hw-native-sys/PTOAS/issues/1648) | CSA-SPMD | LLVM 19 EmitC丢失嵌套位运算括号 | 位运算与布尔表达式括号改变条件语义，属于代码生成正确性问题。 |
| [PTOAS #1649](https://github.com/hw-native-sys/PTOAS/issues/1649) | CSA-SPMD | ResolveBufferSelect扩宽TGATHER/TMOV子视图 | 维护者and0d0于10-22回复“认领”；包含TGATHER读侧及TMOV写侧变体，仍OPEN。 |
| [PTOAS #1650](https://github.com/hw-native-sys/PTOAS/issues/1650) | CSA-SPMD | 共享掩码的value/index成对压缩及真实count | 已有本地成对压缩/满容量直写原型和设备验证；完整CSA收益不稳定，未认定为正式优化。 |
| [PTOAS #1674](https://github.com/hw-native-sys/PTOAS/issues/1674) | CSA-SPMD-TND | MGATHER Row内部等待阻止独立Vector重叠 | 新增性能需求；现MGATHER宏的同步对示例是正确的，期望允许独立Vector重叠，不能删掉真实完成依赖。 |
| [PTOAS #1680](https://github.com/hw-native-sys/PTOAS/issues/1680) | CSA-SPMD-TND | Mat子视图物理尺寸被扩宽，Mat→Right TMOV校验失败 | 新增；独立PTO复现物理行数256被扩成父池384。应用侧显式TEXTRACT绕行，公共PTOAS未修改。 |

### 4.3. CSA：PTO-ISA / PyPTO-lib 3条

| Issue | 来源 | 问题 | 当前进展与限制 |
| --- | --- | --- | --- |
| [pto-isa #352](https://github.com/hw-native-sys/pto-isa/issues/352) | CSA-SPMD | A3 MGATHER ZERO清零与ND2NZ写缺少顺序约束 | 清零与ND2NZ写入的顺序问题仍OPEN，未见明确关闭关联修复。 |
| [pypto-lib #1486](https://github.com/hw-native-sys/pypto-lib/issues/1486) | CSA-SPMD | incore profiling的timeout分钟/秒及外层时限不一致 | msprof分钟参数与脚本秒级描述/外层时限不一致，仍无明确修复关联。 |
| [pypto-lib #1487](https://github.com/hw-native-sys/pypto-lib/issues/1487) | CSA-SPMD | A3模拟器SYS_CNT恒零，计时退避无法退出 | 同二进制真机通过而模型SYS_CNT不推进；模拟专用退出上限不能作为真机同步正确性证据。 |

### 4.4. HCA：PyPTO 6条

六条均来自HCA-TND-MPMD，已在`HCA_TND_MPMD_OPTIMIZATION_LOG.md`确认创建；当前全部OPEN。另向#2717补充复现，不重复新建，见第5节。

| Issue | 来源 | 问题 | 当前进展与限制 |
| --- | --- | --- | --- |
| [pypto #3105](https://github.com/hw-native-sys/pypto/issues/3105) | HCA-TND-MPMD | inline分档helper的正步长在outlining后丢失，合法NZ切片被拒 | 独立CPU对照：捕获局部workers=24失败，直接常量步长24通过；未修改共享SDK。 |
| [pypto #3107](https://github.com/hw-native-sys/pypto/issues/3107) | HCA-TND-MPMD | scope内返回Tensor别名在外消费，生成越C++作用域的SSA | scope外消费失败、scope内对照完整编译通过；不是将局部TaskId直接逃逸出scope。 |
| [pypto #3113](https://github.com/hw-native-sys/pypto/issues/3113) | HCA-TND-MPMD | 热eager重复JIT特化分析，Host提交约7ms | 建议显式prepared callable；区别于#2881进程冷查找和#2629 DP8 TaskArgs打包。当前无实现PR。 |
| [pypto #3115](https://github.com/hw-native-sys/pypto/issues/3115) | HCA-TND-MPMD | constexpr分支含manual_scope/parallel，yield合流位置错误 | 首次Simplify报control-flow body tail must be YieldStmt；已附独立复现、IR及建议。 |
| [pypto #3130](https://github.com/hw-native-sys/pypto/issues/3130) | HCA-TND-MPMD | LowerPipelineLoops克隆外层Scalar IterArg后形成FREE_VAR | 跨query两级FIFO触发；同body改普通range可编译。纯CPU复现，不宣称设备数值错误。 |
| [pypto #3136](https://github.com/hw-native-sys/pypto/issues/3136) | HCA-TND-MPMD | 无循环/phi的未使用Tile初始化仍生成TCI/TCVT/TEXPANDS | unused比clean多出TCI/TCVT/TEXPANDS，真实使用反例保留；只证明生成代码，不直接换算机器码/微秒损失。 |

## 5. 补充或引用的11条既有issue（不计入新建55条）

| Issue | GitHub状态 | 跟进/引用会话 | 修复与覆盖边界 |
| --- | --- | --- | --- |
| [PTOAS #1587](https://github.com/hw-native-sys/PTOAS/issues/1587) | OPEN | CSA-SPMD | 维护者指向[GitCode PR #290](https://gitcode.com/cann/pto-as/pull/290)；CSA追加循环回边不相交行访问变体。直连页面403，本次未确认GitCode合并状态或该变体覆盖。 |
| [PTOAS #1620](https://github.com/hw-native-sys/PTOAS/issues/1620) | CLOSED | CSA-SPMD | 已由[6f5e16af](https://github.com/hw-native-sys/PTOAS/commit/6f5e16afaaacd2ac1324e737235b96f33f12e08c)关闭，版本升至0.69；本会话追加过MTE2→MTE3和动态L1基址变体。关闭不自动等于所有变体设备验收。 |
| [pypto #2951](https://github.com/hw-native-sys/pypto/issues/2951) | CLOSED | CSA-SPMD | 原问题随[pypto #3004](https://github.com/hw-native-sys/pypto/pull/3004)合并关闭；后来追加的双动态维度view变体仍需按准确版本确认，不能由原issue关闭推断其全部解决。 |
| [simpler #2331](https://github.com/hw-native-sys/simpler/issues/2331) | CLOSED | CSA-SPMD | AICore头文件依赖追踪，原作者ChaoWao；[simpler #2340](https://github.com/hw-native-sys/simpler/pull/2340)已合并。Simpler参数容量扩展保留本地，不虚列新增issue。 |
| [PTOAS #1537](https://github.com/hw-native-sys/PTOAS/issues/1537) | CLOSED | CSA-SPMD / CSA-SPMD-TND | 维护者10-22称latest release已修复并关闭。此项是旧runtime offset风险，不代替PyPTO #3116的前端接通需求。 |
| [pypto #2984](https://github.com/hw-native-sys/pypto/issues/2984) | CLOSED | CSA-MPMD-TND | [pypto #3000](https://github.com/hw-native-sys/pypto/pull/3000)已合并；And/Or/Xor/Not官方修复移植到本地26e94e05，已进入公共工具链。 |
| [pypto #2936](https://github.com/hw-native-sys/pypto/issues/2936) | CLOSED | CSA-MPMD-TND | [pypto #2937](https://github.com/hw-native-sys/pypto/pull/2937)已合并；内联动态类型/DeepClone相关修复已选择性移植到本地cda77ec0，非整PR所有功能照搬。 |
| [pypto #2717](https://github.com/hw-native-sys/pypto/issues/2717) | CLOSED | HCA | [pypto #3078](https://github.com/hw-native-sys/pypto/pull/3078)已合并，原问题关闭；HCA在[10-23追加H16→H32环列assemble复现](https://github.com/hw-native-sys/pypto/issues/2717#issuecomment-6363110842)。公共定制版本仍用6×16环绕行，新变体不能仅凭CLOSED判为已通过。 |
| [pypto #1908](https://github.com/hw-native-sys/pypto/issues/1908) | OPEN | CSA-SPMD / HCA | 旧allocator的异形L0区间复用需求；HCA历史分析引用，CSA追加过48+12/32+32KiB容量反例。仍OPEN，不将旧版本结论直接套到当前全部planner。 |
| [pypto #2881](https://github.com/hw-native-sys/pypto/issues/2881) | CLOSED | HCA引用 | 进程冷JIT查找性能问题，已关闭；#3113报告的是同进程热eager重复特化，范围不同。 |
| [pypto #2629](https://github.com/hw-native-sys/pypto/issues/2629) | OPEN | HCA引用 | DP8 TaskArgs不变量打包性能需求仍OPEN；#3113为更早的Python/JIT分析阶段，不合并计数。 |

## 6. SDMA既有PR评论跟进（不是新增issue）

| PR | 状态 | 当前head | CI | 说明 |
| --- | --- | --- | --- | --- |
| [pypto #3001](https://github.com/hw-native-sys/pypto/pull/3001) | OPEN，未合并 | `ed612fbdb7` | 通过（含跳过项） | head仍未包含评论中的配套提交0093d617。 |
| [simpler #2583](https://github.com/hw-native-sys/simpler/pull/2583) | OPEN，未合并 | `97343062d0` | [有失败/取消（1项）](https://github.com/hw-native-sys/simpler/actions/runs/39403313963/job/118563103423) | head包含驻留AIV独立通道修复；不能据此声称PyPTO一侧也已同步。 |

两PR均针对`feat/kernel-mode-integration-test`，不是main已完成全链路。CSA-SPMD的[通道绑定反馈](https://github.com/hw-native-sys/pypto/pull/3001#issuecomment-6269202617)、[公开复现](https://github.com/hw-native-sys/pypto/pull/3001#issuecomment-6269309892)、[隐藏UB scratch覆盖](https://github.com/hw-native-sys/pypto/pull/3001#issuecomment-6269721600)和[配套方案](https://github.com/hw-native-sys/pypto/pull/3001#issuecomment-6288341715)继续保留。PTO-ISA PR #363已不再是该配套方案依赖；历史功能验证不推出完整CSA必有收益。

## 7. 同账号、创建会话尚未确认的2条

| Issue | 状态 | 归属说明 |
| --- | --- | --- |
| [pypto #2948](https://github.com/hw-native-sys/pypto/issues/2948) | OPEN | 跨incore task Scalar/Vector-Cube重叠与early_start；旧审计未找到这三个CSA会话的创建回执，HCA日志也未提供创建依据，保持单列。 |
| [pypto #2949](https://github.com/hw-native-sys/pypto/issues/2949) | OPEN | pipeline深度降为1及MemoryReuse诊断不足；同样不因同账号或被引用而擅自认领。 |

## 8. 本地公共工具链与更新边界

当前本地公共清单为`csa-shared-20261024-group-dims`，状态`validated`：PyPTO `d44925c17d9ff7d7fa8e5067e41c4b4f80bf3063`，PTOAS `914863ed3d0eb3e92a4433a4a706a86cabf4c91f`，Simpler `57e499238fd4ee3548947ad860be456aaeeb16b5`。该清单是本地CSA共用环境的事实，不代表所有GitHub主线修复都已进入该版本，也不改变历史HCA/SPMD结果所记录的SDK。

- #2984官方逻辑表达式修复、#2936相关内联类型修复以及#3138本地Group动态尺寸修复已进入上述公共版本。#3137保留显式incore边界绕行；#3033保留从描述符直接取维度的绕行。
- #3138的本地通过与GitHub OPEN并存；其修复尚无上游PR。此次新关闭的其他issue，不因状态变更就删除本地补丁或自动升级SDK。
- 本地内部工作区版两种TND的单卡图重放已通过；整模型B16/B24实际是eager PTO重新派发，模型级graph replay尚未通过。编译问题解决不能替代模型图重放验收。
- GitCode均使用无代理直连：PTO-ISA #1946由API确认MERGED；PTOAS !333和#290无法取得可核实状态，明确保留未知。
- `csa_three_session_issue_status_20261021/STATUS.json`和CSV保留为10月21日历史快照，不是本次实时状态。当前状态以本Markdown的核对时间和逐条链接为准。
- 本次只更新这份原Markdown；不修改issue状态、标签或评论，不新建issue/PR，不更换依赖、不重新跑设备测试。
