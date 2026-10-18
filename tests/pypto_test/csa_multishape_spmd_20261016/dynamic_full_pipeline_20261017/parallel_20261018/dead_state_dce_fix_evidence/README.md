# Attention 区间策略：修复死状态消除后的验证

2026-10-18。目标是让本地统计仅在有收益的工作量区间启用，其余区间真正
恢复原路径。七档是回归点，不是支持白名单。七档与两个边界均已完成，
最终仅默认启用长历史 G6 且 T<=96；短历史收益不稳定，保留原 GM 路径。

## 源码和独立依赖

- CSA 候选：`../attention_local_stats_policy_evidence/source_candidate_v1/`，
  完整 36 模块，长历史 G6 且 T<=96、短历史 G2 且 T>96 使用本地统计；
  其余使用原 GM 统计。沿既有 Score/G 与 O_A 行区间，不枚举 B4/B8 等测点。
- CSA 对照：`../attention_local_stats_evidence/source_reference/`，正式
  `25de3f32` 的完整快照；与 policy evidence 的 reference 逐文件相同。
- PyPTO：`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-memory-reuse-20261018`，
  在已有独立树上修复 `dead_code_elimination.cpp`，没有修改共享安装。
- 本轮组件修改与回归完整保存在 `compiler_source_v1/`、
  `suggested_fix_v1.patch`。Simpler 沿用独立扩容树，PTOAS 沿用 0.66，均未修改。

## 修复内容与约束

原逻辑把 loop carry 与 if phi 的相互传递也视为读者。关闭 constexpr 后，
768B 统计环虽然没有计算消费者，仍留在循环闭环中并改变 UB 布局。

本轮从运算、条件、返回、描述符和调度用途出发求可达性；纯别名、初值、
循环 yield、if yield 只是依赖边。工作列表收敛后一次性删除无消费者的通道、
别名和原始未初始化 tile.create。保留真实 Tile 读取/写入、零次迭代初值、
while 条件、TaskId 通道、Submit deps/predicate、属性和类型里的表达式。
不对其它调用推断纯度，通信域内部继续保守处理。

## CPU 结果

- 独立构建通过。新增 7 项回归与已有 Simplify、任务依赖、手工 phase fence
  合计 **179 passed**。`tests_v2.log` 的 3 个警告来自导入隔离插件被 sitecustomize
  提前载入，不是编译器警告或失败。
- 原最小复现完整编译后 `ghost_retained=false`，无 ghost/unused 分配。
- long_on、long_off、short_on、short_off、small_on 五类完整符号编译通过；
  每类保持 58 动态轴、130 根入参和一个 MIX。没有传样例 B/T 固化形状。
- 同一修复后编译器重新编译两类原关闭路径，和区间候选生成的 C++ 一致；
  比较仅忽略整行注释，不删声明、地址、事件或计算语句。两类关闭路径均无
  stats_cache 引用。见 `CPU_VERIFICATION.json`。
- 对旧编译器生成物并不逐字相同：修复也清除了其它阶段的死分配。
  `*_old_measured.diff` 原样保留，不能把历史设备结果冒充新编译器已验证。

失败过程也保留：首轮新增测试包含 3 个预期死状态失败和 1 个本地 SPMD
用例缺 block_idx 的编写错误；修正用例后单独确认原编译器的 core_num
对照通过。首轮组件修复回归 177 项通过，2 个子进程误导入共享安装失败；
加入仅测试使用的 sitecustomize 隔离后 179 项全部通过。没有放宽断言。

## 上游反馈

已向现有 PyPTO #3037 补充完整建议 patch、7 项回归、复杂度和保留条件：
https://github.com/hw-native-sys/pypto/issues/3037#issuecomment-6249851423

正文已回读核对，见 `ISSUE_UPDATE_VERIFIED.json`。没有提 PR、没有加标签。
评论发布时明确设备回归仍在进行，不把 CPU 编译当成精度或性能验证。

## 设备验证口径

`DEVICE_TASKS.json` 保存准确命令和任务号。正式 W8A8、layer4、S6，沿用
原多 task / 原动态 / 区间候选三方同进程同根 Tensor 对照，完整 L1 五窗口
中位数；保留输出、Top-K、score、状态、保护区、代际和图重放检查。
不测 Native，不改变数值阈值，不把插桩阶段时间当完整性能。

九个任务均 exit0，完整结果见 `DEVICE_RESULTS.md/json`。长历史 B4/B8/
B15/B16 相对原动态降低 6.01%/2.63%/1.31%/1.23%；短历史 T>96 的两点
分别为 +0.31%/-0.24%，未默认采用。B24/B28/短B16原本就是关闭路径，
实例之间的中位差不能解释为本地统计的收益。

`FORMAL_ADOPTION.json` 核对正式35个数学/编排模块与被测源AST一致，Host
只收窄默认区间。额外CPU编译证明short大T关闭代码与此次已测原动态C++
一致，记录在 `SHORT_LARGE_DISABLED_EQUIVALENCE.json`。正式完整36模块
冻结于 `source_adopted_v1/`。本轮没有再测S1/S5、混合Q或连续轨迹。

PyPTO本轮四文件已独立中文commit，见 `COMPILER_COMMIT.json`；其它依赖
工作树改动未夹带，未push。优化过程和七档仍未达到30%的事实全部保留。
用户再次要求抓大放小，本轮在此收口，不追加小收益候选采样。
