# 无同构历史 anchor 时，从源码构造 CSA 单 SPMD 模板

本目录只做 CPU 绑定、依赖与源码构造。没有运行 NPU，没有修改原静态归档、生产代码、
PyPTO/Simpler 依赖或共享 driver。性能目标沿用上级 `GOAL_AND_CONSTRAINTS.md`，
本目录的成本模型与逻辑检查不构成性能达标证据。

## 结论与接口

可以不依赖新增硬件 trace 构造缺失分支。七个历史档位的 task DAG 可以从原编排中的
显式 TaskRef、后期 IR 的 Tensor 读写效应、实际 Tensor view 与手动作用域合同恢复。
历史 trace 对功能构造并非必需；它仍可以提供离线分核所需的初始成本参考。

```python
from anchor_gap_design_v1.source_schedule import build_source_schedule

schedule, audit = build_source_schedule(
    bindings,       # bind_orchestration.bind 的本轮结果
    bodies,         # shape_specialization.catalog()，带归档源文件的绝对路径
    anchors,        # name -> 历史 *_prepared.json 字典，仅用于同名核成本
    policy="earliest_ready",
)
anchor_for_later_rebalance = audit["cost_seed_anchor"]
```

`schedule` 含 `tasks/call_bindings/per_core_sequences/root_arguments/temporary_allocations`
及已有 prepare/generate/validate 所需字段。它已经调用现有 `balanced_schedule.rebalance`
按本轮 block 数分核，并调用 `rebuild_waits` 生成核间进度等待。后续若需再次分核，可将
`audit["cost_seed_anchor"]` 作为 `rebalance(schedule, anchor)` 的第二参数。
不要把 schedule 本身当作历史成本 anchor：rebalance 已把成员 `end` 转为模型字段。

原七档继续使用原入口是合理的最小集成方式。只有明确允许源码生成、且 `choose_anchor`
无同构候选时，才调用本模块。不能宽泛捕获其他绑定异常后静默转入此路径。

## 为什么 bind 的参数注解本身不够

`bind_orchestration.py` 执行的是 `10_after_OutlineIncoreScopes.py`。该阶段某些写参数仍标为
普通 `pl.Tensor`，例如 `hc_widen_rms` 的 `inv_rms`。仅把普通 Tensor 当只读会漏掉依赖。
`generated/core_body_catalog.json` 从 `25_after_ExpandMixedKernel.py` 导出的参数已经记录
Out/InOut 效应；对 MIX 任务，需要将 AIC 与 AIV 的参数效应并集。

最终模块还将逻辑接口与后期效应做保守并集，并对已有 allocation 的写保守加入读依赖，
覆盖 WAW。自动作用域显式检查三种顺序：

- RAW：生产者写后，消费者读取对应重叠 view。
- WAW：已有 view 的先前写完成后才允许后续写。
- WAR：先前读者完成后，才允许后续写重叠 view。

不能把此算法称为历史 runtime ArgType 的逐项重放。每档约 292 个 Tensor 参数中，
有 12 项后期 Out、历史 trace 为 INOUT；它们不改变本轮七档的最终偏序。
正式依据是完整偏序对照及保守效应合同，而非参数标签恰好相同。

`SourceContracts` 对照实际后期 IR 与 catalog 的参数名及注解，核对每个本轮调用与
逻辑源函数的参数顺序/注解、物理核参数覆盖、scope 与 task 选项。未知调度选项、未知
角色组合、未绑定物理参数、非纯 TaskRef 的 dummy 等不作猜测。

## TaskRef、dummy、手动作用域和 manual_dep

四个 dependency-only 任务不执行数学。其显式依赖递归展开到真实 compute 前驱；空 dummy
不会凭空产生屏障。嵌套列表形式的 deps（最终 8 组输出投影的列表）也递归展开。
TaskRef 使用本轮 ordinal，不能沿用历史 runtime task_id。

Simpler A3 当前 `dep_compute.h` 中：

- `compute_task_fanin` 在 manual scope 直接退出自动查找。
- `register_task_outputs` 在 manual scope 也不注册自动输出。
- `manual_dep` 跳过 modifier overlap map，保留原接口的显式关系。

本构造器保留上述手动合同，不能把八个输出投影组的整个共享 allocation 包围区间当作
真实写集合，给它们添加全串行链。原源码的显式组内链、QK 前驱和最终汇合依赖保留。
新静态工作区在入口前分配并在整个执行期间存活，不需要把历史 allocator/retain-only
任务当作 AICore 数学阶段；如果以后引入工作区复用，必须另行证明生命周期。

## 七档对照结果

`seven_anchor_audit.json` 是最初只读依赖复原证据；最终的 RAW/WAW/WAR 模式证据见
`builder_check_v3_war/summary.json` 与各案例的 `source_audit.json`。

| 项目 | 每个原七档的结果 |
|---|---:|
| 数学任务 | 66 |
| 原图直接边 | 98 |
| 重建图直接边 | 101 |
| 原图与重建图的可达前后关系 | 均为 1377 |
| 缺失偏序 | 0 |
| 新增偏序 | 0 |
| 仅显式 TaskRef 的可达关系 | 222 |

多出的三条直接边均已被原图的传递关系覆盖：17→44、26→28、29→39。
WAR 识别出 17→41、20→21、24→25、26→27、29→30，全部已在原偏序中。
66 个任务并没有排成总序；2145 对可能的先后关系中仅有 1377 对受约束。

旧全图签名拒绝 B17～21，并非数学体缺失。例如长历史 B17 的前段使用 B16 所用的核体，
输出投影使用 B24 所用的核体，最终 HC_post 的 sync_start 选项也随原数学分支变化。
新模块从实际编排生成这些组合，不按名称模糊匹配后套旧参数或等待表。

## 缺失分支与 schema 验证

`missing_branch_audit.json` 检查原审计中 10 个缺少 anchor 的签名代表。

- 长/中历史的缺口仍有 66 个数学任务，可恢复与历史图相同的 ordinal 偏序。
- 短历史 `<8192` 的 Vector Indexer 分支有 65 个数学任务、1333 对可达关系。
  其中 `indexer_score_topk_leaf`、`indexer_topk_single_leaf_publish` 不在原七份 trace 中，
  但完整核体与接口已经存在于冻结 catalog；不能用长历史 Indexer 的核体替代。
- 长历史 B1～3 的特定 Indexer 核也已经在 catalog，尽管不在原七份 trace。

`builder_check_v2` 覆盖七个历史档位与十个缺失签名的完整 schema、分核覆盖、MIX 配对、
工作区边界、等待表、三代际随机推进模型和源码生成。v2 已加载加入 WAR 之前的模块。
`builder_check_v3_war` 使用最终 WAR 版本重建；对已生成案例逐项比较每核执行表、等待、
调用绑定和工作区，不相同就失败，不把旧源码直接当作新版本证据。另补 B17/H65536、
B21/H131072、B3/H16384，覆盖中间长度和缺口边界。最终 v2 的 17 个案例、v3 的 20 个案例全部完成，均为 CPU schema/layout/protocol/source PASS。七个参考档位新增 WAR 后偏序仍无缺失或额外关系。

这些不是 PyPTO lowering 或设备 PASS。CPU 进度模型假定既有发布/获取可见性正确；实际
Scalar DCache、DCCI/DSB、原核内 FFTS、重复重放、精度和性能仍需按现有真机流程验证。

## 成本和性能边界

每个新 task/slot 的初始成本优先从七份历史 trace 中选择同名核、最近 T 的样本中位数。
这些是跨形状调度估计；不会标成同规格测量。完全未测的核使用显式标注的初始模型权重
10.0，可通过 `model_cost_overrides` 指定其他正有限权重，仍不代表实际 μs。

现有 balanced 调度器按这些权重选择并行执行顺序；未添加阶段级全核屏障，也没有全串行
fallback。新分支最终需要测量并调优，不能将“可以生成源码”写成“该区间性能良好”。

本模块使用 bind 中的 allocation/view 身份；它不额外发现调用方不同根 Tensor 之间的
运行时别名。入口仍须遵守已有外部 buffer 不擅自别名的合同。任意新增外部别名、数学体
变更或 allocation 复用都需要重新审计。

## 本次尝试记录

1. 初始从后期效应与显式依赖恢复七档：七档偏序全部一致，证明新硬件 trace 不是功能
   DAG 构造的前置条件。
2. 核对真实 trace 参数方向时发现 Out 与 INOUT 的 12 项差异，改为逻辑接口及后期效应
   的保守并集，额外写读约束；未将参数标签等价当作已证明事实。
3. `builder_check_v1` 在首个案例的 Python 测试脚本中遗漏 `use_local_signals` 的 layout
   和 bindings 参数，TypeError；未进入编译或设备。修正调用后运行 v2。
4. Root 复核指出还需覆盖 WAR；构造器加入先前 reader 到后续 writer 的保守边，v3
   重新检查，不沿用旧版本 PASS 名义。
5. 未提交 NPU 任务，没有改 frozen、依赖或共享入口。集成及真机验证由 root 后续完成。
