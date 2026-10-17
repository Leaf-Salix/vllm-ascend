# CSA 单 SPMD：当前交接

## 当前工作区

已按用户要求迁至 `vllm-ascend-dsv4-pto-0251rc1`，本目录现在是实际开发目录，不再只是文本镜像。
原 `.worktrees/csa-spmd-dynamic-20261016` 保留作历史留存，不再写入新实验。
GitHub 最新 `a919f119` 已合入本地，合并提交 `643d8039`；最新生产 HC_pre 已包含 `f7ee8dda` 的优化。
完整迁移及轻量 CPU 入口检查见 `WORKSPACE_MIGRATION_20261017.md`。没有为迁移重跑 NPU。

## 生效目标

用户已确认所有七档相对同轮动态分 task PTO **耗时降低至少 30%**，即
`SPMD≤dynamic×0.70`。原 128K/B16<700μs 单点目标被替换。
完整约束见 [GOAL_AND_CONSTRAINTS.md](GOAL_AND_CONSTRAINTS.md)。目标保持 active，尚未达到。
分档须覆盖工作量区间；七点是回归点，不是支持白名单。原始粘贴目标和旧约束均保留。

## 当前功能实现

- `range_policy.py` + `device_compare.py --range-policy` 是新增区间入口，使用 B/T/L 选择数学兼容区间和初始排程，保留七个已测点的策略。L含本轮query，H是历史，当前fixture L=H+6。
- 核体目前仍限 S6、T≤384、L≤131200；fixture B≤40。范围选择不等于区间内所有点已通过设备验证。
- `exact_cost_profile` 只用 B/H/S 和所有root shape一致的成本；其他形状仍可生成，参考成本标注为估计。
- 原有同构参考的路径不变。缺参考时由 `anchor_gap_design_v1/source_schedule.py` 从本轮绑定及IR读写效应构造DAG，处理RAW/WAW/WAR、显式TaskRef/dummy/manual合同。无整图串行兜底，无新增全核屏障。
- CLI 禁止与手工实验策略混用，包括显式传默认值；发现、修复和复查均有证据。

## 已完成验证

- 区间分类192点与原编排一致，384端点检查通过；七档选择保持一致。严格成本与CLI审查见 `range_policy_validation_v1/`。
- 源码DAG 20个代表组合通过schema、分片/MIX配对、视图/工作区、通知协议和源码生成检查。七档恢复原1377对偏序，缺失/新增均0；WAR审计也未增加原偏序以外串行关系。见 `anchor_gap_design_v1/builder_check_v3_war/summary.json`。
- B12/H65536、B17/H65536 通过完整生成和PYPTO lowering。后者确实走source_derived_task_dag，未套用B16等待表。
- B12产物：`range_policy_validation_v1/b12_h65536/variants/h65536_b12/`，lowering在`range_b12_h65536_lowering_v1/`。
- B17产物：`range_b17_h65536_cpu_v1/variants/h65536_b17/`，lowering在`range_b17_h65536_lowering_v1/`。

## 当前设备任务（以实时队列为准）

- `task_20261017_140019_112364329866` → `range_b12_h65536_device_v1`。
- `task_20261017_140020_112371518858` → `range_b17_h65536_device_v1`。

两任务均已exit0，完整exact/图重放/信号代际/保护区PASS。B12 L1=690.70 vs714.66μs，降低3.35%；B17 L1=863.08 vs893.52μs，降低3.41%。两点均未达到30%。原始泳道回读及精度摘要见range_entry_device_results_v1.json。当前无本会话设备任务。

## 性能现状及下一步

七档现选完整L1仍为：128K B4/B8/B16/B24=541.92/641.56/807.16/1121.12μs；
8K B16/B24/B32=621.96/836.74/905.72μs。历史收益约2.9%～10.8%，均未达到30%。
本轮回读原始L1起止并计算新门槛，见`target_30pct_historical_gap.json`；不是新测。
`performance_acceptance.py`新增独立的单规格30%门槛，功能PASS与性能达标分开写入结果。

两个代表点真机验证已完成，不重复。下一性能方向在`seven_stage_opportunities_v1/`：
按实际O-A分支保留各自K归约顺序，评估双输出tile输入复用；另调查压缩KV随机行加载的软件流水。
这些是新候选，不能拿逻辑GM请求量或CPU模型当实测收益。

主Q后处理head拆分的新逐成员成本模型调查已结束：已证实成本不均，但模型收益仅0.6～2.26μs，
少量额外开销即抵消，暂不生成设备候选。见`observed_member_cost_v1/`；不要重复旧排程扫描。
更多已关闭方向与历史失败见`ATTEMPTS.md`及`NEXT_STEP_BEFORE_RANGE_ENTRY_20261017.md`。

## 持续约束

正式权重固定 `/data/model/DeepSeek-V4-Flash-0731-w8a8`；PTO对PTO，不跑Native forward。
NPU只经`task-submit --device auto`。计时用完整L1 incore首尾，包含内部等待/空隙。
原静态归档保留，不改依赖仓或共享安装。当前单SPMD仍在本实验目录开发，尚未接入生产算子；过程记录直接在当前工作区保存。
不做hash；大产物/二进制保留本地并列路径。静态已push，动态实验尚未commit；本轮未要求commit。

## query 维度后续限制

只读审计确认S6不只是fixture条件：原Indexer按两query配对，S5会跨请求；request分组的
双AIV使用query_group_size//2，奇数S会漏尾query。冻结IR已固化//6，因此不能仅改入口常量。
后续详见`query_shape_support_audit_v1/`；原核体及当前通过的S6路径保留。

## 新补动态metadata任务

`task_20261017_140653_1348347342` → `range_b17_h65536_metadata_v1`，range-policy + metadata-replay-only。同地址/同图H65536→65535→32766→65536，C点L32772为长历史策略下边界，不采性能。已exit0完整PASS：固定地址、每侧一次图捕获、无新编译单元、信号不在变体间重置；四个变体输出/TopK/state及保护区通过。40秒观察超时后原任务正常完成，未重提。当前设备任务均终态。

## 性能候选补充

O-A独立资源探针存在Left流水退化与真实跨8组NZ绑定未证明问题，暂不直接上卡。QK压缩KV行加载独立Tile流水的CPU审查已完成，预期重叠没有实现，当前写法已停止。短B32当前不启用qk_vector_indices，长B8/B16启用，比较时不能把新增缓存组合冒充短档基线。

该CPU候选的最终报告在 `qk_row_load_pipeline_v1/lowering_audit.json`，不再等待子任务或重复扫描同一 stage 参数。

## 用户最新提示：参考 canonical HC_pre

用户询问独立优化hc_pre.py的参考价值，已读指定文件并保存快照和`HC_PRE_REFERENCE_REVIEW.md`。
当前HC流式/三阶段融合候选不是完整移植；特别是旧流式仍用GM交接，而独立HC_pre用aic_gather供数并复用加宽数据做RMS。
后续优先核对这套数据流/分核设计如何融入现有完整SPMD与norm合同，不以旧候选无收益关闭整个方向。
QK独立行Tile stage2候选已CPU证伪预期重叠并停止：基线实际已有连续TLOAD；候选额外加入逐行等待和TMOV。
详见`qk_row_load_pipeline_v1/lowering_audit.json`。不重测或扫该参数。
