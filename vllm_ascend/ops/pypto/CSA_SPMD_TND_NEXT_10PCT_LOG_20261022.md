# CSA 单 SPMD TND：v122 之后的九档优化过程

## 2026-10-22：启动、永久基线与首轮单变量候选

用户已设置新 active goal：实现 `CSA_SPMD_TND_NEXT_10PCT_GOAL_20261022.md`，
过程中参考 cannbot-knowledge 的 ops。此前暂停状态不再适用。
固定口径为八档 S6 + 128K/B16 固定混长，九档等权 candidate/v122 的 L1
中位耗时比几何平均 <= 0.90。当前尚未有本轮设备结果，不宣称新增收益。

本轮开始于 `a584eded`，worktree 为 `csa-spmd-tnd-20261020`；上游仍是
`origin/dsv4-flash-pto-v0.25.1rc1`。没有执行 push。共享SDK清单仍 validated，
release=`csa-shared-20261021-upstream-fixes`，PyPTO e891351e、PTOAS 914863ed、
Simpler 57e49923。没有修改/重建共享SDK。开始时无本会话未完成设备任务。

### 固化与验收入口

实验目录：`tests/pypto_test/csa_spmd_tnd_opt_20261022`。
已核对 v122 全部41文件；相对 a584eded 的默认TND包仅 `policy.py` 与
`indexer_query_stages.py` 不同。用固定完整提交、两个原始文件与逐文件SHA256
永久保存基线；`materialize_reference.py` 可在不依赖旧 runs 的情况下重建。
未把当前默认包或旧BSH当作新分母。

增量harness新增只接受永久v122清单的引用，记录 `kind=frozen_tnd_v122`。
沿用原完整链路计时、精确比较、metadata/service检查；额外启用已有
`score_arena_unused_untouched` 断言，确认UB路径不写GM Score scratch。
旧BSH八档验收器未修改。新增独立九档验收器，逐档五组交替72核单MIX窗口；
固定混长 `[1,2,3,4,5,6,6,5,4,3,2,1,1,3,4,6]`，T=56，不得换分布或填到96。

### cannbot-knowledge 依据与边界

只读使用本地 `cannbot-knowledge/.agents/skills/knowledge-query/SKILL.md`。
首次 discover 返回退出码2：`artifacts/indexes/knowledge.sqlite3` 缺失。
按技能停止索引检索，不重建索引、不扫描全库兜底，不伪造search/ready receipt。
继续阅读此前已明确定位的具体卡片，以下是工程假设依据，不是已完成知识库审计：

- `knowledge/ops/ascendc/guides/synchronization/resident_softmax_state_slot_recurrence_and_reuse_events.md`：
  draft、未定位平台。借鉴同核状态可驻留、复用必须等最后消费者；不直接套其示意代码。
  本轮应用于短历史半叶UB生命周期审计，实际资源和依赖以A3生成代码及设备验证为准。
- `knowledge/ops/ascendc/guides/synchronization/attention_schedule_slots_and_platform_handoffs.md`：
  draft、未定位平台。保留生产/消费身份、READY/ACK与末尾排空；无效数学可裁剪
  不代表协议可以单边跳过。不照搬950的mode4/flag偏移到A3。
- `knowledge/ops/ascendc/apis/simd_api/basic_api/memory_vector_compute/sort_and_merge_isasi/sort32.md`：
  stable，明确包括A3；32元素块内同分优先原较小位置。半叶UB候选不变更排序树、
  叶地址与归并输入顺序；整树精确性仍由现有TopK/score比较验证。

### 首轮候选与CPU检查

`output_balance_off` 仅令 `OUTPUT_WORK_BALANCE=False`，分别测128K/B1、B16；
不调整Q/L、tile、其他阶段或wait。代表档descriptor断言只有这一个constexpr不同。

`short_half_ub96` 仅把短G2组数上界48扩至96，测8K/B32；每个AIV顺序处理4波。
源码helper在每叶内创建并消费私有22×384 FP32半叶，33,792字节，未按总组数
复制4份buffer。完整生成AIV代码中只有一个该Tile声明，映射固定UB地址；重用前
有MTE3→Vector等待。保留原六个排序分支、FIFO与全部尾部检查。此项只能说明
候选具备编译与初步生命周期证据，不能证明性能或所有边界已通过。

三个descriptor均已通过CPU parse与完整codegen：
`runs/compile_output_balance_off_h131072_b1/result.json`、
`runs/compile_output_balance_off_h131072_b16/result.json`、
`runs/compile_short_half_ub96_h8192_b32/result.json`，均 passed=true、device_execution=false。

已提交三项独立设备任务，使用同一冻结v122引用及共享SDK：

| 候选/场景 | task-submit ID |
|---|---|
| output_balance_off / 128K B1 S6 | task_20261022_020842_313460812168 |
| output_balance_off / 128K B16 S6 | task_20261022_020844_313577826313 |
| short_half_ub96 / 8K B32 S6 | task_20261022_020845_313708213322 |

输出位于实验目录 `runs/measure_<候选>_h<历史>_b<B>`；完整命令与启动器hash在
旧实验目录 `csa_spmd_tnd_20261020/jobs.json`。只用 `--status` 观察，不用带
timeout的 `--wait`，避免观察超时取消pending任务。候选尚未采用，等待实测。

CPU门禁：新九档/重建/真实启动器快照27项通过，旧增量与共享SDK23项通过；
定向ruff与git diff --check通过。format.sh ci因环境缺少pre-commit退出，
未执行完整仓库格式CI，不把定向检查写成全量CI通过。三项设备任务均已观察为running。
