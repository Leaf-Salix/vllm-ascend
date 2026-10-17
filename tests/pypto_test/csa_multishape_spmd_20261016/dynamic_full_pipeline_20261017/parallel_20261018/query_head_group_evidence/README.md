# Query 反量化 head 整波分组：唯一 CPU 检查通过（2026-10-18）

仅在私有 33 文件副本中修改 `query_projection_stages.py` 的
`q_dequant_rope_stage` 分组选择与对应中文说明。正式包、Q_B matmul、
依赖、调度和数学体均未修改。候选和参照已冻结；本子任务只执行了一次 CPU 检查。
主会话已独立提交长历史 T48、长历史 T144、短历史 T192 三个设备代表；
本报告尚不包含它们的精度或性能结果。

## 排重与变化边界

- MASTER 第 7865、7871 行：旧 T96 将 16 头降为 8/4，以减少投影父核依赖的
  纯模型；三个方案关键路径都为 771.53 μs，未上卡。本候选保持原等待，
  T96 仍为 16 头，补全其他整波工作量的 8/32 头，不重复旧依赖缩减。
- VALIDATION 第 4383–4418 行：2026-09-26 迁移独立 head 分组及 Q 展开，
  是旧分 task 路径的既有 4/16/64 分组；不等于本次动态补齐规则的设备对照。
- MASTER 第 14931–14947 行记录动态源抽取及尾块固定 4 头合同。
  本次检索 MASTER、VALIDATION、当前高层源及主 PAR 报告，未定位同一规则的试验。
- `source_snapshot.json`：正式 33 个 Python 文件直接复制到
  `source_reference`；候选只有 1 个文件不同。`candidate.patch` 是完整差异。
  `audit_source_and_owners.py` 还检查除分组选取和模块说明外的 AST 一致，
  包括 Q_B、RoPE 准备、完整/尾部数学及所有 load/store 地址表达式。

## 按实际设备 T 选择

令 `n=ceil(T/8)`。仅当 `T%8==0` 时，从 `64/32/16/8` 依次选择最大的
head 组 `g`，使 `n*(64/g)` 是正的 48 倍数；无匹配或有尾行时保留 `g=4`。
代码使用等价的 `n%(48*g/64)==0` 条件和正工作量检查，不按 B、S、
历史长度或测试点分派，也不把动态 T 冻结成样例数值。

| 实际 T | 原 g→候选 g | 准备任务数 | 每 AIV 的 8行×head 迭代数 |
| --- | --- | --- | --- |
| 48 | 4→8 | 96→48 | 8→8 |
| 144 | 4→8 | 288→144 | 24→24 |
| 192 | 16→32 | 96→48 | 32→32 |
| 240 | 4→8 | 480→240 | 40→40 |
| 336 | 4→8 | 672→336 | 56→56 |
| 432 | 4→8 | 864→432 | 72→72 |

这是 T1..512 内全部改变的值；表格是规则枚举结果，不是白名单。
每个任务共同加载/准备 activation scale、cos、signed sin、swap 及 gather 索引，
每个 head 的 weight scale、512 列反量化/RMS/RoPE 不减少。
上述范围准备次数减半，head 算术次数不变，48 AIV 均保持活跃。
没有把准备次数降低换算成时间，也未假设各 AIV 到达该阶段的时刻相同。

## 所有权、发布和尾块

任务编号 `w=token_block*(64/g)+head_group`，唯一 owner 为 `w%48`；
head 为 `head_group*g+h_inner`。每个 token/head 恰由一个任务写入。
若扩大组，任务数是 48 整波，因此每核 head 数为 `n*64/48`；
原分组在这些区间同样整波，逐核负载不增。
T1..512 的所有 token/head 地址逐项枚举通过，记录见
`source_and_owner_audit.json` 和 `ownership_cases.json`。
该证明只谈工作量，不声称硬件尾延迟相同。

候选仅重排独立 head；每 head 仍按原顺序执行 INT32→FP32、
activation scale、weight scale、512 列归约和 RoPE。
完整八行仍用 rsqrt；T 非 8 倍数时全函数选择 4 头，
尾循环仍固定 4 头、recip(sqrt)、原 valid_shape 和有效 store。
Mixed Q 的请求边界不参与本函数运算；只读取每行自己的 scale/cos/sin，
不跨 token 归约，因此不能将 S6 当成动态 Q，也无需按请求重分组。

现有正式依赖与发布足以覆盖改变后的 owner：

- `vector_schedule.py:562/574` 已等待本轮全部有效 ROPE/QR 行 owner，
  `:755–768` 等完 128 个 QB 列块所覆盖的 24 AIC，调用 DQ 后每 AIV
  发布 PHASE_QUERY；没有按旧 g 推导一份精简等待表。
- `cube_schedule.py:311–312` 先完成 QB 写回再发布；
  `:380–381` 在 Attention 读取前等全部 48 QUERY 发布者。
- `phase_sync.py:84–118` 保留发布前流水完成和 GM 维护、获取后的缓存维护。
  原 DQ GM 二维别名与下游 head 视图保持不变。

## 唯一完整 CPU lowering

命令保存在 `cpu_command.txt`，使用主只读 `reproduce_cpu.sh`、
隔离依赖和 `source_candidate`，`--strategy long` 配合明确的
`controls.json`（G6/N64/1024，OA128/OB96，正式开关）。
一次执行 exit 0；没有补编第二策略或任何设备执行。
`cpu_v1/result.json`、便于归档的 `cpu_result.json` 记录：
58 个动态轴、无样例 shape、完整生成成功、单 MIX、sync_start。

`audit_generated.py` 复用已存在的正式长 B16 生成物作比较，未重编基线：
主 PAR `hc_active_notifications_h131072_b16_v1/build_output/decode_csa_spmd_3kuwaqat`。
其 `hc_active_notifications_evidence/source_candidate` 的 33 源文件
与本次 reference 逐文件文本相同，controls 相同。
审计结果在 `generated_audit.json`：

| 实际规划上界 | 正式基线 | 候选 |
| --- | ---: | ---: |
| DQ Vec 地址末端 | 131072 B | 131072 B |
| 全 AIV Vec 地址末端 | 163872 B | 163872 B |
| AIC Mat | 475136 B | 475136 B |
| AIC Left / Right | 65536 / 65536 B | 65536 / 65536 B |
| AIC Acc | 131072 B | 131072 B |
| AIC Vec | 16384 B | 16384 B |

DQ 所有 Tile 的 shape/dtype/地址/字节数逐项相同；完整 AIC C++
去注释及 SSA/label 名后相同，证明 Q_B 与其他 AIC 流水未变。
DQ work-loop PTO 去路径/行号/SSA 名后相同，保留整数常量值和类型比较；
全链硬件调用计数、局部 set/wait/PIPE_ALL 事件计数相同。
新增的是函数入口处选择 g 的设备 Scalar 判断，未增加 Tile 或内层数学指令。
这不是设备精度或性能验证；owner 改变仍可能影响缓存和阶段接棒时间。

## 归档与复算

可复制本目录中除 `cpu_v1/` 外的文件及两份 33 源快照；
完整 CPU 生成物保留在原私有目录。`paths_and_sizes.json` 列出精简归档和
原始产物的路径、大小。两个轻量审计脚本仅读冻结源/已有生成物，不调用编译器。

```bash
python audit_source_and_owners.py
python audit_generated.py
```

`prepare_candidate.py` 只用于首次物化，拒绝覆盖已存在 source 目录；
冻结之后不要用其重建正在被其他任务引用的候选。

主会话设备登记位于主 PAR 的 `query_head_group_evidence/registry.json`：
长 B8 `task_20261018_043459_392309626395`、长 B24
`task_20261018_043459_392319716700`、短 B32
`task_20261018_043459_392330520909`。本子任务不重复提交或查询这些作业。
