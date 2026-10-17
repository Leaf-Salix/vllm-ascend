# 长历史 G6 的两请求 wave 交接：只读设计审计

结论：在保留当前物理 owner、每次切换排空本组流水的直接方案中，可以实现
第一 wave Attention 与第二 wave Indexer 的跨组重叠，且不形成等待环；但长
B16 的均衡反例中，即使 TopK、plan、通知和切换成本全部视为零，两个大阶段的
总计算链也只打平。当前没有把它选为设备候选的量级依据。

本轮只新增本目录的离线模型和文档。没有修改算子源码、正式默认、共享依赖，
没有重新编译或申请设备。源码只记录路径/字节大小和直接逻辑对照，不做摘要校验。
原有同步候选及其所有原始结果保持不变。

## 历史已做事项

只读历史目录为主仓 `tests/pypto_test/csa_multishape_spmd_20261016`：

- `indexer_query_completion_model.py/json`：当前 B16 的 6 叶成本为
  6/6/6/5/5/5 个 1024 score 块。保留 owner，仅将同 request 提前，可让前八
  请求在每核 11 块后具备根数据；旧整阶段通知仍要等 22 块，消费者不会提前。
- `query_group_schedule_model.py` 和 `query_group_schedule_assessment.json`：
  静态历史已把 Indexer/merge/plan/QK/O-A/quant/O-B/HCpost 都按 48 行拆分，
  建立 1313 成员/1121 单元模型。理想投影半成本下，前半优先、最早就绪、
  关键路径优先分别为 855.325/780.470/813.585 个历史模型微秒，原模型
  771.530；没有采用或上卡。后来允许 MIX 重新放置的模型也未改善。
- `query_group_schedule_optimistic_bounds.json` 已提醒数学关键路径变短不代表
  核工作量下界变短：历史两半模型的 AIC 总工作量下界反而从 665.622 增至
  672.955。以上是旧 CPU 成本模型，不能作为本轮动态源码的实测或硬件下限。

本审计采用当前动态 helper 的实际 owner 与显式两 wave 发布，保留 O-A/O-B
原行块，不再次做历史静态成本排程扫描，也不把单纯叶重排当作提前交接。

## 当前源码决定的依赖与空间

模型通过 AST 离线解释 `indexer_group_stages.py` 的
`indexer_long_leaf_plan`、`indexer_work_item`、`indexer_leaf_range`。
对 H131072/B16/S6，设备描述符给出 16 个 G6 组、6 叶，每核每 wave 11 块，
两 wave 合计 22 块。下列公式的 group 是描述符组，不允许用 `T/6` 代替动态组数。

| 环节 | 当前分工/依赖 | 本轮依据 |
| --- | --- | --- |
| Score 叶 | groups<24 时 item=leaf*groups+group，物理 AIC=item%24；groups≥24 时改为 group*leaves+leaf | `indexer_group_stages.py:235`；score 双侧同一循环 |
| balanced 叶根 | 每 AIV 管 G/2=3 条完整 query；lane=组内 query//3；根在 q*64+leaf | `indexer_score_stages.py:411`、`:435`、`:592` |
| TopK merge | q%48 号 AIV；读取该 query 的全部有效叶，按既有根顺序归并 | `indexer_publish_stages.py:137` |
| Attention plan | 每个 M8 行块归 floor(q/8)%48 号 AIV；先等这八行的 merge | `attention_plan_stages.py:155` |
| Attention | q%24 号 MIX 组；同组 AIC+两 AIV 必须配对 | `attention_stream_stages.py:134`、`:237` |
| Score FIFO | 每组两槽，行 c*2*G+(step%2)*G；事件 0/1；每叶末排空，根写回在 AIV 最后计算之后 | `indexer_score_stages.py:174`、`:375`、`:469`、`:592` |
| Attention ring | 每组三槽 c*3+(work%3)，事件 2..5；AIC +2 拍、AIV +3 拍排空 | `attention_stream_stages.py:141`、`:261`；两份 schedule 的 EVENT_BASE=2 |

Score 和 Attention 的 GM ring 是不同工作区，不同物理组的槽不重叠。
pair、TopK、plan、packed 均按全局 query 行落盘；两 wave 不覆盖对方的行。
一个 Score 叶的 ACK 只证明 FIFO 已被读取，不证明 AIV TopK 根已写完。
中间完成点必须放在最后根写回及原有 release/cache 维护之后，不能在最后 ACK
处提前发布。切换时还需保留本组另一角色的排空关系；独立事件 ID 不会让一个
核同时执行两个 helper。

## 唯一建模方案

以完整有效的 B16 输入作为一个合法反例：wave0 为请求 0..7/query 0..47，
wave1 为请求 8..15/query 48..95。实际实现时切点必须从 device Q/组描述符
求出；此处的 48 仅用于说明一个合法实例，不构造生产规格白名单。

1. 所有核组先处理各自 wave0 的叶。AIC 和双 AIV 都按相同过滤顺序执行，
   保留每叶内部数学及两槽 READY/ACK；每叶末排空。两 AIV 在本 wave 全部根
   写回、release 后，各发布一次独占的 `SCORE_W0` 槽。
2. 每个 AIV 只 merge 自己的 wave0 query，并只等真实 leaf/lane owner。
   如 q0 的六叶来自 AIC 0/16/8 循环，其根写者为 AIV worker 0/32/16；
   q3 则对应 worker 1/33/17。随后只生成 wave0 的 M8 plan，各 query 只被
   对应 owner 写一次，发布 `PLAN_W0`。空 plan 角色也发布自己的本地交接完成。
3. AIC 0..11 及其双 AIV 采用 `Attention0 → Indexer1`；AIC 12..23 及其双
   AIV 采用 `Indexer1 → Attention0`。切换前 AIC 只获取本组两 AIV 的交接点；
   Attention0 再获取自己 query 对应的 plan owner。这里没有新增全核 barrier。
4. 本组上述两个工作都完成后，只 merge 自己的 wave1 query，生成对应 plan，
   发布独立的 `PLAN_W1`；随后执行本组 Attention1。Attention 两次都完整排空。
   O-A 仍按现有完整 packed 行块合同等候，不拆 O-A/O-B，也不假设第一半已经
   能进入原 M128/M96 输出投影。

每个 wave 的 Score/TopK/plan/packed 通知必须用不同阶段槽，或另行证明单调
进度协议；不能在旧槽的同一代际重复写相同 ready 值。当前模型选独立 wave 槽，
仍要求 512B 单写者隔离、发布 fence/cache 维护、等待后业务获取和新工作区初始化。
模型中的 `packed_all_ready` 是既有 O-A 数据输入条件，不是新增设备 barrier。

### 必须切开的真实 helper 范围

当前 `indexer_score_*`、`indexer_pair_publish_stage` 和 `attention_qk_pv_*`
都在一次调用中遍历整段本核工作。方案要求新增 device 范围参数，按原全局
query/组号过滤；只新增 wave 通知而保留旧 helper 循环不能工作。

一个具体等待环是：group0 的所谓 Attention0 若仍运行到 q48，则需要 wave1
plan；q48..55 的 plan 由 AIV worker6 产生，依赖 q48 的 TopK；q48 TopK 由
AIV worker0 产生，依赖 request8/leaf1 的根；该叶仍归 group0，而 group0 的
Indexer1 被排在 Attention0 返回之后。于是出现
`A0_g0 → Plan1(q48) → Merge1(q48) → I1_g0 → A0_g0`。
若不等待而直接读未生成的 mask，也只是漏依赖，不是合法提前执行。

动态边界还必须保留：混合 Q 时 wave 切点未必按八行对齐，plan 需要明确处理
跨切点 M8 tile，不能读取尚未完成的另一半 TopK；空请求和尾 lane 仍完成其
约定的发布/ACK。Attention 的 `EARLY_KV_MIN_TOKENS=144` 判断必须使用原始
全 T，不能把 wave 行数冒充 T 改变 KV/SCORE/PROB 顺序。所有 GM 地址继续使用
全局 q，不能仅把 q_flat 切成半份后沿用原局部编号。

## 可执行模型结果及阻塞条件

`model.py` 建成 301 节点、882 条依赖边，逐核加入程序顺序，检查每核无重叠
占用、DAG 无环、每 query 的根 owner、plan owner、GM 分区和 Score 叶边界的
双槽消息配对。MIX 切换按三角色全部排空建模；TopK/plan/通知耗时先取零，
是有利于候选的抽象极限，不是复用旧性能成本。

把一个 Score 块成本记为 s>0，一个 Attention 块成本记为 a>0。当前合法
均衡输入让所有 24 组都承担 `22s+20a`，任何保留该 owner 且切换前排空的
模型都不能让这组的串行工作短于自身总量。旧整阶段执行已达到同一工作量界：
`22s + 20a`。新方案的每组最终仍是 `11s+10a+11s+10a` 或
`11s+11s+10a+10a`。

以 s=a=1 的抽象工作格显示实际跨组重叠：

| MIX 组 | 0..11 | 11..21 | 21..22 | 22..32 | 32..42 |
| --- | --- | --- | --- | --- | --- |
| g0..11 | Indexer0 | Attention0 | Indexer1 | Indexer1 | Attention1 |
| g12..23 | Indexer0 | Indexer1 | Indexer1 | Attention0 | Attention1 |

旧方案为 0..22 做全部 Indexer、22..42 做全部 Attention。新旧都在 42 结束。
同一方案对 (s,a)=(3,1)、(1,3) 分别都得到 86、82，新旧仍完全相同。
这些数字没有时间单位，不能写成设备耗时、收益或硬件下限。

因此，这个具体方案可以隐藏部分旧 TopK/plan 等待空隙，却不能仅凭两个阶段
在不同组上交错，就把完整 Indexer 与 Attention 的计算时间相减。已有静态
模型同样提示核工作量限制。本轮没有证据显示待隐藏的交接空隙足以支撑目标，
也没有为新增两轮通知、helper 排空或 plan 尾块成本给出实测收益，故不选为
设备候选。

结论只限此固定 owner、三角色排空的具体方案与均衡反例。真实硬件的 AIC/AIV
忙闲互补、角色尾段的细粒度交错或改变分工不在本模型内，不能被该工作格证明
一并排除；本轮也没有把它们假定为已经实现。

## 证据与复现

- `result.json`：完整 owner、301 节点及起止、各类依赖、空间/消息合同与三个
  抽象正成本检查；保留明确的硬件/数值未验证标记。
- `model_stdout.json`：去掉完整节点明细的可读输出。
- `model.py`：仅使用 Python 标准库并读取当前工作区源码，不导入 PyPTO/NPU。

```bash
python /data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_wave_audit_20261018/model.py
```

离线模型及 `ruff check` 已通过；没有设备/数值/性能 PASS，也没有改动任何原始
验证结果。仅交付上述一个有界方案的模型及其性能阻塞条件。
