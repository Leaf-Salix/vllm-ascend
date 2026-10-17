# 保持 leaf/root 数学的 heavy/tail owner 设计（未实现）

状态：有限计数模型可达 B28 41→40 steps，但本轮不推进实现、编译或设备试验。
此设计只改变一个完整 `(group, leaf)` 的 AIC/双 AIV owner。保持原 leaf 边界、
QK/WS、FP16 舍入、候选输入顺序、根槽号、每 query 的根合并顺序及尾组补齐修正。
这消除了主动重划叶导致的平分顺序风险，不等于已经通过浮点 exact 或并发协议验证。

## 有边界的 device gate

仅接受当前 balanced、ROOTS=1、score_tile=1024；groups≥24，leaves≥2，
leaf_tiles=8、extra=0。groups=4/8/16 的既有均衡计划原样回退，不触碰短档。
令 `heavy_total = groups*(leaves-1)`、`r=heavy_total%24`，需要 r>0。

已有唯一 planner 在扫描各组真实 visibility 时顺便求最小值；要求
`min_group_visible >= (leaves-1)*8192`，从而所有组的重叶都真实完整。
用实际最大 group visibility 求尾叶 step 上界 k，要求 `k>0` 且
`ceil(groups/(24-r))*k <= 8`。这仅是尾叶能放入少一个重叶留下的 step 空间，
不是延迟收益门限；任何条件不满足就使用原映射。

不在 host 读取 device metadata，不用 B28 白名单，不根据统一历史上界猜测所有请求
等长。模型中的单长 B28 最小可见 2049、单 stale B28 最小可见 0，均回退原方案。
2685 个有限例子见 `owner_evidence_v1/`，其中强制 G6 的几何例子与真实 policy 例子
在名字上可区分。它们不涵盖所有非法 metadata 或保留 T 容量的 inactive 变体。

## O(1) 每核计数和映射

下面是整数接口草图，模型已有同样表达式，尚未变为生产 helper：

```python
heavy_total = groups * (leaves - 1)
r = heavy_total % 24
light_cores = 24 - r
heavy_count = max((heavy_total - worker + 23) // 24, 0)
tail_count = 0
if worker >= r:
    tail_count = max((groups - (worker - r) + light_cores - 1) // light_cores, 0)

for local in range(heavy_count + tail_count):
    if local < heavy_count:
        flat = worker + local * 24
        group = flat // (leaves - 1)
        leaf = flat % (leaves - 1)
    else:
        group = worker - r + (local - heavy_count) * light_cores
        leaf = leaves - 1
    # 原 score / TopK leaf body 保持；root 地址仍由 group/query/leaf 决定。
```

每核顺序先重叶再尾叶，同组两 AIV 与 AIC 必须调用同一 helper、消费同一 mode。
跨叶双槽 Score 的原排空不能删除：每叶结束必须回收该叶全部 ACK，下叶重新从 slot0
开始。各 AIC/双 AIV 组工作数可不同，但同一组的三核序列必须一致，最终 PHASE_SCORE
发布仍须覆盖全部 worker。旧 root 唯一写者改变但根地址和最终 merge 读取顺序不变。

建议的最小接口是 `indexer_worker_item_count(worker, plan)` 和
`indexer_worker_item(local, worker, plan)`。旧 forrange(worker,total,24) 改为
forrange(local_count)，mode=0 时计算原 `item=worker+local*24` 并调用旧 work_item。
不能仅置换旧固定 140 个 item：B28 在该限制下的最大 step 下界是 41。

新映射每核只做常数次整除/取模、每 item 一次分支；无需每核遍历所有 group/leaf
寻找归属。真正全量扫描仅有已经存在的 planner，增加每组一次 min 更新。
如果不发布 mode 而让 72 个角色重复扫描 groups，会额外引入 O(72*groups) Scalar
和 metadata 读取，本设计不采用这种替代。

## 实际接口代价

当前 `group_plan` 是 8 个 INT32，八个字段已有含义，没有安全空位可复用。
`indexer_group_stages.py:79` 的 PLAN_WORDS=8、`entry.py:232/742` 的 Tensor[[8]]
以及 `dynamic_full_pipeline_20261017/assemble_source.py:103` 的生成绑定均固定 8。
因此 mode 至少需要把现有 plan 扩为 9 个字，也可统一 16 字/64 B；必须同步入口、
生成器、工作区绑定、检查脚本和 helper probe，不能只改一个常量。

模式由现有 planner 写出，并沿原 PHASE_META 发布与读取一致性协议可见。后续 CPU
实现若进行，须检查新字所在 cache 行的可见性和根签名，不能借 descriptor prefix
对齐空位写入未声明字段，也不能偷换 balanced/G/ROOTS 字段。虽然无需新增根 Tensor，
仍会改变已有 plan Tensor 的 shape 合同。这是只省一步的方案需要承担的真实工程代价。

## B28 的精确计数与限制

| 核范围 | 新 steps / 有效叶 / drain / 根写回 |
| --- | --- |
| 0..15 | 40 / 5 / 10 / 30 |
| 16..19 | 36 / 8 / 12 / 48 |
| 20..23 | 35 / 7 / 11 / 42 |

原最忙核为 41 steps、6 叶、11 drain、36 根；core16..19 原来各 6 叶。
总 steps=924、有效叶=140、根写回=840、最终消费根=756、drain=252、ACK=924，
所有 QK/WS 算术和逻辑 GM 交通均不变。新增方案不是减少总根/等待，只是重新集中。
Hadamard 的 owners0..3 仍在 Score 最忙核集合，起跑先后和资源竞争未建模。

即使粗略设每 step 成本为 s、每叶额外成本为 h，新 core16..19 的 `36s+8h`
要低于旧 `41s+6h` 也要求 `h<2.5s`；这不是已测前提。真实 cost 中还有不同比例
的 AIC/AIV 工作、根存储、FFTS、流水气泡和前序任务，不能假设 drain 免费或把
2.44% 的 step 降幅写为 CSA 加速上限/预期。本轮结论是仅归档，不占卡检验这个弱候选。
