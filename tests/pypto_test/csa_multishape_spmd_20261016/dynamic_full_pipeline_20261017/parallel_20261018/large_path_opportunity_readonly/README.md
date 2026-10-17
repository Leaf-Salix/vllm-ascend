# 完整 CSA 较大关键区间：本轮只读结论

2026-10-18。**未找到有充分新证据、值得立即实现的调度或算法候选。** 这不是断言当前实现已最优，也不是要求某个单独优化必须贡献全部 30%；本轮没有找到能可靠削减较大服务区间、且尚未被既有实验覆盖的具体工作。停止提出微小等待或指令删除候选。

仅回看现成源码、阶段时钟与归档报告；未编译、未上卡、未读权重或大型 Tensor/二进制，未新增 CPU 性能模型。前一份 `attention_query_overlap_readonly_20261018` 原报告保持不动。输入文件和字节数、小型时钟摘录见 [evidence.json](./evidence.json)。

## 量级与分母

验收仍是每档完整 L1 相对原动态分 task 降低至少 30%，不能用候选相对旧单 SPMD 的百分比替代。以最新几何缓存消融中**同进程参考**举例：

| 档位 | 原动态 / μs | 正式参考 / μs | 目标 ≤ / μs | 从本轮参考仍需减少 / μs |
|---|---:|---:|---:|---:|
| 128K/B16 | 878.74 | 877.84 | 615.118 | 262.722 |
| 8K/B32 | 969.80 | 1005.18 | 678.860 | 326.320 |

这只是目标差距计算，不是加速下限或新版本预测。原始窗口见 MASTER “本核几何缓存四代表收口”；四代表最终均完整 PASS，但三个回退，正式仍保留 668，不能按 B32 单点启用。

## 本次排重只得到一个明确缺口：Score AIC 的可消除工作尚未被定位

当前长 G6/1024 的 `indexer_score_stages.py` 是每步 16 个 N64 面板：INT8 QK 为 M384/K128，FIX 执行 FP16(ReLU(QK)/1024)，再用 M16/K384 的 FP16 系数矩阵做 WS。QK/current 与 WS/previous 已交错。`indexer_group_stages.py` 先分别将 query scale、head weight 转 FP16，再在 FP16 相乘，系数按六个 64-head 对角块放入 16×384；不能随意移除这些舍入边界。

本轮追查的“第二次 head 归约”**正是既有 WS 路径，不是新发现的重复计算**。已有 `indexer_long_inner_reaudit` 和 `ws_aiv_readonly` 明确：每步有约 50.33M INT8 QK MAC、6.29M 稠密 FP16 WS MAC，而有效六条 64-head 加权和约 0.393M 乘加。后一个比例说明代数稀疏性，**不说明可减少相同比例时延**；QK 和 WS 数据类型、流水及最小矩阵块不同。

直接迁往 AIV 的相近方案已经审计：每步交接 24→768 KiB，单 AIV 接收到的原始 QK 达 384 KiB，超过现有 184 KiB UB 预算，另有 TopK 根/segment；FP16 乘积舍入与 Cube FP32 归约次序也没有 exact 等价依据。既有历史正是从 Vector head 归约转回 Cube 路径，当前没有新的传输或精确归约机制推翻这份反证。本轮不重做此方案。

旧 593 的三轮阶段诊断中，Score AIC `other` 中位约 230 μs，AIV READY 原位区间约 107 μs；它支持继续关注 AIC 供数链，但 `other` 含数学、页访问、局部事件、搬运及发布，READY 也不是纯远端计算耗时。当前 Indexer 五份源码与该快照相同已有逐文本证据，但**不能把旧插桩数值认作当前 668 无插桩服务时间**。

因此最需要区分的是：同轮 Score 中，Key 页解析/GM→L1→L0、QK、FIX、WS、局部等待/排空究竟各自形成了多少关键服务时间。现有聚合端点没有把其中某一段证明为一份可删除的大额重复工作。既没有证据支持重写 WS，也不能据 AIV 等待将排序或其他数学搬过去。此项是证据缺口，不是本轮启动新插桩/编译/任务的要求。

## 其他较大方向为何本轮不再立项

- **跨 Score→Attention 的请求流水：** `wave_audit` 已切开真实 query 范围、叶根发布和三角色排空，合法均衡 B16 中每组仍承担相同的 22 个 Score step 与 20 个 Attention block。仅换执行先后没有消除工作；没有额外核资源、较细角色复用或引擎级互补证据，不能把两个大阶段的重叠包络直接相减。该反例只约束当时固定 owner/整段排空方案，不否定所有未来流水。
- **Attention KV 搬运：** 已有逐 tick 表明压缩部分占每 AIV gather 区间约 63–65 μs，但这是带插桩的包络。当前每 query 仅一个 MIX 组、两 AIV 各采不同 64 行，AIC 已在多个 head/QK/PV 间复用；没有同 query 的重复 KV 读可直接删。directL1、MGATHER、物理行 plan、guard 删除及局部缓存等均有历史。跨 query 的现有数组只来自随机激活 fixture，不能据此提出服务级 KV 重用。
- **HC/pre-QA/输出尾部：** 同轮证据显示 Sinkhorn 已隐藏在 QA 前，DQ 大部已与 MAIN_PROJECT 重叠；尾部 BF16 Tile 已直接交给 HCpost，没有新的同类 GM 物化。已有消费者归约、HC 行分工、逐组输出消费和 OA load 复用记录，本轮不换名复测。

本轮结论限于已有证据：**没有提交新的实现候选，也没有性能结论。** 后续如果新增证据能把 Score 的某份较大服务成本落实为可删除工作，再据其真实 owner、数值顺序、READY/ACK 和存储生命周期设计；在此之前不继续扫 tile、分核或等待。

## 精确既有入口

以下相对路径基于主仓 `tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/`，绝对路径及大小已列入 evidence.json：

- `indexer_long_inner_reaudit/README.md`：当前源与 593 的同源确认及较大省工排重。
- `ws_aiv_readonly/README.md`：WS 稀疏性、32 倍传输、UB 与 exact 限制。
- `indexer_interval_reaudit/README.md`：已做 G/leaf/panel/交接策略，不重复复测。
- `current_clock_h131072_b16_v1/phase_summary.json`：旧 593、三轮诊断，非性能验收。
- `attention_tick_clock_evidence/ANALYSIS.md`、`attention_aiv_d253_audit/README.md`：Attention 区间及生成顺序限制。
- `wave_audit/README.md`：真实 owner 下两波交接、协议和反例。
- `hc_tail_clock_audit/README.md`：HC/pre-QA/尾部的已有重叠和物化审查。
