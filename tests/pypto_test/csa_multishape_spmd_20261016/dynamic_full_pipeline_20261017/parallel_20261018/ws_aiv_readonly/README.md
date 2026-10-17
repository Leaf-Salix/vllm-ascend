# 长历史 WS 移到 AIV：只读排重结论（2026-10-18）

**结论：未找到当前动态单 SPMD G6、仅把 WS 移到 AIV 的 exact 对照；暂不建议沿现有 GM 环直接迁移。** 历史有相反方向的实际实现与设备记录，但不能写成当前方案已经失败。本次只查源码及已有记录，未实现候选、编译或上卡。

## 历史事实及适用边界

[VALIDATION §176–178](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/DSV4_FLASH_CSA_VALIDATION_LOG.md:6814) 记录：v4 将 FIX 后 FP16 的 64-head QK 写 GM，由 Vector 转 FP32、乘 FP32 系数再归约；v7 改为 QK/current 与 WS/previous 流水，FP16 QK 留 L1、Cube 完成加权和，仅最终 FP32 分数交给 Vector。最终保留长历史 Cube 路径；v5/v6 的串行版本不能代表 v7。

[原始七档报告](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_native_cube_matrix_20260927/README.md:9) 中，旧 **CSA 本体均值** v4→v7：128K/B4 为 766.70→749.03 μs，B8 为 968.48→947.36 μs；B16 为 1576.93→1458.16 μs，但 v4 复用了不同 PyPTO 版本。它们是旧分 task 路径，含流水、数学等变化，不能当当前单旋钮同进程结果；旧 Native 零容差差异也未消除。短历史没有切此路径，不能用其变化支持 WS 判断。MASTER 与 VALIDATION 的 WS/AIV/Vector/head 归约记录中，未定位到更新的当前 exact 反向设备消融。

## 当前源码与有限工作量

现有 [score 实现](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/indexer_score_stages.py:331) 已交错提交 QK/current 与 WS/previous，并以 FIX 做 FP16( ReLU(QK)/1024 )；不是完全串行的 QK→WS。长 G6/1024 一步中，QK 为 50,331,648 次 INT8 MAC，稠密 WS 为 6,291,456 次 FP16 MAC，有效六条 64-head 加权和只需 393,216 次乘加：有去掉零元素/填充行算术的空间，但 **MAC 比例不是时延或收益上界**，两种数据类型与流水行为不同。

[双槽绑定](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/binding.py:204) 当前每步交接 6×1024×4 = **24 KiB**；若交接全部 FP16 QK，则为 6×64×1024×2 = **768 KiB，32 倍**。每步额外 GM 写及两 AIV 合计读取各 744 KiB；24 核双槽由 1.125 MiB 变为 36 MiB。这里只算逻辑载荷，不把它等同实际 HBM 流量或带宽时间。

当前 [AIV balanced 分支](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/indexer_score_stages.py:431) 每 AIV 接三条 query，读完即 ACK，然后 key-scale、每两步的 2048 候选排序及 Top512 根合并。迁移后单 AIV 原始 FP16 块就有 **384 KiB**，超过当前 planner 的 184 KiB UB 上限；原有 roots/segments 另占 36 KiB。必须按 head/候选再分块，并重新安排载入、ACK 与排序生命周期。旧阶段时钟的 READY 等待不能当作最新正式源可无代价占用的 Vector 空闲预算。

## 数值与判断

[当前系数](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/indexer_group_stages.py:293) 是 scale、weight 各先舍入 FP16，再做 FP16 乘法；WS 用这份 FP16 系数与 FP16 QK，Cube FP32 累加。AIV 若直接 FP16 相乘会多一次乘积舍入；即使先提升 FP32，Vector 归约树也未证明等同 Cube 累加顺序。数学和相近不足以保证 score、TopK 平分次序及最终 exact。

**推断：** 稠密 WS 的无效算术值得认识，但现有 GM 交接会显著增大数据量、挤占 TopK 的 UB/流水，并新增 exact 障碍；历史保留 Cube 的结果方向亦不支持直接回退。现有证据不足以推进，且不据此否定未来具备不同传输与精确归约依据的设计。

