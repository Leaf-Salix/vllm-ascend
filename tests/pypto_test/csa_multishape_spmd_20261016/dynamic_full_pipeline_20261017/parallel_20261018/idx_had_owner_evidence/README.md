# Indexer query RoPE→Hadamard 精确 owner 等待候选

结论：在当前 24 AIC / 48 AIV、64 heads、RoPE 每 unit 为 8 token×4 heads 的合同下，**每个有 HAD 工作的 AIC 只依赖固定连续 16 个 AIV**。可以将原最多 48 核的等待缩为该集合；没有 HAD token 的核跳过这次等待。这里只完成同源 AST 审计与 CPU owner 枚举，未进行后端编译、NPU、性能测量或正式迁移。

独立基线是正式 `66882634890af3907d8553d91ed4fab28b4fd631`，两侧各冻结 33 份源码；唯一差异为 `source_candidate/cube_schedule.py` 第 298 行起的一处等待。32 份文件逐字不变。删除/还原这一个等待块后，完整模块 AST 与参考完全一致；数学 helper、发布调用、接口、Host 策略、绑定和 runtime ABI 均不变。没有混合 OA pair、小 T 候选或其他优化。

## 1. 精确推导

来源为 `source_reference/indexer_query_stages.py:115–198` 的反量化 RoPE、`:201–217` 的 Hadamard、`runtime_abi.py:16–17` 的 24/48 核合同。

设真实 token 数为 T，AIC worker 为 c（0..23）。Hadamard 的一块是 64 个 head 行，恰好一个 token：

```text
消费 token：t = c + 24*w，0 <= t < T
生产 unit：u = floor(t/8)*16 + h，h=0..15
生产者数量：P = min(ceil(T/8)*16, 48)
生产 AIV worker：u % P
```

当 T≥17 时 P=48，而 `floor((c+24*w)/8)=floor(c/8)+3*w`，所以

```text
u % 48 = (16*floor(c/8) + 48*w + h) % 48
       = 16*floor(c/8) + h
```

因此 c0..7 只读 AIV worker0..15，c8..15 只读16..31，c16..23 只读32..47；不随本核 wave 数变化。准确的 48 生产者阈值是 **T≥17**，不是 T≥24。

T1..8 时 P=16，活跃 c 仅0..7、没有第二 wave，所需集合0..15。T9..16 时 P=32，活跃 c 的集合只能是0..15或16..31。虽然新 wait 参数仍传 VECTORS=48，这两个区间中的所有 owner 均小于真实 P，取模结果相同。T17..23 的 P 已是48，但部分 AIC 没有 token。T0 仅作为空循环模型核对，不扩大正式入口支持承诺。

最后不足8行时，RoPE helper 进入 `for tail_t0 in range(dq_t0, T)`，head group 和 unit owner 都不改变；每个实际 token 的64个heads仍由同一组16核写齐。Hadamard 只循环真实 T，不读取 P16 padding；不能按容量行数代替 T。

## 2. wait 参数的单位

候选保持调用现有 `phase_sync.py:121` 的 `wait_row_owners`，不另写同步实现：

```python
if worker < tokens:
    signal = wait_row_owners(
        signal, PHASE_IDX_QUERY_ROPE, CORES, generation,
        (worker // 8) * 16, 16, 1, VECTORS, GROUPS, 1
    )
```

这里 `first_row` / `row_count` 的单位是 RoPE **unit 编号**，不是 token 行或 head 行；`rows_per_tile=1`。现有 helper 枚举 `owner=(first_tile+ordinal)%48`，再按 `producer_base=24`、`producer_stride=1` 映射到物理 signal 核编号24..71。这与 `vector_schedule.py:368–369` 的 `worker=block*2+subblock; core=24+worker` 一致。

即便某个生产者有多个 wave，也等待它完成整个阶段的同一个代际；没有把第一 wave 的完成误当成之后 token 已就绪。候选没有按单块重复使用完成信号，没有新增 ring、ACK、GM 参数或 phase。

## 3. 发布、缓存、别名与空 worker

- `vector_schedule.py:593–603` 在原 `indexer_query_dequant_rope_stage` 返回后才发布 `PHASE_IDX_QUERY_ROPE`；该 helper 已完成本核所有 unit 和完整/尾行写回。原 `publish_phase` 中的 fence/cacheinvalid/fence、底层 notify 的发布次序不变。
- 16 个所需生产者均满足后，`wait_row_owners` 仍统一执行原 fence/cacheinvalid/fence；信号到了不代替业务数据缓存维护，没有依赖 Scalar cache 一致性或时间延迟。
- `binding.py:194–195,239` 将 BF16 `idx_query_rope[P16,8192]` 的同一连续存储 view 为 `[P16*64,128]`。元素偏移严格相同：`t*8192+h*128+d == (t*64+h)*128+d`。Hadamard 的 FP32 输出 `idx_query_had[T*64,128]` 是另一工作区。合法绑定仅允许声明的 view 别名，不能让自定义 allocator 把独立业务区任意重叠。
- 生产 unit 负责4个连续heads，head跨度256字节，unit宽1024字节；不同生产者写区不重叠。被排除的 AIV 只写其他完整 token/head 区域；不会稍后重写已消费的同一位置。下一次调用仍须遵守原顺序图/流与工作区生命周期。
- c≥T 时 Hadamard loop 没有迭代，不读 `idx_query_rope_heads`；helper 中只读的 Hadamard 常量矩阵没有本轮生产者依赖。原空 HAD 发布仍执行，AIV 后续等待合同不变。
- 随后的主 Q_B 读取 QR/只读权重；它所需的 QR 已在当前 AIC 前面 `PHASE_QR` 等待中获取。这里跳过空 HAD 等待不会让 Q_B 读取未就绪的 QR，也没有修改 Q_B 数学或顺序。
- 所有 AIC 均在此等待之前发布 query INT32 投影；AIV 不需要 HAD 的结果就能发布 RoPE。只删除已证明不读其数据的依赖边，没有把生产移到消费等待之后，也没有新增环。

## 4. 有界 CPU 证据

`check_owners.py` 不导入 PyPTO：先按每个生产者的真实 `range(worker, unit_count, min(unit_count,48))` 逐项建立 unit→worker，再反查每个 Hadamard worker 的全部 token/heads。它没有先把16核结论当成生产映射。

结果 `owner_audit.json`：

- T=0..4096，共4097种、98328个 AIC/T 情况，消费 token 实例8390656个。
- 每个候选集合与实际所需集合完全相等，无漏等或多等；每个所选 AIV 都是本轮真实生产者；原集合始终为超集。
- 覆盖 T8/9、16/17、23/24/25、全部24-token波边界、非8倍数尾行、T1与空循环；保存代表边界完整集合。
- 同时核对数学源码循环表达式、RoPE 尾行循环、原发布位置、合法连续 view 地址和输出有效范围。

T≥24 时，原24个 AIC 合计枚举1152次逻辑 `csa_signal_wait`，候选384次，少768次。T17..23 是1152→16T；T9..16 是768→16T；T1..8 是384→16T。**这是等待调用次数，不是硬件轮询次数或时间收益**。实际生成事件、缓存可见性、精度/图/保护区与完整 L1 仍待验证。

复现：

```bash
# 首次生成，目录已存在则拒绝覆盖
python prepare_source.py
# 只读冻结源，标准库 owner/AST 模型
python check_owners.py
```

`source_audit.json` 保留“一处等待还原后整模块 AST 相同”的证据。没有运行 symbolic/backend lowering；无需把本模型称作 CPU 编译通过。

## 5. 历史排重

只读检查主 `DSV4_CSA_SPMD_OPTIMIZATION_LOG.md`、动态 `ATTEMPTS.md`、两棵实验树中的 Markdown/patch，并直接读 `row_dependencies.py`、`indexer_input_dependencies.py`：

- 旧 row_dependencies 收窄的是 Score→TopK、TopK→plan、plan→QK、QK→OA。
- 旧 indexer_input_dependencies 收窄的是 Hadamard **量化之后** 的 query/scale→coefficients/Score，且为旧冻结 T96 合同。
- 已有 qb_dq_handoff_readonly 是主 QB→DQ 的有限 FIFO/反向依赖问题；ROPE_GATHER_EXPERIMENT 是 gather 数学搬运。

没有找到已实施或设备实测的本条 **Indexer query dequant RoPE→Hadamard 每 AIC 固定16 AIV** 优化。这里不把“所有按行依赖都是新发现”，也不将静态旧实验结果当当前动态候选的设备验证。

## 6. 下一步范围

可以由主会话对这份冻结源码单独执行一次完整 CPU 后端审计，再决定代表设备验证。应检查分支中的缓存维护位置和原 HAD 发布未变、无新增根/轴、空 worker 后续 QB 可达；设备必须使用同轮原分 task/正式参考/候选的完整正确性与 L1。未经这些验证，不采用、不预测百分比、不与 OA 小 T 候选叠加。
