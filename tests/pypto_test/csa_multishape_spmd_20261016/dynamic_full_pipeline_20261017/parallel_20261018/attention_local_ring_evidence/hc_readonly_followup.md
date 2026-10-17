# HC 后续只读筛选：门控偏置直接广播相加

2026-10-18，本轮只读分析。没有改生产源码，没有重新编译或测试，没有继续修改 mi/li 负候选。基于 canonical 当前 `hc_stages.py`、独立 `hc_pre.py`、已归档总账及已有生成产物，最多提出下面一个小候选。不把它写成已获性能收益，也不声称它能独立达到整层 30% 目标。

## 唯一建议

只改 `hc_stages.py:301` 与 `:309` 的门控偏置相加：

```python
# 当前
pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))

# 建议候选，尚未实施
pre_logits = pl.col_expand_add(pre_scaled, pre_base)
post_logits = pl.col_expand_add(post_scaled, post_base)
```

仅两处；不同时改 Sinkhorn、不改数学 tile、不改 RAW 或 consumer-reduce 策略、不改信号与分核。仍计算 pre 的完整八列，包括后续混合不读取的列4～7；不能借此删除原工作区数值，以免混入另一项语义变化。

## 对照来源和确切位置

以下路径均相对 canonical `vllm-ascend-dsv4-pto-0251rc1`：

- `vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/hc_stages.py:295-310`：两路门控各自先乘 inv、再乘 scale、再加 base，然后 neg/exp/add/recip。建议只融合第三步的地址广播，不融合乘加。
- `vllm_ascend/ops/pypto/hc_pre.py:408-411`：独立 HC_pre 的 `TAIL_MAX_ROWS==8` comb 路径已经使用 `col_expand_add`；其余分支仍为显式 expand+add。它证明现有源码有可借鉴的表达方式，不能把其性能结论直接移给完整 CSA。
- 已有 `tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/kernel_hc_consumer_reduce_v3/kernel/kernels/aiv/vector_schedule.cpp:34317-34325`：pre 路径仍明确生成 `TCOLEXPAND`、`pipe_barrier(PIPE_V)`、`TADD`。post 对应 `:34489` 后同样的组合。这个产物用于确认两处共有门控算术的真实指令，不能视作本次候选编译结果。
- 只读 PTO-ISA：`/data/pyptouser/qinchuanyu/pto-eager/pto-isa-src/include/pto/npu/a2a3/TColExpand.hpp`，`TColExpand` 对每个 `validRow` 调用一次 `pto_copy_ubuf_to_ubuf`。
- 同目录 `TColExpandAdd.hpp` 和 `TColExpandBinOp.hpp`：融合广播直接向量加 base，避免先物化完整广播 Tile；算术仍为 FP32 `vadd`，不是 FMA，也不是 recip 变换。

## 已有候选排重

只读搜索并核对 `ops/pypto/DSV4_CSA_SPMD_OPTIMIZATION_LOG.md` 与两份历史实验树中的说明、变换脚本/审计记录后：

- consumer-reduce：已做，当前默认关闭，长档回退、短档小收益；不重复。
- HC row dependencies / norm early：已做，不重复。
- reduce→split→mix 融合、融合内三处 Tile 直传：历史均完整通过但无明确净收益；本建议不做阶段合并或删除通知。
- BF16 原输入直读、FP32 紧凑双槽供数：已有静态全链候选与完整通过记录，不作为新候选。
- mix 整行驻留、分块 UB 复用：已有精度修正与性能记录；不再引入长期 Tile。
- Scalar Sinkhorn：涉及除法/归约数值边界，本次不碰。
- `norm_sq_sum/y_inv` 表面是临时 Tensor，但现有 `cpu_v1/.../37_after_MemoryReuse.py:5572-5580` 已把它们降为本地 Tile，没有对应 GM round-trip，因此不能把“删除该临时 Tensor”包装成减少 GM 搬运。
- 没有找到针对当前 HC 两路门控这两处 expand+add 的独立变换/性能记录。历史 whole-HC 迁移和其它阶段的 `col_expand_add` 不等于验证了这项原子变换的净收益。

## 数学、依赖和动态范围

`pre_scaled/post_scaled` 保留原 `raw * inv` 然后 `* scale` 的两次 FP32 舍入；`base` 的相加仍只发生一次，原 neg→exp→+1→recip 以及 pre+eps/post×2 都不动。两种表达计算的每个有效元素都是 `scaled[i,j] + base[0,j]`。不改变任何归约或 epsilon。

没有新增输入、输出、工作区、loop-carried Tile、phase 或跨核依赖。`HC_GATES` 的发布点、`HC_NORM` 消费点和尾部 scratch 协议全部保持。

源级可适用所有实际 `T>0`，也不依赖 history/B/S：门控本就按 M8 读取已分配的 PAD16 raw/inv 和写 pre，post 用原 `valid_rows` 分支裁切。先评估范围应为 `T%8==0` 的完整 M8 行块，七档都满足：T24、48、96、144、168、96、192。不需要精确 T 白名单，也不增加新的 Host shape 固化。尾行可沿用原表达的有效形状，但仍需将来编译/设备验证，不能沿用七档结果替尾行背书。

## 可确认减少什么，以及不能确认什么

每个 M8 块的两处原广播，各复制八行32字节；合计16次 UB→UB copy，写出两个8×8 FP32展开值，共512字节的局部复制。候选消除的是这些局部广播复制，不减少 GM 权重/输入读写，也不删除 sigmoid 算术。

| 场景 | 实际T | 每层可去掉的行复制调用 | 被省略的UB广播写字节 |
| --- | ---: | ---: | ---: |
| 128K B4 | 24 | 48 | 1536 |
| 128K B8 | 48 | 96 | 3072 |
| 128K B16 | 96 | 192 | 6144 |
| 128K B24 | 144 | 288 | 9216 |
| 128K B28 | 168 | 336 | 10752 |
| 8K B16 | 96 | 192 | 6144 |
| 8K B32 | 192 | 384 | 12288 |

这些是源码和现有 A3 ISA 实现支持的调用/字节计数，不是时间预测，也不保证 UB 峰值下降512字节，因为原临时量可能已复用同一地址。

**主要抵消风险：** 现有生成 Tile 的 valid shape 经常为动态值。`TColExpandBinOp.hpp` 在动态 `ValidCol` 下可能选择逐行 `vadd`（count mode），而原展开后的连续 `TADD` 可以更少的向量调用完成。也就是说，去掉八次局部复制，可能新增更多标量循环/向量发射；不能仅按 API 数减少断言更快。只有后续最小 CPU 产物核对确认指令路径可接受，才值得做完整精度、graph、保护/协议后同进程完整 L1 对照。

本轮到只读建议为止。没有候选实现、CPU 新编译或 NPU 新任务，七档性能状态不变。

根代理本轮决策：不为这两处小门控改动追加实现或完整设备对照；本建议留待后续有充分依据的向量广播批量优化一起评估，当前优先收取 Indexer 的完整实测结果。上表复制次数不折算为预期完整性能。
