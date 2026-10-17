# Q_B 全 K 权重常驻的反方审查

2026-10-18。基线为正式 `66882634`；只读源码、已有生成物和历史日志。本审查没有编译、设备执行、依赖修改或内容摘要计算。当前 `_q_b_nz_stage` 与 `adopted_source_d56c9eb6` 中同名函数的 AST 直接比较一致；因此可以用下述旧生成物说明该函数已有的分配和事件，但不能把它称为新候选的生成结果。

结论：对于 **128<T≤256**，把每个 N256 的 K1024 权重留在 L1、跨两个 M128 复用，有明确的逻辑 GM 读请求减少：64 MiB→32 MiB。它没有减少四段 K256 MMA，也没有减少 Mat→Right 传输；Right 单槽和跨 N 回边仍可能限制收益。320 KiB 的 Mat 容量只是必要条件，生成代码必须确认 QR 两槽、Right 覆盖依赖、首块启动及循环回边。不能据此预记完整 L1 性能。

## 源码与历史边界

路径缩写：

```text
C=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
P=$C/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd
PAR=$C/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018
G=$PAR/compact_priority_h131072_b16_v1/build_output/decode_csa_spmd_ad5etrbf
DEP=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016
ISA=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/simpler-csa-core-args-256-20261016/build/pto-isa
```

正式源 `P/query_projection_stages.py:120–155`：N256 的 owner 是 `n % 24`；M 循环按 `ceil(T/128)`，每个 M 用 K0 初始化 INT32 Acc，再按 K256、K512、K768 追加，store 只写有效行。

历史 `2de2baa6:.../deepseek_v4_flash_dspark/q_projection.py:81–109` 在每个 N256 外提完整 B1024，使用 M64，并逐 M 载入完整 QR K1024。`74be8a6c` 的同文件 `:92–118` 改为 M128/K256 streaming，注释明确针对先全 K 搬完再计算的串行窗口。这不是单独开关 B 常驻的实验：M 和 QR 搬运也变了。

`C/tests/pypto_test/DSV4_FLASH_CSA_VALIDATION_LOG.md` §110 记录过 M64/N256/全 K B 的正向结果，但同时改变 N/M/K 分块。`DSV4_FLASH_HCA_OPTIMIZATION_LOG.md:166–176` 的 v10 记录 M128/K256 streaming 相对旧路径改善：8K 相对 Native 从 +5.87%→+2.82%，128K 从 +0.82%→+0.56%。均不能直接代表本次融合的双 M128 B-only 候选。§110.2 的全 K B 常驻负例属于 **O-B**，不能误记成 Q_B 同方案失败。

另一个直接相关的负例是 `PAR/qb_left_doublebuffer_evidence/`：v1 虽恢复 Left 两槽，却把 QR/weight Mat 都压到地址 65536，新增串行覆盖等待；v2 恢复原 Mat 两槽后，完整设备五窗仍 870.98→876.70 μs，慢 0.657%、5/5 回退。它说明显式搬运与更好的局部槽数均不保证整链收益，不是本次 B-only 的重复实验。

## 容量之外的四个风险

1. **Right 不能跨两个 M 保存全部 K。** 一个 K256×N256 INT8 Right 已是 64 KiB，耗尽 L0B。双 M 仍需每 N 执行 8 次 Right 提取/搬运、8 次 Left 提取/搬运、8 次 MMA、2 次有效行 store。B-only 只能把完整 256 KiB B 的 GM 覆盖从每 M 一次变成每 N 一次；应检查外提的是 Mat load，而不是错误地把可覆盖的 Right 放到 M 循环外。两个 M 的 128 KiB Acc 也不能同时活跃，L0C 总共只有 128 KiB。
2. **完整 B 就绪会推迟首 MMA。** 若一次完整 TLOAD 的就绪事件保护所有 256 KiB，第一段必须等完整 B；当前 streaming 首段只要求 64 KiB B。节省第二 M 重读与增加每 N 启动延迟方向相反。不能仅数少了几个 TLOAD。
3. **完整跨 N 双缓冲不成立。** 两份 B256 KiB 加 QR 两份32 KiB共576 KiB，超过512 KiB。单份 B 只可在最后一次 Mat→Right 读完后覆写；如果生成器让下一 N 的 load 额外等最后 MMA/FIX，隐藏搬运的窗口还会缩小。
4. **流水组数量不是物理槽数。** B 常驻拉长活跃区间后，QR32 KiB 两槽、其他阶段 Mat、Left 等都可能被重新合并或换址。已有 `PH-MR-001` 只是线索，需同时看 `pto.alloc_tile` 地址、实际生产消费与相应 event。比较范围应覆盖 Q_B 以外分配，而非只验证最大地址不溢出。

现有 `G/ptoas/dynamic_csa_full.pto` 中，Q_B K256/K512 的 QR Mat 为196608/229376，B Mat为65536/131072，Left同址0，Right同址0，Acc0。最大 AIC Mat 末端475136；这是全函数实际地址上界，不是本阶段活跃量。CPU 候选不能直接把这些旧地址当预期固定地址。

## 上一 N 最后 MMA/store 是否自然重叠

`G/kernels/aic/cube_schedule.cpp:10837–10843` 的最后 Right 搬运后先归还 MTE1→MTE2 event1，再执行最后 MMA并发 M→FIX、M→MTE1 event0。`:10880–10888` store 后的 N 回边包含：

```text
wait(FIX→M,0)
wait(M→MTE1,0)
wait(MTE1→MTE2,1)
wait(MTE1→MTE2,0)
```

下一 N 在 `:10474–10477` 重新发 MTE1→MTE2 的0/1；随后首 QR/B TLOAD（`:10521–10555`）再等待这些许可。沿 MTE1 队列看，重新发许可排在上一次 M→MTE1 wait 后，所以旧代码没有证明下一 N GM load 可跨越上一 N 最后 MMA；其许可链反而受最后 MMA 限制。FIX→M 是另一条队列边，不能据文本顺序进一步断言 next B load 必等最后 FIX 完成。局部 `wait_flag` 不是 Scalar 完成计时，以上判断基于管线事件边，而非 Scalar marker 时长。

新候选必须报告：最后一次 `TEXTRACT/TMOV(Right,B)` 后，释放 B Mat 的 MTE1→MTE2 set 在何处；下一 N 的 B TLOAD 究竟等它、最后 M，还是最后 FIX。若仍是同样回边，不能宣称“全 B 搬运自然藏在上一 N MMA 内”。不建议在本候选里同时改事件或跨核协议。

## 数学和动态尾块的生成验收点

| T | M 起点 | 每 M 的 valid rows | 必须保留 |
| --- | --- | --- | --- |
| 129 | 0,128 | 128,1 | 第二次首 K 初始化新 Acc，四 K 都只声明1有效行 |
| 255 | 0,128 | 128,127 | 第二次 store 仅127行 |
| 256 | 0,128 | 128,128 | 不增加 t0=256 迭代 |

每个 M 的 INT32 数学顺序必须仍是 K0 的 `matmul` 初始化，再 K256、K512、K768 三次 `matmul_acc`；保持 N owner、M 顺序、valid shape、NZ K/列地址与输出 stride。历史 v10 特别记录直接 `create_tensor` 后以部分 valid rows 做 `matmul_acc` 曾触发 `AccCompactValid`，不能为常驻改回该初始化方式。检查 Acc 的 Compact/valid 属性以及 GM partition 的动态行数，不能只比较 tile 的静态128行容量。

`T>128` 不自动等于两个 M：T257还需第三块。若设计固定双 M/QR容量256，策略必须限制 `128<T≤256`；否则保留动态 ceil(T/128) 的完整循环。T/B仍是符号维度，不按样例冻结。所有核间 publish/wait、QB 消费者和正式数学阈值不在本次改动范围。

## 追加容量问题：QR256全 K + B1024 是否恰好可用

**从本机实际后端与 ISA 合同看，512 KiB 是可用 Mat 上限，没有查到固定 Mat 保留底座。** `DEP/src/backend/common/soc.cpp:119–128` 的910B后端配置 Mat512 KiB、Left/Right各64 KiB、Acc128 KiB；只有 Vec 在 `:133–140` 明确从物理192 KiB扣到184 KiB。`ISA/include/pto/common/buffer_limits.hpp:46–56` 对 A2A3 同样给 Mat512 KiB。`allocate_memory_addr_pass.cpp:368–374` 的 reserved_start 默认0，只有显式 reserve 才抬高。正式33源没有 `reserve_buffer`；已有 AIC PTO `:274–275,3685` 的 Attention qk_q 确实放在 Mat地址0，排除了“所有Mat永久从65536起”的解释。

因此，QR INT8 `[256,1024]`=262144 B 与 B INT8 `[1024,256]`=262144 B，正好524288 B；二者均满足后端128 B对齐。按源级真实生命周期，QR在 QB N循环外、B在 N循环内，可让两个 M128保留原8次MMA，避免QR192/M64方案多50% MMA的问题。不能仅以容量拒绝。

直接提取能力也存在：`DEP/python/pypto/language/op/tile_ops.py:657–689` 的 `tile.extract` 支持 Mat→Left/Right；`ISA/include/pto/npu/a2a3/TExtract.hpp:160–182` 与 `common/arch/memory/textract_common.hpp:432–488` 直接写 L0A/L0B，没有必须分配中间 Mat 的 API 要求。源 INT8 和256/1024维度受支持；row偏移0/128、K偏移0/256/512/768均满足分形对齐。

这是**硬件/API容量可成立、尚未证明完整编译分配成立**：恰好满额，任何额外Mat clone、物化切片或被编译器保守延长到QB的其他Mat都会失败；旧代码中Attention loop-carried Mat地址0不能被当作永久reserved，也不能未经实际生成就断言新两块能复用它。需要生成结果只有两个同时活跃256 KiB Mat，窗口提取目标直接为 Left/Right，并保留动态 `validT`→`min(128,T−t0)`→Acc/store。QR源布局应使用正常固定capacity256的NZ物理步长；不能把动态validT误作物理Rows导致第二窗寻址改变。

QR常驻每worker只读一次，可把现有全体128个N块重复读取的逻辑QR字节 `128×T×1024` 降为 `24×T×1024`（24核各有5或6个N块；128<T≤256）。例如T192为24 MiB→4.5 MiB，减少19.5 MiB；T144为18 MiB→3.375 MiB，减少14.625 MiB。B仍为32 MiB。此为逻辑 GM 请求覆盖量，受缓存影响，不等于减少相同 HBM 流量，也不是微秒收益估计。

本报告交付时没有编译该追加方向；后续任何实现/CPU需独立保存并与正式668、B-only分开对照，不覆盖本次只读依据。
