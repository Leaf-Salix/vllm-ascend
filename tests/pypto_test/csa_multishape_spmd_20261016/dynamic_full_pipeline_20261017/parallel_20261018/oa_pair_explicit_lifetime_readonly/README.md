# O_A 双输出显式生命周期设计审计（2026-10-18）

**结论：有一个与旧探针不同、资源账可成立的有限源码设计，值得在获准后做一次独立 CPU 资源验证；目前没有实现、编译或设备结论。** 推荐显式 K256 L1 当前/下一组三元组，加普通 `range` 内的固定四次 MMA 交错。K512 单层普通 `range` 虽能容纳单组 Mat，但丢失原来的 L1 双缓冲，不作为推荐候选。正式 `66882634` 未修改。

本次只读原源码、旧八次结果与已生成 C++。下面的资源表是生命周期设计，不是新 allocator 实测；不能由此推导整层时延收益，也不能把旧资源探针当成已接入当前动态 CSA。

## 1. 旧尝试真正卡在哪里

旧目录：`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/oa_pair_source_v1`。`evidence.json` 保留各 `result.json` 的准确路径、原错误与大小。

| 次序 | 旧尝试 | 实际结果 |
|---|---|---|
| 1 | nested_v1 | rank-3 Mat 不能直接 `tile.extract`，须先 reshape 成二维 |
| 2 | nested_v2 | 动态 valid_rows 的 Acc 必须使用一致的 compact 布局 |
| 3 | nested_v3 | 外层 K256 stage2 加内层 K128 stage2，Right 131072 B 超过 65536 B |
| 4 | flat_v4 | Left 变量重赋值的 valid_shape/compact 类型不一致 |
| 5 | flat_v5 | 根据微步分支选择更新 acc0/acc1，产生无法 coalesce 的 Acc if-phi |
| 6 | sequential_k_v6 | 首轮加载不匹配的 Simpler 扩展，属于环境失败 |
| 7 | sequential_k_v6_envfixed | CPU 通过；Mat 384 KiB，Left/Right 各 64 KiB，Acc 128 KiB；Right 请求 depth2 实际降到1 |
| 8 | right_lifetime_v7 | 把 Right0→MMA0→Right1→MMA1 写成交错后，内外 pipeline 仍保留，Right 仍 128 KiB，失败 |

所以不能仅把 v7 的语句再移动一次或调整 stage。关键是**取消自动 pipeline 对所有 Right stream 的再次复制，显式表达同时只有两个总 Right 值活跃**。v6 已有两个总 Right 槽，但其内层每轮在所有提取后才依次两次 MMA，再整轮回边；`kernels/aic/oa_pair_n_t96.cpp:461–505` 可见内层回边的 `M→MTE1` 等待，以及分别读取两路 Right 的 MMA。不存在已完成的新交错方案实测。

`hc_pre.py:208–245` 的参考价值只在局部生存期表达：普通 range 中先放两份相邻 Left/Right，计算较早的一份，再准备下一份。该 HC 路径的外层供数靠已有 cross-core 双槽；不能照搬 T96、M32、K1024 配置或声称其 FIFO 是本 O_A 的现成供数机制。

## 2. 推荐的唯一生命周期

保持每输出 N128、每次 K128 的两条独立 FP32 Acc 链；两个输出只共享同一次 Left，不在 N256 Acc 中混合，也不改变 K 的前后顺序。权重仍是八组原 NZ 根布局，Mat 的 group 单例轴先 reshape 二维后 extract。

### L1：显式 K256 当前/下一组，末轮剥离

1. 序言加载第0个 K256 的 `(A, W0, W1)` 三份 Mat。
2. 普通 `range(0,15)` 携带这三份当前 Mat 和 acc0/acc1。每轮先发下一组三份 Mat 的 load，使当前和下一组同时活跃；随后消费当前 K256 的两段 K128，两个 Acc 均无条件更新。
3. yield 下一组 Mat 和两个更新后的 Acc。若后端可以按预期 coalesce，旧 Mat 的最后一个 MTE1 使用结束后才允许下一次 MTE2 覆写其槽。
4. 尾声单独消费第15个 K256，不发第16个不存在的预取；不引入末轮 `if` 选择更新哪个 Acc。

这里是源级方案，**尚未证明 PyPTO 对 Mat loop-carried phi 的分配一定只有两组三元组**。不能靠指定物理地址、全局 reserve 或修改依赖绕过。若它增加 Mat copy/第三组，或者预取因事件合并而被强制先等完，则该实现不合格。

### L0：每个 K256 的固定四次 MMA

用下标 `0/1` 表示该 K256 内的两个 K128，`a/b` 表示两个 N128 输出。可表达为：

```text
准备 L0、R0a、R0b
acc_a = MMA(acc_a, L0, R0a)    # R0a 的最后使用
准备 L1、R1a                  # L0/R0b 仍活跃；R1a 可复用已结束的 R0a 槽
acc_b = MMA(acc_b, L0, R0b)    # L0、R0b 的最后使用
准备 R1b                      # R1a 仍活跃；R1b 可复用已结束的 R0b 槽
acc_a = MMA(acc_a, L1, R1a)
acc_b = MMA(acc_b, L1, R1b)
```

这是执行依赖的说明，**不是说发出 MMA 即可不等待而覆盖 Right**。真实实现仍依赖后端插入正确的 `M→MTE1` 重用事件；`MTE1→M` 也必须保留。两个 Acc 每轮都走同样的直线计算，所以不重现 v5 的“按微步奇偶只更新其中一条 Acc”控制流。Left/Right 不跨普通 range 携带；每轮局部展开四次 MMA，减少类型回边与长期活跃区。

该顺序希望给 R1a 的搬运与后一条 MMA 提供重叠机会。它并不等于两条 N stream 各自独占双缓冲：总计始终只有两个 Right 槽，不能把它描述成四槽吞吐。至少保留每个 K256 内的交错，不承诺跨 K256 边界也能满重叠。

## 3. 资源账与 K512 备选的边界

| 存储 | 单份大小 | 同时活跃份数 | 设计峰值 | v6 既有峰值 |
|---|---:|---:|---:|---:|
| Mat A128×256 BF16 | 64 KiB | 当前/下一各1 | 128 KiB | — |
| Mat 两路W256×128 BF16 | 每路64 KiB | 当前/下一共4 | 256 KiB | — |
| Mat 合计 | — | 6 | 384 KiB | 384 KiB |
| Left128×128 BF16 | 32 KiB | 2 | 64 KiB | 64 KiB |
| Right128×128 BF16 | 32 KiB | 2 | 64 KiB | 64 KiB |
| Acc128×128 FP32 | 64 KiB | 2 | 128 KiB | 128 KiB |

因此本方案没有依赖比旧 v6 更多的片上容量。但 Acc/Right 已到旧探针的边界，完整融合中的其他长活跃 Tile 和跨 helper 分配仍须另查，独立内核资源可容纳不代表整层可容纳。

若改为普通 K512 range，一组三元组是 A128 KiB+两路W各128 KiB=384 KiB，可装下；再放下一组三元组就需768 KiB，超过本地 910B/A3 后端的 512 KiB Mat 容量（`.worktrees/pypto-csa-static-fixes-20261016/include/pypto/backend/910B/backend_910b_handler.h:106`）。只保留一组则把原 L1 预取交给更短的 L0 局部交错，无法凭源码保证抵消。历史也已有 O_A K512 与 N256 方向的性能记录，不能把 K512 字面变化算新优化证据。故不把 K512 普通 range 单独列为另一个候选。

## 4. 数学、动态接口与同步限制

- 每个输出 Acc 的 K 顺序仍为 `0,128,…,3968`，首个 K128 初始化，后续全部累加。只借用旧 pair 的两条链；未来与当前正式 Tensor matmul 的实际 lowering/设备数值仍需核对，旧 CPU 通过不等于整层数值证明。
- 实际行数来自 `min(128,T-row_base)`；两条 Acc 用同一个 compact valid_rows 表达式，禁止一处 `min(rows,128)`、另一处裸 rows 造成 v4 式类型差异。T=0 无任务，尾行 store 只覆盖真实行，不能用探针中的 capacity384 代替 T。
- NZ `W[8,4096,1024]` 根形状及原物理 stride 不改；输出实际 stride8192、不改变 GM 数学布局。动态 M 行块、组和列所有权必须由真实 T 推导，不复制旧静态探针的 `worker//3`、`group*384`。
- 旧探针只计算24个 pair，另16个 singleton 仍是合同设计，不是完整覆盖实现。后续若接入整层，pair 两个输出的消费者必须等待同一 pair owner，singleton 和所有组的量化/零 pad 完成依赖也要一并闭合。不能只替换数学 helper 却沿用旧 N128 owner 等待公式。
- 本轮不修改任何 phase、通知、FFTS 或缓存协议。局部 Mat/Left/Right 生命周期由硬件事件保证，不能以时间早晚替代可见性。

## 5. 是否值得下一步

**可以保留为一个有依据的独立资源 CPU 候选，尚不应直接提交设备或迁入完整 CSA。** 最小检查是一次只验证上述显式 K256 设计：两 Acc 均 compact、两个 Right 总槽、六个 Mat 槽、没有非法 Acc copy，核对真实 MTE2/MTE1/M 事件和原 K 序。若 Mat phi 增槽、Right 再膨胀、或者生成事件把预取和四次 MMA 串得不如 v6，就应止步；不继续换 stage 或扫 tile。

本次没有生成候选源码，没有 CPU 编译，没有 NPU 测量；不存在本方案的精度或性能数字。所有旧错误原文与证据路径见同目录 `evidence.json`。
