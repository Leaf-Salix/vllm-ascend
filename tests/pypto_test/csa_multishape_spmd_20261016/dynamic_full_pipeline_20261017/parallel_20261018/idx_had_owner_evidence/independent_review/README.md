# IDX_QUERY_ROPE → Hadamard 的 16-owner 等待：独立源码复审

**结论：正式 66882634 的现行分工下，所提等待集合成立，未发现尾行、跨 wave、连续别名或后续 QB 的依赖反例。** 最小实现可只在 `worker < tokens` 时等待逻辑 AIV `[16*(worker//8),16*(worker//8)+16)`，物理核加 24；其后的 Hadamard、HAD 发布、QB 及全部发布/获取维护保持原样。此结论只证明源级数据依赖，不预记完整 L1 收益。

本轮只读正式包 `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd`，与主 OA 档案中的 668 参考 33 文件直接字节对照相同。没有写候选、修改正式源、编译或上卡。有限整数核对见 [review.py](review.py)、[review.json](review.json)：24 个 T 边界、576 个 AIC 情况通过；不是设备或后端验证。

## owner 等式与边界

`indexer_query_stages.py:119` 定义 RoPE 总单元数
`U=ceil(T/8)*16`，实际数学 worker 数 `A=min(U,48)`；worker r 按 `range(r,U,A)` 领取单元。单元 `floor(t/8)*16+hg` 写 token t 的第 hg 个四-head组，hg=0..15。

`indexer_query_stages.py:210` 的 Hadamard 每块 64 head，恰好一 token；AIC c 处理 `t=c+24*w<T`。

- **T>16：** A=48，且 `floor((c+24*w)/8)=floor(c/8)+3*w`。RoPE owner 为 `(16*floor(c/8)+48*w+hg)%48=16*floor(c/8)+hg`，所有 wave 都是同一段 16 个 owner。
- **1≤T≤16：** Hadamard 没有第二 wave。活动 c<T 的 `floor(c/8)` 只可能为 0 或 1，其对应 16-owner 段完整位于当前 A=16/32 中。
- **尾部：** T=17、25、33 等最后不足八行仍归同一 RoPE 单元，helper 的 tail 分支只改变有效行循环，不改变 owner；Hadamard 只遍历 t<T，不读 pad16 尾部。
- **T=0 或 c≥T：** 没有 Hadamard 数据依赖，可跳过该等待；不能因此跳过 begin 或已有阶段发布。当前 AIV 即使无 RoPE 数学仍发布该阶段，不能把“非实际 producer”误写成“没有发布”。
- **跨调用：** 依赖集合由本次实际 T 判断，仍读取本代信号；不需要清除旧槽，不更改 72 核代际。

可沿用原 helper 参数：`first_row=16*(worker//8), row_count=16, rows_per_tile=1, producer_workers=48, producer_base=24, producer_stride=1`。物理范围分别是 24..39、40..55、56..71。

证明依赖固定 24 AIC、48 AIV、每单元八 token/四 head、每 token 64 head；Indexer 的 G2/G6 策略不改变这些物理分工。若将 Hadamard 步长改成 16，T33/c0 会读 t=0/16/32，实际需要全部三段 48 个 owner，固定 16 段即错误。因此不能推广成任意 WORKERS 的通用公式。

## 连续别名、业务写入和获取

`binding.py:194,239` 分配连续 BF16 `[pad16(T),8192]`，并建立同址 `[pad16(T)*64,128]` view。地址恒等式是：

```text
2*(t*8192+h*128+k) = 2*((64*t+h)*128+k)
```

因此 Hadamard 读取 `[64*t:64*t+64,:]` 恰好是 RoPE 生产的该 token 全部 64 head、每 head 128 BF16 元素，不跨到另一 token，也不以工作区容量延长有效循环。T 来自 Hadamard 外层实际 Tensor 维度，RoPE 从 qr_scale 第一维取得同一 T；该一致性由未改的绑定合同保证。

每 head 为 256 B，nope/RoPE 两个 128 B 写回由同一 AIV 完成；每 owner 每 token 负责连续四 head、共 1024 B。所有 16 owner 的完成覆盖该 token 全部 16 KiB。其它 owner 段处理不同的八-token行块，未发现源级 payload 读写重叠；输出路径没有对 BF16 RoPE 缓冲做 Scalar 读改写。这里不把 signal 的 512 B 隔离跨度当成硬件 payload cacheline 大小或新的对齐保证。

`vector_schedule.py:593–603` 在整个 RoPE helper 返回、该 worker **所有 wave**完成后才发布一次。必须保持这一位置；不能把通知提前到首 wave 后，否则同一 16-owner 集合并不证明后续 wave 已完成。`publish_phase` 的 fence/cacheinvalid/fence、notify 前 PIPE_ALL 和原信号写侧维护全部保留。新范围仍调用原 `wait_row_owners`，读取同代 16 槽后统一执行 fence/cacheinvalid/fence，连续别名不会绕过获取维护。每个信号仍是按物理核隔离的单写者 512 B 槽。

## 后续 QB 与其它传递依赖

`cube_schedule.py:275–286` 在 Indexer query 投影前已经显式等待全部实际 `PHASE_QR` owner，并做获取维护。`q_r_stage:270–298` 在该发布之前已写完 qr、qr_scale、qr_matmul、qr_scale_pad 的本核全部任务。随后 QB 只消费 qr_matmul 和只读 wq_b，写独立 q_proj；**QB 不需要 IDX_QUERY_ROPE 代传 QR 可见性**，无 Hadamard 工作的 AIC 也保留了此前这次全 QR 等待。

Hadamard 输入为上述 BF16 alias 及只读 Hadamard 矩阵，输出为独立 idx_query_had。`PHASE_IDX_QUERY_HAD` 原发布和 `vector_schedule.py:697` 的后继全实际 HAD owner 等待都不变；这些活动 AIC 的 16-owner 并集恰好覆盖全部实际 RoPE 数学 owner。主 Query 路径本来已显式取得 QR/ROPE，compact Hadamard 也保留自己的 META/IDX_NORM 获取，不需要借本次全 48 RoPE 等待建立额外可见性。

该改动只移除不被本 AIC 消费的等待边，保留原生产者及发布位置，没有引入新的等待环、工作区复用、FFTS 或数学事件修改。源级安全并不等于后端事件序列已验证或性能一定改善。

## 范围与交叉检查

有限核对包含 T0/1/7/8/9/15/16/17/23/24/25/31/32/33/47/48/49/95/96/97/144/192/257/385；后两项用于跨多 wave 的代数校验，不声称已运行对应 Native 场景。T≤8 的原等待本就只有 16 个实际 owner，T9..16 为 32，T>16 才为 48。JSON 中的信号调用数只是源级工作量，不是性能收益。

已将独立 owner/别名/QB 推导发送 hc_tail_scalar_candidate 交叉检查。主会话另有 CPU 与生成代码审核，本报告没有代替或复跑它们。主会话可据各自证据决定是否提交代表档。

