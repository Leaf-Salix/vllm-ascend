# Score AIC 原位 Scalar 累计时钟：首版设计

**仅设计，未写 hook、未编译、未上卡。** 目标是缩小正式 668 长 G6 的 AIC 供数链成本归因，不修改数学或事件，不测量新的候选性能。先采用“全 step 粗分＋每 step 一个固定稳态 panel 细分”，避免 16 个面板全面插桩改变原流水。

## 原物与可复核锚点

唯一设计参照：`PAR/oa_pair_manual_h131072_b16_v1/build_output/decode_csa_spmd_jp7o4g2r/kernels/aic/cube_schedule.cpp`，PAR 的绝对根及文件大小在 [design.json](./design.json)。这是同作业正式 668 reference 的 long/G6/N64/1024 产物；两函数共存的 CPP **只改 `#if defined(__DAV_CUBE__)` 活跃的 cube_schedule**。实际 hook 须按 PTO 注释＋Tile 类型＋调用关系定位，不能硬编码下列 SSA 名/行号。

| 拟记录 | 原 C++ 锚点 | 累计范围 |
|---|---|---|
| `body_total` | `v2564 = (v39)[v267]` 11993；结束在 17545 后 | 所有原 Score 输入获取之后，到原 Score release 之前；含叶计划/准备/步循环/排空 |
| `step_total` / count | `for v4679` 12454，至 `ffts_cross_core_sync(PIPE_FIX,v4145)` 17526 | 全部真实 score step；包括该步复用 ACK，内部数学和 READY 发射 |
| `ack_issue` / count | 12459 的条件 ACK、17537 的叶尾 drain ACK | 两个原 `wait_flag_dev` 站点前后；原分支、次数不变 |
| `page_key_sample` | 第九个 QK 对应 `buf_panel_keys...ssa_v120` 15169，至预取 Key 的 `TEXTRACT` 15378 | 固定 j=8 的页解析、guard、TLOAD、原局部 wait、L1→L0 发射；此处预取的是下一面板，不能叫本 QK 纯加载耗时 |
| `qk_fix_sample` | 15414–15417 的 QK；15430–15436 的原 FIX wait/TINSERT | 两个不相邻区间相加，**跳过夹在中间的 WS**，不挪动原语句 |
| `ws_sample` | 15399–15406 的前面板 Right 提取；15424–15429 的 WS；15437–15469 的输出设置/TSTORE | 三个不相邻区间相加；保留 QK→WS→FIX→store 的原交错顺序 |
| `local_wait_sample` / count | 只限上述 j=8 区域的原 `wait_flag` | 与 page/math 区间嵌套，单列辅助解释，不重复加总；不把前后 Scalar 时间命名为引擎完成时间 |

j=8 是现有 16 面板中一个**编译时固定站点**，不新增运行时选择分支；每个实际 step 都执行一次，覆盖不同叶/step 的供数状态。不是各 panel 的代表性证明，**不能乘 16 推算整个 Score**。先保留总 step、叶数、ACK 次数与 sample 次数作合同核对；若首版不能解释差异，应据结果决定是否需要其他观测，不能预设继续扩探针。

## 槽与可见性

本地每调用清零 7 个 `uint64_t` 累计量和少量 count，只保留一个临时时钟；不使用 UB Tile、不建大 Scalar 数组。对 `get_sys_cnt()` 使用现有空 `asm volatile("" ::: "memory")` 编译重排约束；**没有硬件 fence**。仅在 Score 所有叶结束后、原 17546 的 release 序列开始前，一次写本核 `signal[34, physical_aic,32:64]`。指针从首次 BEGIN 的 `TNOTIFY(v933,v936,...)` 的 `v933.data()` 导出，再加 `34*72*128+32`；原常量 `v312=313344=34*72*128` 可交叉核对。不重新猜 AIC/AIV 编号。

拟布局：32=schema、33=generation（最后写），34=leaf_count、35=step_count；36..49 为上述七项 uint64 的 low/high；50=ACK_count、51=sample_count、52=local_wait_count，53..63 写零保留。24 个 AIC 各写自己的 512B 槽；不碰信号项 0，兼容原 phase-clock 的 16..21。**本轮不同时启用 Attention tick hook**，后者会使用各 phase 的 32..63。

写入位置在原 `pipe_barrier/DSB → DCCI(ENTIRE_DATA_CACHE) → DSB → TNOTIFY` 之前，使用原 release 发布诊断数据，不新增 DCCI/DSB/PIPE_ALL；无需设备消费者轮询。Host 仅在 kernel/graph 同步完成后读取，检查代际和计数。空 owner 也写清零计数和本轮代际，不能误读上次结果。这里的 512B 是既有槽隔离合同，不称为硬件 cacheline 大小。

## 判读及进入实现前的约束

这些量描述 Scalar 发射区间及区间内原有阻塞；MTE2/MTE1/M/FIX 异步重叠，WS 可早于前一 QK/FIX 完成，局部 `wait_flag` 也不能未经验证当 Scalar 的完整停顿。只用 `*_issue`/`*_sample` 命名；总 step、nested wait 与子区间不相加冒充纯硬件服务时间。采样会增加 Scalar 工作、寄存器压力与指令布局，必须标记 `eligible_for_performance=false`。

如后续获准实现，最小审查应要求：去标记后原 CPP 逐字恢复；原 Tile/算术/store 与完整同步序列不变；只改 cube 分支；48 AIV 无新写；独占槽/代际/计数匹配；真实 CCEC 产物核对时钟顺序与新增 spill，失败即停止。当前没有实现/生成物 PASS，也不承诺可从本设计分离 QK、WS 的纯引擎耗时。
