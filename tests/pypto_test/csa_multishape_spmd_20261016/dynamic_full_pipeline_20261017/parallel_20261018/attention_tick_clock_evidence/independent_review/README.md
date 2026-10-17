# Attention 逐 tick 时钟：独立只读审阅

审阅日期：2026-10-18。本文归档已经完成的审阅，不增加编译、设备任务或性能测试。
对象是 `ultra_attention_tick_clock_20261018` 的 hook、导出驱动及其已有 `cpu_v1`
生成 C++，限定冻结 d253 源的 128K/B16、S6 诊断。正式业务源码和依赖未修改。

**结论：未发现本档诊断的业务或同步阻塞。** 本结论是源码和现有生成代码审阅，
不代替设备精度、图重放、记录完整性检查，也未验证本次新二进制的时钟指令次序。

## 记录区、所有权和容量

- `signal` 为 `[60,72,128]` INT32，每个 `(phase, physical_core)` 独占 512 字节。
  AIC 使用 `core=block`；AIV 使用 `core=24+2*block+subblock`，两条 AIV lane 不串写。
- hook 从本函数第一次 phase0 通知的 Tensor 获取已带本核偏移的指针：实际 CPU
  生成物中 AIC 是 `v932`，AIV 是 `v1522`，没有另猜设备核编号。
- tick 记录写 `[32:64]`；真实信号在 `[0]`，原 phase clock 在 `[16:20]`，
  溢出标记在 phase0 的 `[64]`，区域互不重叠。
- 已完成的有限地址枚举覆盖 `60×72=4320` 个记录槽，全部唯一；最大访问 word
  为 `552895`，小于 `60×72×128=552960` 的容量。各记录起点按 64 字节对齐。
- `tick>=60` 不越界写记录，只写本核溢出标记；导出遇到非零溢出标记即失败。
  导出另要求本档每核 tick 集合完整：AIC 为 `0..21`，AIV 为 `0..22`。
  这是单档诊断约束，不能解释成所有动态形状已覆盖。

## 原文、同步和可见性

对实际 `cpu_v1/kernel/ptoas/dynamic_csa_full.cpp` 进行只读文本插桩/剥离，
原 C++ **逐字恢复通过**，原同步调用序列相同。未新增业务通知、等待或 pipe fence。
两处 SCORE wait 和两处 KV READY send 分属互斥的大小档分支，不会在同 tick
先后覆盖同一个字段。

实际 AIV wrapper 的 59993–60013 行保留原阶段 `PIPE_ALL / DSB / DCCI ENTIRE /
PIPE_ALL / DSB`，随后发布 `ATTN_VECTOR`。AIC wrapper 的 18199–18229 行保留
原循环排空、整缓存维护与 `ATTN_CUBE` 发布。因此逐 tick 普通 Scalar GM 写入
仍由原阶段发布序列完成可见性维护；没有用“时间已经过去”代替缓存协议。
驱动在 stream 同步后取回记录，A→B→A 保留 signal 代际，不逐轮清零。

“没有新增 fence”仅描述 tick hook 本身。组合使用的既有 phase clock 仍在每次
通知后写入自己的记录，并执行该记录缓存行的 DCCI 和 DSB；这些开销仍在诊断中。

## 时间字段的准确边界

- `tick_begin` 在本轮局部记录初始化之后；`tick_end` 在 128 字节记录写入之前。
  记录写入成本会落入相邻 tick 的间隔或整体阶段时间，不能把各 tick 区间相加
  当作完整 Attention 耗时。
- 最后一轮 `tick_end` 在原循环外 drain 之前，AIC/AIV 的最终排空未包含在它内。
- `ring_store_issue_end`、`l1_load_issue_end` 是 TSTORE/TLOAD 发射后的 Scalar 位置。
  `publish_end` 是最后 TEXPANDS 发射后的位置；均不是 DMA 或 Vector 完成时间。
- wait 的前后时间包含原位依赖和插桩影响；条件未执行的字段保持零，应解释为
  “本 tick 未执行”，不能当成耗时为零。不能据此把剩余区间命名为纯数学耗时。
- 空 inline asm 只提供编译器内存重排约束。本次审阅不把 C++ 顺序当作新二进制
  SYS_CNT 指令顺序的证明；所有时钟结果只供归因，不能作为性能收益证据。

## 已发现的归档路径问题

审阅当时 `cpu_v1_codegen/audit.json` 的 `raw_cpp_path` 写为
`cpu_v1/kernel/dynamic_csa_full.cpp`，该路径不存在；真实 hook 前原物为
`cpu_v1/kernel/ptoas/dynamic_csa_full.cpp`。已通知负责代理修正或补记实际路径。
这是归档指针问题，不是编译或业务错误。本文保存发现时的事实，不推断之后是否
已修正，也不为归档重新审阅源文件。

完整绝对路径、归档时字节数及证据角色见 `evidence_inventory.json`。不复制大生成物，
不做 hash；本目录只新增本文和路径清单。
