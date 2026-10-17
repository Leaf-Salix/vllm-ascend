# DQ 完整八行块的本地交换索引候选

最终状态：**停止于 CPU，不采用、不上卡、不扩测。** v1 因 A3 不支持 INT32 TXORS 失败；唯一修正 v2 完整 CPU 通过，但 INT16 TCI 与 TXORS 的额外向量工作和事件变化削弱了收益依据。没有设备性能结论。

本目录是独立候选：以正式 `9d5327e5` 的全部 33 份源为基准，保留已采用的整波 head 分组，不叠加已否决的连续分工或小 T QA。只替换 DQ 完整八行块的索引准备，保留共享 rope_swap 的生产、KV 消费、DQ 尾行及浮点数学、owner 和跨核同步。本轮没有 NPU 执行、精度或性能结果，也没有修改正式仓、依赖或安装。

## 源码与数学合同

`source_before/` 是 9d 的 Git 原文；已与当时正式目录及既有 `query_dequant_group_20261018/source_candidate/` 逐字比较，全部 33 文件一致。`source_candidate/` 是 v1；`source_v2/` 是唯一获准修正版。只修改各自的 `query_projection_stages.py`，差分分别在 `candidate.patch`、`candidate_v2.patch`。文件开头及索引准备处有中文边界说明。

完整八行局部序号 `j=64*r+c` 满足 `j XOR 1 = 64*r+(c XOR 1)`；512 项逐整数相同。这里依赖完整 CSA 的 prepare 合同，不能把 helper 当作支持任意自定义 swap 的通用 gather。KV 继续使用原 GM 行内 0～63 置换；DQ 仍等原 RoPE 发布，因为 sin/cos 不变。索引 Tile 仅在本次 dq_work 内准备，不提升为跨动态循环的长活跃值。

`source_owner_audit.json` 保存全 33 源比较、替换索引片段后全部执行 AST 相同以及 T=0～384 的 owner 唯一写覆盖；`source_v2_audit.json` 保存修正版同样的 AST/32 文件检查与 FP32 中间转换的 512 个精确整数检查。未改变实际 T 动态轴、尾行条件、GM 输出范围或 runtime root。

## v1：INT32 TXORS 被 A3 后端拒绝

唯一首次完整 CPU 的日志为 `cpu_v1.log`，结果为 `cpu_v1/result.json`：符号 parse 通过，58 个动态轴保持；PTOAS 在 `source_candidate/query_projection_stages.py:223` 拒绝：

```text
'pto.txors' op expects A2/A3 txors src and dst element type to be i8/i16
```

这是前期只读设计的漏查：PyPTO `xors` API 与通用 INT32 Tile 类型并不代表 A3 实际支持。实际限制在 `/data/pyptouser/qinchuanyu/pto-eager/ptoas-src/lib/PTO/IR/PTOTile/PTOTransposeXorPrintMatrix.cpp:141–157`；A5 允许 i32，A2/A3 仅 i8/i16。没有绕过校验、改依赖或尝试其它编译路径。

`ir_audit_v1.json` 只审计失败前已经生成的 PTO 规划产物：三个 TXORS Tile 的物理区间互不重叠，src=65536/2048 B、tmp=67584/2048 B、dst=36864/2048 B；其它阶段全部 Tile 分配序列及完整地址包络与基线相同。DQ 少一处 2 KiB TLOAD、五次 TCVT、两次 TMULS、一次 TADD，增加一处 TXORS。由于没有可运行 C++，这些不是实际指令/事件或性能证据。

## v2：INT16 XOR 后显式两次精确转换

获得一次有限修正授权后，只在独立 `source_v2/` 用 INT16 `arange(512)` 和 TXORS，再显式 `INT16→FP32→INT32`。A3 `TCI.hpp` 有 b16 路径，PyPTO `tile_ops.py:849–889` 的 arange/ci 合同也明确支持 INT16；所有索引 0～511 都在 INT16 范围内。

额外核对 A3 `TCvt.hpp:753–812` 后发现没有 INT16→INT32 直接转换分支，不能依靠宽松的 TCvt verifier 或 CPU 生成成功假设它实际发出转换。v2 因此使用已实现的 INT16→FP32 与 FP32→INT32 两条转换；0～511 在 FP32 中精确表示，整数索引不改变，但两次 TCVT 是真实代价。不会为节省源码行数使用不支持的直接 cast。

复现脚本 `run_cpu.sh` / `run_cpu_v2.sh` 读取同一 `controls.json`（G6、平衡单根、OA128/OB96、关闭 query metadata 与 consumer reduce）。它们拒绝覆盖输出；本次总计只允许 v1 与 v2 两次完整符号 CPU，不重新编译基线，不进行第三轮或设备测试。

v2 的最终结果及真实生成审计在后面持续追加，不能把本节的预期写成已通过。


## v2 最终生成审计及停止理由

`cpu_v2/result.json` 完整 PASS：130 个 runtime 根参数、58 个符号动态轴、一次 MIX 提交、原 sync_start；没有提供实际 B/T 样例静态化，也没有设备执行。`source_v2/` 与两份旧源均已冻结，不再修改或第三次编译。正式仓后来纳入的小 T QA 条件不在本独立消融中，本候选仍严格比较 9d。

`ir_audit_v2.json` 给出真实规划地址：TXORS src=67328、tmp=65536、dst=68352，各 1024 B，三者互不重叠；最终 INT32 gather 索引地址仍为 36864、2048 B。TCI 临时区为 65536～67328（1792 B），完成 TCI 后才复用其开头为 XOR tmp；TCI 输出从 67328 开始，没有与该临时区重叠。其它阶段全部 Tile 地址/类型/尺寸/分配顺序与基线相同；完整地址上界为 AIC Mat=475136、Left=65536、Right=65536、Acc=131072、Vec=16384 B，AIV Vec=163872 B，均未增加。这些是实际 allocator 地址包络，不是测得的同时驻留峰值。

实际 C++ 的索引准备删除一次 2 KiB GM TLOAD，五次旧 TCVT 改为两次新 TCVT，删除两次 TMULS 和一次 TADD，增加一次 TXORS。AIC C++ 归一化后完全相同；DQ head/尾行及其后全部阶段的 C++ 去除同步后归一化相同，浮点数学及 store 未变。原始片段在 `baseline_prep.cpp.txt`、`candidate_v2_prep.cpp.txt` 和 `prep_generated.diff`，整体结果在 `cpp_audit_v2.json`。

不能由这些表层调用变化直接认定更快：

- A3 TXORS 仍展开为六个 Vector 操作阶段和五个内部 PIPE_V。
- INT16 TCI 使用 448 个 FP32 scratch（1792 B），原 INT32 用 192 个（768 B）。固定 512 项、按当前模板循环次数做源码调用模型，TCI_b32 为 21 次 Vector intrinsic 调用，TCI_b16 为 34 次；其内部 PIPE_V 从 8 个到 10 个。这不是反汇编机器指令计数，也不是周期测量。
- 整个 AIV C++ 显式 PIPE_V 少 4 个，但新增 TXORS 的 5 个及 TCI 的 2 个内部 PIPE_V；这几个明确变化合计反而多 3 个 PIPE_V 调用。没有把不同 TCVT/TLOAD 模板内部开销合并成一个不可靠的全程序计数。
- 局部事件重排超出了索引几行：完整 AIV 的 set/wait 静态调用各少 2 个，DQ 前的 MTE3 事件分配、DQ head-loop 相关 EVENT_ID、DQ 后的等待位置均有变化。DQ 发布并完成后续 producer 等待后、进入 Indexer group-plan 前，原一处 MTE3→V EVENT5 等待变成 PIPE_ALL；后续另一处 MTE3→MTE2 等待前移。完整外层 PIPE_ALL 总数不变，但位置改变，不能声称“所有同步逐字相同”。`event_changes_v2.json` 保留精确差异序号和调用内容。源码中的 phase notify/wait、FFTS READY/ACK、缓存发布没有修改。

据此，本轮仅得到“删除一次重复索引读取、资源未超，但增加向量准备与局部同步代价”的结论，缺少继续消耗设备验证的充分依据。经主会话确认，停止在 CPU，候选不合入正式；也不把它称为已测退步或精度通过。原 v1 后端失败、v2 编译产物和全部审计均保留。

## 文件与可复现范围

- `prepare_and_audit.py`：从 Git 9d 提取 33 源、生成 v1 与源级/owner 证据；拒绝覆盖冻结源。
- `prepare_v2.py`：复制 v1 为独立 v2，仅作已说明类型修正和源级精确性检查。
- `run_cpu.sh`、`run_cpu_v2.sh`：各自完整符号 CPU 命令，已运行各一次；本次不再执行。
- `audit_ir.py v1/v2`：只读现成 PTO 分配与操作；v1 结果明确后端失败。
- `audit_cpp_v2.py`：只读现成 C++；准确保留事件不相同的事实。首次审计的“所有变化只限准备片段”断言未成立，已改为报告实际差分，没有修改候选以满足断言。
- `isa_support_excerpt.txt`：本次依赖版本的实际接口/实现片段，仅小文本副本。
- `LOCAL_ARTIFACTS.json`：源、结果、日志和编译产物真实路径/字节清单；不含 hash。大型完整产物留在本目录本地，归档时由主会话按清单选择，不复制权重或二进制到 Git。
