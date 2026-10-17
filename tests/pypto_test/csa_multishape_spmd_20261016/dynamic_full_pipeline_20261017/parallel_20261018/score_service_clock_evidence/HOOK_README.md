# Score AIC 服务区间 hook：离线实现与冻结交接

2026-10-18。**测试 hook 已冻结，已有 formal668 原始 C++ 与 wrapper 的离线检查通过；未进行新的 PyPTO/CCEC 编译或设备执行。** 代码仅用于下一次诊断，不是性能优化。四个接口已经与独立 runner 对齐：

```python
instrument_cpp(raw) -> (marked, audit)
compilation_probe(directory)  # contextmanager，退出恢复后端
export_clock(signal, destination, phases) -> dict
check_probe_padding(signal, generation) -> bool
```

实现为 `score_service_clock.py`，纯标准库离线审计为 `check_offline.py`。不会在 import 时导入 Torch/PyPTO 或启动任务。`compilation_probe` 只在调用方已有编译发生时改本次 C++ 返回值；不安装依赖、不切共享环境。完整设计原文仍在相邻 `score_inner_clock_design_20261018`，本记录说明最终实现差异。

## 实际插桩和记录合同

从 C++ 的函数、循环嵌套、PTO 数学注释、Acc Tile 类型及 matmul 定义/使用关系定位唯一长 G6/N64/1024 的 16 QK＋16 WS；不依赖 SSA 序号/历史行号。仅 Cube 活动区插入 `CSA_SCORE_PROBE_BEGIN/END`，Vector 函数逐字保持。一个 wrapper 含双函数，不能要求 AIV wrapper **全文**没有 Score 字样。

- `body_total`：输入 acquire 后的 Score plan/所有 leaf 准备、step、原排空和末端局部等待，止于原 release 开始前。
- `step_total`：每个实际 step 的全部原业务，从条件复用 ACK 前到 READY 后；按本轮实际循环计数。
- `ack_issue`：原 step 复用及 leaf 尾部排空两个 FFTS ACK 站点，二者累计。
- `page_key_sample`、`qk_fix_sample`、`ws_sample`：每 step 固定 j=8 的分段。QK/FIX 两段、WS 三段保持原交错，Key 采样含为下一面板准备的预取；不得将这些值乘 16 外推。
- `local_wait_sample`：上述采样区内原 10 个 local wait 的嵌套辅助记录；不从别的区间中扣除，也不再相加当纯计算。

**20 个起始变量各自独立、仅声明/相减一次**；body、step、segment、ACK、local wait 不共享 start。7 个 uint64 累计和 5 个 count 在每次 kernel 入口清零，不跨调用复用。无 UB Tile、大 Scalar 数组或逐 panel GM dump。

仅在 Score 结束后的原 release 序列前一次写 `signal[34,AIC0..23,32:64]`。从原 BEGIN 的 GlobalTensor.data() 加 `34*72*128+32`，不另猜 coreid；24 核共 768 个 UINT32 地址唯一。schema/代际在32/33，leaf/step在34/35，七个64位累计在36..49，ACK/sample/local_wait在50..52，53记录固定10个local wait站点，54..63保留零。epoch最后写。

原业务 release 的 DCCI(ENTIRE_DATA_CACHE)/DSB 和 TNOTIFY 原样保留，用来发布新增padding。Score hook不新增硬件 fence；原 `phase_clock` 16..19 记录沿用它既有的维护。此次不启用FFTS20/21，也不能混用Attention tick的32..63。

`check_probe_padding` 检查所有phase/core的1..15、20..31、64..127为零，其他所有32..63为零；检查schema/epoch、24核发布代际、实际sample=step、ACK=step、local_wait=10×sample、leaf数量及保留字。**不要求每核22步**。ACK计时包含step外的leaf drain，所以只要求它不超过body，不错误要求它不超过step累计。空工作核仍写本轮schema/epoch与零计数。

## 本轮已有证据

`raw_hook_audit.json` 对应实际 `_compile_pto_module` 返回文本所在 `ptoas/dynamic_csa_full.cpp`；`wrapper_hook_audit.json` 对应同一668参考的 `kernels/aic/cube_schedule.cpp`。具体绝对路径/字节见 `offline_checks.json`。两者均确认：

1. 删全部 Score 标记后，原文逐字恢复；原事件序列不变。
2. Vector函数完整不变；新增代码只位于Cube活动guard内。
3. 16个QK/16个WS、固定j8和10个local wait锚点成立；20个start变量无重复或漏配。
4. 唯一一次32-word写回在原release之前，不碰其他物理核或旧phase时钟。

标准库槽模型覆盖实际steps为0/1/2/3/7/33的合成情况，17个保留槽/schema/epoch/计数反例均被拒绝；额外检查“leaf尾ACK时间大于step累计”仍合法。`synthetic_export/` **全部是人工构造信号的导出烟测**，不是设备时间、实际shape或精度报告，不得拿其中数字作分析。

复现仅处理既有文本：

```bash
cd /data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/score_service_clock_20261018
python check_offline.py
```

离线首轮raw已通过，随后对wrapper检查因函数声明拼写变成 `static __aicore__` 而拒绝定位；辅助函数已兼容两种已有文本，复核通过。这是只读定位合同修正，不是新的编译失败。还在冻结前修正了ACK与step的嵌套检查。失败与更正单独记入 `OFFLINE_CORRECTIONS.md`，没有覆盖设备/编译失败证据，因为本轮根本没有新编译或设备执行。

## 已知边界

Python AST/语法与C++词法/结构检查通过，不代表CCEC语法或设备执行通过。当前没有可读机器码反汇编的限制继续保留；不能证明最终get_sys_cnt指令的次序、寄存器数量或spill。后续只报告现成工具实际能取得的生成物/编译结果，不能把无可读反汇编自动记作新编译失败。

所有时钟都是Scalar代码区间和原有阻塞的观察，包含插桩扰动；未把MTE/Cube/FIX异步的纯硬件时间拆开。`eligible_for_performance=false`。独立runner拟做的A→B→A仅改变hidden正负，不能表述为history/页表/metadata变化验收。本模块未提交任何任务；是否进入下一步由主会话统一处理。
