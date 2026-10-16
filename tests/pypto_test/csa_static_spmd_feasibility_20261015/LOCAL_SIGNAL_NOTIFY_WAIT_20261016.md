# 普通 GM 核间 notify/wait：临时实现与验证

用户于 2026-10-16 明确要求先实现临时接口，未来主线
[PyPTO #3020](https://github.com/hw-native-sys/pypto/issues/3020) 落地后替换。
此前因 Scalar 非一致缓存讨论而暂停的实现工作已恢复。

本次只在独立 PyPTO worktree 增加接口，并在测试目录建立 128K/B16 候选；
没有切换共享安装，没有改 Simpler、PTOAS、PTO-ISA 或生产 CSA。
旧 `static/h131072_b16` 完整保留。

## 接口与替换边界

```python
pl.system.signal_notify(signal, [offset], value)
pl.system.signal_wait(signal, [offset], expected)
```

- signal 是普通一维 GM INT32 Tensor，不要求 DistributedTensor 或通信窗口。
- offset 支持设备标量；value / expected 必须为 INT32。Python 整数越界立即拒绝；
  动态标量必须由调用方保证范围，不能把溢出的 INT64 静默当成正确进度。
- notify 是单写者 Set；wait 是非消费式 >= 等待。多个消费者可以独立等待同一信号。
- 每条信号缓存行只允许一个写者，不与 payload、其他写者共享。
- 仅用于明确 AIC/AIV 核体；没有新增自动 MIX lane 放置规则。
- 所有必要生产者须可继续执行，不能用阻塞轮询等待尚未驻留的后续调度波次。
- 首次执行前初始化；代际单调递增。并发调用使用独立工作区，溢出前排空后重置。

CSA 通过 [local_signal_adapter.py](local_signal_adapter.py) 中两个 inline 函数调用。
主线接口可用后，优先只替换这两个函数；若主线改用不同信号类型或槽布局，
同时调整候选生成器的 progress section，不能机械替换名称。

PyPTO 改动位置（均在独立 `pypto-csa-static-fixes-20261016` worktree）：

| 文件 | 职责 |
| --- | --- |
| python/pypto/language/op/system_ops.py | pl.system 的 DSL 入口 |
| python/pypto/ir/op/system_ops.py | IR 构造、整数常量范围检查 |
| src/ir/op/sync_ops/sync.cpp | 普通 Tensor / INT32 / 一维 offset 类型验证；SyncOp 副作用 |
| src/backend/common/pto_ops_memory.cpp | 普通本地 GM 单元素视图，下沉到现有 TNOTIFY/TWAIT |
| tests/ut/language/parser/test_system_ops.py | round-trip、错误类型和越界检查 |
| tests/ut/codegen/test_pto_codegen_ops.py | 等待/通知顺序保留，无通信上下文、无新增 syncall |
| docs/en/dev/ir/05-operators.md、docs/zh/dev/ir/05-operators.md | 同步更新临时接口合同 |

## 底层复用及缓存语义

当前 A3 `pto::comm::TNOTIFY(..., Set)` 执行：
单行 DCCI → volatile GM 写 → 单行 DCCI → DSB_DDR → PIPE_ALL。
本原型只用 Set，未使用 AtomicAdd；单写者缓存行不需要并发加法。

`TWAIT(..., GE)` 使用 volatile GM 读取，并在每次检查前对信号行做 DCCI；
等待期间周期执行 PIPE_ALL，底层有 100000000 次轮询的断言上限。
它不是 ld_dev 版本，也不能从 volatile 推导出硬件 cache 一致性。
两者只接受 INT32，所以没有直接套在旧 INT64 进度 Tensor 上。
底层已有能力的复用不等于 A5 或任意协议已经验证。

**信号到达和 payload 可读分开处理。** 信号接口不承诺自动刷新所有业务数据：

1. 生产者完成该阶段计算后执行 fence → 全缓存 cacheinvalid → fence。
2. CSA notify 薄封装先执行 `bar_all()`（本核 PIPE_ALL），排空本核前序流水，
   然后执行 signal_notify。这一步是小例对照后补齐的，不能只依赖第 1 步的 gm fence。
3. 消费者等待必要前驱：signal_wait。
4. 消费者进入计算前：fence → 全缓存 cacheinvalid → fence。

第 4 步最后一个 fence 是本次对旧候选的补充，明确等待缓存维护完成。
第 2 步的 PIPE_ALL 只作用于当前物理核，不是 syncall，也不要求 72 个核集合；
它会排空当前核前序工作，后续性能优化需要在正确性成立后再评估更细粒度的等待。
全缓存 DCCI 会清理并失效，可能写回 dirty line；并不是任意多写者数据都可安全套用。
当前 B16 的数学体 Scalar 写落在 temporary_14/15/54：
前两者分别独占 64 B、由同一个 row-offsets worker 写；
temporary_54 每行 16 个 INT32（64 B），按 8 行整块分配，
没有两个 Scalar 写者共享该行。真实 arena 基址另加 512 B 对齐检查。
上述核对是当前候选的使用前提，不外推为通用缓存协议证明。

## B16 候选

[prepare_local_signal_candidate.py](prepare_local_signal_candidate.py) 生成独立目录
`static_local_signal/h131072_b16/`：

- history=131072，B16，S6，T96；正式权重沿用现有驱动。
- progress 从 4608 个 INT64 改为 9216 个 INT32；总字节数仍为 36864。
  每 worker 仍独占 512 B，工作区其他 offset 和总量不变。
- 代际仍为 `generation * 1024 + completed_sequence`。
  测试前检查计划代际未超出协议 INT32 范围。
- AIC 40 个、AIV 44 个计算段的 AST 数学体逐段相等；
  只换通知/等待并补消费者 fence，没有改 Tile、矩阵计算、原事件或依赖表。
- 外部 130 Tensor、Core 132 Tensor；仍 native JIT 直调，不走 Torch schema。
- CPU 生成代码确认单个 24 组 MIX，sync_start=true，无新增全体 barrier。

原静态版本及本候选都不是最终 dynamic batch/seqlen 实现。

## 当前验证记录

| 项目 | 结果及边界 |
| --- | --- |
| 独立 pypto_core 构建 v1 | 常量重载歧义，未通过；改为显式 int64_t |
| 独立 pypto_core 构建 v2 | 通过；没有覆盖共享安装 |
| 定向单测 v1 | xdist worker 误导入共享 PyPTO，5 项失败；不是新接口逻辑失败的证据 |
| 定向单测 v2 | bootstrap 隔离 + 单进程，5 passed；262 deselected |
| 错误类型收窄 v3 | 两项 dtype 拒绝用例使用 ValueError，2 passed；55 deselected |
| 完整 B16 CPU kernel 编译 | static_b16_local_signal_cpu_v1，通过；确认 TNOTIFY/TWAIT 已进入 C++ |
| 完整 B16 设备 | task_20261016_174712_328046030630 首轮正常返回；72 核进度及保护区通过，输出和 cache/state 失败，exit=1 |
| 独立 payload 小例 CPU v1 | INDEX/INT32 算术生成类型不一致，未上卡；测试显式区分 INDEX 计算与 INT32 通知阈值 |
| 独立 payload 小例 CPU v2 | local_signal_payload_cpu_v2 编译通过 |
| 独立 payload 小例设备 v1 | task_20261016_175638_384060212210 正常返回但数据错，5376/18432 个 checked 元素不符 |
| 小例诊断 v2 | task_20261016_175919_40304807000：72 信号正确、最终 mm 正确，但消费者读到旧矩阵值 -999；对应 Vector 输出 -998 |
| 小例修正 v3 | task_20261016_180315_11129226453：notify 前加入本核 PIPE_ALL，三代际信号、数据、保护区全部通过 |
| 完整 B16 修正复测 | task_20261016_180421_21978513378，exit=0；三次调用的输出、Top-K、六份 cache/state 对冻结动态 PTO 逐项完全相同，进度及保护区通过 |

完整设备 v1 原计划检查三次串行调用；首轮失败即保存现场，第二/三轮没有执行。
该失败版输出有 20480 个非有限值，不能当成普通末位误差，也不能仅因进度正确排除 payload 问题。
1.8 GiB 的 failure_states.pt 留在原结果目录，文本诊断见 summary.json /
failure_by_rows_cpu.json。v1 不记完整 CSA PASS。

补齐通知前 PIPE_ALL 后，v2 的完整 128K/B16 **三次调用通过**。
生成 C++ 的 86 个 TNOTIFY 静态调用点都确认前置本核 PIPE_ALL。
三轮均与冻结动态 PTO 的输出、Top-K、六份 cache/state 完全一致，无超差；
72 核代际、Native/metadata 未写区域及工作区保护区均通过。
这是完整 attention 半层的静态模板、正式权重与合成历史验证，
没有 MoE/EP16、动态模板覆盖、ACLGraph 或性能结果。
每次调用前恢复 fixture，signal 保留并进入下一代际；不是连续 decode 状态轨迹验证。
与 Native 的关系仍以冻结动态基线自身的差异记录为准，不把本轮 PASS 外推为
Native 与 PTO 全部逐 bit 相同。结果目录 summary.json 清楚保留这层区别。

[test_local_signal_payload.py](test_local_signal_payload.py) 独立验证 24 组 MIX 中
AIC→AIV（两个消费者）、AIV→AIC、跨组 AIV0→AIV1 三种交接；payload 跨多缓存行，
覆盖 Scalar 写、矩阵/MTE 写、Vector/MTE 写、MTE 读与后续 Scalar 读。
连续三个代际保持信号地址不变、不清零旧 payload，独立检查输出与首尾保护区。
该小例没有 CSA 数学，不能替代完整 B16 验证。

小例的首轮失败不是缓存行多写者冲突：Scalar payload 正确，最终矩阵输出也正确，
但部分 AIV 在接收 AIC 信号后读到了尚未更新的矩阵数据。生成的 AIC gm fence 是
MTE2/MTE3/FIX 分别的 pipe_barrier 加 DSB，缺少 Scalar 发布前的本核流水汇合。
只在 notify 前加入 PIPE_ALL 后，小例三个代际通过；这个对照支持上述定位，
不代表任意协议、A5 或完整 CSA 已经通过。

官方语义参考：[CANN asc_sync_pipe](https://asc.gitcode.com/api/SIMD-API/c_api/sync/intra_core_sync/asc_sync_pipe.html)
说明 PIPE_ALL 等待本核此前提交的各流水操作完成。不能把单流水 barrier 或 Scalar DSB
直接等同于这个完成条件。旧 adapter 保存在 local_signal_variants/v1/，
各结果目录另保存实际使用的 adapter 源码。
本轮不测性能，不重测 B4 或其他档位。

原始构建/单测日志归档于 session_logs，完整结果在 evidence 对应目录；
大文件与设备二进制保留原路径，清单见 artifact_inventory.json，不做 hash 校验。

## 复现命令

以下在运行 worktree 中执行；原输出目录已经存在，不要覆盖，复测用新的目录。

```bash
root=/data/pyptouser/qinchuanyu/pto-eager
csa_work="$root/.worktrees/hc-pre-layer-integration-20261012"
csa_case="$csa_work/tests/pypto_test/csa_static_spmd_feasibility_20261015"
csa_out="$csa_work/tests/pypto_test/results/csa_static_spmd_feasibility_20261015"
task-submit --device auto --max-time 1200 \
  "bash '$csa_case/run_static_jit_device.sh' '$csa_out/static_b16_local_signal_device_NEW' '$csa_case/static_local_signal/h131072_b16'"
```

脚本显式加载独立 PyPTO 与扩容 Simpler，校验 binding 所在路径及 ABI，
只允许由 task-submit 分配设备后执行。
