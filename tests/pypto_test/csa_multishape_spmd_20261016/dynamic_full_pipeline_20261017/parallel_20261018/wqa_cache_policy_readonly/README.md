# wq_a 的 DEFAULT/BYPASS：只读排重与实际生成核对

结论：**当前不值得追加仅把 wq_a 声明改成 DEFAULT 的设备候选。** 没找到单独 wq_a 的动态单 SPMD 设备消融，但已有两次“恢复权重 BYPASS 后 C++ 不变”的记录；本轮实际产物再次确认 QA 的 BYPASS 留在 IR，最终调用是普通 `TLOAD`，未生成显式 NotAlloc hint 或地址偏移。不能把源码写着 BYPASS 当作实际绕过 L2，也不能宣称普通 TLOAD 已经获得缓存命中。先等 QA 分工结果；如果以后工具链确实生成不同缓存访问路径，才有单张只读权重的策略对照意义。

本报告仅读已有文件。没有改候选、正式源码或依赖，没有编译、设备测量、全局缓存开关或 hash。证据与可重复提取脚本为 [evidence.json](evidence.json)、[inspect_existing.py](inspect_existing.py)。

## 历史排重

在本会话 `MASTER`、动态 `ATTEMPTS.md`、两树 Markdown 和已有缓存专项中，未找到只改 wq_a 的 DEFAULT/BYPASS 设备试验。找到的相关工作是：

1. `tests/pypto_test/csa_multishape_spmd_20261016/WEIGHT_CACHE_POLICY_EXPERIMENT.md`：旧融合实验恢复多张权重声明，IR 带策略 TLOAD 从 2 处增到 44 处，但完整 C++ 2,784,561 字节相同，未上卡。
2. 正式高层动态链 `dynamic_full_pipeline_20261017/kernel_weight_bypass_v2/cache_policy_audit.json`：五张权重声明恢复后，IR 有 25 处 BYPASS，C++ 没有 hint；仅删除整行 `//` 注释后与 `kernel_schedule_overlap_v1` 完全相同，因此没有做等价代码的设备性能复测。主日志 `vllm_ascend/ops/pypto/DSV4_CSA_SPMD_OPTIMIZATION_LOG.md:15500` 已记录这一纠正。

这两项不等同于 wq_a 单独缓存收益测量，但足以否定“尚未尝试，所以单改声明就值得上卡”的推断。

## 本轮现成产物检查

当前 `cube_schedule.py:194` 仍用 `(1,1)` 为 wq_a 声明 BYPASS；其 ABI 参数 1 对应 `wq_a[1024,4096]`。沿真实生成的 NZ GlobalTensor 地址链核对，权重切片来自原 wq_a 指针及 N/K 切片偏移；没有额外 L2 别名地址调整。

| 现成产物 | QA 的 IR BYPASS TLOAD 数 | 显式 offset | C++ TLoadL2Hint / TLOAD 模板 / l2Cacheoffset |
| --- | ---: | ---: | ---: |
| `current_clock_h131072_b16_v1/.../decode_csa_spmd_jqlk5eey` | 8 | 0 | 均 0 |
| sync 小 T M16 候选 `cpu_v1/kernel` | 10 | 0 | 均 0 |
| 同作业原分 task `_decode_csa_tp1_layer_0leidxnz/ptoas/qr_proj_matmul_0` | 4 | 0 | 均 0 |

精确路径、字节数、IR 行和代表 C++ 均在 evidence.json。长 B16 的示例是 `cube_schedule.cpp:6412` 的 NZ 权重视图，随后 `:6414` 为 `TLOAD(v1067, v1079)`；对应 IR `:466` 有 `l2_bypass`。小 T 候选只读已完成的 CPU 产物，没有改变其运行中设备任务。

当前隔离 Simpler 所带 PTO-ISA 的根路径为：

`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/simpler-csa-core-args-256-20261016/build/pto-isa/`

可直接核对这些实现：

- `include/pto/common/pto_instr.hpp:261`：无模板参数的 `TLOAD` 传 `TLoadL2Hint::NormalFirstVictim`。
- `include/pto/common/type.hpp:331`：`NormalFirstVictim=0` 是 DEFAULT；NotAllocKeep/Clean/Drop 是另三种值。
- `include/pto/npu/a2a3/TLoad.hpp:335`、`include/pto/common/arch/memory/tload_common.hpp:413`：模板参数进入 GM→L1 搬运地址处理。
- `include/pto/npu/a2a3/l2_cache_hint.hpp:56`：只有 NotAlloc 三种模式才加运行时填写的 `g_opL2CacheHintCfg.l2Cacheoffset`，普通模式返回原地址。是否能通过当前 Simpler 加载方式实际填写该配置，本报告没有验证。

本地 `ptoas-src/lib/PTO/Transforms/PTOToEmitC/LoadStore/TLoad.cpp:77` 则要求 `L2Bypass && op.getOffset()` 才调整源地址；随后仍发普通 `TLOAD`。对应 `test/lit/pto/tload_cache_policy.pto` 明确检查没有 `TLoadL2Hint`、无 offset 时普通 TLOAD。这解释了当前看到的产物，但本地源码 HEAD 不被冒充为已安装 PTOAS 二进制的精确来源。PyPTO 文档中“直接生成 `TLOAD<NotAllocKeep>`”的描述与这批产物不一致，以产物为准。

## 只读权重复用机会与副作用

理论上，多个 M 组的相同 N/K 块会重复读取同一 wq_a 字节；允许普通缓存分配可能复用这些读取。原小 T 分工可能由同一 AIC 顺序重读，新 M16 分工则让多个 AIC 读相同列块。每张完整 BF16 权重是 8 MiB，只说明数据量，不据此猜测 A3 L2 容量、驻留比例或节约带宽。

反面是缓存分配可能与并发的 Indexer 权重、输入及中间数据竞争，造成额外替换；重复加载是否及时命中还受组间到达时序影响。这不是改变 L1/Mat 生命周期、减少 TLOAD 指令或消除显式重复读取，不能保证完整 CSA 更快。wq_a 在本次调用中只读，改变其读策略本身不改变数学和依赖；它也不能替代任何业务数据发布/等待协议。

然而，本工具链的普通 TLOAD 路径已经是上述 DEFAULT 模板。因此暂不创建另一个只改声明的候选，也不扩展到所有权重。只有先证明确有不同生成路径及可用运行时地址配置，才应讨论 wq_a 单张权重的硬件消融；当前没有这项新增前提。
