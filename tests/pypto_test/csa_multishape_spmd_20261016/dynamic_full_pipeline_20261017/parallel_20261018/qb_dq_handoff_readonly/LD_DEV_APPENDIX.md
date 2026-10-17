# ld_dev 信号读取历史/API 补查（只读）

结论：这不是未经尝试的新方向。旧融合 CSA 已有连续 `ld_dev` 和失败后 25 tick 退避的完整负例；当前动态包保持 100 tick、仅替换读取的精确组合未找到设备对照。API 与当前 INT32 单写者信号兼容，但不能由少一条 DCCI 推定收益。本轮依主会话要求收口，不实现、不编译、不上卡。

## 历史与实际产物

来源：[ATTEMPTS.md:55](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/ATTEMPTS.md:55)、[同文件:212](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/ATTEMPTS.md:212)、[同文件:397](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/ATTEMPTS.md:397)。以下均为 128K/B16、各侧三窗完整 L1；各行属于不同任务，禁止把跨行总时差换算成因果收益。

| 旧目录 | 读取/失败退避 | 原分 task 中位 μs | 融合 SPMD 中位 μs | 原始融合三样本 μs |
| --- | --- | --- | --- | --- |
| b16_bypass_wait_v1 | ld_dev / 0 | 901.88 | 1366.36 | 1332.88, 1379.46, 1366.36 |
| b16_balanced_lddev_v1 | ld_dev / 0 | 897.08 | 1327.28 | 1108.82, 1327.28, 1465.98 |
| b16_backoff_25_v1 | ld_dev / 25 | 900.10 | 1146.62 | 1146.62, 1205.64, 999.92 |
| b16_balanced_plain_v1 | 原 TWAIT | 902.28 | 1067.36 | 1061.38, 1067.36, 1140.28 |
| b16_cached_backoff_100_v1 | DCCI + volatile / 100 | 907.02 | 851.32 | 850.94, 851.32, 854.06 |

原始结果分别在上述目录的 `result.json`；历史记录判定 ld_dev 候选未优于原等待，全部精度、状态、保护区、图重放及 30 代信号通过。它们是旧按 shape 分配成员的融合程序，不能冒充当前 33 源完全动态版本的结果，也不包含 Native 或连续 decode 验收。

本次扫描 `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016` 下现存 `change.json`，reader=ld_dev 只命中前三项：84/68/68 个站点，0/0/25 tick。未找到 ld_dev100。同级 [signal_polling.py:15](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/signal_polling.py:15) 清楚保留 TNOTIFY、payload 维护及每 64 次失败 PIPE_ALL；25 tick 仅增加未就绪分支中的本核 SYS_CNT 等待。

逐文本检查了三项的 `signal_polling/original.cpp` 与 `bypass_wait.cpp`：移除插入 helper、把调用名恢复为 TWAIT 后，首个原始模板之后的完整源码逐字相同，include/namespace 前缀也相同。实际 `build_output/entry_*/kernels/aic/csa_static_aic.cpp` 与 `aiv/csa_static_aiv.cpp` 都含 ld_dev helper，实际调用数分别仍是 84/68/68。由此可确认旧试验确实只替换等待读取；未仅依据参数标签认定生效。

## ABI 和可见性

正式 [wait_codegen.py:30](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/wait_codegen.py:30) 只接单元素 INT32 GM 视图、GE 比较。每次检查前 DCCI，随后 volatile 读；首次没有退避，失败后 100 SYS_CNT tick，每 64 次失败 PIPE_ALL。它是显式编译 hook，导入模块本身不改变默认后端。

A2/A3 的 [syncall_soft.hpp:43](/data/pyptouser/qinchuanyu/pto-eager/pto-isa-src/include/pto/common/syncall_soft.hpp:43) 使用：
`static_cast<int32_t>(ld_dev(reinterpret_cast<__gm__ uint32_t*>(counter), 0))`。
本机 CANN 9.0 目录的 [cce_aicore_intrinsics.h:1373](/usr/local/Ascend/cann-9.0.0/aarch64-linux/ccec_compiler/lib/clang/15.0.5/include/cce_aicore_intrinsics.h:1373) 将 ld_dev 声明为 `__builtin_cce_ld_dev` 别名，C++ 返回 uint64_t。这个返回寄存器类型不能解释为应把信号改成 INT64；本地 A3 官方实现及旧已执行 C++ 均传 uint32 GM 指针、零偏移并转回 int32。

若未来重新评估，局部替换边界是 signal.data() 的地址已指向该单元素槽，转为 `__gm__ uint32_t*` 后以 offset=0 调用，再转 int32 与 expected 作 GE；不对 volatile int32 指针机械替换调用，不添加其它偏移。保持当前四字节对齐 INT32、512 B 槽隔离、单写者、代际不回绕合同；不改 72 核 begin_invocation。无需新增外层参数、根地址、Tensor、GM/UB 或运行时任务。

写侧 [TNotify.hpp:38](/data/pyptouser/qinchuanyu/pto-eager/pto-isa-src/include/pto/comm/a2a3/TNotify.hpp:38) 的写前 DCCI、volatile store、写后 DCCI、DSB_DDR、PIPE_ALL 全部保留；外层发布前的数学 drain/payload 维护和等待后的业务获取也保留。ld_dev 只绕过本核 Scalar DCache 观察信号，不是原子 RMW、C++ acquire、业务缓存失效、DMA 完成证明或必然 HBM 访问。[A2/A3 retirement 文档:109](/data/pyptouser/qinchuanyu/pto-eager/.worktrees/simpler-csa-core-args-256-20261016/docs/troubleshooting/a2a3-worker-retirement.md:109) 也明确将 bypass 观察与后续 DSB/发布协议分开。

可保留正式读取前后的空 asm memory clobber，在同一位置用局部 observed 接收 builtin 结果，保持失败退避、spin 上限、周期 PIPE_ALL 的控制流原样。这些空 asm 是编译器约束，不是新增硬件 acquire/fence。不能复制 syncall 的全部 barrier 算法替换原点对点依赖。

## 汇编边界与结论

本轮只读既有 C++ 和本地声明，未生成新机器码。旧完整设备 PASS 支持该调用形态在旧 A3 工具链可执行；可变参数 builtin 声明本身没有公开完整的指令编码、对齐/偏移约束或编译器内存属性，不能据此宣称当前包的最终机器码已经核对。任何未来实现仍须在实际 AIC/AIV 产物确认轮询回边每次执行正确宽度的设备读取、未退回普通缓存 load、未被提升/合并，且写侧和业务维护未变。本轮不为此补编译。

历史已有负例，当前没有证明“已就绪检查的固定成本由 DCCI 主导”的观测；100 tick 精确组合未测只是证据缺口，不构成继续试验的正证据。依据主会话指示，仅保留此总结。

