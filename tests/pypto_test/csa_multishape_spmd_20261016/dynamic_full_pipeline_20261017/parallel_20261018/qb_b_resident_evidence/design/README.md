# Q_B 多 M 权重复用：66882634 上的有界设计

最终结论：**仅 B 全 K 跨 M 常驻不采用**。它保留 M128、N256、QR K256 双槽和四段 INT32 累加，但两档完整设备对照均回退。v1 在 Right 分配失败；按明确重排原因修正后，v2 完整符号 CPU 与两档设备精度通过。完整经过与同轮结果见 [b_resident_cpu/README.md](./b_resident_cpu/README.md)。以下保留原有限设计，字节数是逻辑载荷，不是 HBM 测量或收益证明。双 M/N128 和 QR+B448KiB 两个设计变体在本子任务未实现。

## 历史不是空白

- `2de2baa6:vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/q_projection.py:97–108` 已把完整 `[1024,256]` B 放到 M64 循环外。真实原文保存于 [historical_qb_source.txt](./historical_qb_source.txt)。因此“B 跨 M 常驻”并非新原理。
- [CSA VALIDATION §110.1](</data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/DSV4_FLASH_CSA_VALIDATION_LOG.md:3818>) 记录旧实现曾保留：`task_20260926_155100_79968024243`，另有 B40/T240 边界任务 `task_20260926_155902_8638095291`。这些是旧分 task、M64、整 K 高层 matmul，不能当当前方案的 exact/L1 结果。
- 后来 `74be8a6c` 改为现 M128/K256 流水，明确指出旧整份 B 加载与计算串行。[HCA 日志 §7](</data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/DSV4_FLASH_HCA_OPTIMIZATION_LOG.md:166>) 记录 v10 的重分块收益；既改变 M/K 又改变搬运，不能据此定量归因 B 常驻本身。
- 旧 [双 N128 列块](</data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/QPROJ_L1_COLUMNS_EXPERIMENT.md>) 是 T96、20 worker、QR96KiB 常驻、双列累加器；同进程仅差 0.36% 未采用。不是双 M，也没有当前两次 M 的 B 重读可删。
- 当前 [QB Left 双槽](</data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/qb_left_doublebuffer_evidence/DEVICE_RESULTS.md>) 已回退 0.657%；[M64/N512 审查](</data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/qb_tile_readonly/README.md>) 指出增加 B 重读和 Right 超容量。均不是本次候选。

没有找到“动态单 SPMD、仅双 M128 区间、N256 与 QR K256 流水不变、显式 B 常驻”的独立已执行对照。不将检索未发现说成全部历史的不存在证明。

## 当前真实资源及三案比较

正式 [query_projection_stages.py:128](</data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/query_projection_stages.py:128>) 按 N owner 遍历全部 M128，最后一块 `min(128,T-t0)`。已有 QA CPU 产物的该文件全文与正式 66882634 相同；[existing_qb.pto.txt](./existing_qb.pto.txt) 和 [existing_qb.cpp.txt](./existing_qb.cpp.txt) 保存真实证据：Acc 0/128KiB，Left 0/32KiB，Right 0/64KiB；两份 B Mat 和两份 QR Mat 活跃窗口共192KiB，地址范围65536..262144。全 AIC 现有最大 Mat 地址末端475136B，不能与新方案活跃容量直接相加。

| 设计 | Mat 活跃模型 | Acc / Right 单槽 | 对双 M 当前路径的代价 |
| --- | ---: | ---: | --- |
| 双 M128、N128，每 K 共享一个 B | 192 KiB | 2×64 / 32 KiB | QR载荷×2；MMA/FIX调用×2；最大有效列1536→1408 |
| **仅 B1024×256 常驻，M128** | **320 KiB** | **128 / 64 KiB** | 首个256KiB B加载前置；QR/MMA/FIX/owner不变 |
| QR192×1024 与 B1024×256 常驻，M64 | 448 KiB | 64 / 64 KiB | MMA/FIX调用+50%；L1→Right逻辑载荷64→96MiB |

后两项是容量模型，不是已生成的分配。A3 Mat/Left/Right/Acc 上限512/64/64/128KiB，来源为本地 `pypto-csa-static-fixes-20261016/include/pypto/backend/910B/backend_910b_handler.h:96–106`。

对 `128<T<=192`，当前两 M、拟 QR 常驻三 M，所有方案有效 MAC 和输出 INT32 字节数相同：

| T | 当前 B+QR / MiB | 双 M/N128 | 仅 B 常驻 | QR+B 常驻 |
| ---: | ---: | ---: | ---: | ---: |
| 144 | 64+18=82 | 32+36=68 | 32+18=50 | 32+3.375=35.375 |
| 168 | 64+21=85 | 32+42=74 | 32+21=53 | 32+3.9375=35.9375 |
| 192 | 64+24=88 | 32+48=80 | 32+24=56 | 32+4.5=36.5 |

仅 B 常驻使权重 GM TLOAD 从1024次64KiB变128次256KiB；QR仍1024次，MMA仍1024次，FIX仍256次。每个原列块的一次256KiB载入必须先完成，之后才能提取首个K256；相对流式首块多前置192KiB，失去原后续 B DMA 与计算的重叠。不能用节省32MiB直接换算加速。

双 M/N128 的总输入读节约为 `128KiB*(256-T)`，T256归零；其峰值有效点积降8.33%是所有权更均匀，并非时间。QR+B常驻比仅B再省14.625/17.0625/19.5MiB逻辑 QR 读，但多32MiB L1→Right、512次MMA和128次FIX。当前 QR 使用默认缓存、B使用 BYPASS，QR重复请求可能命中缓存；不能把额外逻辑节约全算 HBM 收益。因此本轮先选改动与归因更小的仅 B 常驻。

## 接口、有效尾与依赖

可用现有 `pl.load(..., target_memory=Mat)` 明确 B 生命周期，再用 `pl.tile.slice(B,[256,256],[k0,0])` 作为 Tile matmul 操作数。本地 `CanonicalizeTileSlice:20–43,869–920` 将 Mat slice 融合成 Mat→Right TEXTRACT，不能把片段物化为不支持的 Mat→Mat 拷贝；也不能把完整1024×256 B直接 move 到 Right。该规则已有测试，但本候选需看真实生成结果。

`tile_ops.py:396/657/2347` 支持实际 valid load、extract 与 slice；`transform.cpp:782–834` 将 extract 有效形状与 parent 相交，部分 Mat→Left 设 compact，A3 `textract_common.hpp:432–456` 使用有效行。故 QR192 的三个 `[64,256]` 物理窗口0/64/128合法；T128以下和T192以上必须回旧分支。这是第三案可表达的证据，并非已编译证明。

最终 CPU 候选由 Host 现有 T 形状描述符选择 `128<T<=256` 的 `QB_B_RESIDENT` constexpr，默认 False；不读 host Q/L，实际 M 循环与 valid 仍取 device T，不以工作区容量作为有效行。B全K常驻，M仍0/128、末块valid=`T-128`；每个累加器仍按0/256/512/768更新，INT8点积绝对和≤16,777,216，不溢出INT32。由首个 matmul 创建 compact Acc，不以固定 create 后盲加尾行；直接按有效矩阵store，绝不把两 M 拼入一个 Acc 父视图。

N256 owner仍为`(column//256)%24`，输出无交叠。已有 [cube_schedule.py:313–314](</data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/cube_schedule.py:313>) 在返回后按原协议发布 QB；[vector_schedule.py:761](</data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/vector_schedule.py:761>) 等全部24个 owner。所有输入获取、发布与后续 DQ 保持。S1/S5/mixed Q仍按相同真实 T 处理矩阵，不改变已有效/无效行的既定规则。

[analyze.py](./analyze.py) 仅以标准库复算14个代表/边界 T，分解枚举T1..512的行、列唯一覆盖，输出 [evidence.json](./evidence.json)。没有用固定 S6=6 代替动态 T，也没有预测时延。
