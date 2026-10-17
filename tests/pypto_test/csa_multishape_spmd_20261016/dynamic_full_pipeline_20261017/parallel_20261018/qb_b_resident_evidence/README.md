# Q_B 整 K 权重跨 M 常驻：失败、修正与否决

最终不采用。v2 完整符号 CPU、长 B24 与短 B32 全部精度检查通过，但对正式 66882634 的同轮完整 CSA 中位数分别回退 **0.275% / 1.522%**。只减少逻辑 GM 权重请求没有带来整层收益；本路线停止，不扩大区间、不扩七档、不追加参数变体。

## 候选范围与冻结源

参考为正式 **66882634** 的33个 Python 源，逐文件直接文本核对后复制到 [source_reference](./source_reference)。正式源、依赖和共享安装均未修改。失败版 [source_candidate](./source_candidate) 与修正版 [source_candidate_v2](./source_candidate_v2) 独立冻结，后者相对失败版只改变 `query_projection_stages.py`。

相对参考共四个文件有改动：`query_projection_stages.py` 实现 NZ Q_B 搬运；`policy.py` 根据已有 Host T 描述符在 `128<T<=256` 选择 `QB_B_RESIDENT=True`；`entry.py` / `cube_schedule.py` 透传该默认 False 的 constexpr。实际 T、M128循环、尾行 valid、N256列 owner、INT32四段K256累加与全部跨核发布保持。False分支仍是原实现，Vector源没有修改。

[audit_source.py](./audit_source.py) 验证3个内核文件在 False 折叠后与参考AST相同；3072组T1..512、Q1/5/6、长短历史策略验证仅新字段改变，开启区间精确为128<T<=256。该检查不是 False 新二进制等价证明，本轮未重编 False。ND helper仅兼容接收新参数，数学没有改变。

本方案逻辑 B 请求在两 M128 区间由64MiB降至32MiB；QR、有效dot、MMA、FIX计数不变。整 B256KiB 载入必须在首个 K256 前完成，且下个 N 没有权重预取。历史区别、收益可能被抵消的原因及有限模型见 [上级设计报告](../README.md)。这些是工作量分析，不能换算为时间收益。

## v1：明确的 Right 生存期失败

唯一首轮命令见 [run_cpu.sh](./run_cpu.sh)，输出 [cpu_v1/result.json](./cpu_v1/result.json)：58个动态轴源解析成功，完整 lowering 在 `AllocateMemoryAddr` 失败，Right实际需要131072B，超过65536B。没有生成最终 PTO/C++，没有设备执行。

不是 first Right 被抬出 M 循环，也不是 K128 变成 K256。原正式生成已经使用四个K256和单Right64KiB。证据是：

- [32号pass片段](./v1_pass32_qb.txt)：K256 extract→K256 MMA→K512 extract→K512 MMA，源级顺序尚未重排。
- [33号pass片段](./v1_pass33_qb.txt)：`CanonicalizeIOOrder` 将K512 extract提前到K256 MMA之前，两份Right真实重叠。
- [37号pass片段](./v1_pass37_qb.txt)：first、K256、K768共用 `mem_right_380`，K512单独 `mem_right_386`，每个64KiB。QR仍为两32KiB槽，B父Mat只加载一次，没有Mat→Mat重复物化。

本地 [IO分类源码](./io_category_rule.txt) 明确把 Mat→Left/Right 的 `tile.extract` 当作 Load，原高层路径的 `tile.move` 则属于 TileCompute；[重排范围源码](./io_pipeline_scope_rule.txt) 只对存在 Pipeline祖先的 SeqStmts生效。普通直线与普通 `pl.range` 不被这一pass提前。因此仅统一首段到同一个stage2流水，或把slice换成显式extract，都没有解决双Right重叠的依据。

## v2：一次有据修正 CPU 通过

获准修正只采用固定算法 K=1024 的四段直线供数，不展开任何实际 T：

`QR0 → QR1 → R0/MMA0 → QR2 → R1/MMA1 → QR3 → R2/MMA2 → R3/MMA3`

不使用 `pl.pipeline`，每次Right消费后才提取下一段，QR消费后才覆盖同槽。首段 `tile.matmul` 依据实际valid M创建compact INT32 Acc，后三段 `matmul_acc` 保持0/256/512/768原序。源码生成及AST检查见 [prepare_v2.py](./prepare_v2.py)、[source_audit_v2.json](./source_audit_v2.json)。

第二轮命令 [run_cpu_v2.sh](./run_cpu_v2.sh) 输出 [cpu_v2/result.json](./cpu_v2/result.json)：`passed/full_codegen_passed/single_mix=true`，58个动态轴，无样例shape、无设备执行。本轮只有首轮失败及获准的这一次修正，没有 stage 扫描。

真实生成分配见 [v2_qb.pto.txt](./v2_qb.pto.txt)：

| 对象 | 地址 / B | 容量 / B | 生命周期 |
| --- | ---: | ---: | --- |
| B1024×256 INT8 Mat | 65536 | 262144 | N内、动态M循环外；两M复用 |
| QR0 / QR2 Mat | 327680 | 32768 | 消费QR0后覆盖为QR2 |
| QR1 / QR3 Mat | 360448 | 32768 | 消费QR1后覆盖为QR3 |
| 四个 Right256×256 INT8 | 0 | 65536 | 顺次复用一个槽 |
| Left128×256 INT8 | 0 | 32768 | 顺次复用一个槽 |
| compact Acc128×256 INT32 | 0 | 131072 | 四K同一累加器；FIX后供下一M |

Q_B同时活跃Mat容量320KiB；全AIC最大地址末端 Mat475136、Left65536、Right65536、Acc131072、Vec16384B，与同控既有基准相同，不能把各阶段容量相加。无额外GM根Tensor。

[v2_qb.cpp.txt](./v2_qb.cpp.txt) 保留实际事件：QR覆盖前 `MTE1→MTE2`；每个Right覆盖前 `M→MTE1`；MMA前 `MTE1→M`；首段及下一M首段等 `FIX→M`，最终 `M→FIX` 后TSTORE。N/M计算体内部没有 `PIPE_ALL`，阶段边界的原发布排空仍存在。每次QR GM view、Left、compact Acc及输出GM view共用 `min(T-t0,128)`；B提取窗口0/256/512/768均在完整1024行之内。

这证明源码生成和静态事件链合理，未单凭CPU宣称设备数学或运行时同步已经通过。

## 非 Q_B 变动审阅

[audit_cpu.py](./audit_cpu.py) 只读现有产物，结果 [cpu_audit.json](./cpu_audit.json)。比较对象不是 OA128/OB96 的旧9d CPU，而是已有设备基准：

`主PAR/query_linear_h131072_b24_v1/build_output/decode_csa_spmd_5e_gzb5p`

该次 `source_matrix/result.json` 明确标为 `spmd_reference`，原13个controls与本轮完全同值，包括 **OA256/OB128**。它来自9d源；先将668新增QA在False下折叠、删除未调用helper，32个非policy源AST与该参考全部相同。本轮 `QA_M16_PARALLEL=False`。不把这份旧产物冒称为新编的668二进制。

比较结论：

- 非Q_B AIC的535个 `alloc_tile` 按生成顺序，其地址、空间、类型、行列及字节数完全相同。
- 去掉Q_B区域后，AIC完整PTO在只去loc/SSA名称、展开常量声明后直接相同；保留了全部数学、控制流、常量值、布局和地址。整个AIV也相同。
- `cube_schedule` 函数体非Q_B的4722条局部 `set_flag/wait_flag/pipe_barrier` 序列完全相同。
- 原设备C++额外注入100tick wait-backoff helper，本CPU没有；原始文件头曾因此多一个PIPE_ALL。未将此差异归因候选，也未声称整份C++、GM等待helper或二进制全文相同。当前设备矩阵双方均使用相同wait-backoff。

空文件 [non_qb_normalized_pto.diff](./non_qb_normalized_pto.diff) 和 [non_qb_events.diff](./non_qb_events.diff) 是上述直接文本对照结果，不是内容摘要校验。

## 同轮设备结果与停止结论

主进程分别提交冻结v2和正式668的三方同进程矩阵，原分task为额外精确参考。两档均完整PASS：原PTO精确对照、图重放、状态/保护区及72核代际检查通过。性能仅完整L1 incore首尾、五窗中位数，双方相同100tick失败轮询退避。

| 历史 / B | task ID | 原分task / μs | 正式668 / μs | v2 / μs | 相对668降低 | 更快窗 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 131072 / 24 | task_20261018_055057_11658413779 | 1170.54 | 1164.80 | 1168.00 | −0.275% | 2/5 |
| 8192 / 32 | task_20261018_055057_116588019925 | 979.06 | 997.16 | 1012.34 | −1.522% | 0/5 |

权威完整样本、角色、任务和路径见主仓 [DEVICE_RESULTS.md](</data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/qb_b_resident_evidence/DEVICE_RESULTS.md>) 及同名JSON。本报告只引用同轮数据，不与另一个QR+B候选的异轮绝对值排名。

**不采用，不扩测。** B逻辑请求64→32MiB是事实，整层加速未发生；256KiB前置加载、改变原B/QR搬运重叠可作为与结果相容的原因，但没有阶段测量，不能作确定归因。没有新增S1/S5探针、False编译或其它变体；现有两档PASS不代表所有动态边界已验收。

## 归档

必要报告、脚本、JSON、源patch和片段均在本目录。源、完整CPU passes、PTO、C++、日志继续留在原私有目录。精确路径和大小、建议复制文件见 [LOCAL_ARTIFACTS.json](./LOCAL_ARTIFACTS.json)，由 [make_inventory.py](./make_inventory.py) 按路径与文件大小生成。
