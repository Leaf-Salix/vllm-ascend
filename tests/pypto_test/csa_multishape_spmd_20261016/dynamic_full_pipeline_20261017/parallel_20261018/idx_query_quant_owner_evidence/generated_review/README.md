# Indexer Query 量化单 owner：一次完整 CPU 编译与生成物审阅

结论：**CPU 与生成物审阅均 PASS，可以交给主会话进行设备验证。** 本目录只证明编译和指定生成合同；尚未以此候选取得设备精度、graph、保护区或性能结论。没有修改冻结的 33 份候选/参考源码、正式源码或依赖，没有提交 NPU 任务。

## 实际执行与来源

- 唯一一次完整编译由 [`../run_cpu.sh`](../run_cpu.sh) 启动，日志 [`../cpu_v1.log`](../cpu_v1.log)。包装脚本检测已有输出并拒绝重复运行。
- 编译参数为 `--strategy long --attention-query-metadata off --spmd-source ../source_candidate --controls-json ../controls.json`，OA128 / OB96 / QA_M16_PARALLEL=false；完整 controls 保存于 [`generated_audit.json`](generated_audit.json)。
- [`../cpu_v1/result.json`](../cpu_v1/result.json) 确认 full_codegen_passed=true、130 个 runtime 根参数、58 个动态轴、single MIX 与 sync_start，未传实际样例 shape、未执行设备。
- 对照同轮 `oa_pair_manual_h131072_b16_v1/source_matrix/result.json` 中 **spmd_reference** 对应的 `decode_csa_spmd_jp7o4g2r`，不是 OA pair 候选。其 controls 与本次完全一致，33 份参考源与本目录 source_reference 相同；基准为正式 66882634。完整路径和大小见 [`INPUT_ARTIFACTS.json`](INPUT_ARTIFACTS.json)。

## 真实生成合同

AIV 的逻辑 worker 为 `block * 2 + subblock`。实际 token 数从 `x_fp32.shape[0]` 绑定 `CSA_FULL_T`；生成 C++ 使用 `v133`。候选在 `worker < T` 时仅等待物理 AIC `worker % 24`，**不加 24**。信号字地址为 `18 * 72 * 128 + (worker % 24) * 128`。AIV 编号的 `+24` 仅用于后面的 AIV 发布及消费者等待，不混入本次 AIC owner。

同一 AIV 量化的 token 是 `worker + 48*w`，Hadamard 的 AIC owner 为该 token `%24`，始终等于 `worker%24`。`worker>=T` 没有量化读写，因此可跳过本次获取；其原 QUANT 发布仍保留。获取后原有 `fence → cacheinvalid → fence`（C++ 对应 PIPE_ALL/DSB/DCCI 等）位于第一个 64×128 FP32 Hadamard 结果 TLOAD 前，不能以任务先后替代这些维护。

完整量化循环的两个 TSTORE（scale 和 INT8）及循环后的原 PHASE_IDX_QUERY_QUANT 发布保留。随后 coefficient 前仍获取全部 `min(T,48)` 个有效 AIV QUANT owner，物理核为 `24+owner`、phase=19；原 weights 等待也未删除。因此本次不是将后继的全量依赖一起收窄。

关键行号、真实表达式与顺序可查：

- [`wait_quant_successor.pto.txt`](wait_quant_successor.pto.txt)：PTO 原件第 8366–8477 行，包含候选获取、完整量化、QUANT 发布及后继等待。
- [`wait_quant.cpp.txt`](wait_quant.cpp.txt)：C++ 原件第 45146–45438 行，包含实际分支、地址构造、维护与量化发布。
- [`cpp_constants.txt`](cpp_constants.txt)：24、128、phase18 偏移等实际常量。

## 同控差分结果

| 项目 | AIC | AIV |
|---|---:|---:|
| Tile 分配项数 | 558 | 1812 |
| 地址、dtype、shape、空间及大小 | 全部相同 | 全部相同 |
| 局部 set_flag/wait_flag/pipe_barrier 条目 | 1711，次序与 ID 相同 | 3088，次序与 ID 相同 |
| 归一化 PTO | 整个 AIC 函数相同 | 仅排除本次 HAD→quant 获取块后，其余完整函数相同 |

归一化按函数作用域处理 SSA 名，并仅展开同值同类型常量、去除 loc；没有删除数学、T*64 动态维度计算、TSTORE 或后继等待。四份 `.diff` 文件为空。

实际分配地址上界（bytes）为：AIC Mat 475136、Left 65536、Right 65536、Acc 131072、Vec 16384；AIV Vec 163872。上述值均与参考相同，是生成 Tile 地址区间的上界，**不额外推断编译器未报告的资源或实际占用率**。

## 复核与限制

只复核现成产物，不重新编译：

```bash
python3 /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/idx_query_quant_owner_evidence/generated_review/audit_generated.py
```

CPU 使用默认 TWAIT，设备参考使用 wait-backoff 辅助实现；本审阅不宣称完整机器码逐字一致。局部事件静态序列相同，不等于其动态执行次数相同——条件获取和单 owner 正是本次实验变量。CPU 日志中的 ScalarWriteLineShared 警告来自现有共享 planner 路径，本次没有通过修改依赖或屏蔽警告处理它。

大 PTO/C++、编译目录与二进制仅登记原路径和字节数，不复制大文件、不做 hash。主会话负责接下来的完整设备精度、graph、保护区和同进程性能矩阵。
