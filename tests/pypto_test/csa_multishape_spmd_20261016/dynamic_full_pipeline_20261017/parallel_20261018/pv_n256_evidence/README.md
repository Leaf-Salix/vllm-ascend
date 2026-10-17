# Attention PV N256 独立候选

2026-10-18。基于正式 e581（生产 33 源与 d253 相同）冻结副本，仅将 `attention_stream_stages.py:101` 的 `PV_N_TILE` 从 128 改成 256；没有叠加已否决的 gather guard 删除或其它候选。源码从复制起冻结。

初步源级判断：H=64、K=128、输出 512 列、三槽与事件顺序保持。PV 从四个 N128 变成两个 N256，两个同时存活的 FP32 Acc 各 65536 B，总计 131072 B；单块 Right 为 65536 B，不允许规划出两个同时驻留的 Right。每个有效 PV 块的提取、matmul、FIX 写回位点各 4→2，总运算量和写回字节不变。是否有额外等待或资源退化，以唯一一次完整 CPU lowering 的实际产物为准。

历史排重：MASTER/VALIDATION、当前 Python/Markdown 和全部 Git refs 中的 PV_N_TILE/pv_n_tile/PV_N_COLS 256 常量搜索均未见同方案。VALIDATION §185–186 是 N512→N128 和 N128 双 Acc；QK_ROLE_SPLIT/QK_MERGE_STATE 中的 256 列为 AIV 归并状态，不能当作 PV N256 已测。没有据搜索未命中声称穷尽所有未命名实现。

CPU 只用 long/G6、OA128/OB96、metadata off，完整符号编译不提供样例 shape。编译命令和结果随后追加。此候选未上卡，CPU 通过也不代表数值、硬件时序或完整 CSA 性能通过。无散列校验。

## 唯一 CPU 结果及真实分配

`cpu_v1/result.json` 完整 PASS，58 动态轴、130 根入参、single MIX、sync_start 保持；未提供样例 shape，没有第二次编译。命令：

```bash
E=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-indexer-20261018/tests/pypto_test/ultra_indexer_20261018/pv_n256_20261018
D=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017
bash "$D/reproduce_cpu.sh" "$E/cpu_v1" --strategy long \
  --attention-query-metadata off --spmd-source "$E/source_candidate" \
  --controls-json "$E/controls.json" > "$E/cpu_v1.log" 2>&1
python "$E/audit_generated.py"
```

已有同源基线使用 `parallel_20261018/hc_active_notifications_h131072_b16_v1/build_output/decode_csa_spmd_3kuwaqat` 的 hook 前 PTO/C++；先直接文本核对其 33 源与 `source_reference_e581` 相同，没有重新编译基线。详见 `baseline_artifact_reference.json`。

| 实际资源，单位 B | 基线 N128 | 候选 N256 |
|---|---:|---:|
| PV Acc 两槽 | 0/32768，各 32768 | 0/65536，各 65536 |
| PV Right | 各次提取同址 0，大小 32768 | 两次提取同址 0，大小 65536 |
| 全 AIC Acc 末端 | 131072 | 131072 |
| 全 AIC Right/Left 末端 | 65536/65536 | 65536/65536 |
| 全 AIC Mat 末端 | 475136 | 475136 |
| AIV Vec 末端 | 163872 | 163872 |

全局资源上界来自 PTO `alloc_tile`，并用 AllocateMemoryAddr 的 MemRef 交叉核对。QK/PV 共用的三槽 L1 在 65536，大小 384×512×2=393216 B；Query 在 0，大小 65536 B；概率在 458752，大小 16384 B。未增加 L1 或 UB 工作区。AIV 完整 C++ 去注释及统一 SSA 编号后全文相同，不能把只有上界相同说成所有 AIC 临时地址完全相同。

## 生命周期、数学与写回

以下 C++ 行号属于 `cpu_v1/kernel/ptoas/dynamic_csa_full.cpp`，完整 Attention 片段另存 `candidate_attention_cube.cpp.txt`：

- QK 的 64×128 FP32 Acc 为 32 KiB。QK TSTORE 后在 17805/17809 行执行既有 FIX→M 事件交接，再进入 PV；因此不能把该 QK Acc 与两份 PV Acc 简单相加成 160 KiB。
- 第一 PV 的 TMATMUL 在 17888；17890 发 M→MTE1/1，17898 等同一事件，17901 才复用 Right 地址 0 提取后 256 列。原 N128 也只有同址 Right，不是删除一个既有预取槽。
- 第二 TMATMUL 在 17911，首块 TSTORE 在 17931；两份 Acc 同时存活且地址不重叠。第二 TSTORE 在 17957，17959 仍在 FIX 上发布原 PV_READY。
- tick 尾部 17964 发 FIX→M/0，下一 tick 开头 17591 等同一事件；末尾 17966 排空。前后拍 Acc 复用仍受保护。上述边界事件也存在于基线，并非本次新引入的串行化。
- 每个有效 PV 块的 TEXTRACT/TMATMUL/TSTORE 各 4→2；右输入仍来自同一 L1 的行 `(work%3)*128`、列 0/256，左概率仍 64×128。PTO 写出 `[64,256]` 两矩形覆盖列 `[0,256)`、`[256,512)`，行 stride512，完整 64×512 无遗漏、重叠或覆盖别的 slot。每块 MAC 仍 4194304，FP32 GM 写出仍 131072 B。K128 及每个输出的归约输入顺序未改；实际硬件逐 bit 等价仍须设备验证。

整个 AIC 生成调用差分只有：TEXTRACT/TMATMUL/TSTORE 各减少 2、TASSIGN 减少 6、局部 set_flag/wait_flag 各减少 8、PIPE_FIX barrier 减少 2；没有新增调用。PIPE_ALL、FFTS、QK、其它阶段调用保持。详细事件方向与编号在 `generated_audit.json`，没有仅凭同步总数相同推断生命周期。源码中原有“N128”说明暂未改，以严格保持候选仅一行常量差异；若最终采用需同步文档。

这仅证明源/生成物的有界资源与提交次序合理，不证明实际重叠幅度、最终 ISA 顺序、数值通过或完整 L1 收益。没有新 NPU 动作由本代理提交。

## 冻结、主进程设备任务与归档

根代理已提交唯一长 128K/B16 三方矩阵：`task_20261018_035635_317824730121`，主仓 `parallel_20261018/pv_n256_h131072_b16_v1`，registry 在 `pv_n256_evidence`。该任务尚不在本报告预记 PASS 或收益；由根代理收取。候选和参照继续冻结，不扩短档、不组合其它变更。

建议复制本目录除 `cpu_v1/`、`__pycache__/` 外的文本与两个 33 源目录；其中 `cpu_result.json` 是唯一 CPU 结果的直接文本副本。原始完整 lowering 留在 `cpu_v1/`；`paths_and_sizes.json` 分列归档文件与原始产物的路径、大小，不记录校验散列。
