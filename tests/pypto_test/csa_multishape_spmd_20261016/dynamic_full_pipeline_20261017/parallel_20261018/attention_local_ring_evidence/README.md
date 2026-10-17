# 2026-10-18：Attention 同核 softmax 统计缓存候选

本目录属于独立 worktree `csa-ultra-hc_attention-20261018`，Git 基线为 `593c5d25`。只修改本 worktree；没有修改主仓、PyPTO、Simpler 或共享安装，没有提交或 push。当前候选尚未通过完整 CPU 后端编译，尚未运行设备，因此没有新的精度或性能结论。

## 本轮目标与选题依据

整体目标仍为 S6 七档完整 CSA L1 incore 首尾耗时相对原动态分 task PTO 降低至少 30%：128K B4/B8/B16/B24/B28 与 8K B16/B32。目前七档均未达标。本候选只尝试减少 Attention 的同核重复搬运，不把局部节省等同于完整 CSA 收益。

只读参考：

- canonical `tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/diag_h131072_b16_clock_v2/phase_clock/phase_clock.json`：旧诊断 `ATTN_VECTOR` 的最高 `other_us` 为 176.70 μs；该结果包含当时版本及插桩开销，仅用于选方向，不充当当前候选或当前基线计时。
- `attention_stream_stages.py` 原实现：每个有效块由同一 AIV 产生 `qk_mi/qk_li`，写入三槽 GM；两拍之后同一 AIV 又分别从 GM 读出 `pv_m/pv_l`。没有别的核消费这两个统计量。
- 不重测已经有结论的 `HC_CONSUMER_REDUCE` 和 query-metadata 开关；不与并行代理的 Attention 入口冗余等待删除或 Indexer G2 score 缓存工作混合。

## 候选语义与约束

生产文件仅改 `vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/attention_stream_stages.py`。

每个 AIV lane 建立自己独占的三槽 FP32 mi/li 缓存，生产时仍写原 GM 工作区，供完整输出/工作区验收；在线归并改读同一 lane 的本地缓存。原参数、工作区、phase 数、实际动态 T、Q/L/positions/pages、分核方法、softmax 与在线归并数值顺序不变。原所有 `sync_wait/sync_set/set_ffts` 和 GM `pl.store` 保留。没有把阶段级通知当作逐槽 ACK，没有改变 A3 的发布/缓存维护。

时序为：work `w` 的 softmax 在 tick `w+1` 产生统计量，merge 在 tick `w+3` 消费；下一个同槽 work `w+3` 到 tick `w+4` 才写入。先写后读与覆盖前已读都成立。query 之间 work 继续计数；末尾三拍 drain 保持。有效块谓词同时控制写和读，失效块不读取未初始化 UB。缓存没有跨 lane 共享。

## CPU 源码和协议审计

运行：

```bash
python tests/pypto_test/ultra_hc_attention_20261018/check_candidate.py
```

`source_protocol_audit.json` 记录：98 项原赋值 RHS 不变；仅 `pv_m/pv_l` 从 GM load 改为局部 FP32 slice/reshape；函数签名、全部 GM store、全部 FFTS 调用相同。穷举 0～3 query（每 query 五块）的全部 33,825 个 mask，共 250,960 次有效消费，覆盖全无效、孔洞、跨 query、槽复用、尾排空。这个 CPU 标签模型不模拟 A3 舍入或 DMA，不替代设备保护区/graph/代际检查。

## 尝试 1：两个独立 loop-carried 统计环

两个 `[3,32]` FP32 本地 Tile，逻辑共 768 B。将它们与原在线 m/l/oi 一起显式循环携带，分支中只更新有效块。

完整动态 CPU 解析成功，没有提供样例 B/T。后端在 MemoryReuse / AllocateMemoryAddr 阶段失败：

```text
Vec buffer usage (255232 bytes) exceeds platform limit (188416 bytes)
capacity-gated reuse could not fit memory space Vec at any double-buffering depth;
fell back to the legacy packing (which also overflows)
```

证据 `cpu_v1/result.json`、`cpu_v1.log`、`cpu_v1/source/attention_stream_stages.py` 和 `cpu_v1/kernel/passes_dump/37_after_MemoryReuse.py`。后者确认两环分别为 384 B，同一环的迭代/assemble/slice 已被关联到同一 MemRef；不能简单声称新增的 768 B 自身被扩成 66 KB。可确认的是完整函数内存规划未找到容量内布局；原因可能涉及新增长期存活对象与既有 UB 高峰的干涉，尚未将其认定为编译器 bug。

未提交 NPU。失败产物保留。

## 尝试 2：合并为一个交错统计环

遵循本轮最多一次有针对性表达调整的约束：改为一个 `[6,32]` FP32 loop-carried Tile，slot `s` 的第 `2s` 行保存 mi，第 `2s+1` 行保存 li。仍为 768 B，不新增 concat 或改变数学，减少一个跨整个 Attention 循环存活的独立对象及其分支合流关系。先完整 CPU 编译；若仍超容量则保留负结果，不为该小范围搬运候选修改依赖或反复绕编译器。

复现命令（工作目录为本 worktree）：

```bash
D=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017
P="$PWD/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
bash "$D/reproduce_cpu.sh" "$PWD/tests/pypto_test/ultra_hc_attention_20261018/cpu_v2" \
  --attention-query-metadata off --spmd-source "$P"
```

`cpu_v2` 为本次真实输出路径，重放时须换全新目录，不能覆盖既有失败证据。

尝试 2 最终失败：完整动态解析仍通过，UB 规划为 255,104 B，超过 188,416 B 上限；相对尝试 1 仅减少 128 B。证据 `cpu_v2/result.json` 和 `cpu_v2.log`。这说明合并两个统计对象没有解决整体 UB 规划限制。按约定停止本候选，不继续改变归约、缩 tile 或修改编译器来强行容纳。当前生产文件保留尝试 2 的隔离实现，便于审阅，不建议迁入主仓。

## 本轮最终边界

- 源码数学与通知审计、三槽时序 CPU 模型通过。
- 两次完整动态 CPU 后端编译均因 UB 超容量失败。
- 没有提交 NPU，没有新增精度/graph/保护区或性能数据；七档验收状态不变。
- 没有改 ABI、phase、GM 工作区、共享主仓或依赖。没有提交、push。
- 可复用结论：Attention mi/li 同核回读在语义上可去除；在当前完整内核的 UB 生命周期和规划约束下，本次 768 B 本地缓存表达尚不可部署，不能把它记为优化完成。
