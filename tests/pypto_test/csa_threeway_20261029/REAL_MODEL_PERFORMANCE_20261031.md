# 8K真实模型：Native、我方精度修复TND与nalinaly性能TND

## 测试目的与复用

比较完整CSA替换后的输出token、接受事件与性能。HCA、FFN、采样和调度框架保持公共版本。
复用已验收Native金标，不重复运行Native2或128K。
已有我方61,440个输出token与Native完全一致；原观测要求40请求同时活跃，实际采用8请求分波，
因此旧记录没有有效性能样本。本轮每方只新增一次模型运行，补采匹配实际入场的计时。

## 固定配置

| 项目 | 设置 |
|---|---|
| 模型 | DeepSeek-V4-Flash-0731-w8a8完整真实权重，43层 |
| 并行 | TP1，DP=EP8；本轮临时批准物理卡8–15，0–7禁用 |
| 请求 | 每rank相同40种不同问题，分5波×8；不是B40同时活跃 |
| 上下文/输出 | 固定前缀8192，加真实问题；每请求192输出token |
| 生成 | 温度0，DSpark出5验6；确定性level1和HCCL确定性开启 |
| 图/缓存 | FULL_DECODE_ONLY，capture sizes 6/240，KV blocks=4000 |
| 权重/并行优化 | NZ2、atomic add关闭、EPLB关闭 |
| 替换范围 | 完整21个CSA层HC-pre至HC-post；20个HCA层全部Native |
| 公共框架 | `41ba392f482bf88f7b6a797e2542443c81497338` |
| 我方CSA数学包 | `65bebf7f6b2e4f3f6492fd6463f73106506f3496` |
| nalinaly性能CSA包 | `79ea6d3c8460338a172cb5570d14999d5a544262`，本轮GitHub确认仍为最新 |
| SDK | 共享已验收`csa-shared-20261024-group-dims`，不更换编译器或运行库 |
| PTO容量 | 双方同样window2048，heap=(256,512,256,256)MiB，默认dependency pool |
| 时间上限 | 每任务内部720秒、调度780秒；首个输出token差异停止 |

## 计时与精度口径

1. **目标模型forward**：过滤真实非dummy、非draft、非capture的实际B8/FULL240步骤，
   丢弃8个预热步骤，采20个NPU Event样本/rank。每条保留实际T、padding T和positions。
   包含整模型CSA、HCA、FFN等；不是单层CSA耗时。
2. **公平性能门禁**：与本轮Native的实际请求数、实际token数、padding、positions逐样本核对；
   同时要求替换覆盖、完整有效采样、零抢占。缺样本或入场轨迹不一致时不计算公平速度比。
3. **分波客户端墙钟**：计时`generate_aligned_batch`调用，止于外部完成屏障之前；
   包含内部暂停/入队/入场协调/唤醒、草稿生成、验证、CPU逐token检查，不能称为纯设备生成耗时。
   每波取8个rank最大值后求和。仅作端到端辅助指标。
4. **不用相邻采样cycle作主要指标**：选中样本间可能夹有未选中步骤或分波等待。
5. **精度**：逐请求、逐token核对既有合格Native金标；token或输出长度首差停止。
   接受事件差异独立保存，不把事件不一致自动称为输出token不一致，也不把token一致称为内部逐bit一致。
6. **冻结**：只在快照旁新增内容hash的runner、observer、mixed副本；不改原冻结数学包、
   runner、scheduler或Native金标，不重装SDK。严格原postflight的事件失败仍保留。

## 任务和复现

| 实现 | 任务 | 输出标签 |
|---|---|---|
| Native计时 | `task_20261031_085454_241324827611` | `last8-native-8k-wave-perf-v1` |
| 我方精度TND | `task_20261031_085811_249497130046` | `last8-ours-8k-wave-perf-v1` |
| nalinaly性能TND | `task_20261031_085811_24943951347` | `last8-nalinaly-8k-wave-perf-v1` |

同[主报告](RESULTS.md)的环境和数据根目录。先source公共环境及共享SDK环境。
`run_eight_card.sh`在进入模型前强制核对`TASK_DEVICE=8,9,10,11,12,13,14,15`。
Native使用`eightcard-source-v5/native`；我方使用同cohort的`ours`；
nalinaly使用`eightcard-nal-perf-v1/nalinaly`，由同Native框架副本只替换完整CSA包创建。
三方均传`--measure-waves`；两份PTO实现另传`--inspect-token-sequence`，
并显式设置`PTO_CSA_RING_TASK_WINDOW=2048`，reference指向`last8-native-8k-waves8-v5`。

```bash
# 在共享环境下、无设备运行CPU门禁，bootstrap参数同主报告。
pytest -q tests/pypto_test/csa_threeway_20261029/test_wave_measurement.py \
  tests/pypto_test/csa_threeway_20261029/test_runtime_audit.py \
  tests/pypto_test/csa_threeway_20261029/test_token_diagnostic.py \
  tests/pypto_test/csa_threeway_20261029/test_summarize_waves.py \
  tests/pypto_test/csa_threeway_20261029/test_postflight.py

# 完成后无设备汇总；参数使用绝对结果目录。
python tests/pypto_test/csa_threeway_20261029/summarize_waves.py \
  "$native_performance_root" "$native_performance_root" "$ours_root" "$nalinaly_root" \
  --output "$summary_json"
```

CPU门禁已在227通过：观测/审计/token诊断7项，汇总及postflight17项。
独立审阅确认冻结与首差机制保留；上述客户端墙钟和非连续cycle限制已纳入口径。

## 结果

任务排队中，尚未产生本轮真实性能结果。
