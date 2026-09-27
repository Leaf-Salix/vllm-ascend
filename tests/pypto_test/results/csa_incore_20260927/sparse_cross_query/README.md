# Sparse Attention：跨 query 延续核内流水

2026-09-27，基底 `c160cabe`，最终改动见 [candidate.patch](candidate.patch)。
只改性能版 `decode_sparse_attn_csa.py`，精度版、任务数和跨任务依赖不变。
本目录记录先导与代表档；随后同一源码 `da2e2368` 的[七档单卡结果](MATRIX.md)已补齐。
七档共28个泳道见cases.json；没有16卡整模型验收。

## Native 策略与实现

Native `sparse_attn_sharedkv_scfa_kernel.h` 的 `gloop` 在 batch/gS1 循环外延续，
只在本核最后一个 query 的 `isEnd` 条件下追加 `extraLoop=2`。
原 PTO 对每个 query 都执行5个候选块，然后 AIC/AIV 分别追加2/3轮排空。

当前改为每核一个连续工作序列：QK处理k、softmax处理k−1、PV处理k−2、PV合并处理k−3。
各阶段分别推导所属query及局部块编号；三槽编号按全局工作项递增。
每个query最后一个PV完成后写出其独立结果并重置归约状态，下一query的QK/softmax保持在途。
每核仍按 `core + query_idx*24` 取query；没有混入已撤回的连续query分配候选。
已有 `T>=144` 提前KV发布选择继续保留；块内BF16量化及每query的归约顺序保持原样。

B40/S6每核10个query：AIC循环从70轮变52轮，AIV从80轮变53轮。
这只是控制流轮数，不是可直接换算的性能收益。

## 功能检查与首版问题

- CPU完整根算子编译通过，最终日志 `compile.log`。
- 固定Native 8K/B40输入，最终Sparse输出与基底PTO的7,864,320个元素逐bit一致。
  `run.sh` 的 `torch.equal` 断言已通过，任务日志包含
  `CROSS_QUERY_FIXED_INPUT_AND_MIXED_TAIL_BIT_EQUAL_PASS`。
- B9/S6/T54解析用例：query 24～47全部无效；部分核经历有效→无效→有效，
  各核2/3个query不均分。1,769,472个元素与解析值逐bit一致。
- B3/S6/T18解析用例：6个AIC没有query。589,824个元素与解析值逐bit一致。
- 两个解析用例不是Native实测参考；旧通用诊断入口将所有 `--input` 统一标作Native，
  本目录的 [cases.json](cases.json) 明确区分参考来源。

首版把初始sink tile留作重置值，但生成代码将它与循环中的m状态分配到同一UB地址。
第一个query以后重置读到的已不是原始sink，固定输入出现465个差异元素；这是状态错误，不能放宽容差。
最终改为每次重置重新从GM加载sink，恢复逐bit一致。
[初版补丁](initial.patch)、[差异](fixed_input_diff.json)、[地址别名证据](sink_alias_codegen.txt)保留定位过程。
这证明首版实现不安全，不据此推断所有PyPTO循环tile都存在相同问题。

## 计时口径与结果

A3/CANN9、固定正式layer4权重、单卡S6/TP1/mode2/atomic1/确定性0、EPLB关闭。
本体为HC_pre→norm→CSA→HC_post，复用第二层metadata；拆分与写回另计。
5次warmup、20次无profiler设备计时；核内为4个独立DFX窗口各自的block均值范围。
Native是完整kernel，PTO是单block均值；不能直接计算等范围加速比或相加AIC/AIV。

| H / B | 参考本体 μs | 当前本体 μs | 当前 p50 / p95 μs | 参考 qk_pv AIC μs | 当前 qk_pv AIC μs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K / 40 | 1361.40 | 1342.68 | 1335.63 / 1401.54 | 301.01–318.44 | 283.06–288.68 |
| 8K / 16 | 796.12 | 795.04 | 794.33 / 813.26 | 124.04–137.00 | 122.51–125.71 |
| 128K / 16 | 1304.87 | 1269.53 | 1203.15 / 1549.14 | 170.48–181.13 | 172.52–180.50 |

B40参考来自上一保留实现；B16本体参考分别为最终早通知分派版和V10。
B16参考核内均为V10，最终早通知分派版8K/B16没有采DFX，不能冒充同轮数据。
B40核内范围明显下降，本体下降1.38%；8K/B16本体基本持平。
128K/B16的核内范围与参考重叠；长尾仍在，没有剔除慢样本，不认定该档稳定获益。
两档B16的merge_norm分别为24.55–25.26 / 23.68–23.78 μs，高于V10的
20.92–22.56 / 20.58–21.09 μs；该任务源码未改，不能只挑qk_pv单项宣称本体必然改善。
所有轮次独立采集，不把各轮最优数据拼成一份新的七档矩阵。

目前8K/B40完整PTO1683.07 μs、同轮Native1427.96 μs，完整路径仍慢。
当前代表档的保护区、索引结构与非有限值由 [cases.json](cases.json) 逐项记录；
对Native的既存浮点差异仍在，零容差FAIL不能写成精度验收通过。
本轮没有16卡逐token/DSpark验收。

## 复现与证据

- [run.sh](run.sh)：固定输入、B9混合无效/尾部、B40计时及DFX。
- [run_followup.sh](run_followup.sh)：B3零工作量核、8K/B16与128K/B16计时及DFX。
- [summarize.py](summarize.py)：只读现有报告和trace，不运行设备测试。
- [cases.json](cases.json)：逐项误差、计时、四窗口原始泳道路径。
- 首版失败任务 `task_20260927_164957_21969421834`，在固定输入断言处退出1，未跑完整CSA。
- 修复后任务 `task_20260927_165242_22114026823` 退出0。
- 代表档任务 `task_20260927_165609_22367535890` 退出0。
- 剩余四档任务 `task_20260927_170134_22993926622` 退出0，见 [run_remaining.sh](run_remaining.sh)。

通过 `task-submit --device 0 --max-time 900 'bash <上述脚本路径>'` 提交。
大张量与编译目录保留在本地，不加入Git；固定输入来源为
`../sparse_pmu/native/native_sparse.pt`，生成办法见该目录的README。
