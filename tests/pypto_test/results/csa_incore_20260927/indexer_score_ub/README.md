# Indexer缩放Score驻留UB：两档核内无收益，撤回

基底ec12e0e9，性能版独立候选。Native QLI `ProcessVec1` 将Cube结果读入UB，乘key scale后直接排序并维护Top-512。
最新pypto-lib2164563仍将缩放分数写入`score_arena`再读回排序。本候选只消除第二次GM中转，
Cube→Vector的`buf_score_transfer`和最终Top-K pair写出仍保留；没有实现Native完整的2048候选流式Top-K。
保留双query、768候选步长、24MIX任务、原sort32/mrgsort分支和tie/半区合并顺序，精度版不变。

## 结果

两档各5预热/20计时＋4个独立DFX窗口，四窗口block核内均值的平均如下，单位μs。

| 项目 | 8K/B16基底 | 候选 | 128K/B16基底 | 候选 |
| --- | ---: | ---: | ---: | ---: |
| Score AIC均值 | 27.14 | 29.46 | 343.35 | 351.86 |
| Score AIC四窗口范围 | 25.37–30.67 | 25.07–38.25 | 339.50–349.12 | 348.58–356.96 |
| Score AIV均值 | 31.70 | 34.12 | 352.64 | 362.10 |
| Top-K merge均值 | 9.90 | 8.23 | 18.34 | 16.87 |
| 本体均值 | 786.57 | 782.82 | 1278.32 | 1305.68 |
| 本体p50 | 788.71 | 780.89 | 1197.83 | 1213.97 |
| 本体p95 | 803.64 | 795.24 | 1541.40 | 1546.34 |
| 同轮Native均值 | 935.92 | 927.60 | 1309.96 | 1326.72 |

长档Score AIC+2.48%、AIV+2.68%；短档均值也未改善，范围有重叠，不能宣称稳定收益。
Top-K merge虽较快，但其代码未改，不能归因为它的算术优化；本体短档约−0.48%，Native亦约−0.89%。
撤回，不扩测七档，不以逻辑GM流量减少代替性能证据。

## 表达成本和检查

每AIV用2×4224 FP32存两query半leaf，4224容纳11×384完整物理panel，有效候选最多4096。
当前PTOAS的非Mat TMOV要求源/目标物理形状匹配；直接小tile拼大tile被拒绝。
最终候选以相同大物理形状、有效1×384的源加载/缩放，再取小逻辑子视图写入目标对应偏移。
生成C++与ISA确认每次TMOV只搬有效384元素，不覆盖已有分数；增加的是UB占用。
排序前用UB gather得到512/1024/2048/4096的紧凑tile，再沿用原padding/sort逻辑。
这些复制和gather也是成本，没有证据把退化精确归到其中某一条指令。

`pl.jit.inline`本地特化未保留Tile参数注解，因此最终在唯一调用点展开排序体，未改PyPTO。
曾试TSCATTER追加，但当前A3 ISA实现会先清空目标且逐元素标量复制，不能作为累积缓冲。
该任务task_20260927_201501_387130811520已主动终止（exit130），数据排除且已删除，未纳入对照。
没有修改PTOAS/PTO-ISA。最终有效任务task_20260927_201836_388760529573退出0。

单卡A3/layer4正式权重＋合成输入/历史，第二CSA metadata复用，S6/TP1/mode2/atomic1/deterministic0，无EPLB。
保护区/Top-K结构通过、非有限值0；Top-K替换仍为366/670；Native零容差FAIL。
输出max_abs=0.03125/0.0390625，RMSE=0.0033201523/0.0041770386，未做新的整模型token/DSpark验收。

[补丁](candidate.patch)、[CPU重建](compile_candidate.py)、[运行](run.sh)、[四窗口和原始路径](report.json)。
统计：`python tests/pypto_test/results/csa_incore_20260927/compare_candidate.py incore_indexer_score_ub_v2 tests/pypto_test/results/csa_incore_20260927/indexer_score_ub/report.json`。
下一项改为长历史每轮1024候选、短历史保留768，先减少Score循环和跨核通知，不再叠加UB驻留候选。
