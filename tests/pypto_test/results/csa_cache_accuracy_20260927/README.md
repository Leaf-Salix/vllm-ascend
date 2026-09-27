# Native原布局内部读写：单卡cache精度通过，16卡看护进行中

用户要求先完成P95长尾及精度检查，再交替进行调度/incore优化。当前生产源码固定f76b3ad4，不叠加性能候选。

## 单卡隔离检查

对照3d1f0f65（外部拆分＋PTO＋Torch写回）与f76b3ad4（内部原生页读取/提交＋长档整组准入）。
两者主算术、正式层4权重、seed1024、mode2、B4/S6相同；atomic0、Native确定性1/HCCL=true，无EPLB。
用8K/128K两档覆盖短/长Score分支；原七档的atomic1单层误差/保护区检查不等同此隔离比较。

原合成case的全部Indexer scale为0.01，无法充分检出scale取错页。因此通过仅测试用的
[accuracy_case.py](accuracy_case.py)，将scale换成随物理行变化的251种可精确表示FP16正数，
保留原非连续/反序物理页和0/64B相位；两侧同步更新各自初态，不改生产代码或正式计时fixture。

完整检查8类逻辑输出/状态：x_out、idx_topk、swa、compressed、state、Indexer key/scale、Indexer state。
Native侧也必须改造前后逐元素一致，作为输入/环境对照；PTO固定规约前后零容差比较。
两侧检查自身重复执行、保护区和metadata；新版本另检查同址A→B→A图重放。
已有H32767同图4→3→1→4补位结果仍见`../csa_cache_panel_read_20260927/padding/`，不重复未受影响的padding矩阵。

[run.sh](run.sh)、[compare.py](compare.py)。任务`task_20260927_221629_57292629657`。
旧任务`task_20260927_221254_52367321182`尚未开始即取消，原因是补充有区分度的scale输入；不是因观察超时重启。
该任务退出0，两档逐元素比较均PASS：

| 档位 | Native对照前后 | PTO改造前后8类输出/状态 | 当前PTO A/B/A图 | metadata/保护区 |
| --- | --- | --- | --- | --- |
| 8K/B4 | 全部零差异 | 全部零差异 | PASS | PASS |
| 128K/B4 | 全部零差异 | 全部零差异 | PASS | PASS |

[8K完整比较](h8192_b4/comparison.json)、[128K完整比较](h131072_b4/comparison.json)。
原始两侧报告和states.pt保存在各档位/提交子目录，大张量不入Git。
Native/PTO固有量化与Top-K差异仍在；这里证明所测case的cache改造数值中性，不是两种算术已对齐。

## 之后的整模型看护

单卡两档通过后已提交[run_model.sh](run_model.sh)，任务`task_20260927_222524_66096122013`：正式W8A8、TP1/DP=EP16，
H8192/B16、DSpark出5验6、mode2、FULL_DECODE_ONLY、EPLB关、PTO性能版atomic1、两侧Native确定性0/HCCL=false。
复用已有通过audit的真实prefill bank；两侧各生成96 token/请求，比较16×16×96=24576个token及各rank的DSpark总计数/逐位置统计。
这项是当前新算术/cache路径的模型看护，不能以旧整模型结果代替；也不把一次token看护外推为完整七档模型性能或EP16尾部验收。
