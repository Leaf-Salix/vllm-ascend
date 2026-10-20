# BF16布局完整CSA候选：首差与停止记录（2026-10-20）

## 结论

同分支`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`新增独立候选
`deepseek_v4_flash_csa_tnd_precision_spmd_bf16layout`。
短T（小于64）保留原布局，大T使用已完成前缀对拍的BF16布局；原43＋14＋6个源码不变。
完整根、归约、舍入、缓存和同步逻辑均沿用已验收预打包版，仍只把前缀融合为SPMD。

**完整验收未通过，候选不作为精度交付版，不修改默认路由。**
T21 FULL通过；新增T126混合请求发现首差，按约定停止T240及8K/B40、128K/B24整模型测试。
不能把此前前缀3%左右的收益或旧版整模型token验收转授给本候选。

## 已执行与未执行

均使用真实C4层权重、8K合成历史/hidden、确定性Native、SK0、独立cache；无整模型Native2。

| 测试 | 请求长度 | 结果 | 说明 |
| --- | --- | --- | --- |
| CPU及编译门禁 | T1～256资格范围内的源码合同 | 通过 | program、CCEC、kernel ABI、Torch64入口及prepare2注册通过 |
| 完整CSA FULL，T21 | 1,2,3,4,5,6 | 通过 | A/B输出、TopK、六类cache/state逐bit相同；T21↔T6交替replay通过 |
| 完整CSA FULL，T126 | 上述长度重复6次，共36请求 | **未通过** | 第86行TopK重排，最终输出发生差异；首差后未计时 |
| 原预打包版同输入对照 | 同一T126 | **同样未通过** | 输入、初始cache、全部Native及候选边界与新候选逐字节相同 |
| T240 FULL | 40请求各6行 | 未执行 | 首差门禁停止扩展 |
| 真实8K/B40、128K/B24 | warm192＋3×192计划 | 未执行 | 不绕过精度门禁测性能 |

T21的8个配对计时窗口仅用于本用例：Native平均561.596μs，候选841.291μs。
这不是大T布局优化的性能验证，因为T21仍走旧路径，也不是真实整模型计时。

## 首差定位到什么程度

T126没有每请求补S6：真实行数和FULL图行数均126，而补S6需要216行。
TopK唯一出现差异的query为第86行（从0开始），绝对position=8192。

- TopK有28个位置不同；排序后所选key集合完全相同，其余query的TopK相同。
- 最终输出有1164个元素不同，位于第86、103两行；最大绝对差0.015625。
- 六类cache/state边界逐bit相同。
- 原预打包版和BF16候选的两份远端快照，在hidden、positions、所有初始allocation、
  八项Native结果及八项PTO结果上全部逐字节相同。

这排除了**本例中**BF16搬运引入新增边界差异；它揭示了原实现尚未覆盖的T126 FULL精度问题。
旧版已测8K/128K真实模型token一致的事实仍成立，但不能推导任意TND输入都与Native逐bit一致。

目前最早观测到的外部差异是Indexer TopK。**尚未证明是并列分数排序规则的唯一根因**：
分数上游的query/scale/weights及头维规约也可能令近邻分数次序变化。
第103行的TopK相同而输出不同，进一步说明TopK排序不能独自解释全部输出差异。
该行还需继续检查Q、attention及O-proj边界，当前不能宣称所有误差都来自Indexer。
Native源码的head规约是逐query的K64，而当前PTO使用分组块对角乘法；
这是需单因素验证的算术边界，不能仅凭源码不同就认定为根因。
不得通过重排已知索引、容忍这28个位置或放宽误差阈值通过测试。

## 观测限制与下一步

独立诊断副本尝试`return_value=True`导出Native QLI分数，当前安装版本拒绝：
`input attr returnValues only supported False`。
该任务属于观测接口不支持，没有得到可采信的分数对拍；不是新增布局的精度失败。
未修改SDK、Native tiling或系统环境来绕过限制。

下一诊断应在支持的接口内保存实际Native query/scale/weights，
再做相同输入的PTO分数链对拍，区分上游舍入与分数规约/排序。
必要时把head规约单独恢复为Native的逐query K64做单因素验证。
确认并修正首差后，须重新通过混合T126、T240 FULL，才进入8K→128K整模型门禁。

## 复现、证据和运行纪律

环境及权重沿用[已验收预打包报告](DSV4_FLASH_CSA_TND_PRECISION_SPMD_PREPACK_REAL_MODEL_20261020.md)。
独立variant可通过`PTO_CSA_VARIANT=pkg:deepseek_v4_flash_csa_tnd_precision_spmd_bf16layout`选择，
**仅供本实验复现，未获得交付验收**。固定权重、串行设备执行工作区和T≤256限制不变。
没有重装vLLM/PyPTO/Simpler，没有手写C++数学；生成C++仅作编译证据。

证据位于`evidence_tnd_precision/spmd_bf16layout_first_difference_20261020/`：

- 正式七源编译receipt、父六源receipt、实际生成前缀及SHA。
- T21通过结果、T126失败及原实现同输入失败、观测接口不支持记录。
- 三套实际助手及SHA、CPU全快照比较脚本、逐tensor SHA。
- 小型NPZ含两次运行全部TopK及两行差异输出，可离线重算28个重排、集合相同和1164个输出差异。
- 两份完整428MiB首差快照保留在原实验服务器；未把完整cache张量加入Git。

实际助手和生成源码按原字节保存在`raw-helpers.tar.gz`，避免格式化归档改变实测SHA。
T21通过任务为`task_20261020_081938_414474326474`；T126候选及原实现对照分别为
`task_20261020_082243_11799323982`、`task_20261020_082640_58589323616`。
分数观测不支持的任务为`task_20261020_083424_39693534952`。

先运行无设备编译/注册门禁，再分别通过task-submit执行对应`run-full-csa.sh`：
T21参数为`1,2,3,4,5,6 4 8191`，T126为该列表重复6次、层号4、历史8191。
原实现对照使用`bf16layout-control-full-tools`，不切换或修改原checkout。
映射助手中的独立实验根及冻结基准路径后，在新输出目录执行，禁止覆盖已有结果。

所有任务均设置780秒调度上限、内部750秒timeout，已结束并释放卡。
另一次启动在argparse阶段因缺层号退出，修复为source环境前保存参数并显式传4/8191；
无任何精度结果。所有失败都保留，不记作通过。

离线证据审计（验证报告与首差证据一致，不表示候选精度通过）：

```bash
python -m pytest tests/pypto_test/test_precision_spmd_bf16layout_cpu.py \
  tests/pypto_test/test_precision_spmd_bf16layout_first_difference_evidence_cpu.py -q
```
