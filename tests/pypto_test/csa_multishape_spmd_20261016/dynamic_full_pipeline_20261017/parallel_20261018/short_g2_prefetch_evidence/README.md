# 短历史 G2/N128/768 的单 panel Key 预取候选

状态：从正式 `d253d627` 复制 33 个 Python 源文件，私有候选只修改 `policy.py` 的
PREFETCH 区间；唯一一次 medium/G2 完整符号 CPU 编译 PASS，生成地址与操作数审计通过。
设备精度及性能不属于这个 CPU 结论，主 agent 已另行提交两个完整矩阵，结果尚未纳入本文。
正式包、冻结的早期 score 缓存候选、依赖及共享安装均未修改。此候选不含早期缓存改动。

## 有限历史审查

在 MASTER `DSV4_CSA_SPMD_OPTIMIZATION_LOG.md` 和其早期验证记录中检索短档、G2、768、
PREFETCH/预取，并读取对应实现报告，未找到“完整动态单 SPMD、短 G2/N128/768、仅
PREFETCH 0→1、TO_L0=true”的设备试验。这个结论只覆盖本地已检索记录，不宣称不存在
任何未归档试验。

- `tests/pypto_test/DSV4_FLASH_CSA_VALIDATION_LOG.md:9626`，§346，以及
  `tests/pypto_test/results/csa_score_key_l1_pair_20260928/README.md`：旧双 query 预取
  明确只改长档，N1024 每步 8 个 panel；短 B16 执行核不变，只是控制组。因此其短档
  变慢不能当作本候选的否定结果。旧任务为 `task_20260928_184711_373603922282`。
- 同一验证日志 §349（9708行）：三 query 预取只到 L1，同样明确短档执行核不变；
  不是本候选的 TO_L0=true/G2 流水。旧任务 `task_20260928_190508_19636832316`。
- MASTER:7683–7700 的两 panel Key 预取是长 G6/M384/N64、1024 交接，任务
  `task_20261017_111540_229620229969`；不是短 G2 单 panel 提前。

本次不重复 N128 已为 G2 默认的核实，不重试长 G6 的 N128/双 Acc/双 panel 方案。

## 唯一策略改动与 CPU controls

`source_reference_d253/` 是正式 33 源冻结副本；`source_candidate/` 仅把
`prefetch=1 if long_history else 0` 改为
`prefetch=1 if long_history or group == 2 else 0`，并增加中文调度注释。
因此短 G2 开启、短 G6 仍关闭、长档不变；其他策略及 helper 源文本相同。
`source_freeze.json` 记录路径、大小和直接文本比较，`policy.py.diff` 保留唯一差异。

精确 CPU 策略见 `controls_medium.json`：G2、N128、tile768、PREFETCH=1、
TO_L0=true、BALANCED=false、ROOTS=2，OA128/OB96，其余 flags 与 medium 基线一致。
只解释 policy 的五组有限描述符检查见 `controls_audit.json`，没有执行内核或扩展大模型。

重要 harness 事实：主 `check_cpu.py:75` 的 medium 分支硬设 PREFETCH=0，
`--spmd-source` 只决定导入哪份源码，不自动调用该目录的 select_policy。
所以本次同时传 `--controls-json controls_medium.json`；最终必须从
`cpu_medium_v1/result.json` 核对 controls、source_package 以及无样例 shape 的符号维度，
不能把单改 policy 后生成的旧策略编译误报为候选通过。

CPU 命令（已执行一次，产物目录不可覆盖重跑）：

```bash
AUDIT=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-indexer-20261018/tests/pypto_test/ultra_indexer_20261018/short_g2_prefetch_20261018
DRIVER=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017
bash "$DRIVER/reproduce_cpu.sh" "$AUDIT/cpu_medium_v1" --strategy medium --attention-query-metadata off --spmd-source "$AUDIT/source_candidate" --controls-json "$AUDIT/controls_medium.json" > "$AUDIT/cpu_medium_v1.log" 2>&1
```

## 源级半叶、别名与尾部检查

G2/768 的 AIC 每轮交接两个384列半步。令 step=s、lane_span=L、panel=p，原非 balanced
地址为 `leaf_begin+s*384+(p//3)*L+(p%3)*128`。开启预取后先 prime panel0，再在当前
QK 之前读取 panel p+1；六个 panel 的最终消费地址仍依次是 0/128/256 与
L/L+128/L+256，加同一个 step/leaf 偏移。384 可被128整除，不会有单 panel 跨半叶。
L 为 `min(ceil(valid/768)*384,4096)`，两种边界也均可被128整除。

既有 Key L1 池为 `[256,128]` INT8，prime 写前128行，随后按下一 panel 奇偶写0/128。
当前 Key 已提取到 Right，下一 Key 提取到另一 Right，当前 QK 消费后才更新 current 引用；
源码意图不覆盖仍待使用的 Key。是否真实分配为独立地址、同步保护是否合理，仍须检查
本次生成物，不能仅根据 Python 变量名声称 alias 已验证。

每页逻辑地址继续按原 cache 长度 clamp，再检查页表列数、物理页范围和 Key0/Key64
视图；没有更改页地址表达式。最后 panel 不越界预取，第六次 QK/WS 的算术顺序不变。
每 query 的真实可见行仍由原 AIV 的 valid shape 与半叶 TopK 控制，根地址和 merge
顺序不变。S5/live1 的 A3 L1 补满物理 M 修正完整保留，GM 仍只读取有效 query。
不合法历史页仍遵守原入口合同，本审查不扩大非法页支持或代替设备保护区检查。

## 片上上界与待核对项

仅按源码逻辑热块计算：Left Query16KiB+coeff4KiB=20KiB；Right 上一 Score 的
WS 输入32KiB+两份 INT8 Key各16KiB=64KiB；Acc QK64KiB+WS8KiB=72KiB。
L1 Query/coeff20KiB、双 Score64KiB和 Key池32KiB共116KiB，不包括可能残留的
未使用声明或完整 MIX 其他阶段。TO_L0 保持 true，prefetch 是既有分支，无新根 Tensor、
FFTS 事件或长期 AIV 缓冲。

上述理论容量不证明 GM/MTE1 工作被隐藏。实际生成检查如下；没有第二次内核编译。

## 唯一 CPU 与生成物结果

`cpu_medium_v1/result.json` 为 PASS；session21528 正常退出0，controls 与
`controls_medium.json` 完全相同，PREFETCH=1、TO_L0=true。source_package 指向冻结
候选，33个编译时源码副本逐文件直接文本相同。symbolic_parse、full_codegen、
single_mix、sync_start、动态根 ABI 和内部 Key 派生视图检查通过；未提供样例 shape，
未运行设备，未宣称 exact 或性能收益。

`audit_codegen.py` 只解析本次既有 PTO/C++，结果为 `codegen_audit.json`；
逐 panel 的 QK、上一 panel WS、下一 Key 的地址和行号均可追溯。六次 QK、六次 WS、
六次真实 Key TEXTRACT 没有减少或增加算术调用。首 Key Right 地址0只在 prime 使用；
稳态 Key 地址为32768/49152，各16384B，WS Right为0..32768；同时活跃的 Key/WS
操作数不重叠。声明里未使用的初始 current/prefetched 对象同址0，不可误当活跃双槽。

QK Acc为0..65536，WS Acc为65536..73728。L1 Key池65536..98304，常态 Score
98304..131072和131072..163840；早期 Query/coeff 位于 Key池范围，但已搬至 Left
后才被 Key DMA 复用。最后一份 Key 提取后，Score 复用该 L1 区域：
`kernel/ptoas/dynamic_csa_full.cpp` 第14210行 TEXTRACT，第14211行发出
MTE1→FIX EVENT0，第14262行等待该事件，第14266行才 TINSERT。此处显式保护尾段
L1 复用。Key环复用也保留 MTE1→MTE2、M→MTE1 等局部事件；典型片段见
`codegen_excerpt.txt`。上述是生成源码的地址/次序证据，不是最终机器指令或真实重叠证明。

完整 MIX 各空间分配末端为 Mat475136B、Left65536B、Right65536B、Acc131072B、
Vec163872B，未超当前约束。下一 Key 的 TEXTRACT 在当前 QK 前发射，也仍有
MTE2→MTE1 wait；不能从代码次序直接承诺隐藏了多少 DMA 延迟。

离线摘录脚本首轮未识别 `ui32`，补充类型宽度后继续；随后修正跨函数重名数字 SSA
使总空间末端统计不被覆盖。仅重读同一产物，CPU编译次数仍为1，未修改候选源码。

主 agent 提交的设备作业为 B16 `task_20261018_031515_236275422318`、B32
`task_20261018_031515_236279311548`；本 agent 不提交或重复这些任务。
两侧源目录从提交前继续冻结。后续最小边界建议见 `BOUNDARIES.md`，没有自行运行。

## 完整设备矩阵结束：不采用短 G2 预取

主会话确认两个同进程完整矩阵均已结束并通过全部检查。每侧采用5个完整CSA L1窗口，
下表单位为µs；相对变化以同轮正式参照为分母，不能与早期跨轮数字拼接。

| 档位 | 原分task | 正式参照 PREFETCH0 | 候选 PREFETCH1 | 相对正式参照 |
| --- | ---: | ---: | ---: | --- |
| 8K/B16 | 680.90 | 682.86 | 693.48 | 慢1.555%，全部5窗均慢 |
| 8K/B32 | 991.36 | 1002.38 | 999.38 | 快0.299%，仅弱改善 |

B16任务为 `task_20261018_031515_236275422318`，B32任务为
`task_20261018_031515_236279311548`。原始报告已由主会话归档；本节按其最终确认追加。

结论：不采用短G2的PREFETCH0→1。CPU和生成地址检查通过没有转化为稳定整链收益，
B16明确退化，B32的弱改善不足以支持整个策略区间，也不建立B32特化或与其它候选组合。
`BOUNDARIES.md` 中的追加建议不执行，不再增加S1/S5尾probe、CPU编译或设备任务。
两份33源参照/候选继续冻结保留。本节追加于原CPU归档之后，仅更新本README的最终结论。
