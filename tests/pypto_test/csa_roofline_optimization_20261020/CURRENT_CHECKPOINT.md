# 当前交接：B1优化已采用，继续优先128K/B1、B16、B28

目标active：相对PTO多task完整incore L1提升10%，公式(多task−SPMD)/SPMD。
当前重点三档均未达成，不能标记complete。保留八档、严格精度/TopK/score/
状态/保护、动态metadata/padding及图/代际合同。B1允许单独策略，其余区间
策略；20μs最低门槛已取消。不push。所有本轮设备与CPU任务均已终态。

## 当前正式源码与采用

正式37源在ops/pypto/deepseek_v4_flash_csa_spmd；当前快照
source_after_b1_half_v2，记录SOURCE_AFTER_B1_HALF_V2.json。
只有policy/group/score/topk/publish五文件改变。B1在query容量5..6、
8192<压缩历史上界<=65536时，独立分派原动态半叶；每AIV保留4×1024
分数，按原六条排序分支处理，累计根留UB，原末叶及同分规则不变。
B/T/长度/位置/页表仍动态，默认wait150，未新增外部适配或任务。

task_20261021_033259_218820330623完整确认：多task387.60、正式改前
401.30、候选371.94μs，净省29.36，5/5窗口更快。相对多task4.21%，
不足10%。实际S1/cap6任务21878753496及S5全同分335495412701通过；
完整精确/状态/保护/metadata/padding/图/代际通过，见B1_HALF_SCORE_ADOPTION。

最终guarded源仅将UB初始化限定在B1。第一次G6 CPU发现无用16KiB填充，
已修正。B1/G6完整符号CPU均通过；B1 PTO IR仅去loc/SSA拼写/空白后与
已上卡v5逐字相同（没有删操作/数值/地址），G6不再有legacy_half_scores。
见B1_GUARDED_IR_AUDIT和B1_G6_INIT_GUARD。不额外重跑相同B1设备验证。
source_after_b1_half是修正前快照，已标记废弃；后续只从v2继续。

## 本轮其它结果与已结束候选

WS列池150tick两档完整PASS：B16多task840.66/正式786.66/候选781.52μs，
4/5快；B28为1214.14/1151.22/1153.14μs，3/5快。分布混合，不把B28
中位慢1.92μs说成确定性回退，也不足以整体采用。保留B16正向证据，
暂不整体采用、不扩档、不扫池宽，不写B16单点白名单。

pair直接压缩首轮B28省21.22，反向仅2.84，同址慢3.16；B16中性，已停止。
B1旧4K流式版13.38%被同分FAIL拦住：2560 TopK和57510输出不同，
max_abs0.17431640625，score及状态通过。原因是末叶半跨度及tie顺序改变，
不是组件bug。原半叶GM修复后378.66、再UB376.10是各自轮次，不相加。
当前正式采用数据只用上面的直接配对371.94。

## 基线、工具链、复现

公共多task当前冻结frozen_pypto_goal10_b1update=37c47753，含4d5b2ed9 B1
优化。旧frozen_pypto_goal10=9aec2d92用于旧轮，不能当新B1基线。
两manifest可从Git恢复；原record-name错误已修复，没有改任何已测源码。
后续若共享多task再有数学更新，应新增快照，不覆盖这两份。

PyPTO独立pair树8c360367、PTOAS独立pair树本地c04ecdf9、Simpler独立
core-args-256树。共享安装不变。PTOAS#1650评论6297714053、6298459118
保留直接压缩实现和负性能证据；#1649评论6299434307新增TMOV列子视图
最小复现/行布局对照及修改建议。无PR/标签/push。

测试脚本和所有尝试均在本目录；源码候选保留，失败的大IR仅列本地路径。
prepare_goal10_archive.py只选择本会话轻量证据/代码/日志，不纳入权重或
二进制，不做hash。上一过程提交8aca9c38；B1采用代码与最终证据另行中文提交。

## 下一步

继续优先B16/B28，兼顾B1距离目标仍约19.58μs。B1新半叶可评估逐页scale
读取成本；原连续scale策略形成于这次半叶改造前，须先说明机制差异再试。
B16列池仅是正向证据，若收窄必须按资源区间，并验证边界。B28继续找主耗时，
不要恢复用户已放弃的Attention数学方向或原样扫描已失败组合。

旧source_score_pool_goal10基于source_after_wait150，不能整目录覆盖当前
正式源，否则会丢掉本次B1。下一候选一律从source_after_b1_half_v2构造。
尚未把本次变更做八档整轮复测，不将CPU死分支检查冒称其它档设备性能通过。
