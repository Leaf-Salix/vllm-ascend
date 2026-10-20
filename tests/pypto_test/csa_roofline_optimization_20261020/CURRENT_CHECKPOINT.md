# 当前交接：优先128K/B1、B16、B28，B1采用前确认中

目标active：动态单SPMD相对PTO多task完整incore L1提升10%，公式
(多task−SPMD)/SPMD。保留八档、精确输出/TopK/score/状态/保护、动态
metadata/padding/图/代际合同。B1可分档，其余只允许区间。取消20μs门槛。
正式SPMD数学暂未改，默认150；不push。详细结果GOAL10_FOCUS_PERFORMANCE。

## 仅需继续收取的原任务

- task_20261021_033259_21878753496：B1真实S1、候选capacity6；
  b1_half_score_s1_cap6_v1；无性能采样。
- task_20261021_033259_218820330623：B1/S6分数留UB对正式source_after_wait150，
  b1_half_score_formal_confirm_v1；新多task/正式SPMD/候选三方五窗。

命令见B1_ADOPTION_CONFIRM_TASKS.json。不因观察等待重提。
source_b1_half_score_resident是当前完整候选，prepare脚本可逐字重建37源。
上述两个任务通过且正式配对有稳定整体收益，再考虑采用B1；不因旧13.38%
宣称达标。通过一个S1不等于整个请求生命周期或八档覆盖完成。

## 已完成结果

B1固定4K/流式2048旧候选普通精度通过，但H139264/S5全同分FAIL：TopK2560项
和x_out57510项不同，max_abs0.17431640625，score及状态通过。根因是末叶
原半跨度动态且原排序分支tie顺序不同；不是组件bug。旧13.38%不允许采用。

原半叶修正版保留旧动态半跨度、原六条排序分支和空根、UB累计根。全同分及
完整S6通过；三方五窗381.86/399.18/378.66μs（多task/正式/半叶GM），
净省20.52但相对新多task仅0.85%。

source_b1_half_score_resident再将每AIV半叶分数留在4×1024 UB，reshape到
1×4096后按原分支排序。完整CPU58轴/单MIX、Vec声明地址上界163872B不变。
H139264/S5同分和完整S6两任务032530_335495412701、335557216998均exit0。
新轮多task388.40/半叶GM378.70/UB376.10μs，4/5快，净省2.60μs；相对
多task3.27%，不能与另一轮20.52相加。正式源/S1确认尚待上面两任务。

B16/B28 WS列池150tick复核均完整PASS。B16多task840.66/正式786.66/
候选781.52μs，4/5快；B28 1214.14/1151.22/1153.14μs，3/5快。B28
分布混合，不能称确定性回退，也不足以整体采用。保留B16证据、不整体采用、
不扩档、不扫池宽，不给B16单点白名单。此前20μs门槛不再适用。

pair直接压缩的首轮B28省21.22，反向仅2.84，同址反慢3.16；B16中性。
已停止，不原样重测。A/A独立工作区差13.18μs，同址3.54μs；地址不是
已确认唯一原因，同址诊断不能覆盖采用试验。

## 基线与组件记录

旧目标快照frozen_pypto_goal10=9aec2d92只用于旧轮。
共享多task已合入4d5b2ed9 B1优化，当前完整轮统一使用
frozen_pypto_goal10_b1update=37c47753（之后共享提交只记录测试，不混入本轮）。
manifest分别为SHARED_SOURCE_GOAL10.json、SHARED_SOURCE_GOAL10_B1UPDATE.json。
冻结脚本record-name的旧错误已修复，旧manifest恢复，新manifest按实际快照补回；
任何已测源码均未改变。两快照可按记录从Git恢复，无hash。

PyPTO独立pair树8c360367、PTOAS独立pair树本地c04ecdf9、Simpler独立
core-args-256树；共享安装未改。PTOAS#1650评论6297714053、6298459118
已补实现和负性能证据。

UB候选v1路径错误/v2类型不完整/v3分支SSA/v4列subview失败全部保留。
v5改为行布局通过。纯PTO列版在ResolveBufferSelect后TMOV目标物理列被
扩大、row版通过，已补PTOAS#1649评论6299434307，附最小复现与建议。
没有新增PR/标签/共享依赖修改；本地算子绕行不删除同步或改变同分规则。

阶段详细中文本地commit，归档脚本prepare_goal10_archive.py只选本会话
源码/脚本/轻量原始证据；大编译物、二进制和重复快照仅列LOCAL路径。
