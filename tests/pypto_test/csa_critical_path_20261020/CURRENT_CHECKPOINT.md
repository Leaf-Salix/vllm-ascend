# 当前检查点：正式Score区间版；五档30%目标仍未完成

## 正式源与基线

正式目录：vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd，37个Python文件。
已采用代码为13259f92，完整source_after_score_split与正式逐字相同。
旧source_reference是E49历史对照，**不能再当当前正式**或累计旧收益。
后续候选必须相对source_after_score_split同轮三方五窗测量。

正式采用的Score：长历史G6且既有T容量>=32，双48KiB QK Acc提前FIX，
原完整六行系数K192+192累计到独立4KiB WS Acc，一次写回。仍是单MIX，
T/B/长度/页表/有效行/循环范围动态。新增constexpr为SPLIT_SCORE_ROWS；
CPU controls和诊断必须从当前policy取得，漏flag会误测原Score。
采用轮B8/B16/B24/B28净省7.30/15.38/40.72/43.80μs，B4保留原代码。
七档全冻结、真实B1/S1、B16/S5、T30关闭/T36开启边界均通过。
原始结果及采用映射分别见SCORE_SPLIT_ALL_PERFORMANCE.json、
SCORE_SPLIT_CURRENT_PERFORMANCE.json；不把它们称为每个档位最后一次测量。

## 新正式诊断与已结束候选

CURRENT_SCORE_PRESSURE_ANALYSIS.json保存新正式B16/B28全72角色端点及
READY/ACK，均单次精确/状态/保护/协议通过。Score非外层GM等待中位
B16 AIC192.64/AIV181.67，B28 AIC329.82/AIV324.70μs；Attention分别
153.47/158.41、304.19/304.77μs。均含内部等待/流水/探针，不能当正式性能。

KV分配置换候选已完整结束，不采用、不扩档：B28同轮正式1152.44、候选
1156.20μs，慢3.76，2/5窗快；B16关闭代码相同，未重复上卡。
详见PREFIX_KV_BALANCE_DESIGN.md及PERFORMANCE报告，记录提交de041136。

N128宽面板候选也已完整结束，不采用、不扩档、不扫更多tile。两档全冻结
检查通过，实际96KiB QK/8KiB WS和FIX/WS交错机制成立，完整L1仍各0/5快：
- B16：原多task866.58、当前正式795.54、候选823.50μs。
- B28：原多task1300.30、当前正式1175.42、候选1223.84μs。
这些是截至本检查点新正式B16/B28的最新同轮reference数据；不得与其它轮
参考相减计收益。宽面板实际私有N128与公共policy选择条件64明确区分，
只留测试源source_score_wide_pipeline；详见SCORE_WIDE_PIPELINE_DESIGN.md。

**本轮全部CPU/NPU任务已终态，无悬挂任务，无须继续等待或重提。**

## 下一步与验收边界

下一大方向审查在NEXT_WAVE_COMPOSITION_REVIEW.md：重做跨请求Score/
Attention流水时，当前物理plan/MGATHER、local统计、连续scale和N64早FIX
必须真正进入活跃step，不能搬旧交错文件后以policy字段存在声称已生效。
旧861.06对811.10负例仍保留，group计划规则实际未变。尚未创建新的交错
候选、编译或提交设备任务；不要误把只读设计当已执行。

目标仍为128K B4/B8/B16/B24/B28相对多task完整L1降低>=30%，当前0/5；
优先B16/B28，8K不明显退步。每候选至少一重点档较同轮当前正式净省20μs，
另一档无明确退步才扩档。保持正式W8A8、冻结精确/状态/保护/metadata/
padding/图/代际及S1/S5边界；非逐bit数学采用前另行确认。

工具链仍为固定隔离PyPTO/Simpler/PTOAS，没有新增依赖修改或issue/PR。
所有仓库禁止push，须用户再次明确允许。阶段中文本地commit跳过提交检查，
不提交权重/二进制/大编译快照，不做hash；大产物登记LOCAL_ARTIFACTS。
本检查点只保留最新状态，全部历史尝试仍在统一优化日志和各专项DESIGN。
