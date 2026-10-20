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
这些是宽面板验证轮的同轮reference数据；不得与其它轮参考相减计收益。
宽面板实际私有N128与公共policy选择条件64明确区分，
只留测试源source_score_wide_pipeline；详见SCORE_WIDE_PIPELINE_DESIGN.md。

上述两组负例及下述wave候选任务均已终态，没有悬挂任务。

## 下一步与验收边界

跨请求Score/Attention的新候选已在wave_current_math/source_candidate完成，
尚未采用到正式ops。当前物理plan/MGATHER、local统计、连续scale和N64
早FIX已真正进入活跃step；保留旧动态wave、首wave原KV和三阶段发布获取。
旧861.06对811.10负例仍保留，group计划规则未变，不能跨轮相减算收益。

两档完整CPU均通过58动态轴/130根/单MIX，AIC Mat 496KiB、Left/Right
各64KiB、Acc128KiB，AIV Vec184KiB，未放宽预算。实际生成审计确认两个
wave调用都走当前数学/数据路径。首次迁移误判merge-file冲突返回码已
记录并修正，非组件问题。详见wave_current_math/README.md及CODEGEN报告。

设备B16 task_20261020_074844_4911330778、B28
task_20261020_074844_491555648均completed/exit0。完整精确/状态/保护/
metadata/padding/图/波次代际通过。同轮三方五窗完整L1：
- B16多task878.56、当前正式807.88、wave850.52μs，净省-42.64，0/5快。
- B28多task1290.20、当前正式1167.66、wave1369.50μs，净省-201.84，0/5快。
这是wave轮的reference测量，不与其它轮次相减计优化。
两档未达门槛，结束该组合、不采用、不扩档、不扫wave宽度或局部参数。

实际设备C++匹配CPU审计版。离线读取已有metadata/IR发现：每叶一次的
60KiB Left恢复变为每step一次，额外操作字节模型B16/B28为25.3125/
45.9375MiB；B28整段最大41步、各wave最大步数之和56，提示切开全局工作
范围会暴露局部不均衡。不是实测带宽/关键路径，也不是wave全核集合。
其它片上竞争、启动排空、Attention两L1槽成本未定量分离，不编造唯一
回退原因。详见wave_current_math/README.md、WAVE_WORK_ANALYSIS.json。

## 新一轮终态：当前硬件计数已完成，主投影共享Left无收益

cc220e36已详细中文本地归档wave负例，未push。随后补当前正式B16/B28的
同进程原多task/单SPMD PipeUtilization各三步轮转，两个任务
task_20261020_081507_372121430574、task_20261020_081507_372128211895
均completed/exit0，完整单次精确、两图及采样后数值/状态/保护/代际通过。
报告current_pipe_pmu/PIPE_COUNTERS.json，**不是新的L1性能或验收数据**。
当前AIC MTE1/MTE2中位B16 254.981/376.591、B28 451.874/527.553μs；
Scalar已低于多task，不能继续用旧版额外Scalar解释当前差距。占用可重叠，
Scalar含等待、miss率不是stall时间，不相加估计完整收益。

主Compressor实际源码K512被后端拆为K256，value/gate重复恢复同一Left。
main_shared_left首版改源码K256，完整CPU通过并共享输入，但Right退成单槽，
没有上卡。v2显式保持原两K512 Mat预取和双Right，Left每K1024由8次减4次，
每输出的K256顺序和动态compact行距保留。两完整CPU均58轴/130根/单MIX；
实际Mat475136B、Left/Right各65536B、Acc131072B、AIV Vec163872B。
只改该候选的主投影文件，正式37源和所有依赖仍不变，无新增外部适配。

设备B16 task_20261020_082812_19733772806、B28
task_20261020_082812_19742111343均completed/exit0；结果目录
main_shared_left/device_b16_v2、device_b28_v2。完整冻结精确/状态/保护/
metadata/padding/图/代际通过，输出max_abs=0，实际设备CPP匹配CPU审计。
同轮三方五窗完整L1：B16 883.58/797.80/799.40μs（多task/当前/候选），
净省-1.60、2/5快；B28 1283.02/1160.72/1176.26μs，净省-15.54、0/5快。
这更新了正式B16/B28的最新reference测量，不与其它轮次相减算收益。

未达20μs门槛，结束、不采用、不扩档、不扫K/缓冲或追加局部探针。真正
减少Left操作不等于完整性能收益，也不能据全链数据定量判断局部快慢。
新PMU诊断与主投影全部任务均终态，无待收任务；正式源仍13259f92。
后续回到Score/Attention等较大工作段，用新硬件证据及既有全核流水审查
实际计算/搬运；当前没有新的替代候选或设备任务，不原样重跑已有负例。

目标仍为128K B4/B8/B16/B24/B28相对多task完整L1降低>=30%，当前0/5；
优先B16/B28，8K不明显退步。每候选至少一重点档较同轮当前正式净省20μs，
另一档无明确退步才扩档。保持正式W8A8、冻结精确/状态/保护/metadata/
padding/图/代际及S1/S5边界；非逐bit数学采用前另行确认。

工具链仍为固定隔离PyPTO/Simpler/PTOAS，没有新增依赖修改或issue/PR。
所有仓库禁止push，须用户再次明确允许。阶段中文本地commit跳过提交检查，
不提交权重/二进制/大编译快照，不做hash；大产物登记LOCAL_ARTIFACTS。
本检查点只保留最新状态，全部历史尝试仍在统一优化日志和各专项DESIGN。
