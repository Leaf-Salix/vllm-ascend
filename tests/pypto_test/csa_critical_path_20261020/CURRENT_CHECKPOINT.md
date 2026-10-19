# 当前正式检查点：Score区间已采用，最终30%目标未完成

正式目录仍为vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd。完整37文件
快照为source_after_score_split，与正式文件逐字相同。旧source_reference
是本轮开始时的E49，只供历史三方对照，不能再当作“当前正式”做新候选门槛。

本轮改变四文件，长历史G6且已有Host T容量>=32启用双48KiB QK早FIX和
单次WS写回。T/B/有效长度/页表仍动态，仍一个完整MIX；小T和其它资源域
沿原Score。所有验证源source_score_split_interval与正式算法AST一致，
正式仅更新解释性docstring。新增编译常量为SPLIT_SCORE_ROWS。

新候选必须相对source_after_score_split同轮测量，不能再相对E49累计
旧收益。CPU controls须从新policy.constexprs()取得；继续传旧JSON会因
新flag默认False误测旧路径。诊断脚本同样必须加载新源/新policy；旧独立
Score probe若不传新增flag，会继续执行原helper，不能作为新正式诊断。

七档原始数据在SCORE_SPLIT_ALL_PERFORMANCE.json；按当前实际代码选择
映射的数据在SCORE_SPLIT_CURRENT_PERFORMANCE.json。开启长B8/B16/B24/B28
分别相对本轮旧正式净省7.30/15.38/40.72/43.80μs。B4保留原代码；8K
生成代码未变，原测量波动完整保留。正式采用依据SCORE_SPLIT_ADOPTION.json。

七档完整冻结检查、真实B1/S1和B16/S5、T30/B5关闭图回归、T36/B6开启
三方完整检查通过。S1/S5参照原动态PTO，不扩大为新增Native/整网token
精度结论。T36性能仅基本持平。所有本轮NPU/CPU任务均已终态。

目标仍是128K五档相对原多task完整L1降低至少30%，当前0/5，不得标完成；
优先B16/B28，8K兼顾。新候选仍以重点档至少净省20μs且另一档无明确
退步才扩档。后续应从更新后的关键路径继续，不再重试已结束的KV驻留、
Q_B驻留和叶旋转小改动。旧阶段时戳/背压属于E49，不能假定仍代表新Score。

工具链仍为固定隔离PyPTO/Simpler/PTOAS；未改依赖，无新issue/PR，
所有仓库禁止push，只有用户再次明确允许才能push。阶段中文本地提交，
不做提交检查，不提交权重、二进制、大编译快照，不做hash。
