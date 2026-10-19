# 当前动态前段与PTOAS范围同步下重验Score/Attention交错

参考正式289cc072；不使用冻结静态shape入口。旧score_attention_interleave
候选曾完成128K/B16精确/图验证，但当时889.96μs未获得收益，保留该负例。
目前已采用前段协作、行依赖和PTOAS L1子区间同步改进；本轮重新组合动态
逐step Score与前wave Attention，是否足以抵消暂停状态/Left重装/两Mat槽
的成本，必须由同轮完整L1验证，不能用旧889.96与当前约800作直接差值。

先对原候选相对旧参考的变更做三方迁移预览，再在正式动态文件中接入和
解决新旧调度冲突，冻结44文件后精确恢复37份正式源码。候选的非交错路径
完整沿用当前计算和更细行依赖；既有G6/N64/1024资源组合与device多wave
条件才启用，query/group/wave边界均来自当轮metadata，没有batch白名单。
入口维度、权重与状态保持动态；signal扩展按Host已知T容量预留，实际波次
和写者仍在device确定。首尾drain、原数学和A3业务发布/获取保留。

初始交错数学helper沿旧已验证的逻辑plan、逐页scale、GM mi/li实现；
当前物理plan/MGATHER、连续scale和本地mi/li暂只作用于非交错路径。
这是明确的组合取舍，不宣称policy开关存在就等于交错路径已经使用。
前段协作保留；若收益不足，不会以省略门槛或把旧时延当基准强行采用。

CPU在原独立pypto-csa-memory-reuse-20261018与PTOAS v3下验证；
不依赖双行DMA的gm_view原型，不修改共享安装。当前仅CPU准备。

CPU v1在解析时发现迁移的旧分支使用core_group/subblock命名Scalar，而
当前入口已经改成直接核心表达式。只在交错分支恢复同值的命名别名，
保留v1源码与失败；这是迁移接线遗漏，不报告成组件bug。v2待编译。

CPU v2完整通过58动态轴/单MIX。SCHEDULE_MIGRATION_AUDIT核对Cube/Vector
前段、非交错分支、输出后段的AST与当前正式实现完全相同；交错分支的
逻辑plan/GM统计等成本明确保留。唯一128K/B16设备任务见DEVICE_SUBMISSIONS，
包含当前harness的真实wave写者/代际/保护和metadata/padding，不预记PASS。

首个迁移任务在候选首次调用返回507018；没有数值或性能结论，原任务
未开failure-snapshot，故缺少阶段快照。准备不改算子、仅失败后ACL读取
控制区的诊断，另用历史已验证40份源（与旧真机快照逐字相同）+当前
工具链做对照。诊断首版wrapper错误地将参考源码置None，初始化元信息
读取glob失败，两任务未进入kernel；保留失败。改为保留参考源码字段，
仅在显式diagnostic-only下允许1次调用/0个性能窗口，不修改精度检查。

诊断v2因缺run_device.sh在初始化失败；v3补齐18个依赖文件后两侧均执行到kernel。
当前迁移v3任务188794822312在候选同步失败，ACL读回全部控制区成功：
phase13(KV norm)及14(原cache发布)在72核均为0，首wave ROOT/TOPK/PLAN已经发布。
旧源码对照188800824528在同NEWP/v3工具链下完成kernel，phase13/14和后续wave均发布；
但新harness访问旧policy不存在的scale_page_coalescing属性，在数值比较前失败。
因此旧对照只证明本次kernel能够执行结束，不记精度、完整功能或性能PASS。

根因是本仓迁移遗漏，不是已有证据指向的PyPTO/PTOAS/Simpler错误：
协同前段把original_kv_vector_stage推迟到Score之后，交错分支却只复制了
Score/Top-K/Attention，遗漏原KV消费者。非交错前后段AST相同不足以保证
所有跨分支生产者仍存在。source_candidate_v2在首wave完成Score并发布root后，
按原行owner执行一次原KV norm/RoPE/cache写回，之后才允许下wave消费Attention。
不新增全核集合、不改缓存协议、不省略数学、不使用虚假信号解锁。
修改先作用于ops动态源码，再冻结44文件并恢复当前已采用的37文件。
诊断摘要与原始快照路径见DEADLOCK_DIAGNOSIS.json；完整真机验证待执行。
CPU v3启动引用不存在的cpu_controls_on.json，尚未编译；v4使用实际controls_on.json。

修复版CPU v4通过58动态轴/单MIX；完整真机task_20261019_104350_24961196376
已completed/exit0。三次A→B→A输出、Top-K、score、全部cache/state和保护区
精确一致；metadata/padding同址图、基础图及真实wave/代际检查全部通过。
同轮同工具链五窗中位数：原多task872.48、正式MGATHER参考811.10、
交错候选861.06 μs。相对参考回退6.1595%，相对多task仅降低1.3089%。
因此只归档、不采用、不扩大七档。该结果不能证明所有交错方式无效；
此版明确还使用旧逻辑plan/GM统计，且多了wave控制、Mat→Left重装等成本，
没有独立分解出各项的因果占比。后续更优数据路径应由同轮整层验证决定。
