# 正式动态CSA的Score服务区间诊断

本目录准备一份128K/B16/S6、正式668的单基线诊断，用于区分Score供数链内部区间。不是新优化候选、性能验收或七档扫描；不得用插桩时间替代完整L1。

首版设计保存在initial_design；实施时修正其“共用单个临时时钟”表述：body、step、数学分段与嵌套local wait必须分别保存起始值。实际hook与runner由两个独立agent并行实现，第三agent只读审阅。当前尚未冻结、未提交设备任务。

只采每step固定第九个panel的细分，并统计整个stage/step与ACK；不逐panel写GM，不将固定panel样本乘16外推。仅一次写各AIC自己的phase34预留槽32..63，沿用原release发布；活动Vector函数不改。没有新增硬件fence。当前可用工具未提供可读A3反汇编，不预记最终时钟指令顺序或无spill。

## 已提交，待结果

正式33源逐字保持668。两个agent分别实现测试C++ hook和runner，第三agent独立审阅。复用已有formal668原始C++做离线检查，没有新启动PyPTO CPU编译。固定128K/B16/S6、long G6/N64/1024，仅本次诊断限制，不改变正式动态支持范围。

hook记录整个Score body/step与ACK，并只在固定第九个QK面板附近采样页/Key、QK+FIX、前一面板WS和原local wait。20个独立起始变量避免嵌套覆盖，独立review逐项核对同一C++大括号作用域。原同步语句序列、Vector活动函数不变，剥离标记逐字恢复原C++。每核只在phase34自己的32..63列一次写入，768唯一word，schema/epoch与计数严格检查；17个反例、0/1/2/3/7/33动态step离线模型通过。仅保留原phase-clock16..19，不混入Attention tick或FFTS20/21。

原phase release承担可见性；本hook没有新增硬件fence，组合的旧phase-clock仍有其既有DCCI/DSB。Scalar区间包括发射、原依赖和插桩扰动，邻接区间含计时开销，不能称纯异步引擎耗时、相加为完整延迟或将j8样本乘16。当前工具无法读取最终A3机器码，未核实CCEC时钟指令次序和spill。

runner复用正式权重、原动态PTO精确oracle、状态/保护区及图代际检查，三轮A→B→A只改变hidden正负，不声称本次切换了history/page metadata。无DFX/L1性能采样窗，eligible_for_performance=false。主提交包装初次字段名不匹配在task-submit前报错，修正为实际ready键后才执行唯一任务，未重复占卡。

唯一任务 `task_20261018_073047_323892912778`，见registry.json。


## 2026-10-18：Score服务诊断完整通过，保留发射与依赖解释边界

唯一任务task_20261018_073047_323892912778完成exit0。eager、graph预热及三轮A/B/A共5次完整检查全部PASS，原PTO精确输出/Top-K/score/cache/state、保护区及72核代际正确，最终generation=5；正式668的33源未变。本次A/B仅hidden正负，未切换history/page，不冒充额外动态metadata验收。

下表是每核本次累计量再取24核中位数（us），不是单panel或完整L1性能。每轮每核均4个叶、22个step、22次ACK、22个固定j8样本和220处local wait。

| 轮次 | body | step | ACK | j8页/Key | j8 QK+FIX | 对应WS | 嵌套local wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 232.28 | 214.24 | 30.91 | 3.83 | 0.20 | 7.44 | 6.42 |
| 1 | 239.39 | 219.23 | 32.27 | 3.84 | 0.20 | 7.43 | 6.28 |
| 2 | 233.21 | 215.76 | 29.71 | 3.87 | 0.18 | 7.50 | 6.33 |

生成事件映射确认，采样实际为预取Key9、QK8、WS7；10处local wait分为页3处、QK2处、WS5处。最后WS的M→FIX事件位于先发QK8、再发WS7之后，保障此前M队列前缀完成，不能把约7.5us当WS独占计算。6.3us是10站合计，不能归给某个wait，也不与WS直接相减出纯compute。QK+FIX接近0仅说明这段Scalar发射区间短，不代表QK硬件耗时为0。不能将j8乘16、相加嵌套区间或将探针端点当完整性能收益。

同次阶段端点与独立统计见DEVICE_DIAGNOSTIC_CHECKS.json和analysis/；实际事件/生产消费来源见wait_mapping/。三个核组body与ACK呈相同分组趋势，而固定j8页/WS无相同单调趋势，最多支持进一步核对消费相位/背压，不能据此归因硬件慢核。历史全局G3/N128已测并后被G6替代，当前观察不足以重试；另一个G6内部双M192/N128负例独立列出，不混淆两者。

完整探针、runner、源快照、初稿修正、离线/独立审阅、真机结果/日志均在score_service_clock_evidence。无额外L1窗口、无七档重测、无新增正式优化。当前仅继续只读评估不同请求波次的Score/Attention块交错，与旧whole-stage两wave方案排重，核对有向事件顺序和同时live资源；未开始实现或编译。
