# 2026-10-26 工具链对齐和阶段观测检查单

控制实现11b380e3f；实验报告已推送123c12c63。nalinaly公开分支c346573a。

- [x] 七个候选性能均失败，封存原始证据，不合入，不扩测精度。
- [x] 确认nalinaly最新PyPTO d44925c1、PTOAS914863ed公开可取；官方0.69并非相同PTOAS提交。
- [x] 复用自有LLVM/Simpler产物，仅在新目录构建公开精确提交，不修改SDK源代码，不覆盖旧环境。
- [x] 查看目标源码AGENTS/CLAUDE及构建约束，固定来源、提交、模块路径、二进制SHA。
- [ ] 编译控制实现与nalinaly最新完整SPMD（无设备），不得修改生成C++。
- [ ] 经task-submit最小smoke；然后单层8K/B32受控性能，区分计时与DFX。
- [ ] 性能稳定收益才做Native逐bit八边界/ABA、反向重复，再128K/B16、mixedT126。
- [ ] 明确attention compute/move/wait证据边界，不能将DFX跨度直接称为纯计算。
- [ ] 独立审查、记录结果，正常commit push。每项设备外780秒/内750秒，队列期间继续监测。

## 新进展

- PMU观测任务task_20261026_175944_38902877599：1759提交，1804采集完成，exit0。旧SDK三方，不作为新SDK性能结果；需检查原始CSV是否完整。
- d449 PyPTO新目录构建成功，旧验收SDK未改。PTOAS默认Python模式缺少LLVM Python来源，改用914公开提交已有的PTO_ENABLE_PYTHON_BINDING=OFF原生CLI模式，不改源码。
- 当前正式后缀没有GM signal_wait，不能给FFTS sync_wait直接套backoff参数。新工具链先用于最新nalinaly完整SPMD受控比较；旧owner2的GMsignal后缀才可单因素测试退避。
- 新d449源码导入/二进制/固定Simpler ABI经严格finder检查；core/test_error和runtime/test_kernel_abi合计107项CPU测试通过，16.54秒，未初始化NPU。fork多线程警告保留原始日志。
- SWA旧SDK组合候选已提交task_20261026_184226_283356519449；已有源/实际生成物独立审查资格，尚待性能门槛。
- SWA v1 task184226已exit0，0.481214%/12wins/L1改善，但未过0.5%门槛，accuracy_executed=false，不合入。没有自有NPU占卡。
- SWA v2两臂64/load/noNPU已通过，待UB别名/Tile stride生成审查；仍未提交设备。

## 2026-10-27 续测

- [x] 确认task194324 completed exit0；SWA v2无收益，不做精度扩测或合入。
- [x] GitHub ls-remote仍c346573ab，不把失败的整库fetch写成成功。
- [x] 最新nalinaly构造/内部scratch/debug TopK入口经独立源码审查。
- [x] 修正空allocations导致的guard资格误报；最终manifest a63f21b39153ac30656da21acafe10a65539246524226b8238f1c1c6d7171ff9。
- [ ] 最终两臂CPU64/57资格，审查实际编排容量，再提交8K三方同SDK单卡测试。
- [ ] 取得普通计时、Native A/B8和ABA；计时与DFX/PMU分开，不比较不同范围的counter。

- [x] 新SDK8K/B32三方task081907完成，ours A/B8逐bit，所有ABA通过。均值Native1233.611/Nal1081.373/ours1592.640μs。
- [ ] 纯DSL累计截断诊断：Nal完整、止于TopK+plan、止于heads；7个前段边界A/B必须与完整Nal逐bit。先4根CPU资格及独立截断依赖审查，再设备20×100；差值只作为累计诊断参考。

## 2026-10-27 09:45 分段定位及单因素准备

- [x] 累计普通计时090855和独立L1任务092122均exit0；截断前7边界A/B与完整nalinaly一致，我方A/B8、ABA、guard通过。
- [x] GitHub最新1685da69相对c346共4提交，下载changed源码+compare证据；主要为三张Native NZ权重，公开NPU只比较旧版，不算Native精度证明。
- [ ] 单独dep任务093646完成后，检查CPU/JIT二进制、L1实际func_id以及所有task-token覆盖，形成我方分段。
- [x] WKV-NZ独立候选源码review：保留K256 order、matmul调用和BF16边界，只有物理layout/Native storage变化。
- [x] 无设备两根CPU64资格、实际TLOAD/MMA/资源审核，保留原环境和控制实现。
- [ ] 若前半段仍为重点，单卡8K/B32性能筛选：12×100、交替顺序，>=0.5%/10wins/L1更低后才验Native A/B8与ABA；反向重复，单任务780/750秒。
- [ ] 无稳定收益不合入；稳定收益先128K/B16和混合T126，再整模型8K/B40、128K/B24。

- [x] HC-ready / projection-tail独立调度候选：源码审查、CPU两臂64参、FFTS/Norm只读依赖/作用域容量实际生成审核；不与NZ组合。

- [ ] 两候选实际NPU性能筛选尚未执行；唯一dep任务093646排队，16卡CI占满。保留所有原精度门槛。

## 新鲜依赖与热点

- [x] dep093646于10:11:34启动/10:14:20退出0，166秒；新SDK实际63个leaf SHA/名字与L1构建相同、所有task token覆盖、两窗映射一致。
- [x] 同边界entry→TopK+plan我方722.41us；plan→heads441.87us；heads→end337.85us。Nal独立累计cut429.35/276.83/222.28us仅诊断参考。前段优先，不能当可直接节省的纯计算。
- [x] WKV NZ首轮1614.663→1584.650us，+1.8587%/12wins/L1降低，随后Native A/B8逐bit与ABA通过。
- [ ] WKV NZ独立reverse-repeat；通过后128K/B16及mixedT126，再整模型。
- [ ] 删除QA无用清零单因素：QR_OK1、ATOMIC_ADD0、STORE_ATOMIC.None硬约束，原MMA/K/round不变。先两臂CPU/尾pad完整写和producer依赖审核，再性能筛选，赢后T95/126/168和短T、poison/ABA。

## 2026-10-27 11:33 当前状态（上方为逐时历史）

- [x] 新SDK三方、累计cuts、L1、单独deps和全部映射已完成并独立审核。前段优先结论成立。
- [x] WKV-NZ反向重复0.4534%低于预设0.5%门槛，未采用；HC-ready回退2.62%，未采用。
- [x] QA no-seed v1额外95路由在CPU审查时否决，v2严格从冻结来源派生。
- [x] QA v2两次独立8K/B32性能降低1.2720%/0.6260%，各12wins与Native A/B8+ABA通过。
- [x] 128K/B16单层降低1.6751%，Native A/B8+ABA通过。
- [x] QA真实权重poison单元9形状27子场景，整个ceil16输出逐bit、尾保护、逐phase只读与ABA通过。
- [x] guarded正式95/126/168/generic CPU两根64参通过；65共同leaf记录SHA相同，4处seed位于if0，原始GM alias和Normalize依赖保留。
- [x] 完整T95与混合T126各Native A/B8+ABA通过。
- [x] 原验收SDK下独立完整CSA候选CPU+selector1～256+8K bank/Native gold无设备预检通过。
- [ ] 真实整模型8K/B40 warm192+192 token，任务112507正等待16卡；其通过后才128K/B24。
- [x] nalinaly GitHub最新1685da69，11:21再次确认；相关公开变更快照已同步。
- [x] 独立Indexer Q_B NativeNZ候选两根CPU64通过，仅idx_qr_proj_matmul变化，其余62leaf记录SHA一致；设备前审查中，未组合QA/WKV/HC。
- [ ] 最终正式迁移、独立审查、CPU递增检查、文档/证据commit push；未达门槛不得提前采用。

## 当前结果补充：正式QA候选未采用

- [x] IndexerNZ仅0.3835%低于门槛，无精度扩测、不合入。
- [x] guarded正式QA首轮0.7624%/Native A/B8通过；独立reverse仅0.1624%，未过性能门槛。硬删除v2的通过不能替代正式候选。
- [x] 已取消未获卡whole8K/B40任务112507，没有新增整模型token资格，没有提交128K。
- [ ] Query岛只做CPU编译/生成审查；先修正缓存声明作用域，再检查GM view、FFTS、实际scope容量。未提交NPU。
