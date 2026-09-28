# 小batch长档：S6 Key复用与leaf分配候选

基线为d8627207，已含2048排序与UB中间根；两侧整包私有副本，完成三档验证后合入性能版。
借鉴最新AscendC的Key跨M子块复用，使用现有已验证的S6 M384/N64内核，扩大到长档B4/B8。
不是重试历史退化的4+2算子拼接；这里减少每个请求不同query组对同一Key的重复读取。

- 忽略padding等边界，B4由三个双query组变为一个S6组，Key读取量约为原来的1/3；B8由两个三query组变为一个S6组，约为1/2。
- 为防止query组变少导致24个worker闲置，B4 leaf数向6的倍数取整，B8向3的倍数取整；超出现有pair容量则保留原划分。
- Score和最终merge共用相同leaf计划，并同步query组选择，避免读错half-leaf根。
- 原B16及更大长档、全部短档保持原策略；B<4仍用双query。精度版、Native cache、调度开关未改。
- 先128K/B4与8K/B24两档，若核内获益且状态/图通过，再补受影响长B8及已保留root的B24。

两侧生产/测试根的_get_dep_graph()解析、完整CPU编译/load、ruff及shell语法检查通过。
等待其他会话16卡任务结束后完成编译；首轮task_20260929_021428_30679553859完成exit=0。
128K/B4 Score AIC134.025→71.036μs、AIV141.809→81.382μs，CSA700.056→685.302μs（−2.108%）。
8K/B24 CSA946.372→977.518μs（+3.291%），长短8:2为−1.028%。
短档算术/分工未改，仍如实记录回退，不把它假定为测量噪声。
两档八类跨版本状态零差异、A→B→A、保护区均通过；P95未出现远离主体分布的异常。
受影响B8补测task_20260929_022538_341885330935完成exit=0：
Score AIC168.931→137.435μs、AIV173.838→144.219μs，CSA819.977→794.187μs（−3.145%），
P95 833.060→805.280μs；八类状态、A→B→A及保护区通过。
两组长档内部等权、再长短8:2，CSA−1.443%、Score AIC−26.516%/AIV−23.508%；
merge核时+20.474%单列。按核内收益及8:2规则保留，生产两根解析通过。
正式文件与已测私有候选一致；精度版迁移、新七档和整网token/DSpark仍待阶段验收。

## Native与pypto-lib的实现依据

最新本地ops-transformer 28f40354的arch22 `QLIV2Matmul::ProcessQk`（第201行起），
只在`s1gL0LoopId==0`时执行KeyNd2NzForPA/KeyNd2Nz；同一Key在后续M子块中复用，
直到最后一个M子块才释放Key L1缓冲。这里吸收的是跨query/M复用Key的策略，
没有直接复制其M分块或动态metadata调度。

当前本地pypto-lib 2164563的`deepseek_v4_flash_dspark/decode_indexer.py`仍按单query×leaf分工，
使用独立key/scale分页缓冲；普通路径在AIV做FP32 head规约，大batch长档才使用FP16双缓冲Score传递。
本PTO性能版沿用此前已验证的Native式第二次Cube head规约，S6一次覆盖同请求六个query，
直接读取Native交错物理cache。不能把旧725μs泳道或单query源码当成本候选的同配置性能对照。
本轮只改变长B4/B8的query组及leaf计划，没有进一步降低算术精度或改cache布局。

## 读数限制

B4 DFX中Score AIV包络149.790→91.640μs，Sparse首次接收435.81→382.75μs，
末HC_post结束640.23→600.28μs；完整CSA另用无profiler图事件计时。
这说明核内收益沿依赖链传到下游，但两种采集不能直接相减来归因“剩余调度开销”。
本轮Native列仅作为手工图环境控制，不等同新增的真实编译半层基线。
完整状态、核时及四窗口泳道路径见[RESULTS.md](RESULTS.md)与[evidence.json](evidence.json)。
