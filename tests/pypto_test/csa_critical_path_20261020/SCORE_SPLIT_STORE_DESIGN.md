# G6 Score双QK早FIX与每面板一次WS写回

## 范围与候选依据

上一轮独立64KiB Attention供数方案完成完整验证但无收益，已结束；新证据
改变后续行动，属于有进展。正式37文件重新核对仍为E49，没有待处理旧任务。
本候选直接基于source_reference，不叠加任何未采用的KV、Q_B或叶分配改动。

已有B28背压观察：AIV等READY中位约151.17μs，AIC等ACK约38.14μs，
支持继续检查Score生产侧；这些含插桩的Scalar区间不是可直接省掉的时间。
此前提前FIX/独立WS Acc完整净省7.14/14.84μs，没有达到20μs扩档门槛。

旧M192/N64双Acc已完整验证且变慢，详见NEXT_SCORE_REVIEW.md。旧版每半组
分别写三行，每1024候选32次WS STORE，而且QK1先于FIX0。这次的区别是
FIX0立即发出后才发QK1，以及WS分K累计后只写一次原六行；不能把“双Acc
可容纳”当作新发现或收益保证。额外MMA/FIX指令可能抵消这两个变化的收益。

## 实现和动态边界

source_score_split_store只修改indexer_score_stages.py。为复现保留原完整
Cube helper，原AIV函数逐字不变；新增wrapper只按既有BALANCED/G6/N64/
1024资源组合选择新helper，其余组合走原helper。没有batch/history测试点
白名单，T/B/实际历史、页表、尾query安全填充、叶计划和FFTS全保留。
候选若有完整收益，正式迁入前再收敛重复公共逻辑；目前不改ops默认源码。

- QK按行分两个M192，K128不拆，两份INT32 Acc各48KiB长期持有明确句柄。
  每次init_cond=True重新初始化，第一份FIX紧跟QK0，QK1用另一份Acc。
- 原FP16系数保留16行和六query的位置，按K0..191、192..383截成两份。
  WS两次MMA按原K段顺序累计到同一4KiB FP32 Acc，再做一次原六行STORE。
  未改变FP16量化/ReLU边界，但K分段的真实舍入仍必须通过完整精确检查。
- query Left共48KiB、系数Left共12KiB；两个WS Right各24KiB，与两份
  8KiB Key同时存在恰64KiB。原预取、分页guard和1024分数交接不改。

## 编译与实际生成证据

CPU首轮cpu_score_split_store_b28_v1在ConvertToSSA报分支Tile越出定义域，
没有设备执行。两个current Tile改为分支前建立，沿原helper的写法绑定
输出句柄，保留失败源source_score_split_store_scope_v1与FAILURE_SUMMARY。
PyPTO控制流文档明确分支新值须从两臂yield；这是本次写法修正，不报组件bug。

B28 v2、B16 v1均完整编译通过58动态轴/单MIX。实际地址：QK Acc分别
[0,49152)、[49152,98304)，WS为[98304,102400)；Left总61440B。
两份WS Right实际互不重叠，共49152B，整体Right不超过65536B。

audit_score_split_store.py核对：每1024候选32次QK MMA、32次QK FIX、
16次WS初始MMA、16次WS续加MMA、16次STORE。16个面板的FIX0均在QK1
之前，释放事件与QK1等待事件不同；后15面板的当前FIX1也不阻塞上一面板
WS。最后面板可能没有再复用该Acc的FIX→M释放事件，最初审计误把它判成
失败，已修正为允许末面板无该释放；算子未因此修改。真正的旧槽复用依赖
及两次WS之间的同Acc依赖均保留，不删除PIPE_M屏障。

这些是提交顺序/地址证据，不等于硬件重叠时长或性能收益。

## 真机任务与停止条件

- B28：task_20261020_050736_244883023864，score_split_store_h131072_b28_v1。
- B16：task_20261020_050812_247206811437，score_split_store_h131072_b16_v1。

任务使用同一冻结工具链、正式权重、原三方五窗完整L1，包含A/B/A、精度、
Top-K/score、cache/state、保护、metadata/padding图与真实代际。任务期间
源与工具链冻结，尚无设备终态或性能结论。

精确检查失败则停止，不按数值等价自行放宽；只有至少一档相对同轮正式
完整L1净省20μs且另一档无明确退步，才扩其它目标档及S1/S5/区间边界。
若未满足门槛，结束该组合，不扫描更多M/N/K。最终五档30%目标保持，
未使用局部阶段计时或跨轮最优值替代完整收益，所有仓库禁止push。

## 三档终态收益与扩档门槛

B16/B28均completed/exit0：完整精确、状态、保护、metadata/padding同图、
图重放与代际通过。实际C++与CPU审阅版相同。原三方五窗完整L1中位：

|128K档位|多task μs|正式SPMD μs|候选SPMD μs|净省 μs|更快窗数|
|---|---:|---:|---:|---:|---:|
|B16|873.04|817.68|802.30|15.38|4/5|
|B28|1264.44|1209.84|1166.04|43.80|5/5|
|B24|1096.70|1076.02|1035.30|40.72|5/5|

B28达到20μs门槛且B16无明确退步，才扩至B24/B4/B8及两档8K。
B24任务task_20261020_051859_287537028516随后完整终态通过，收益见表。
B24所有constexpr与已审计B28相同，复用无样例shape的CPU编译，设备仍
单独完整验证。三档原始窗口全部保留，最终五档相对多task降低30%尚未达成。

系数构造也已核对：从全零16×384矩阵只写query行的对应64头对角块，
每query的64个非零K全部落在0..191或192..383之一，未把同一query的
非零点积拆到两段。整数映射证据见SCORE_SPLIT_COEFFICIENT_SUPPORT.json；
这解释分段结构，不替代已经完成的真实精确验证或保证任意非法系数输入。

## v2：未选资源分支合法化

B4/B8开启组合的完整CPU和16面板地址/事件审计通过。两档8K的第一轮
CPU因未选G6 helper仍收到G2参数而拒绝192行extract超过128行源Tile，
未上卡。另建source_score_split_store_v2，只在G6专用调用处显式传
G6/N64/1024/BALANCED=True四个本来就必须成立的资源常量；原helper、
新helper与AIV的算法AST均不变，没有固定实际T/B/历史/位置值。

修订后G2两种资源策略完整CPU通过，G6 B28生成完整C++与v1去注释空白
后逐字一致，见SCORE_SPLIT_LEGAL_ENABLED_COMPARISON.json，故不重测
已完成的G6整轮。B4/B8和8K两个首次完整任务改用v2，均含原冻结检查
及五窗。所有七档task及实际source版本在SCORE_SPLIT_TASKS.json记录。
正式源码仍未替换；扩档和动态边界检查尚未结束。

## 七档初轮收齐：保留全部原始结果，转为区间选择

七档均完整精确/状态/保护/metadata/padding图/代际通过，raw数据全部在
SCORE_SPLIT_ALL_PERFORMANCE.json。补充四档正式→候选完整L1（μs）：
128K/B4 486.54→492.10，B8 635.06→627.76；8K/B16 652.26→661.00，
8K/B32 930.12→935.06。B4仅2/5窗快，B8为5/5。8K两档实际完整C++
与同轮正式版本去注释空白后逐字相同，SCORE_SPLIT_G2_EXECUTION_EQUIVALENCE
记录这点；原始波动保留，不归为代码收益，也不因这一波动修改8K数学。

最终区间候选source_score_split_interval新增SPLIT_SCORE_ROWS，按既有Host
容量描述符：长历史、G6且T>=32才启用。32沿QA小M16前段资源区间的
既有边界，不枚举B4/B8，不证明QA就是小档回退根因，也不声称阈值理论
最优。真实有效组/长度/页表仍在device。关闭路径保留原Score，新增的是
编译期选择，不增加Host回读、Tensor根或外部task。

四类完整CPU通过：长B28/B8开启代码与已测候选相同，长B4关闭代码与
原正式相同，短B16关闭代码与v2原G2相同；均比较完整C++、无hash。
准确的Host T31/T32选择也已核对；数学shape没有固定为这两个T。
证据：SCORE_SPLIT_INTERVAL_CODEGEN_COMPARISONS.json、
SCORE_SPLIT_EXACT_HOST_BOUNDARY.json。未重跑已证明相同的整组性能。

真实S1/S5沿旧query_shape_pair脚本，使用同一正式权重与Native Tensor
绑定，不补成S6：B1/S1 task_20261020_053619_356347016565、B16/S5
task_20261020_053619_356351127622均completed/exit0，eager与A→B→A
图重放的完整输出/Top-K/score/allocation、保护、地址和代际通过。参照
是原动态SPMD，不扩写成新增的Native或整网token精度结论；这两项不测性能。

区间两侧的真实S6边界任务仍待终态：T30/B5关闭路径做query形状图回归，
task_20261020_055437_4699711584；T36/B6开启路径做原三方五窗完整验证，
task_20261020_055437_4712624285。B6所有constexpr与已审计B8相同，
复用无样例CPU编译。任务与folder在SCORE_SPLIT_BOUNDARY_TASKS.json。
正式37文件暂仍E49，边界完成前不预记正式采用；所有仓库未push。

## 区间边界通过，正式采用

T30/B5 task_20261020_055437_4699711584 completed/exit0，eager及A/B/A图
的完整输出/Top-K/score/allocation、保护、地址、代际通过；未测性能。
T36/B6 task_20261020_055437_4712624285 completed/exit0，完整原三方
精确/状态/保护/metadata/padding/图/代际通过。多task594.80、正式590.72、
候选589.62μs，净省1.10μs，3/5窗较快，视为边界基本持平，不宣传明显收益。

随后正式迁入四文件：indexer_score_stages.py、cube_schedule.py、entry.py、
policy.py。写入前确认正式37文件仍等于E49；签名默认关闭先落地，最后
更新policy启用。仅更新解释性docstring，37文件算法AST与已验证区间候选
一致，记录SCORE_SPLIT_ADOPTION.json；未修改任何依赖或适配层。

当前选择的数据见SCORE_SPLIT_CURRENT_PERFORMANCE.json：开启档复用原
候选完整配对数据；B4保留旧实现，采用该轮reference整组数据并明确非新测；
8K原执行代码相同，仍保留candidate原测量，不换成较低的reference数字。
所有原始失败/回退与105个七档窗口仍在ALL报告和各目录中。七档原门槛、
真实S1/S5及区间S6边界通过不等于最终性能目标：128K五档30%仍0/5。
本轮全部任务已终态，正式采用范围为长G6且T>=32，其余旧路径；未push。
