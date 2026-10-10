# TND 精度版：对齐 Native 规约与双槽读取顺序（2026-10-10）

## 修复背景

工作分支：`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
精度源码基线：`99854b2b591975356d2fd090089348227ddc5600`；历史定位记录提交：`49ad5e1ba`。
只修改独立 `deepseek_v4_flash_csa_tnd_precision` 的 sparse attention 规约、双槽读取顺序及对应契约测试。
性能包23文件仍以 `98eadc516a785e33d51dc0714bf23dd34390ece8` 为备份基线。

此前 history259、B4/T11、请求长度 `[6,3,1,1]`、seed62、真实C4第4层权重的实际 Q、
完整 cache/state、TopK 均逐 bit 一致，但 inverse RoPE 前已出现3个1 ULP差异，
最终输出210元素不同、最大298 ULP。只替换 K65/66 分母规约后同进程差异降为0。
详见[首次失败及单因素证据](DSV4_FLASH_CSA_TND_STABILITY_20261010.md)。

## 原生逻辑与修复

Native `NewReduceSumLastNDImpl` 以实际有效 K 分档，512列物理缓冲不能替代逻辑长度。
本轮将 raw 与 compressed 的长度分别从 replay 当次 position 读取：
raw为 `min(position+1,128)`，compressed为 `min((position+1)//4,512)`。

| 实际 K | 本轮精度版与 Native 对齐的顺序 |
| --- | --- |
| 0 | planner 的无有效块判断跳过计算，保留上一块 sum/max |
| 1…63 | 64列切片的 valid_shape设为实际K，`vcadd` mask为K |
| 64…511 | 复制首64列，按64列块依次相加；尾部仅保留有效通道的相加结果；再64→8→1 |
| 512 | 保留此前验证的连续8元素partial，后续两级8规约 |

短 K 使用动态 valid_shape，避免固定64/512列补零代替真实 mask。
中等 K 的首64列经GM暂存逐位复制到compact64缓冲，
不通过浮点加零或乘一实现复制。暂存按24个core/64个head独占，固定384 KiB；
两个AIV使用不同head行，token及block之间串行复用。每个后续64列块按 Native 顺序折叠；
尾部 `LT` mask + `TSEL` 保留无效通道的原始 bits，包括 subnormal。
采用静态展开与显式 `tile.slice`，兼容当前 PyPTO 的 tile 下标限制。

max、exp、alpha、概率 CAST_ROUND、PV、最终除法和 BF16 舍入边界保持原计算。
TND 仍使用真实请求边界，总 token 数为请求长度之和，不要求 padding 到 B×6。
精度包仍显式复用 Native HC/norm/O-proj，PTO 执行其余CSA链路。

## 预检与独立审查

候选快照为 `native-sum-v31`，23文件逐 SHA 校验后执行注册实际 signature/ABI、完整 lowering、
PTOAS 生成与 orchestration 编译，并实际编译修改后的 AIC/AIV incore 二进制。
v31当时的26项增量契约UT通过；后续v43加入双槽边界后完整39项通过。
UT执行真实规约 AST，覆盖live position正向及倒退更新；使用NaN无效尾验证 mask 屏蔽。
这些CPU测试只证明路由与合约，NPU数值结果单独记录。

生成CPP独立审查确认：

- 短 K 源Tile仍为物理stride512，valid列数取当前K；
- 首64列复制目标紧密排列，没有退化成strided alias；
- 七段64列折叠与phi均紧密，尾部比较为LT，false通道取前一acc；
- 临时缓冲没有覆盖exp，后续概率转换仍读取原exp。

实际编译使用的PTO ISA HEAD与pin均为 `327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d`。
`TROWSUM_IMPL → TRowReduceInstr → OneRepeatProc → SetContinuousMask(K) → vcadd`
的短K链已核实，静态大矩阵优化不会命中该动态8行切片。

早期无设备候选分别被动态下标、Tensor/Tile混用、TMOV物理形状、SSA返回契约拒绝。
v29完整PTOAS/编排预检通过，但首项在设备注册时被CCEC拒绝：TSEL要求源/目标完整Tile类型一致，
不能直接将物理512列视图复制到64列目标。该任务自动停止，precision数值尚未执行，
剩余9项未运行。由此补上无设备真实AIC/AIV编译门禁，v31两份设备二进制均编译通过。
这些失败不作为精度测量，也未绕过任何编译或测试门禁。

## 硬件回归协议

固定227单卡A3、确定性Native level1/HCCL true、atomic0，验证真实static+SuperKernel。
环境为 CANN/ATB9.2.0-beta.2、Torch2.10.0、Torch-NPU2.10.0.post2、vLLM0.25.1、
vLLM-Ascend0.25.1rc1 lineage、PyPTO3e87a843、Simpler a54c0509、PTOAS0.66。
复用框架与二进制，未重装。使用真实模型第4层权重及合成hidden/history。

所有臂独立cache/graph owner，PTO禁止回退；Native观察副本须全状态等同未修改Native。
候选须与Native在实际heads及全部八类输出/state逐bit一致，包含扰动hidden后的B输入；
compiled/raw、A/B/A、100 replay、reset、保护区与非有限值门禁都要通过。
首个失败即保存结果并停止批次，先定位，不继续剩余完整测试。

边界批次顺序为 history259、507、1019、2043、251、63、126、3、0、96，
均B4/T11、请求长度 `[6,3,1,1]`、seed62、layer4。
它同时覆盖压缩K分档及raw短窗口；具体完成范围以结果表为准。
各档分别捕获图，A/B只扰动hidden。live长度正向/倒退变化目前由AST和CPP验证，
尚未在同一captured graph中改变position/seq_lens并跨K分档重放。

## 第一轮硬件结果

任务 `task_20261010_102106_59460414141` 使用 v31，设备0，首失配自动停止。

| history | A 输入 heads/完整八状态 | B：hidden+0.125 | 后续稳定性 | 结论 |
| --- | --- | --- | --- | --- |
| 259 | 逐bit一致 | heads与完整八状态逐bit一致 | 100 replay、reset、保护区、SK资格通过 | 修复了此前3个heads及210个最终输出差异 |
| 507 | 逐bit一致 | 完整八状态（含最终输出）逐bit；heads有1元素差1ULP | 首失配停止，未执行后续100次/计时 | 未达到严格验收，继续定位 |
| 其余8档 | 未运行 | 未运行 | 未运行 | 不能计作通过 |

history507的B差异 max_abs为3.814697265625e-6、heads relative L2为1.7043808891e-7。
最终输出一致不能替代中间heads的逐bit门禁。
已核对Native在线sum及PV更新都是先乘后加，最终使用vdiv和CAST_RINT；
当前生成代码对应相同边界，不据此猜测FMA或倒数近似。
新诊断仅在同一507输入的同captured graph导出actual Q及inverse RoPE前BF16 heads，
并要求未修改/观测四路A/B状态与heads保持一致，以确定首差阶段。
任务 `task_20261010_105504_26105056987` 已完成：观测副本与原版在A/B的heads和八状态逐bit，
实际Q的360448元素全exact；preinverse与heads的首差都为 `[8,18,284]`，
PTO为-0.0006103515625、Native为-0.000606536865234375。该NOPE坐标不参与inverse RoPE，
因此首差已收敛到attention本体，不能归因于Q投影或inverse RoPE。
诊断执行成功、100 replay/reset成功不代表候选已逐bit通过Native。

下一项仅将PV输出切成4个N128 slab，使当前生成核的K32变为K128，
对照Native的N128/K128累加边界；概率、softmax、QK、sink和舍入边界不变。
同进程保留未改control，要求它复现B的heads1差异与完整八状态exact。
候选另行记录heads/完整八状态的逐bit通过标志。
任务 `task_20261010_110653_33781312532` 已完成：baseline与N128候选均在同一NOPE坐标差1ULP，
候选Q及完整八状态仍exact；该因素没有修复作用，不合入正式源码。
有效K候选 `task_20261010_111618_398341222532` 已完成：真实MAD消费live K127、跳过未用part，
仍在相同NOPE坐标差1ULP。其Q、完整八状态和observer fidelity通过，但heads不通过，未合入。
Native sink首次注入、raw窗口382…509的顺序已由安装源码核实，与PTO相同。

进一步发现Native `BlockReduceSum` 使用counter mask的`vcgadd`，
当前PTO小块规约使用`vcadd`，不能只由8元素分组推定逐位等价。
独立最小原语对拍使用完全相同FP32 bits，比较原PTO、normal mask vcgadd、
Native counter mask vcgadd及counter原地写法。
四路signature/ABI/lower/orchestration与真实AIV已通过，私有生成CPP严格检查布局/替换数量并记录SHA，
不改框架、共享ISA或正式CSA源码。
第一项原语任务因启动脚本变量被环境脚本覆盖而失败，没有执行数值计算。
修正后normal mask使用位图而不是错误的literal64，改用捕获一次后replay，避免每次注册重复编译。
任务 `task_20261010_115433_201751129561` 已完成：20组相同FP32 bits，
覆盖N8/64/512/1024及uniform、exponential、cancellation、subnormal、rounding输入；
四路FP32输出全部逐bit，compiled/raw、三次replay、输入更新/恢复、输入未修改及保护区通过。
这项有限原语测试未发现需替换规约指令的证据；不宣称所有输入普遍等价，也未合入指令替换。
不因诊断任务exit0而宣称候选逐bit通过。

## FP32 舍入前继续定位

独立v38诊断导出attention最终max、sum、log(sum)、LSE、FP32 PV分子和除法结果，
同时仅在Native观察臂打开真实LSE输出。两份正式精度副本保持原数学，性能包保持不变。
完整signature/ABI/lower/orchestration及实际CCEC AIC/AIV均通过，生成merge核独立审查通过。
FP32 division拷贝完成后才允许原地BF16 CAST_RINT；LOG和LSE使用独立UB地址，不改sum。

任务 `task_20261010_122656_352943719831` 已完成，exit0，仅固定history507的同一失败输入。
Native LSE开关及PTO导出均须通过A/B heads与八状态的观察一致性、真实SK profile资格；
导出数据须通过compiled/raw、各缓冲随B更新、A/B/A、100 replay/reset及finite/guards。
LSE等于log(sum)+max，LSE相同不能独自证明原始sum逐bit相同。
全部观察可靠性门禁通过；B目标 `[8,18,284]` 的PTO FP32除法结果为
`-0.0006084442138671875`，精确处于两个BF16数的舍入中点。
其sum=`71.42652893066406`、max=`1.4120025634765625`；
PTO LSE=`5.680671691894531`，Native LSE=`5.6806721687316895`，差1个FP32 ULP。
A的704个LSE元素有7个不同，B有5个不同；此观测不能把根因直接归为sum或max。
不能通过修改最后cast或硬推一侧来修复；下一步私有单算子观测Native真正的核内统计。
私有改名未插桩clone必须先与installed Native在同一A/B heads和八状态逐bit一致，
插桩后再次验证观察一致性；不重装框架、不覆盖原生算子或共享库。


## Native 核内统计的受控观测

先构建私有改名、未插桩单算子（不修改安装版），再构建带统计输出的同源单算子。
未插桩任务 `task_20261010_125110_90358025729` 和统计任务
`task_20261010_130607_16745277122` 均完成 exit0：同一次 Native 的 Q/cache/TopK 输入下，
私有算子的 BF16 heads 与安装版 inverse RoPE 前输出逐bit；统计版 LSE 与安装版 LSE 逐bit；
导出的 FP32 division 经 BF16 舍入也与安装版逐bit。
观察臂的最终输出、八状态、A/B/A、100 replay/reset、保护区及 Native SK 资格均通过。
统计值来自私有同源重执行，不冒充对安装二进制内部的直接截获。

| 字段 | A | B |
| --- | --- | --- |
| max（704元素） | 逐bit | 逐bit |
| PV分子（360448元素） | 逐bit | 逐bit |
| sum | 30元素不同，最多2 FP32 ULP | 26元素不同，最多1 FP32 ULP |
| FP32 division | 14040元素不同，最多3 ULP | 12345元素不同，最多2 ULP |

B目标 `[8,18,284]` 的max两边为 `1.4120025634765625`，PV两边为
`-0.04345905780792236`（bits一致）。PTO分母为 `71.42652893066406`，
Native统计分母为 `71.4265365600586`。PTO除法恰落BF16中点，Native除法
`-0.0006084441556595266` 比该中点向零偏移1个FP32 ULP，形成最终1个BF16 ULP差异。
因此继续定位每块exp、块内sum、old sum和alpha更新；不修改最终cast或添加epsilon。


## 压缩 KV 双槽搬运顺序的定位

分块观测任务 `task_20261010_134812_292649614266` 完成 exit0。所有观察保真、
有效exp覆盖、尾部NaN、各块old/new状态连续性、显式FP32先乘后加重构、
实际最终max/sum身份、A/B/A和100 replay/reset门禁通过。
第一项诊断因Torch schema65参数超过64限制失败，未得到分块数值结果；
修正为单个66304列观测Out后，实际Torch CPU注册64参数及完整CCEC通过。

| 分块字段 | Raw A/B | Compressed A | Compressed B |
| --- | --- | --- | --- |
| old max、old sum、alpha、new max | 逐bit | 逐bit | 逐bit |
| exp | 逐bit | 值的集合逐bit，槽位顺序不同 | 值的集合逐bit，槽位顺序不同 |
| block sum | 逐bit | 59元素不同，最多2 ULP | 49元素不同，最多1 ULP |
| new sum | 逐bit | 30元素不同，最多2 ULP | 26元素不同，最多1 ULP |

直接核对当前Native SCFA `ProcessVec0L → CopyInKv`：它按相邻两槽合并搬运，
在合法stride下从较低的**物理地址**开始读取，不保证TopK槽位原次序。
当stride溢出或为负、任一索引是最后有效索引时，会退回两次单独搬运，保留原序。
当前fixture物理页表本身逆序，不能用逻辑index大小替代实际页表寻址。
用同次B的实际TopK和fixture真实页表规则预测这项双槽交换后，
11个token×64个head的全部有效压缩exp逐bit匹配Native。该证据解释了不同求和顺序。

v43仅修改内部稀疏KV读取顺序：按原生双槽物理顺序及回退条件整理内部索引；
公开TopK、cache写入槽位、QK/exp/规约/舍入指令不变。
当前ABI及adapter已明确只接受连续 `[pages,32,1,512]` BF16 KV页，
不宣称支持页间padding；实际CPP以INT64计算物理地址和stride。
新增13个CPU边界回归测试，完整39项通过；planner实际AIV CCEC通过。
修复任务 `task_20261010_140715_338612730209` 完成 exit0：A/B 两组输入的两块
全部有效exp及六项更新统计逐bit；最终FP32 max、sum、PV、division、LSE全部逐bit；
实际Q、preinverse BF16、heads与完整八状态全部逐bit；观察保真及100 replay/reset/guards通过。
它修复了此前507 B唯一的1个heads ULP，未经插桩的精度臂也通过相同heads/完整状态门禁。
没有改变最后舍入或加入epsilon。

| history507字段 | 修复前B差异元素数 | 双槽顺序修复后A/B |
| --- | --- | --- |
| block sum | 49 | 0 / 0 |
| final sum | 26 | 0 / 0 |
| FP32 division | 12345 | 0 / 0 |
| BF16 heads | 1（1 ULP） | 0 / 0 |
| 最终输出及完整八状态 | 0 | 0 / 0 |

10档有效Key边界任务 `task_20261010_141206_362656324182` 已完成，exit0。

| history | 请求长度 | A heads/完整八状态 | B heads/完整八状态 | 100 replay/reset/guards/SK |
| --- | --- | --- | --- | --- |
| 259、507、1019、2043、251、63、126、3、0、96 | `[6,3,1,1]` | 全部逐bit | 全部逐bit | 全部通过 |

没有未跑边界被记为通过。复测总共10个独立配置，均为B4/T11、seed62、真实C4第4层权重。
它覆盖空块、短块与64/128/256/512附近长度；不证明其他batch、权重、随机种子及整模型。
原47条配置宽覆盖另由 `task_20261010_144607_102448830529` 执行，首失配立即停。
47条中有46个不同配置；仅对配置、原始结果精度标志、runner SHA与完整manifest完全相同的
已有结果复用，记录明确区分复用与新硬件运行。该任务已完成exit1：
前两项（history8191/B4/S6、history8186/B4/长度`[3,4,5,6]`）A/B全部通过；
第三项history8186/B4/长度`[5,5,5,5]`在首次A检查停下。
该输入heads655360元素、O-proj81920元素、最终输出327680元素和全部六类cache/state逐bit；
TopK10240位置中2个不同，来自1行的顺序交换，选中集合相同。B及后续44项未运行，
不能将输出一致当作完整八状态验收通过。当前保持严格TopK逐bit门禁，单独定位分数/排序差异。

不能用预检通过宣称数值逐bit一致。
该验证仍是单层合成输入，整模型、多层累积、DP/EP16、EPLB和吞吐另需验收。

## TopK同分顺序定位（首失配后）

官方binding虽有return_value参数，但已安装Native tiling仅接受False。
首项观察任务因此exit1，没有产生分数测量；未将接口失败记为精度结果。
随后在独立目录构建唯一命名的原生QLI诊断算子，仅在最终排序后导出已有分数。
原Cube、系数、scale和排序均保持原样，且与未修改的实际Native强制对拍保真。

任务`task_20261010_153331_27882291554`完成exit0：

- 观察臂与未修改Native的A/B TopK、实际heads、全部八状态逐bit一致。
- compiled/raw、A→B→A、100 replay、reset及持久分数buffer guard全部通过。
- A输入按index对齐的10240个selected score逐bit一致，差异不来自打分。
- 唯一差异row12第357/358位：Native顺序495→1564，PTO顺序1564→495。
- 两候选同为FP32 bits939743030；B输入的完整精度检查也通过。

此输入有效候选2047个，旧PTO按AIV半区拆成1152列与895列，
再将右root置前合并。495在左半区，1564在右半区，因而同分时右侧优先。
Native则对完整连续2048排序块做SortAll，两个候选属于同一块，优先级不同。
这项证据不支持修改WS或给score加入epsilon；修复方向是恢复原生排序边界。

独立v46候选保留QK/WS/scale原计算与双AIV搬运，将短档score存为连续query行，
待完整score任务结束后按连续2048候选块排序与合并。
8192上限时4096半区并非384整数倍，已额外裁剪每次store至其half边界，避免跨AIV竞写。
69项CPU测试、ABI/lower/编排及修改核实际CCEC通过；
严格单项复测`task_20261010_154746_345415414166`已完成exit0：
A/B两组10240个selected scores、TopK全部顺序、实际heads、完整八状态逐bit；
compiled/raw、A→B→A、100 replay、reset、buffer guards及Native实际SK均通过。
这修复了此前eq5的两个同分索引交换，未增加epsilon或修改打分/最后舍入。
新增17项关键边界回归`task_20261010_160603_13278418371`已提交，首失配立即停。
长档代码保持原样，其他短档仍需实测，不能以静态检查或诊断完成宣称全矩阵通过。
