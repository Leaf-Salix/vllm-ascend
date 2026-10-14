# TND 精度版整模型低接受率验证（2026-10-10）

初始轮源码冻结于 `ff2bb7b1690e503ce4a4aa38a984916a12ed0d9d`，显式选择
`deepseek_v4_flash_csa_tnd_precision`。性能包未修改；HCA始终使用Native。

参考用户指定的 nalinaly `low_acceptance_20261001/RESULTS.md`，原计划先做8K/B40，
再做128K/B24。当前按用户要求只验证8K/B40，不执行后续大配置。每条输出192 token，TP1/DP=EP16，DSpark提出5验6，greedy、ignoreEOS。
40条不同问题复用共同Native历史前缀，后缀真实prefill；固定完整target/draft cache和state。
旧前缀来源为CANN9.0.1，后缀本次由CANN9.2生成；各臂使用同一冻结状态。
不能把该口径写成完整历史都由CANN9.2重算。

## 最新结果（2026-10-14）

测试源码提交 `a9e0ae0d9`；性能版未改，HCA使用Native。

| 验证 | 结果 |
| --- | --- |
| 全部CSA首步 | 21层×16rank，336条记录的14个已观测边界全部逐bit一致 |
| Native两次192token重复 | 640条请求、122,880个token ID及完整acceptance事件均零差异 |
| 精度版对Native 192token | 同上，token ID、acceptance事件、DSpark统计均零差异 |
| 低接受率覆盖 | 接受长度0–5均覆盖，平均接受步长3.7035909921，无抢占 |

RMSNorm首差定位为512通道FP32平方和归约树；已在完整行和尾行统一为Native的8×64分组顺序。
完整行Tensor和Tile数学路径的2560行统计、1,310,720个BF16归一化元素均逐bit一致。
整模型任务分三次独立提交，每项上限780秒，实际运行约6–7分钟；模块任务更短。
本轮全部任务结束，未测试128K或新batch。192token一致不等于所有后续中间tensor都已逐bit验证。

详见[192token验收证据](evidence_tnd_precision/low_acceptance_rms_fix_decode192_20261014.json)及下方历史记录。

不等长CSA整层图的后续验证独立记录于
[2026-10-14 ragged FULL报告](DSV4_FLASH_CSA_TND_RAGGED_FULL_20261014.md)，
不与上面的等长整模型192token结果混为同一验证范围。

## 2026-10-14：按nalinaly方法补做真实模型重复验证

沿用nalinaly `low_acceptance_20261001` 的固定输入/状态、不同题目和逐轮接受事件口径，
再次执行当前精度版全模型decode。正式模型DeepSeek-V4-Flash-0731-w8a8的75个权重分片
实际加载；输入来自40条不同问题及Native实际导出的target/draft cache/state。
本次不使用单层变长测试中的合成hidden/history。

配置：TP1、DP=EP16、8K历史加实际问题后缀、每题192 token、greedy、DSpark出5验6；
只替换21层CSA，20层HCA Native。确定性level1/HCCL true、atomic0、NZ2、EPLB关闭，
FULL_DECODE_ONLY档位[6,240]。源码仍为a9e0ae0d9，数学/性能源码未修改，SDK与缓存全部复用。

| 对照 | 不同token/总token | 逐请求DSpark统计差异 | 逐轮事件差异 |
| --- | --- | --- | --- |
| Native两次，重新严格核验已有基线 | 0 / 122,880 | 0 / 640 | 0 / 640 |
| 本轮精度版对Native | 0 / 122,880 | 0 / 640 | 0 / 640 |
| 本轮精度版对今天上一轮精度版 | 0 / 122,880 | 0 / 640 | 0 / 640 |

16rank均完整完成，40种题目对应640条副本请求，不称640种独立输入。
实际接受长度0–5均覆盖，平均接受3.7035909921，无抢占；actual runtime选择、图覆盖、
静态包安装、READY复用及源码/bank/worker共同条件均通过严格检查。
新提交的NPU任务max-time780秒，正常exit0，已结束；未运行128K。

[真实模型重复验证证据](evidence_tnd_precision/real_model_precision_repeat_20261014.json)。
原始三组严格比较JSON的SHA也记录于该证据，完整明细保存于本地reports。

两边恢复相同Native状态；旧前缀来自CANN9.0.1，问题后缀由CANN9.2实际prefill生成，
不声称本轮重新计算了整个前缀。这与nalinaly固定bank方法一致。
本轮证明真实模型token和接受轨迹在这些输入下可复现地与Native一致，
不推出192步的全部浮点中间tensor逐bit一致，也不声称覆盖不等长整模型FULL：
本轮各rank仍未观测到同一步ragged query；不等长CSA整层图证据另见相邻报告。

## 判据与次序

1. Native1、Native2分别独立运行，确定性level1、HCCL true、atomic0、NZ2、EPLB关闭。
2. 先核验每worker实际环境、41层选择、静态包安装、FULL捕获和重放、192 token完整性。
3. 检查Native两次逐token和逐请求DSpark proposed/accepted事件序列，不只比较总直方图。
4. Native重复通过后，只替换21个CSA层；20个HCA层继续Native，再比较首个分歧。
5. accepted0…5必须全部覆盖，主批平均接受3…4；未满足时不宣称低接受率测试通过。
6. 首个token或接受事件失配立即停止后续配置，定位同输入阶段。首分歧后的轨迹差异不当作同输入浮点误差。
7. CPU query bounds记录真实请求长度，总token不是graph档位；不强制每请求padding到S6。

整模型通过仍不能推出全部中间tensor逐bit一致；中间逐bit判据由单层阶段测试另行验证。

## 时间与静态编译

每个CPU/NPU任务独立提交，max-time780秒，预留两分钟清理裕量；约五分钟查询一次，
不以持续等待替代小任务拆分。没有创建定时任务。

第一轮Native因1410项静态编译超时，无输出token结果，不计精度失败或通过。
后续先分8个不占卡任务运行官方op_compiler，再通过测试私有桥复用官方包。
当前捕获的完整JSON、CANN/SOC、实际确定性、框架/编译器SHA、输入到成功产物关联均硬核验，
不完整时导出缺失项并立即退出；原安装、barrier、reselect流程保留。

特别记录：op_compiler可能部分失败却返回0。chunk00实际198/200，不能宣称200项全覆盖。
两项CompressorMetadata的const指针兼容错误在独立OPP副本修复，仅增加只读参数const；
补编2/2成功。旧失败两项从旧包覆盖剥离，由新成功包覆盖。未改共享OPP或HCA计算。
首阶段75项由原框架成功编译/安装；合计1485个唯一描述符、10个官方包。
原始包、成功/失败日志、关联清单和header前后SHA均保留。

## 2026-10-10 状态（历史记录）

- 40条固定Native请求状态准备完成，审计PASS。
- 官方离线静态编译、两项兼容补编完成；CPU复用门禁6项通过，独立复核通过。
- Native1重试在安装官方包时退出：测试复制丢失执行位，尚未进入生成。
- 本地改为保留权限的copy2，CPU门禁再次6项通过；远端上传自动审批连续超时，未执行。
- 所有本会话任务均已结束。尚无Native重复一致性或Native/PTO token误差结论。
- 下一步上传权限修复，先完成Native1、Native2并比较，再执行仅CSA替换的精度臂。

## 2026-10-12：恢复运行与固定容量控制

用户批准继续，上传复制权限修复后，Native自动容量重试在约5分钟退出：
严格静态JSON门禁发现1179项未覆盖，生成未开始。
最近差异主要来自不同进程的自动KV页数和零拷贝view偏移，少量为dtype别名。
此结果是测试静态覆盖问题，不能记录为Native精度失配；没有放宽JSON门禁。

只在私有测试入口给8K/B40三臂共同设置`num_gpu_blocks_override=16384`，
使用原生allocator；不修改Native/CSA/HCA算术，不把此容量套用到128K。
全部16个rank确认实际override16384，逻辑容量386893 token，最大并发44.52，
超过本轮40条请求。完整数据见
[容量审计](evidence_tnd_precision/low_acceptance_capacity_20261012.json)。
后续验收还要求零抢占，无OOM；不能仅凭请求参数推断容量足够。

固定容量捕获发现340项未覆盖，硬件任务退出，未生成token。
按150/150/40分成三项CPU任务，均逐项成功；每批均检查唯一n/n记录、
无错误日志、输入JSON精确集合及SHA、非空官方包、编译器与header来源。
340项成功产物已追加到原清单，共13组；旧清单保留，不用工具exit0替代逐项成功。

CPU querybounds观察上限从32扩大到256，覆盖192 token生成的收尾变长边界，
只读取已有CPU metadata，不加入NPU同步；后续三臂共享同一观察版本和源码SHA。
正式精度代码仍冻结于本页列出的ff2bb7b，性能包23文件与备份逐字节一致。

Native固定容量第一轮已重新提交，正在排队；每项max-time780秒，
继续每约5分钟监控。当前仍无Native重复一致性或Native/PTO token误差结论。

## 2026-10-12：官方静态缓存入口与第一轮完成

固定容量后的再次独立启动仍新增385个静态描述符，尚未生成token便退出。
只读追踪确认，Native `_align_memory` 根据真实tensor地址对齐到2 MiB，
因此即使KV页数相同，cache view的storage offset仍会随进程分配地址改变。
逐次扩大完整JSON白名单不能形成稳定的复测入口。

后续三臂共同使用当前Torch-NPU已有的官方缓存分支：
`use_cache_compile=1`，实际CANN版本`9.2.0.beta2`，确定性level1，
同一个私有缓存目录和冻结的13个官方包。测试入口检查包与框架SHA、SOC、
目录所有权、执行位及缓存元数据，保留官方安装、状态广播、重选与barrier。
安装后核验13个包全部登记；直接包裹真实CPP重选调用，仅在其成功返回后记成功。
禁止占卡过程中再执行op_compiler。CPU七项门禁测试和独立复核通过。

该口径证明共同官方包已加载且重选成功，不证明每个算子都命中静态kernel，
也不宣称与此前逐JSON新编译路径逐bit等价。三臂必须共享这一基线，
不把旧入口结果混入新的Native重复或Native/PTO对拍。

8K/B40 Native第一轮已exit0完成，核验通过：

| 项目 | 结果 |
| --- | --- |
| DP rank / 不同问题 | 16 / 40 |
| 合计请求 / 输出token | 640 / 122880（每请求192） |
| 同次跨DP输出token差异 | 0 |
| 同次跨DP接收事件序列差异 | 0 |
| 平均接受draft token | 3.7035909921 |
| 接受长度0/1/2/3/4/5次数 | 1696 / 2432 / 2816 / 2592 / 2240 / 14512 |
| 抢占 | 0 |
| FULL 240-token图重放计数 | 每rank 50 |
| 官方包 / 编译错误 | 13 / 0 |
| 已观察不等长query的rank | 0 / 16 |

这是第一轮运行资格与同次跨DP检查，**不是独立Native重复验证**，
也尚无本轮Native/PTO精度结论。真实CPU query记录未观察到不等长请求，
不把本轮宣称为变长TND整模型覆盖。
独立Native第二轮已提交，保持共同源码、固定状态、静态缓存和运行参数；
通过后才执行CSA精度版，HCA继续Native。

## 2026-10-12：独立Native重复通过

第二个独立Native进程也exit0完成。冻结源码、固定bank、容量参数、静态缓存SHA、
实际worker配置及启动参数核验相同；仅排除通信运行ID和观察输出目录。
两轮各640条请求、122880输出token，逐token差异0，逐请求逐轮接收事件差异0；
接收统计同样一致，全部0…5类别覆盖，平均接受3.7035909921，零抢占。
逐请求证据与运行条件见
[独立Native重复结果](evidence_tnd_precision/low_acceptance_native_repeat_20261012.json)。

两轮的原始rank结果、request_stats JSONL和日志均已另行保存。
第一轮独立复核也确认逐轮事件与Prometheus汇总对账一致。
第一轮每rank观察84或87个query样本，包含收尾请求数减少及graph padding；
有效请求query长度仍全为6，没有不等长请求覆盖。

在此Native基线通过后，已提交TND精度版CSA单独替换任务，HCA仍Native，
复用相同固定状态和官方缓存。任务max-time780秒，按约五分钟间隔监控。
本段只证明Native自身重复稳定，尚无本轮Native/PTO误差结论。

## 2026-10-12：CSA初始化超时与共享JIT缓存修复

旧环境CSA任务触发780秒上限，exit130，未生成任何rank结果。
第一个Graph档位完成，第二个仍在PyPTO orchestration二进制编译等待；
13个官方静态包均已成功安装/重选。此次不是数值失配，精度判定为未测试。

只读核验发现共享缓存已开启，但官方工具链身份检查拒绝
`pypto.pypto_core`的外部导入重定向：Python包来自源码树，原扩展来自editable安装目录。
持久缓存的identity-unusable分支直接build，不检查对象缓存；本轮产生463个构建目录，
451个compiled metadata具有同一ABI，448份binary manifest。
ABI相同不能证明生成代码相同，没有把其中任意产物强行替换到其他编译请求。

修复仅在私有测试目录复制完整PyPTO Python包及原来已编译的两扩展，206个文件逐项SHA一致。
使用限定于pypto命名空间的普通PathFinder，保留Torch/Simpler既有选路；
两扩展仍选中原依赖库路径。原官方工具链身份检查未修改，重新核验usable=true、failures为空。
没有重装或重编译vLLM/PyPTO/Simpler，也未修改原安装目录、性能包或精度算术。

官方无设备kernel artifact接口完成signature-mode预编译及READY发布。
第一次诊断误查不变的GENERATED output_dir而退出；随后改查官方runtime.handle的BINARY_READY状态，
不弱化二进制SHA、ABI与加载检查。独立新CPU进程约3.1秒命中READY，
miss/bypass/generation/binary/storage计数全0。证据见
[运行包身份与READY命中](evidence_tnd_precision/low_acceptance_runtime_package_20261012.json)。
CPU命中仍不能证明NPU真实特殊化全部命中；后续逐rank核查cache_stats，拒绝bypass、
强制重建、身份/存储错误及重复构建，并记录实际模块路径。

这是新的环境基线，三臂必须共用同一运行包manifest SHA和同一观察源码SHA。
旧Native重复PASS保留为历史证据，不与新环境CSA混作主对照。
已提交新环境Native第一轮，之后独立Native第二轮，再仅替换CSA；HCA继续Native。
每项任务仍max-time780秒，继续约五分钟监控。

## 2026-10-12：完整运行包环境Native重复通过

完整包新环境的Native1、Native2均exit0完成，各640请求、122880输出token。
逐token、逐请求逐轮接收事件全部一致；源清单、运行包SHA、worker配置、
除运行ID/输出目录外的启动参数均相同。实际package/core模块路径均在私有包中。
零抢占，平均接受3.7035909921，0…5类别全部覆盖。
结果见[新环境Native重复](evidence_tnd_precision/low_acceptance_package_native_repeat_20261012.json)。

已提交同环境仅CSA精度版任务；使用CPU准备好的同一官方READY缓存，
逐rank核查实际替换、FULL重放、模块路径和cache_stats。
每项仍max-time780秒，约五分钟监控。尚无新环境Native/PTO精度结论。

## 2026-10-12：整模型 CSA 精度版首档失配，暂停扩展

同一运行包环境的仅 CSA 精度替换任务完成（exit0），16rank结果完整。
40题各生成192token，DP16各自重复同一40题；性能版与HCA未修改。
Native两次独立重复已逐token/逐接受事件一致。

| 检查 | 结果 |
| --- | --- |
| 输出token分歧 | 26/40个不同问题（416/640条DP请求） |
| 接受事件分歧 | 32/40个不同问题（512/640条DP请求） |
| 最早分歧 | q09、q38的第0个输出token |
| q09首token：Native / PTO | 46543 / 24313 |
| q38首token：Native / PTO | 3648 / 9854 |
| 平均接受：Native / PTO | 3.703591 / 3.656043 |
| PTO内部16DP复现 | token与接受事件一致，抢占0 |
| 每rank官方cache_stats | requests105、object_hits62、ready_hits1 |
| 重建/旁路/身份或存储错误 | 所有rank均0 |

[完整比较记录](evidence_tnd_precision/low_acceptance_package_precision_fail_20261012.json)。
首个token不同后生成上下文已分叉，后续45824个不同位置是轨迹差异，不能解释为同输入的浮点误差计数。
此前17项单层逐bit一致不能推广为本次真实整模型通过。

独立复核另发现测试overlay的观测缺陷：CSA与已缺失的HCA包处于同一个import异常分支，
HCA导入失败导致CSA捕获计数器也未安装。因此空capture_time_selection不证明未执行PTO。
声明的21CSA精度runtime、实际PyPTO缓存调用与FULL重放均存在；真实token失配仍成立。
修复为独立安装CSA计数器、仅允许可选HCA模块本身缺失、真实依赖错误向上传播；
无NPU回归覆盖缺失HCA、继承HCA不重复计数、缺失真实依赖三种情况。

按首个失配即暂停规则，不提交128K或后续大配置。
下一项只对首步T40/B40、首个CSA层2的同一实际hidden/positions/cache初态进行模块对拍，
恢复声明会写的物理页后执行Native参考，保留PTO原输出和PTO cache继续执行。
诊断仍保持短任务、复用已编译产物，优先定位最早算术或接口差异。

## 2026-10-12：首步 layer 2 同输入对拍

使用同一40题fixed8192 bank，只生成1个token，恢复各slot声明写入的物理页后即时运行Native参考。
16rank完成且诊断指标一致；实际rank0日志区间22:34:04至22:39:54，长排队不计入占卡执行时间。

| 边界 | 结果 |
| --- | --- |
| HC/norm后的输入 | 逐bit一致 |
| raw KV有效写入 | 请求2/channel183、请求13/channel305、请求20/channel61各差1 BF16 ULP |
| 压缩KV、Indexer缓存、两类state | 逐bit一致 |
| O-proj前heads、O-proj输出、整层输出 | 逐bit一致 |
| 两臂对hidden的修改 | 无 |

[首步数据与有效写入差异](evidence_tnd_precision/low_acceptance_first_step_20261012.json)。
3个差异均在NOPE通道，而非RoPE或padding；最大绝对差3.814697265625e-6。
raw KV差异本步尚未改变该层heads或输出，不能声称它已解释整模型首token分歧。
按照最早误差优先规则，先补抓Native KV投影/RMSNorm边界并转单卡模块回放，暂不扩大配置。

这里的参考是同输入即时Native调用，不能替代此前独立生产Native图模式验收；
此处单层输出一致也不覆盖后续20个CSA层。

## 2026-10-13：首步 raw KV 差异定位到投影

补抓同一次Native的WKV BF16输出与KV RMSNorm输出，两次首步采集的normalized hidden与Native heads逐bit一致。
独立单卡导出projection FP32的诊断，最终KV与原PTO路径逐bit一致，排除导出本身改写算术。
投影在BF16发布边界即出现同一3个坐标差异，后续差异仍只在这3个坐标。

| 单因素 | 投影BF16差异 | RMSNorm/RoPE后的raw KV差异 | 结论 |
| --- | --- | --- | --- |
| 原K256 | 3 | 3 | 最早在投影 |
| 改K64 | 3 | 3 | 与原PTO逐bit相同，无效，不合入 |
| 注入同一次Native投影 | 0（注入保证） | 0 | 后续RMSNorm/RoPE可以对齐 |

[单卡证据与任务退出说明](evidence_tnd_precision/low_acceptance_kv_projection_20261013.json)。
原单卡任务在保存有效结果之后因误调用shutdown模块退出1；清理修正后K64与注入任务均exit0。
诊断缓冲的静态容量、准备tile二维/32字节对齐问题均在CPU生成检查阶段解决，没有为此占卡。

据此只在精度包接入原Native WKV模块的BF16输出，作为新的动态TND输入`native_kv_projected`。
后续KV RMSNorm/RoPE/cache写入仍由PTO执行；性能包保持原样。
旧直接kernel fixture必须补充该输入，不能把旧READY当作新ABI证据。
新ABI无设备注册/生成/CCEC/READY检查通过，复制任务以同一GM缓冲的inout写入、RMS任务以input读取，
尾行保持有效形状。独立源码review未发现绑定、scope或生命周期阻断。
正式整层候选已提交相同首步复测，结果未出，尚不能宣称修复已在整模型通过。
该方案增加一个Native投影边界，减少融合程度；本轮没有测性能。

## 2026-10-13：Native WKV发布边界修正的首步整层验证

新ABI同档首步任务exit0，16rank结果完整且指标一致，全部已观测边界逐bit一致：
Native投影BF16、归一化输入、raw/压缩/Indexer缓存、两类state、heads、O-proj与整层输出。
此前3处raw KV误差降为0，两次运行均没有修改输入hidden。

[修正后数据](evidence_tnd_precision/low_acceptance_native_wkv_fix_20261013.json)。
主进程日志10:22:18至10:28:19，约6分钟；占卡连同前面的预检约7分钟，未超过15分钟。
提交至结束42分28秒包含排队，不应解释为占卡42分钟。

附加检查40题首token均与Native基线一致，q09/q38的首token分歧消失。
候选只生成1token，旧参考生成192token，此结果仅为首token控制检查，
不视为完全同参数独立生成验收，也不能证明后续step/其他CSA层的逐bit稳定性。
当前正式源码修正仅涉及精度包三个文件；性能包未改，没有性能提升结论。
后续仍只在首步相同输入扩展阶段定位，128K和后续大配置继续暂停。

## 2026-10-13：扩展到所有CSA层的首步验证协议

正式精度版源码为`fc437f41f4aa0a78b3bb751ba7077ced2291cbe3`。
仍使用fixed8192、B40、TP1 DP/EP16、确定性level1/HCCLtrue、Native HCA，
只生成1token；FULL_DECODE_ONLY配置保持6/240两档。此次即时Native参考在实际首步中运行，
不能视为独立Native生产图模式验收。

| 门禁 | 要求 |
| --- | --- |
| 层覆盖 | 21个CSA层，模型索引2、4、…、42 |
| rank覆盖 | 16rank，每个rank/层唯一记录，共336份 |
| 逐bit边界 | normed、projection BF16、raw KV、压缩KV、Indexer packed cache、两类state、真实TopK、heads、O-proj、输出、两臂hidden保持，共13项 |
| 输入与cache | 同一实际hidden/positions；Native前恢复声明写入的物理页，Native后恢复PTO状态 |
| 首个失配 | 各层通过已有EP CPU组MAX同步；任一rank失配，全部rank停止后续层对拍，正常结束首token防止EP缺参与者 |
| 失败证据 | 先发布STOP，再保存失败层张量；passing层只保存JSON |
| 192token放行 | 无STOP、336份完整唯一记录、全部必需边界逐bit一致且有效输出无非有限值 |
| 占卡时间 | 每项任务上限780秒，预留清理时间，禁止扩展128K等后续大配置 |

无设备精确导入预检及正式ABI生成/CCEC/READY检查已通过。
本次READY identity为`a86ef21bbce42bc9fcbfe8e156b421b43e67d94a2062c837d608af06e5388251`。
本地覆盖门禁检查包括完整记录通过、缺样本/重复/失配/STOP/缺必需TopK均拒绝。
独立复核确认QLI wrapper返回原值、finally恢复原算子，EP CPU同步顺序未发现阻断。

[首个失配与16rank一致性证据](evidence_tnd_precision/low_acceptance_all_csa_first_stop_20261013.json)。

| 首步层范围 | 16rank结果 |
| --- | --- |
| 2、4、6、8、10、12、14 | 全部13项逐bit一致 |
| 16 | 仅TopK有15个位置差异；其余12项包括heads与最终输出逐bit一致 |
| 18、20、…、42 | 首个失配后停止，未对拍 |

首个差异发生在第12个请求（从0计数）的TopK排列；逐行排序后两臂完整512项逐bit一致，
没有新增或丢失选点。16rank全部复现同一结果。不能据此认定同分排序或score误差根因，
需要继续观察实际Native QLI输入与PTO score；不会通过忽略TopK差异放行。
实际主进程日志11:55:20至12:01:32，约6分12秒，加预检仍小于15分钟。
模型正常完成首token；任务exit1来自首个失配/覆盖门禁，非模型启动异常。
共保存128份记录（16rank×8层），按预期不具备336份完整通过资格；192token复测未提交。

## 2026-10-13：layer16 TopK排列差异定位到head-weight投影

只对同一真实首步的layer16新增采集，未跑192token或后续配置；任务exit0，
主进程13:13:01至13:19:00，约6分钟。16rank再次复现同一TopK排列失配，
声明写入cache/state、heads和输出仍逐bit一致。

[QLI实际输入、score分析及单卡投影证据](evidence_tnd_precision/low_acceptance_indexer_weight_projection_20261013.json)。
PTO重排位置的score均不相等，因此不能归因于同分排序规则。
Native算子不支持返回score；CPU模拟明确不是Native实测score，也不作为逐bit验收。
其诊断回归将512个已选score误差收敛到head20系数，随后用同一正式
`indexer_weights_project`、T40实际normalized hidden和checkpoint存储BF16权重单卡回放。

| 检查 | PTO | Native实际QLI输入 |
| --- | --- | --- |
| 40×64中不同元素数 | 1 | — |
| 请求12/head20的FP16权重 | 0.042724609375 | 0.04248046875 |
| 其余2559个权重 | 逐bit一致 | 逐bit一致 |
| 官方READY复用 | hit1，无重建/旁路/身份错误 | — |

单卡任务exit0，上限180秒。独立源码复核确认回放保持原K顺序、BF16投影/缩放及FP16发布，
40行尾块只导出实际8行。checkpoint权重转置与原adapter一致；
下一轮整层会补实际runtime权重SHA与checkpoint的直接核验，避免仅凭名称判定加载后参数相同。

修正候选仅在精度包保留原`impl.weights_proj`、Native缩放表达式及
`DeviceOperator.prepare_dsa_indexer_weights`输出，新增FP16动态TND输入；
PTO内只精确拓宽为FP32，后续score、TopK、sparse attention仍执行原代码。
原投影函数保留用于诊断，性能包未修改；不宣称性能改善。

新ABI无设备生成与CCEC通过，真实C++使用实际有效行进行TLOAD/转换/TSTORE，
Native输入明确注册为input，score依赖搬运后的weights。
14项边界的首步复测已提交，336份完整唯一PASS才放行192token。

### Native head-weight边界修正：首步复测结果

[16rank复测及首个新失配](evidence_tnd_precision/low_acceptance_native_indexer_weights_first_step_20261013.json)。
模型正常结束首token，严格覆盖门禁exit1，未提交192token。

| 层范围 | 结果 |
| --- | --- |
| 2、4、…、24（12层） | 16rank全部14项逐bit一致；layer16旧TopK误差已消失 |
| 26 | cache/state、投影、Native Indexer权重及TopK逐bit一致；heads/O-proj/输出失配 |
| 28、30、…、42 | 首个差异后未对拍 |

layer26全部差异集中在请求31：heads的105个元素（均在head45）、
O-proj的1161个元素、整层输出429个元素不同。heads relative L2为1.49535e-5，
整层输出relative L2为4.99834e-5；这些结果仍不满足逐bit目标。
全16rank复现同一指标，共208份记录，不能冒充全部21层通过。

layer16实际runtime的BF16投影权重SHA与checkpoint转置后的逐字节SHA一致：
`ef523dd976546e43444a285fe9ff09b686350c7966554548a8f4ebc1d211d9a3`。
这补齐了单卡投影回放的参数身份核验。
下一步只定位layer26 heads前的Q与sparse attention阶段，后续层和192token继续暂停。

### Layer 26：Q 输入观测（2026-10-13）

在 Native Indexer weights 发布边界修正之后，首步已验证 CSA 层 2–24 的所有已列边界逐 bit 一致；首个差异停在层 26，没有继续层 28–42 或 192 token。

独立 Q 导出诊断完成，16 个 rank 的保真门禁均通过：新增观测前后的 hidden、positions、TopK、weights、cache/state、heads、O-proj 和输出与此前失败样本逐 bit 相同。Native Q 在 sparse attention 入参处观测，PTO Q 在原有 BF16 发布之后导出。

差异仅为请求 31、head 45 的 4 个 NOPE 通道：

| 通道 | PTO Q | Native Q |
| --- | ---: | ---: |
| 88 | 0.23046875 | 0.2294921875 |
| 232 | 0.23046875 | 0.2294921875 |
| 342 | 0.921875 | 0.91796875 |
| 419 | -1.84375 | -1.8359375 |

Q 的整体 relative L2 为 7.7254e-6；该结果将最早已观测差异前移至 attention 的 Q 输入，尚未区分 Q 投影发布与 RMSNorm 计算边界。下一项仅导出该 head 的投影发布值、平方和、variance、inverse RMS，以及 Native RMSNorm 前后值，确认数学路径保持不变后再定位。生产源码未因此修改；不将源码中的平方 dtype 差异直接认定为根因。

### Layer 26：Q RMSNorm 之前逐 bit 一致

同一首步的第二次诊断保真门禁在 16 个 rank 全通过，两侧 Q、heads、输出及其他已观测字段均与上一诊断逐 bit 相同。请求 24–31、head45 的全部 8×512 个 Q 投影发布值与 Native RMSNorm 入参逐 bit 一致。PTO 在请求31的 FP32 平方和为 `1.2337054014205933`，variance+epsilon 为 `0.0024105808697640896`，inverse RMS 为 `20.36756706237793`。

对实际 Native 输入使用 CPU FP32 平方和/rsqrt 公式得到 inverse RMS `20.367565155029297`，归一化后的8×512个BF16值与已观测 Native 全部一致。两种 inverse RMS 相差一个 FP32 ULP。加入 BF16 平方发布边界的 CPU 候选反而有103个元素不一致，因此不能依据 Triton 源码表面的 dtype 推断来补 BF16 平方舍入。

这些 CPU 数值只用于收窄归因，不代替 NPU 修正验证。后续单卡对拍共用原 FP32 平方和归约，仅比较高精度 rsqrt、基础 rsqrt、sqrt→reciprocal，并要求完整 Native RMS 重放与首步观测相同。

### Layer 26：实际 Native 统计纠正归因（2026-10-14）

单卡同输入反平方根对拍中，高精度 rsqrt 和 sqrt→reciprocal 均复现生产版本的4个归一化差异，基础 rsqrt 有769个差异。epsilon、variance 和高精度 inverse RMS 与生产捕获值均逐bit一致；因此不合入这些候选。

新增 Native 统计导出后，全部2560行归一化输出与未加观测的 Native 首步快照逐bit一致。实际 Native 的平方和已经与 PTO 不同，**最早差异是归约顺序，不是已经证实的反平方根实现错误**。此前 CPU rsqrt 对拍只用于收窄范围，不能代表 Native 的实际 variance。

| 请求31/head45 | Native 实际值 | PTO 实际值 |
| --- | ---: | ---: |
| 平方和 | 1.2337055206298828 | 1.2337054014205933 |
| variance+epsilon | 0.0024105811025947332 | 0.0024105808697640896 |
| inverse RMS | 20.367565155029297 | 20.36756706237793 |

CPU 对全部2560行尝试明确的FP32求和顺序，`stripe64-halves-then-adjacent` 的平方和与 Native 全部逐bit一致：8组64通道先依次组合 `(0+4)`、`(1+5)`、`(2+6)`、`(3+7)`，再 `((0+4)+(2+6))+((1+5)+(3+7))`，最后归约64通道。仅凭CPU模拟不能认定PTO的64列row_sum具有相同指令顺序；下一项NPU单因素对拍对平方和、variance、inverse RMS、mean以及归一化输出的全部2560行设置逐bit硬门禁。

### Q RMSNorm 归约修正：Tensor 和 Tile 模块实测通过

单因素 NPU 对拍已通过，全部2560行的结果如下：

| 路径 | 平方和 | variance | inverse RMS | BF16归一化输出（1,310,720元素） |
| --- | --- | --- | --- | --- |
| Tensor 8×512→8×64分组 | 全部逐bit一致 | 全部逐bit一致 | 全部逐bit一致 | 全部逐bit一致 |
| Tile 8×512→8×64分组＋高精度scratch | 全部逐bit一致 | 全部逐bit一致 | 全部逐bit一致 | 全部逐bit一致 |

两项均先校验未改动Native独立重放与真实首步快照相同，并设置统计与输出失配硬失败。Tile模块验证数学路径，不能替代实际部分行crop的整层验证。正式修改仅在精度版 `qkv_proj_rope.py`：main/tail采用相同7次FP32加法树，tail使用匹配尺寸的row_sum与高精度rsqrt scratch，原有效行crop和RoPE保持不变。性能版、ABI、框架版本未改。完整CSA根的真实CCEC/BINARY_READY通过，随后提交全部21层首步门禁；本节的模块通过不放行192token。

### 修正后全部 CSA 层首步逐bit验收（2026-10-14）

8K/B40、TP1、DP=EP16、Native确定性开启。层2、4、…、42共21个CSA层的全部336条首步记录通过；覆盖门禁没有缺失、重复或失败记录。

每条记录的14个边界全部逐bit一致：normed、raw KV、compressed KV、Indexer cache、两类压缩state、attention heads、O-proj、最终输出、TopK、Native Indexer weights、Native WKV BF16发布值、PTO/Native分别不修改hidden。此前层26的heads/O-proj/输出差异消除。14项不是所有内部临时Tensor的枚举，也不证明后续decode精度。

本轮仅一个实际首步，不将此结果写成192token已经通过。下一步仍是同源码、同缓存与图配置下分别执行两个Native重复对照，再执行精度版192token；各任务独立且上限780秒，发现差异不扩展配置。

### RMS修正后的192token整模型验收（2026-10-14）

全部CSA首步门禁通过后，冻结源码与共同manifest，分别提交Native1、Native2和精度版。
固定8K/B40、TP1、DP=EP16、提出5验6、192token、Native HCA，动态EPLB关闭。
每个任务独立，max-time780秒；没有合成一个长期占卡任务。

| 比较 | 请求记录数 | token数 | token失配 | 接受事件失配 | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| Native1 vs Native2 | 640 | 122,880 | 0 | 0 | PASS |
| Native1 vs 精度版 | 640 | 122,880 | 0 | 0 | PASS |

40条不同问题在16个DP rank中复现，640不是640条不同问题。
两侧平均接受长度均为3.7035909920876446；接受长度0–5完整覆盖，抢占数均为0。
实际41层选择、Native确定性level1/HCCL true、无atomic、13个官方静态包、FULL graph重放、
共同缓存状态与配置、私有PyPTO包来源及精度版READY复用均通过严格门禁。
精度版没有重新生成或编译框架/kernel产物；不是误走Native CSA的结果。
CPU query bounds一致性门禁通过，graph容量仍为共同240档位。
但16个rank均未记录不等长请求的FULL样本：接受长度0–5变化不等于同一步query长度不等长。
首步的真实T40/S1已由PTO参与逐bit对拍；本次192token结果不宣称覆盖ragged FULL模式。

结论限于本次8K/B40测试：全部21层首步的14个边界逐bit一致，192token生成和完整接受事件一致。
不外推到128K、其他batch或所有后续内部临时tensor。当前任务全部结束；后续大配置未提交。
