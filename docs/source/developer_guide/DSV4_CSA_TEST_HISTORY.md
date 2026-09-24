# DSV4 CSA 分支测试历史与记录规范

更新日期：2026-09-24。分支：`Leaf-Salix/vllm-ascend:dev/pypto-dsv4-csa-main-20260922`。

## 阅读结论

**当前没有可验收的生产整模型精度或性能结论。** 本文保存历史测量及其局限，不能只摘取最优延迟或最小误差。

- 正式代码最后一个已提交的算法版本为 `39993cd43`；后续数值对齐是未提交实验副本。
- 已确认原生 forward 中 indexer 权重发生变化，导致若干整层比较不满足同权重条件；这些精度数字仅保留作诊断。
- scatter 修正签名后的确定性对照已完成：主 compressor scatter 前权重差异0，后509；过滤负 slot 后全程0，确认本次权重污染与负 slot 写入有因果关系。首轮过滤包装的结果仍作废。
- 历史性能组未显式开启确定性。后续强制 level=1、HCCL 确定性，并检查权重不变性。
- 固定 B4/S6 的手工 Graph 不覆盖生产选档、padding、空 rank 或 dummy capture。

## 提交与实验的对应关系

|代码身份|内容|验证范围及限制|
|---|---|---|
|`dd24be2fe`|上游 main 基点|不是本分支的 CSA 测试结果|
|`6c9d552a1`|初始 main 集成 CSA|PR5仅为kernel来源；未找到可单独归属此提交的完整测量|
|`22beafb39`、`45c0e1ae4`、`3b45322e7`|存储接口、RoPE与cache分配兼容修正|没有独立逐提交性能数据；不得套用后续数据|
|历史 overlay：40参数 / metadata50 / direct-consumer|三阶段适配精简|无精确Git身份，保留阶段名称，不伪装成某commit|
|`362073abab53c00ad53644b4ba583af38ce161af`|50参数原生metadata消费精简集成|25项CPU测试；历史overlay硬件证据不是该commit逐字节复测|
|`39993cd43beaea042571b56e00c2c2b82665db57`|TP1 O-proj原生量化对齐|27项CPU测试；从父/新commit导出并校验源码的Graph复测|
|39993之后的Q/KV/Indexer候选|逐阶段补齐BF16/FP16计算边界|未提交实验；单层诊断与Graph，整体精度未验收|
|capture-size选项工作区修改|默认S6对齐，限DSpark FULL_DECODE_ONLY|12项CPU测试，未部署、未做生产NPU验证；不属于39993|

没有实测的提交明确记为未测；历史文档不为每个提交推测成绩。

## 共同测试口径

模型为 DeepSeek-V4-Flash-0731-w8a8 第2层C4真实权重，TP1、B4、S6、block128。
输入hidden/history为seed62合成数据，8K/128K表示结束上下文长度，起始位置8186/131066。
单层attention forward不含整模型MoE、调度器、采样、DP16通信与EPLB；未测完整DSpark接收率。

环境：CANN9.2.0-beta.2、Torch2.10.0+cpu、Torch-NPU2.10.0.post4、vLLM0.29.0、PTOAS0.63。
PyPTO `54957491ede07ad5d5015f5e69874f367113cf45`（含PR2867）；Simpler
`32dff953d07f6bd2aacab8532860f28aca6df931`及已记录的退出兼容补丁。
复用现有二进制及对应干净SDK；进程级退出版本门禁调整保留真实清理hook。
这不是未经修改的纯上游软件栈。

relative L2是输出差值L2/参考L2，表中百分数乘100；不是token正确率。
Graph延迟单位ms/attention forward，排除编译、预热、capture、cache恢复；插桩诊断不用于性能。

## 性能总表：只在同一实验组内比较

|实验组|代码/阶段|8K原生|8K CSA|128K原生|128K CSA|方法|
|---|---|---:|---:|---:|---:|---|
|G1|direct overlay|0.5925|0.5746|0.7956|0.8757|6轮×30 replay|
|G2|40参数|0.592708|0.580760|0.794936|0.839502|10轮×100|
|G2|metadata50|0.590550|0.576239|0.776765|0.852211|同卡串行|
|G2|direct-consumer|0.584183|0.573062|0.776335|0.815882|同卡串行|
|G3|O-proj实验旧版|0.580237|0.577603|0.799445|0.810254|12轮×100三路轮换|
|G3|O-proj实验新版|0.580237|0.585628|0.799445|0.819125|同进程|
|G4|精确父提交362073|0.567689|0.576975|0.795839|0.829223|12轮×100三路轮换|
|G4|精确新提交39993|0.567689|0.590259|0.795839|0.836976|同进程|
|G5|39993基线|0.584411|0.676060|0.788136|0.903950|12轮×100三路轮换|
|G5|Q/KV/Indexer候选|0.584411|0.672552|0.788136|0.938798|同进程|

G2最近一步128K改善约4.26%，但不能外推为全流程收益。
G4 O-proj修改代价为8K +2.30%、128K +0.93%；G5候选为-0.52%、+3.86%。
同一39993跨组延迟明显漂移，不能把不同组的最优值拼接计算收益。
权重变化问题解决、确定性开启后，以上需重新建立可验收基线。

## 精度与因果证据分级

|证据|历史观测|当前解释|
|---|---|---|
|G2三阶段CSA输出|两组输入逐元素相同|支持该测试下精简未改变输出，不代表对原生已对齐|
|G4整层39993|8K 1.5679%、128K 1.6919%|未通过allclose；同权重条件后来发现风险，暂停算法归因|
|G5候选整层|8K 1.2608%、128K 0.3780%|同上，不作为精度验收|
|O-proj同heads|T24/T6新版逐元素一致，旧版约1.61%|局部边界对齐证据，仅限这些输入|
|Q/KV逐步对齐|Q 0.6727%→0.03789%；KV 0.2861%→0.007072%|局部hook诊断，非整模型结论|
|同输入RoPE|逐元素一致|Indexer投影输出固定后的局部证据|
|同输入Hadamard+量化|query及scale逐元素一致|不证明上游输入一致|
|同输入Indexer top-k|集合100%一致|集合一致不等于排列完全一致|
|同输入attention|heads误差约0.0292%|仍非bitwise，需进一步验收|
|CPU权重审计|原生top-k后0差异，主scatter后509元素变化|该运行中CSA尚未执行，不能归为CSA引入|

## 失败、更正与未完成事项

- 初版O-proj FP32[8,1] tile行宽不符合PTOAS要求；改[1,8]后重新测试。
- 磁盘JIT缓存导致反复编译的一轮eager计时作废。
- Indexer诊断inline Out签名导致编译失败，改普通Tensor引用并在顶层导出后重测。
- 权重快照最初仅保存在NPU，存在被覆盖风险；后续立即复制到CPU并重新检查。
- 首轮scatter过滤对照误读updates作为slots，不能用“无变化”作根因证据；新增整数indices断言后重跑。
- 未显式确定性、权重不变性失败的整层数据，不再用于宣称精度或生产性能通过。
- 当前等待：正确负slot对照；精确compact行数的隔离确定性Graph。后者绕开多余compact行，只用于隔离，不能代替生产修复。
- 新capture选项仅配置测试完成。仍需compact动态边界、补位、dummy、档位切换、请求退出、空rank验证。
- nalinaly固定历史对照不是同环境性能结果，其最新padding修正应单独评估，不能直接视为本分支已通过。

## 后续强制记录规范

每次有意义的精度、性能或回归测试，在对应代码提交时追加本文件或链接新的日期报告。
保留旧结果；更正必须注明旧结论为何失效，不能静默覆盖。

每条记录必须包含：

1. 代码完整commit SHA；dirty实验注明基点及diff/source SHA256，不能用分支名代替。
2. 模型版本/层、真实或合成输入、seed、TP/DP/EP、B/S、上下文、block、量化、确定性、Graph模式与实际档位。
3. 依赖版本、补丁、原生与CSA开关、fallback检查、权重不变性与有效cache槽检查。
4. 运行命令、warmup/轮次/replay次数、计时边界、原生同组基线与原始样本。
5. relative L2、max/mean绝对误差、阈值、是否通过；将局部和整层结论分开。
6. 失败、跳过、环境阻塞、未验证范围，下一步；exit0不等于数值通过。
7. 可公开的证据文件相对链接。不得提交凭据、用户绝对路径、内网地址或大权重。

### 新记录模板

```text
日期 / 测试编号：
代码SHA / dirty diff摘要与hash：
与哪个历史实验同口径：
模型、输入、环境、Graph档位、确定性：
命令与测量方法：
原生/CSA精度、阈值、权重与cache检查：
原生/CSA延迟及原始样本：
结果：通过 / 失败 / 阻塞 / 仅诊断
限制、更正、证据链接、下一步：
```

## 历史详细记录

以下为当时报告的归档摘录，保留数值与失败过程。涉及旧精度结论时，以本文开头的更正为准。
本机路径、调度任务标识已去除；这些归档不是一键可复现实验包，早期overlay没有完整Git身份。

### 归档 1：dsv4-native-direct-ab-20260923

> 历史报告；如与最新更正冲突，以本文结论与证据分级为准。

### 原生 vLLM-Ascend 与精简 CSA 对拍结果

#### 结论

两组真实权重对拍完成。当前 CSA 最终输出尚未对齐原生，relative L2 约1.83%；
图重放在8K仅有约3%的单次实验收益，128K反而慢约10%。不能据此宣称已获得可用的整模型加速。

#### 测试范围

- 权重：`<运行环境路径>`，第2层C4注意力的真实全部权重。
- 单卡 TP1，B4，S6，24个query token；结束长度8192和131072，原生128槽物理页。
- hidden和历史KV为固定seed=62的合成输入；双方独立相同缓存快照，保留原生共享allocation别名。
- 测量边界：同一 `AscendDSAImpl.forward`，环境开关0/1；每次断言CSA调用计数增量正确。
- metadata类型/QLI v2/压缩metadata调用来自当前0.29原生接口。每次forward重新建立context，
  包含compact compressor metadata生产；不包含scheduler、SAS/QLI metadata builder、MoE、采样、通信。
- 因果S6 attention测试，不是完整DSpark出5验6流程；没有测接收率3.8、EPLB或DP16吞吐。
- 旧official-l3量化格式与当前kernel合约不符，因此未将其回退原生当作CSA结果。
- 当前单层已有精度差异，未继续提交完整模型任务。

#### 性能

单位ms/attention forward。5次预热；eager各30次，交替路径顺序；graph各6轮×30次，
轮次交替。编译、capture、缓存恢复不计入时间。graph为固定metadata/位置重复重放，非序列推进。

|上下文|原生 eager 中位数|CSA eager 中位数|原生 graph 中位数|CSA graph 中位数|CSA图延迟变化|
|---|---:|---:|---:|---:|---:|
|8K|2.226|69.891|0.5925|0.5746|-3.02%|
|128K|2.508|69.026|0.7956|0.8757|+10.07%|

8K单轮图均值范围：原生0.5898～0.6017ms，CSA0.5717～0.5979ms。
128K：原生0.7921～0.8033ms，CSA0.8553～0.8950ms。
只有一组正式实验，8K的3%不应解释成稳健的普遍收益。

eager的约69ms是在确认每个进程只编译一次后测得，包含Python/JIT分派和检查开销，
不能当作kernel纯设备时间。图重放避开这部分Python分派，因此差距明显缩小；
尚未对69ms做CPU逐函数profile，不把原因全部归到metadata适配层。

#### 精度

比较当前调用实际写入槽位，不用大块未变历史稀释差异。以下为relative L2：

|项目|8K|128K|
|---|---:|---:|
|最终attention输出|0.018278026|0.018362285|
|raw KV|0.0028605651|0.0028991177|
|压缩KV|1.1290598e-06|0|
|主压缩state|3.043687e-07|2.9526334e-07|
|内压缩state|3.1202082e-07|3.1876741e-07|
|INT8 index key|0.0063260921|0.0063513765|
|index scale|0.0026132355|0.002856495|

- 输出最大绝对差：8K 0.01953125，128K 0.021484375；cosine均约0.99983。
- 两组输出均不通过 `torch.allclose(rtol=1e-2, atol=1e-2)`。
- index key最大差为1个INT8量化级；该INT8 allclose仅是诊断，最终以输出差异判断。
- 主/内state及压缩KV接近一致；rawKV和index量化边界仍有差异，
  不能仅凭当前数据确定最终1.83%误差的唯一根因。
- graph replay先将输出填NaN，确认实际写回且全部finite；输出保护区通过。
- graph输出relative L2分别为1.8002%和1.8362%，精度问题同样存在。

#### 版本和证据

CANN9.2.0-beta.2，Torch2.10.0+cpu，Torch-NPU2.10.0.post4，vLLM0.29.0；
vLLM-Ascend为当前native-direct源码overlay，不是未修改的远端commit。
PyPTO `54957491ede07ad5d5015f5e69874f367113cf45`（含PR2867）；
Simpler32dff953d07f6bd2aacab8532860f28aca6df931，host二进制带先前退出兼容补丁。
未升级/重编译上述依赖。测试进程放宽2.10.post4退出版本门禁，保留真实清理hook。
使用同revision干净SDK读取头文件，二进制链接回原build/lib；PTOAS0.63。

任务：

- 8K：`[调度标识省略]`，completed exit0。
- 128K：`[调度标识省略]`，completed exit0。

exit0表示测试流程完成，不表示数值通过；JSON里accuracy_pass均为false。
完整结果：`8k.json`、`128k.json`；脚本：`ab.py`、`run.sh`。
早期8k-v7.json开启磁盘缓存导致反复编译，其eager计时作废，正式结果使用进程内缓存。
环境问题和处理方法同步到 myskill/pto227-vllm-cann92/references/native-direct-ab-20260923.md。

### 归档 2：dsv4-adapter-graph-20260923

> 历史报告；如与最新更正冲突，以本文结论与证据分级为准。

### 适配层精简前后的 Graph 对比

#### 配置与范围

同一卡串行六组；完整模型DeepSeek-V4-Flash-0731-w8a8第2层真实C4权重，
TP1、B4、S6、128槽物理页，8K与128K历史。固定seed62的合成hidden/history。
只统计NPUGraph replay，预热5次、每路径10轮×100次，轮次交替native/CSA顺序。
编译、预热、capture、缓存恢复均不计时；每个版本各自捕获原生和CSA图。
每次capture真实执行原生compact metadata生产；不包含scheduler、MoE、采样和通信。
这是固定attention步的微基准，不是整模型或完整DSpark生成。

#### Graph延迟（ms/forward）

|版本|8K原生|8K CSA|128K原生|128K CSA|
|---|---:|---:|---:|---:|
|精简前：40参数|0.592708|0.580760|0.794936|0.839502|
|第一步：原生metadata，50参数|0.590550|0.576239|0.776765|0.852211|
|当前：进一步精简消费端|0.584183|0.573062|0.776335|0.815882|

#### 解读

- 最近一步（metadata50→current）：8K CSA延迟降低0.55%，但原生也降低1.08%，无法认定收益；
  128K CSA降低4.26%，原生只降低0.06%，支持本次消费端精简有约4%的局部收益。
- 整轮（before40→current）：8K CSA降低1.33%、原生降低1.44%；128K CSA降低2.81%、
  原生降低2.34%。因此跨整个序列的净收益很小，不能声称适配层精简带来了明显整体加速。
- 当前128K CSA仍比同组原生慢约5.09%；8K则约快1.90%。
- 这是一次同卡串行实验；10轮反映进程内波动，不等同于多次独立重复。

#### 精度保持情况

三阶段保存的首次执行CSA输出（outputs.pt）在8K、128K下均逐元素完全一致，
相对before40的relative L2=0。精简没有改变本组输入的结果。
各组graph输出相对原生的relative L2仍约1.83%，不通过rtol=atol=1e-2；
该已知精度问题没有被本次精简修复。graph replay用NaN填充输出后检查全部finite，
并断言真实CSA调用计数；未把fallback计入CSA。

#### 源码与环境

- before40：远端backups/native-metadata-20260923，加载该目录的dsa_v1、pto_attn和pto_kernels。
- metadata50：backups/native-direct-20260923，使用当前dsa_v1及该目录的pto_attn、pto_kernels。
- current：当前src/vllm-ascend的native-direct版本。
- 通过attention包搜索路径选择备份，未覆盖现用源码。原生计算路径相同，hook/ABI按各阶段配套加载。
- 沿用上一轮CANN9.2 beta2、Torch2.10、torch_npu2.10.post4、vLLM0.29、
  PyPTO5495749及Simpler32dff95；退出门禁、干净SDK和提前加载OPP沿用已验证方式。
- 只编译各CSA kernel，未重编译或重装vLLM/PyPTO/Simpler。
- 任务[调度标识省略]，设备2，completed exit0。
- 脚本ab.py/run.sh；完整结果results/<version>-<start>/result.json。

### 归档 3：dsv4-precision-hooks-20260924

> 历史报告；如与最新更正冲突，以本文结论与证据分级为准。

### CSA 分模块精度诊断（2026-09-24）

#### 范围与方法

227、CANN 9.2 beta2、Torch-NPU 2.10.0.post4、vLLM-Ascend 0.29。
使用 DeepSeek-V4-Flash-0731-w8a8 第2层真实权重，B4/S6、8K上下文，
固定合成 hidden states 和历史缓存。不是整模型生成或接受率测试。

原生侧hook Q/QR、O-proj入口、indexer，并读取当前token的KV实际写入槽；
CSA私有诊断副本将内部中间tensor改为调用者提供的Out参数，保持形状、dtype和计算。
没有修改正式fork或生产overlay。不用插桩运行衡量性能，也未证明其输出与未插桩版逐位一致。

#### 实测

任务 [调度标识省略] 成功；完整数据 causal.json。
relative L2定义为差值L2/参考L2，下表已乘100转为百分比。

| 边界 | CSA对原生relative L2 |
| --- | ---: |
| QR INT8 | 0.6574% |
| Q（RoPE后） | 0.6727% |
| 当前KV | 0.2861% |
| heads（逆RoPE后、O-proj前） | 0.9859% |
| 最终输出 | 1.8123% |

topk集合重合率98.9909%。以上是累计差异，不能把相邻百分比相减作为模块贡献。

#### 因果实验

##### QA舍入边界

CPU FP32 QA matmul后直接做RMS/INT8量化，参考与CSA QR相差0.0305%。
同一参考先将QA舍入为BF16再做RMS/INT8，所得INT8 QR与原生逐元素一致；
scale相对L2约1.07e-7。源码也确认原生QA输出BF16，CSA保留FP32进入RMS。
这证明该输入上QA BF16边界是QR差异的明确来源，不代表原生在数学上更精确。

已在diagnostic-qa-round私有副本补两处QA读取的FP32→BF16(rint)→FP32，
任务 [调度标识省略] 已提交。后续已成功完成：QR relative L2降至0.01759%，Q降至0.28732%，最终输出仍为1.79919%；
相同heads的O-proj差异仍为1.60844%。补丁作用于预期边界，但整体精度尚未对齐。
证据见qa-round.json。

##### O-proj独立误差

将完全相同的CSA heads送入原生O-proj，CSA最终输出与这一路原生输出仍相差1.6088%。
所以O-proj存在独立差异。原生O-proj(CSA heads)与完整原生输出相差1.4888%，
也证明前段误差仍需处理。

源码：原生dsa_v1._forward_o_proj调用npu_transpose_batchmatmul后reshape为T×8192，
wo_b的W8A8DynamicLinearMethod.apply再调用npu_dynamic_quant。
CSA decode_o_proj按8组分别处理1024维OA结果、计算各组scale再汇总，且保留FP32中间值。
这是量化分组/舍入语义差异；尚未用独立补丁分离二者各自贡献。

#### 修复顺序与待验收

1. 完成QA单边界实验，确认QR及后续误差变化。
2. 对齐Q量化matmul、KV投影、RMS到RoPE之间原生BF16边界，逐处单因素验证。
3. O-A输出按原生BF16舍入；O-B使用完整8192维每token量化范围，保持跨组scale语义。
4. 若仍有误差，继续hook compressor、indexer、Hadamard和attention内部，区分topk离散跳变。
5. 同输入验证插桩与未插桩CSA，再以无hook Graph模式复测8K/128K精度和性能。

当前结果是误差定位证据，不是精度通过证明。不能要求融合内核所有FP32归约逐位等同原生，
但必须先对齐显式dtype、量化范围、scale和cache等语义边界，再制定可接受误差阈值。

### 归档 4：dsv4-oproj-global-20260924

> 历史报告；如与最新更正冲突，以本文结论与证据分级为准。

### O-proj 拼接后量化：精度和性能（2026-09-24）

#### 结论

TP1实现可行。相同heads、真实权重的T24/T6两个用例中，修改后的O-proj与原生输出逐元素一致，
旧版约1.61%的relative L2差异消失。完整attention层仍有1.56%～1.69%的误差，
未通过rtol=atol=1e-2。完整层同进程交替Graph测量显示约1.1%～1.4%的延迟代价。

#### 实现

基于main集成分支当前精简版本的私有运行副本；PR5只是kernel来源。
正式fork提交362073abab53c00ad53644b4ba583af38ce161af未修改。
`baseline`与`global-v2`的适配层及其余kernel一致，只修改`decode_o_proj_tp1`：

1. 等全部8组O-A计算完成。现有缓冲已经是T×8192，无Python拼接或量化。
2. 消费O-A时FP32→BF16(rint)→FP32，恢复原生BF16边界。
3. 每token对全部8×1024维归约amax，以同一个scale量化各组。
4. 保留各组INT8 matmul并行；先相加INT32 partials，再统一转FP32、乘激活scale和权重scale。
5. 显式padding生产者；O-B等待全局量化及padding完成。

这次对齐包含舍入边界、量化范围和累加顺序三项，不应把全部收益归因于单独改变amax范围。
各组INT32最坏累计绝对值不超过8192×128×128=134217728，不会溢出INT32。
只修改TP1实验路径，TP>1及原文件末尾旧语义golden helper未改，后者未用于此次验收；
因此当前目录是实验副本，不能直接当作已完成生产集成的补丁。

#### 测试环境和方法

- 227、CANN9.2 beta2、Torch-NPU2.10.0.post4、vLLM/Ascend0.29，复用已安装二进制。
- PyPTO `54957491ede07ad5d5015f5e69874f367113cf45`；Simpler运行库32dff953d07f6bd2aacab8532860f28aca6df931及已有退出兼容补丁，SDK使用对应干净副本。
- `<运行环境路径>`第2层C4真实权重，B4、S6、block128、TP1。
- 固定合成hidden/history，start_pos=8186/131066，对应末端8K/128K；不是整模型decode吞吐或DSpark接受率验证。
- 完整层无hook；原生、旧CSA、新CSA在同进程同卡预热/捕获，然后12轮×100次轮换replay。
- cache恢复、编译、预热排除在计时外；延迟为NPU Event均值/轮的中位数。
- 单模块用此前保存的相同CSA heads；覆盖T24和非8倍数T6；旧/新各自10轮×100次，并配对原生。
- 单模块各版本在独立进程测量，有基线漂移；完整层同进程测量是主要速度结论。

#### 同进程完整层结果

relative L2已乘100转成百分比；这是输出差异，不是token准确率。

| 上下文 | 旧CSA误差 | 新CSA误差 | 原生Graph ms | 旧CSA Graph ms | 新CSA Graph ms | 新比旧延迟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 1.8345% | 1.5598% | 0.580237 | 0.577603 | 0.585628 | +1.39% |
| 128K | 1.8362% | 1.6919% | 0.799445 | 0.810254 | 0.819125 | +1.09% |

按每轮配对计算变化的中位数分别+1.37%/+1.17%，与中位延迟比值一致。
新CSA比原生分别慢约0.93%/2.46%。两组输出guard通过；两种CSA均未通过整层allclose。
这些小幅性能差异属于本机本配置测量，不外推到DP16完整服务。

#### 相同heads的O-proj

| Token数 | 旧relative L2 | 新relative L2 | 原生ms（新进程） | 旧CSA ms | 新CSA ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 24 | 1.6088% | 0（逐元素一致） | 0.093674 | 0.185952 | 0.206403 |
| 6 | 1.6095% | 0（逐元素一致） | 0.091627 | 0.176213 | 0.196286 |

Graph与普通单次输出均检查，guard均通过。单模块延迟增加约20微秒；
独立模块计时包含共同的启动依赖节点，不能与整层延迟做简单相减。
两组逐元素一致不代表所有输入、层、batch和硬件都保证bitwise相同。

#### 失败、修复和证据

- 初版任务：[调度标识省略]、[调度标识省略]。
  新版quant_global编译失败：FP32的[8,1] full tile行宽4字节，不符合PTOAS32字节对齐。
  修正为[1,8]行向量，消费时reshape；数值算法不变。原始失败副本保留在global。
- 修正版任务：[调度标识省略]，exit=0，证据results-v2。
- 交替测量：[调度标识省略]，exit=0，证据results-paired。
- 首轮跨进程原生延迟有漂移，因此增加交替测量；不使用首轮绝对差作为最终性能结论。
- 脚本：ab_v2.py、module_v2.py、paired.py及run_v2.sh/run_paired.sh。
- 源码SHA256：source-sha256.json。最终三种输出保存在results-paired各子目录outputs.pt。
- 远端：`<运行环境路径>`；
  最终输出`logs/oproj-global-paired`。没有修改生产overlay、依赖安装或正式fork。

#### 后续

建议保留这条对齐方向，继续修Q/KV投影、RMS→RoPE等前段边界。
性能优化必须维持共享scale与BF16边界，可优化padding块写和全局归约调度；
不能恢复按组独立量化来换速度，否则会重新引入已确认的语义差异。

### 归档 5：dsv4-oproj-commit-20260924

> 历史报告；如与最新更正冲突，以本文结论与证据分级为准。

### O-proj 独立提交与原生 Graph 延迟复测

#### 提交

- 仓库：Leaf-Salix/vllm-ascend。
- 分支：dev/pypto-dsv4-csa-main-20260922。
- 新提交：39993cd43beaea042571b56e00c2c2b82665db57，已正常push。
- 父提交：362073abab53c00ad53644b4ba583af38ce161af。
- 范围：TP1 O-proj原生BF16边界、整行8192维量化、INT32汇总后反量化，以及对应golden和回归测试。
- 可单独执行`git revert 39993cd43`；本次没有执行revert。
- TP>1路径未改，QA舍入实验未混入本提交。

27项CPU测试通过，独立review通过。CPU测试验证golden/接口，NPU验证由实际kernel对拍提供。
增量pre-commit的代码检查通过；Gitleaks因本机缺少gitleaks及wget未执行，不能称全套hook通过。
本地工作树干净。两份测试源码均从Git提交直接导出，227侧38个文件SHA256匹配source-manifest.json。

#### 复测方法

[调度标识省略]，exit=0。
同进程、同卡，原生/父提交/新提交三组轮换；各12轮×100次ACLGraph replay。
排除编译、预热、cache恢复。NPU Event记录每轮平均延迟，表中取12轮中位数。

DeepSeek-V4-Flash-0731-w8a8真实第2层C4权重，B4/S6，TP1，block128；
固定合成hidden/history，start_pos8186/131066，覆盖8K和128K。
环境CANN9.2 beta2、Torch2.10.0+cpu、Torch-NPU2.10.0.post4、vLLM0.29.0。
精确运行期导入路径保存在results/*/result.json。
本次是单层attention Graph测试，不是整模型DP16服务TPOT或接受率测试。

#### 最新延迟

| 上下文 | 原生ms | 父提交CSA ms | 新提交CSA ms | 新CSA相对原生 | 新提交相对父提交 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K | 0.567689 | 0.576975 | 0.590259 | +3.98% | +2.30% |
| 128K | 0.795839 | 0.829223 | 0.836976 | +5.17% | +0.93% |

新CSA比原生慢约22.57/41.14微秒；其中本次O-proj修改相对父提交增加约13.28/7.75微秒。
各组round_ms均保留在JSON中。不同轮运行存在基线波动，应使用本轮内部配对比较；
不能把新CSA与原生的全部差距都归因于O-proj对齐。

#### 精度与边界

| 上下文 | 父提交输出relative L2 | 新提交输出relative L2 |
| --- | ---: | ---: |
| 8K | 1.8360% | 1.5679% |
| 128K | 1.8362% | 1.6919% |

两组输出guard通过，完整层仍未通过rtol=atol=1e-2。
相同heads的O-proj T24/T6逐元素一致证据见上一轮reports/dsv4-oproj-global-20260924。
本提交TP1 kernel与该已测global-v2函数AST一致，只补齐golden和回归测试。
后续仍需处理Q/KV等前段误差；本次保留对齐方向不代表整体精度已验收。

#### 产物

- results/8186、results/131066：result.json和三组outputs.pt。
- source-manifest.json：父提交、新提交、源文件SHA256。
- ab.py、paired.py、run.sh：实际复测脚本。
- 远端源码：`<运行环境路径>`。
- 远端结果：同ROOT下`logs/oproj-commit-39993`。

查询期间SSH曾返回Exceeded MaxStartups；退避后恢复连接，任务不受查询断开影响并正常完成。

### 归档 6：dsv4-qkv-align-20260924

> 历史报告；如与最新更正冲突，以本文结论与证据分级为准。

### CSA 剩余精度：逐模块原生数值边界对齐

> **最新更正：整层精度结果暂不能用于验收。** CPU固定快照确认，原生forward中top-k结束后、进入O-proj之前，indexer WqB权重最后一行末尾512字节出现509个元素变化。CSA尚未执行，原生与CSA读取权重已经不同。以下历史误差表保留作定位证据，不再视为严格同权重A/B结果；Graph精度也需增加权重不变性检查后重测。

#### 范围与状态

正式分支保持在 `39993cd43`；本轮均为独立诊断副本。复用现有 PyPTO/Simpler/vLLM 二进制，仅新增 kernel JIT。

DeepSeek-V4-Flash-0731-w8a8 第2层 C4 真实权重；固定合成输入与历史 cache，B4/S6、8K、TP1、block128。不是整模型输出质量测试。CANN9.2 beta2、Torch-NPU2.10.0.post4。

#### 已完成逐阶段 hook

下表均为 relative L2 百分数；top-k 为集合重合率。步骤累计增加原生舍入边界。top-k/heads 受近似排名及归约影响，不能将相邻误差相减当作模块贡献。

|阶段|Q|QR|KV|heads|最终输出|top-k重合率|
|---|---:|---:|---:|---:|---:|---:|
|oproj|0.672720%|0.657385%|0.286057%|1.008427%|1.497364%|98.950195%|
|qa|0.287317%|0.017588%|0.286057%|0.888207%|1.393507%|99.218750%|
|q_mm|0.110909%|0.017588%|0.286057%|0.945208%|1.434108%|99.088542%|
|q_rms|0.037894%|0.017588%|0.286057%|0.885981%|1.390674%|99.218750%|
|kv_mm|0.037894%|0.017588%|0.082274%|0.887704%|1.345977%|99.088542%|
|kv_rms|0.037894%|0.017588%|0.007072%|0.821178%|1.297674%|99.218750%|
|idx_mm|0.037894%|0.017588%|0.007072%|0.871004%|1.303611%|99.121094%|
|idx_hadamard|0.037894%|0.017588%|0.007072%|0.696764%|1.188518%|99.430339%|
|idx_weights|0.037894%|0.017588%|0.007072%|0.715515%|1.174220%|99.414062%|

- QA/Q-B/Q RMS/KV投影/KV RMS：补上原生 BF16 中间边界，Q误差从0.6727%降到0.03789%，KV从0.2861%降到0.007072%。
- Hadamard：移除初始化时折叠的归一化，改为原生矩阵乘→BF16→归一化→BF16；该用例 indexer key 与 scale 逐元素一致。
- Indexer weights：投影→BF16→缩放→BF16。最终误差降至约1.1742%，仍不能判定精度通过。

#### 已提交后续验证（等待设备）

- 同输入 attention 对拍与 attention输出 BF16→inverse RoPE：`[调度标识省略]`。
- 进一步对齐 indexer query scale、weight输入、系数乘积和QK打分 FP16 边界：`[调度标识省略]`。
- 去掉全部hook/中间输出后，同进程原生/39993/candidate Graph，8K与128K：`[调度标识省略]`。

native arch22 QK INT32先乘1/1024再舍入FP16并ReLU，随后half权重系数乘法归约；候选保持该内部score尺度，不能直接与旧版未缩放score逐值比较。query scale的FP16舍入只改变存储的反量化scale，不改用来产生INT8 query的量化scale。

候选Hadamard ABI与旧版不同：旧版传已归一化矩阵，新版传原生矩阵。Graph脚本分别缓存两套权重字典，capture前选择，计时不含准备操作。

脚本对各诊断子例记录JSON error，不能仅凭wrapper exit=0宣布所有子例成功。源码SHA256见source-manifest.json。

#### 同输入 Attention 对拍（已完成）

固定 CSA 的Q/KV/top-k，分别调用CSA与原生attention：

|版本|同输入heads误差|原生attention消费CSA输入，相对原生heads|整层误差|
|---|---:|---:|---:|
|idx_weights|0.088694%|0.711068%|1.174220%|
|attn_rope|0.029262%|0.711068%|1.137392%|
|idx_score|0.029182%|0.718157%|1.154416%|

因此当前attention内部自身差异已经较小，较大的heads误差主要随上游输入进入；不能把整体误差全归因于attention实现。FP16打分对齐尚未明显改善8K top-k，需要进一步hook query/weights和同输入top-k。

#### 无hook Graph 对拍（已完成）

同进程三路轮换，12轮×100次重放，重置cache/编译/权重准备均在计时外。candidate包含idx_score全套对齐。

|结束长度|原生延迟ms|39993延迟ms|candidate延迟ms|candidate相对39993|candidate相对原生|39993误差|candidate误差|
|---|---:|---:|---:|---:|---:|---:|---:|
|8192|0.584411|0.676060|0.672552|-0.52%|+15.08%|1.5598%|1.2608%|
|131072|0.788136|0.903950|0.938798|+3.86%|+19.12%|1.6919%|0.3780%|

两种长度的输出均finite、guard通过，但allclose_1e-2仍false。并行存在其他设备任务；以本次同进程共同原生基线作比较，不与旧任务绝对延迟直接拼接归因。

新增indexer中间输出诊断首轮因inline Out签名改变返回绑定而未编译；改为inline普通Tensor引用、只在顶层Out导出，重跑[调度标识省略]。失败仅在诊断插桩，Graph候选不含此改动。

#### Indexer同输入因果隔离（已完成）

任务110131、110421、110903、111134均完成，对应JSON见results/attention-idx_*。

1. 当前query/scale/weights送入原生top-k：与CSA top-k集合100%一致；原生自身重复运行100%一致。weights逐元素一致。
2. 以原生QR/scale替换CSA的QR/scale后，indexer query仍有1.3521% relative L2；QR原本仅token10一个INT8元素不同，因此不能归因为上游QR。
3. 同一个RoPE后张量进入Hadamard＋动态量化，INT8 query和scale逐元素一致。
4. 相同投影输出进入RoPE，结果逐元素一致。
5. 相同原生QR/scale进入indexer投影，输出仍相差1.08598% relative L2。

因此本轮已将主要残余误差缩小到indexer INT8投影/反量化，后续用CPU INT32精确乘法及两种反量化顺序核对（[调度标识省略]）。不能将上述同输入结论扩大为全层精度通过。

#### 关键根因线索：输入权重发生变化（2026-09-24 11:25）

CPU INT32参考：CSA输出与当前绑定权重精确参考逐元素一致；原生输出与其调用时权重参考仅约0.000626%误差。
原生调用的QR/scale与CSA注入值逐元素一致，但原生调用时权重与之后CSA绑定权重有509元素不同。

首次NPU快照诊断存在快照本身可能受影响的证据缺口；已通过立即 `.cpu().clone()` 固化基准重新审计，
任务 `[调度标识省略]` 成功，无JSON error。

|快照时点|相对调用时CPU权重快照的不同元素数|
|---|---:|
|indexer rotate前/后|0 / 0|
|top-k前/后|0 / 0|
|进入O-proj前|509|
|O-proj后、原生forward结束后|509|
|CSA参数绑定后|509|
|CSA执行后|509|

变化位置集中 `weight[1023,7680:8192]`。当前可以定位发生区间，尚不能断言具体算子或越界机制。
原生自身用当前权重重复matmul也会与先前输出不同；这不是scale舍入顺序能够解释的问题。

进一步拆分native attention与inverse RoPE的任务已提交：`[调度标识省略]`，
脚本 `diagnose_weight_writer.py`，输出目录 `logs/qkv-align/writer-idx_proj_hook`。
11:25仍pending，前面有正在运行及排队的16卡任务。没有中断其他任务。

下一步必须先定位并处理权重变化，再用不可变权重快照和独立cache重新跑原生/CSA精度及Graph，
不能为凑齐误差阈值继续修改CSA舍入，也不能把受污染的整层误差报告为精度已通过。

#### 用户追加确定性要求

后续native精度对拍强制 `HCCL_DETERMINISTIC=true` + `set_deterministic_level(1)`，
初始化完成回读level=1；排队writer和后续Graph均纳入。旧结果没有显式设置，
不构成确定性基线。待writer运行后首先核验日志 `DETERMINISM_VERIFIED`，再看权重变化是否复现。

#### 同事补丁db6f3e1核对

已读取nalinaly提交db6f3e1094cffd54e9e2ff405b81544f6eaf8585：QLI长度/余数更新应在metadata cache命中判断前；scatter batch-copy/长行路径应跳过负linear index。当前源码两处均未包含修复。
当前手工metadata测试直接填充QLI长度，没有经过共享builder缓存路径，第一项不是本用例首要解释。
负scatter slot则是强相关候选：单流indexer.forward在top-k后调用主compressor scatter，恰在已确认的权重变化区间。
更正之前范围遗漏：该窗口不止SAS/输出复制/inverse RoPE，还包含compressor scatter。
排队writer已增加scatter负slot计数及调用前后CPU权重快照；尚未应用补丁或编译，根因待运行证据确认。

#### scatter包装参数错误更正

首轮writer包装误将(cache,slots,updates)解释成(cache,values,slots)，负slot统计实际读了updates；过滤组不能用于因果判断，撤回该组结论。原生未过滤调用保持原始参数顺序，其权重时序仍可参考。已修正签名与调用顺序，增加整数indices断言，重跑[调度标识省略]，两组均仅native、确定性开启。

#### 14:46确定性复测排队

- [调度标识省略]：正确scatter(cache,slots,updates)签名的原生/过滤负slot对照，两组独立进程，仅native，level1+HCCL确定性。首轮过滤证据无效已撤回。
- [调度标识省略]：graph-exact-deterministic，8K/128K，native/39993/candidate三路；将固定fixture的num_compressed_tokens设为实际boundary数，避免多余compact行，增加QA/QB/indexerQB/KV/OA/OB权重全量CPU快照不变性断言。
- 后者仅隔离固定形状微测，不是负slot kernel修复，也不是生产padding或capture档位验证；新platform选项尚未部署。
- 截至14:46均pending，16卡被CI占用，前面还有16卡任务。

## 2026-09-24：负 scatter slot 因果确认与 kernel 修复

### 修正包装后的确定性对照

两组均回读 `deterministic_level=1`、`HCCL_DETERMINISTIC=true`。输入为上述第2层真实权重，固定B4/S6。

|检查点|原生未过滤|仅过滤负slot更新行|
|---|---:|---:|
|主compressor scatter前，indexer权重不同元素数|0|0|
|主compressor scatter后|509|0|
|forward完成后|509|0|

主compressor更新为 `[10,512]`，两个slot为 `[-1,127]`，在128槽页布局下线性索引为-1。
index key/scale scatter也含相同两个无效slot，但其前后被监控的indexer权重没有变化；这不证明其他allocation未受影响。
本对照确认本次权重污染的scatter原因，与nalinaly的db6f3e1负索引修复一致；不代表整层数值误差全部消除。

### 原生metadata契约更正

尝试将compact行数从10缩到实际闭合边界数8，导致 `aclnnCompressor` 在8K首次warmup失败：
`ropeSin shape dim 0 ... should be ... 10, but got 8`。128K未执行，没有新有效精度或延迟。
原生要求保留10行容量，其中8行有效、2行sentinel；应修复scatter跳过sentinel，不能缩减metadata容量。

### 最小kernel修复及待验收项目

在 `scatter_nd_update_hp.h` 的批量复制和长行分片两条路径中，于输出地址计算前检查signed线性索引。
批量路径跳过写入仍推进源偏移；分片路径在事件等待及ping-pong切换之前跳过，保留同步协议。
适用范围是负线性索引padding，未扩大为任意非法多维坐标或正向越界保护。

16个scatter编译变体已单独生成，复用现有host tiling、Torch扩展、vLLM、PyPTO和Simpler。
测试使用独立完整OPP目录，原安装产物保留。静态独立审查通过。
新增40组NPU回归：混合/全padding，INT32/INT64索引，对齐/非对齐/长行，连续/真实stride0 view，
每组检查eager及3次Graph replay后的整个backing，包括前后guard页。
上述40项回归全部通过；无Python过滤器的真实层writer中indexer权重全程不变。后续Graph对拍也已完成，结果如下。

### G6：修复scatter后的确定性Graph对拍

所有三路共用修复后的scatter；正式CSA算法仍取39993，最新候选仍为未提交数值对齐实验，
包含Q/KV/Indexer的BF16、FP16舍入与Hadamard边界调整。此表不代表这些实验已合并到正式分支。

同卡串行，真实第2层权重，TP1/B4/S6/block128；每路径12轮×100次Graph replay，
轮换计时次序，编译、warmup、cache恢复、CPU权重检查均在计时外。
native开启level1及HCCL确定性，QA/QB/indexerQB/KV/OA/OB六组权重在各路径输出后及计时后均未变化。
8K与128K均保留原生compact容量10，结果JSON的compact_rows=8仅表示有效闭合边界数。

|结束长度|原生ms|正式CSA 39993 ms|对齐候选ms|候选比原生|候选比正式CSA|
|---|---:|---:|---:|---:|---:|
|8K|0.576343|0.614856|0.619637|慢7.51%|慢0.78%|
|128K|0.795113|0.846685|0.871587|慢9.62%|慢2.94%|

|结束长度|正式CSA relative L2|候选relative L2|候选max abs|正式CSA allclose|候选allclose|
|---|---:|---:|---:|---|---|
|8K|1.4350%|0.4447%|0.00988770|false|true|
|128K|1.6919%|0.3780%|0.00439453|false|true|

allclose阈值为 `rtol=atol=1e-2`。通过这个阈值不等于逐元素一致，也不等于整模型精度验收。
候选最终输出仍有约0.38%～0.44% relative L2残差，性能仍慢于共同原生基线。
两组输出finite、输出guard通过，权重不变性通过。旧受权重污染的数据继续保留为诊断历史。

### QLI共享metadata的原生接口修复

独立审查确认 `_build_qli_metadata` 在共享cache hit时未更新当前builder独有的
`qli_seqused_k` / `qli_cmp_residual_k`。将这两组buffer的div/remainder刷新移到缓存判断外，
仍复用metadata生成结果，保留持久buffer地址和原接口，不增加CSA参数或适配包装。

CPU回归覆盖两builder共享cache、下一步长度及请求数变化、INT32/INT64输入、buffer地址不变；
本机Torch2.12与目标环境Torch2.10均2项通过。该测试执行真实方法，mock硬件metadata算子，
不等价于整模型多builder/生产Graph验证；G6直接构造metadata，不覆盖该builder缓存命中路径。

scatter修复与QLI刷新已部署到私有overlay，原二进制及源码已备份。
尚待验证：同一捕获桶内请求数变化、padding/空rank、不同请求起始位置余数、跨步compact行数变化。
默认6倍数capture选项仍是独立工作区修改，尚未部署，本次数据不能替它提供生产验证。

### 本轮静态检查

Ruff、format、codespell、typos与diff whitespace检查通过。Gitleaks因环境缺少gitleaks及wget未运行，
应视为工具缺失，而非扫描通过。

### G6证据指纹

- `graph-baseline` Python文件SHA256清单的规范JSON摘要：`f9326ca9da9b32467b00e2597f29479890fe8d44d78e2c6434bdc2811b9cc2b8`。
- `graph-candidate` Python文件SHA256清单的规范JSON摘要：`7df6108a34bad0a7446572962f963d9ed203c4cbffa2073a2c1aebd841284c27`。
- `131066/result.json` SHA256：`6ca16fdb78c48a0096d89c8abdda70f7120136d95c8689447115d1835bd64d73`。
- `8186/result.json` SHA256：`2a3455eabe21d088374524f2fee081e7546ba1a2d950bd950c9ecbc29f02a7f7`。

## G7：单因素舍入与固定heads的O-proj归因（2026-09-24）

基线为 `6c4236be9ca9bc6b86c86ea76aaf265474b237bb`，未将实验数值修改合入正式CSA。
26组NPU Graph对拍完成，TP1/B4/S6/block128、8K/128K、真实C4权重和合成输入，原生确定性开启。
完整设计、每组QR/Q/raw KV/heads/output/cache指标及证据hash见
[单因素报告](DSV4_CSA_SINGLE_FACTOR_20260924.md) 与 [完整JSON](DSV4_CSA_SINGLE_FACTOR_20260924.json)。

- Q-A BF16令QR不同元素数从1397/1355降至1/0。
- 相同QR输入下，Q-B与Q RMS两个舍入边界对齐后，Q relative L2为0.000986%/0.004372%。
- KV projection与KV RMS边界对齐后，raw KV relative L2为0.007072%/0.009388%。
- 固定heads的O-A BF16+8192量化，其OA/INT8/scale与原生逐元素一致；最终输出仍保留报告中的小残差。
- 跨组输入与原生输出hash、无关分支逐元素不变、全部权重不变断言通过。

这些是分模块因果对照，不能宣称所有因素合并后的整层精度已通过。本轮没有正式延迟结果，
插桩实验kernel不能与G6性能相混；Indexer仍有差异，动态padding/生产metadata刷新仍未验证。
