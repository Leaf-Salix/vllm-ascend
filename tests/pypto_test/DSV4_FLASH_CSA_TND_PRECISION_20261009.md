# 独立 TND 精度版（2026-10-09）

## 当前结果（v23）

- 8K/B4/T11、8K/B4/T24、8K/B16/T60、128K/B4/T24：actual heads、最终输出与全部cache/state、TopK均与确定性Native逐bit一致。
- history96/B4：与原v13精度版的独立heads及完整状态逐bit一致，保留短窗口原计算。
- 正式精度包仍显式复用Native HC/norm/O-proj，PTO执行Q/KV、compressors、Indexer与attention；保持真实TND，不要求补到S6。
- 性能包23文件逐字节未改；当前精度版较Native慢，仅验证单个C4层，未外推整模型或DP/EP16吞吐。
- 增量UT26项通过。下文按阶段保留历史失败、定位与改进；早期“未对齐”描述仅对应当时阶段。

## 身份、备份与目标

工作分支：`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
性能版基线：`98eadc516a785e33d51dc0714bf23dd34390ece8`。
备份分支：`backup/pypto-dsv4-csa-tnd-performance-20261009`，已正常推送到 Leaf-Salix。
性能包 `deepseek_v4_flash_csa` 的 23 份 Python 源码逐文件 SHA 保持不变。
独立精度包通过 `PTO_CSA_VARIANT=tnd_precision` 选择；旧 `precision`/默认入口仍指向 BSH 包，旧 `performance` 行为不变。

本轮目标是尽量逐 bit 对齐确定性 Native。候选与自身重复运行一致不等于与 Native 一致；
相同输入某一模块逐位一致也不能推断整层或整模型一致。
TND 使用 `query_start_loc`/`token_request` 的真实请求边界，总 token 数为各请求长度之和，
不要求补到 B×6。每请求上限 6 是当前 DSpark 的支持范围。

## 可复核环境与协议

CANN 9.2.0-beta.2，ATB 9.2.0-beta.2 cxxabi1；Torch 2.10.0、Torch-NPU 2.10.0.post2；
vLLM 0.25.1，vLLM-Ascend 0.25.1rc1 lineage；PyPTO `3e87a843`、Simpler `a54c0509`、PTOAS 0.66。
复用原框架及二进制，隔离 Python overlay，不重装框架或改服务器全局环境。

单卡 A3，真实 `DeepSeek-V4-Flash-0731-w8a8` 第4层权重，seed1024 的合成 hidden/history。
比较范围为 HC pre + norm + CSA + HC post；尚不是整模型 DP/EP16、EPLB、实际接收率/吞吐验收。
确定性 level1、`HCCL_DETERMINISTIC=true`、atomic0；Native 真实 static+SuperKernel 的调用与设备 profile 均核实。
各臂独立 cache、operator 注册名和图 owner；PTO 回退明确禁止。

硬门禁包括：compiled/raw replay、连续重放、100次连续重放、恢复初态后重放的全部输出/状态逐 bit 不变，
非有限值与保护区检查。诊断 probe 必须与被测候选的全部八类输出/状态逐 bit 一致。
计时采用平衡次序，100次连续 replay 轮均值与单次 replay 分开，profile 单独采集。

## 逐阶段结果：8K/B4/T24

下表为各阶段对同轮 Native 的结果。百分比仅帮助量化；精度是否达标以 bit/ULP 为准。

| 阶段 | output relative L2 | 不同 BF16 元素 | ULP P99 | TopK 集合不同的行 | 替换 index 数 | 完整稳定性门禁 |
| --- | --- | --- | --- | --- | --- | --- |
| v1：整 token O-proj 量化 | 0.003300463547 | 160660 | 26 | 24 | 78 | 通过 |
| v2：Q/KV 舍入及规约对齐 | 0.002889308453 | 143570 | 21 | 23 | 53 | 通过 |
| v3c：512 attention + 索引 task 边界 | 0.002889347073 | 143517 | 21 | 23 | 53 | 通过 |
| v4c：HC/norm 使用 Native 边界 | 0.002840771464 | 138854 | 21 | 23 | 51 | 通过 |
| v5c：v3c + Indexer 数值边界 | 0.002784942796 | 133127 | 20 | 21 | 50 | 通过 |
| v6c：Native HC + Indexer 对齐 | 3.200457187e-05 | 50 | 0 | 0 | 0 | 通过 |
| v7c：再对齐 Compressor 数值顺序 | 3.200457187e-05 | 50 | 0 | 0 | 0 | 通过 |
| v11：Native O-proj 边界复用 | 0 | 0 | 0 | 0 | 0 | 通过 |

### 已有模块归因证据

- v1 同 PTO norm 的 Q relative L2=0.00648299、KV=0.00284527；v2 同 norm 两者逐 bit 一致。
  Q 参考来自 Native eager 算子助手，不能称为导出了 SK 内部 Q。
- Native 助手的 factual KV 与实际 Native graph 写入的 raw cache 逐 bit 一致，验证了 KV 助手归因。
- v4c 的 raw KV 和压缩 KV cache 与 Native 全张量逐 bit 一致；state 的 FP32 仍有微小差异。
- v4c 的 Indexer key/scale 和 TopK 仍有差异。仅修正 O-proj 或 Q/KV 不能宣称整层对齐。

## 失败与修复记录

1. 初始注册失败：kernel 返回内部 tensor，违反 external alias 要求。修改返回外部输出；
   编译门禁加入与 `register` 相同的 `kernel_signature_for_program(entry.specialize())`。
2. 第一份诊断读取 `RopeDataProxy.keys()` 失败。改用 Native 真实 layer_name 索引；不归因于 kernel 精度。
3. v3/v3b 的初始与连续 replay 一致，但 reset 后输出变化，cache/TopK 不变。
   v3b 保存逐状态差异：仅 x_out 不同 267514 元素，relative L2=0.0208892。
   生成代码确认同 task 内 Scalar GM store→MTE2 load 中间没有 DCCI，kernel末尾才刷新。
   v3c 拆成独立 checked-index task，任务末尾 flush 后，planner 显式依赖并读取。
   生成代码、独立审查和完整 NPU reset 门禁均通过。

## 早期阶段待完成的精度检查

8K/B4 的 HC+Indexer、Compressor 修正已完成；继续检查 O-proj 及真正变长/128K。
当前上述阶段尚未达到整层逐 bit 对齐，不以 allclose 或百分比阈值替代目标。

## v8b：真实 Native SK heads 与 PTO O-proj 单因素诊断

实际 O-proj 输入（inverse RoPE 后 heads）786432 元素逐 bit 一致。
Native 观察臂与未修改 Native 的全部八类输出/状态逐位一致；PTO 导出臂与候选亦逐位一致。
观察对象的 shape=[24,64,512]、stride=[32768,512,1]、设备地址稳定；
改变 hidden 后 raw replay 使 heads 改变，恢复 hidden/cache 后 heads 及全部状态逐 bit 恢复。
Native 两臂实际 static+SK 及 profile 均通过，不用 eager helper 冒充 SK 内部结果。

PTO O-proj 直接消费 Native heads 后，仍有相同的50个整层输出差异，最大75 ULP；
因此本例剩余误差定位到 O-proj，不支持此前对 SWA/attention 的推测。
v7c 中所有 cache/state 和 TopK 已逐 bit 对齐；仍不能称整层逐 bit 对齐。

初始 v8 诊断注册失败，原因是导出函数在 Device scope 读取 Out tensor 的维度，
使 lowering 判定为 InOut。改为传入已有 token 数后完整 artifact ABI 门禁通过。
ABI 错误详情同时列出的 NZ 物理 reshape 是合法的，未修改框架或绕过门禁。
以后预检必须调用 `_compile_impl(..., _kernel_abi=abi)`，不能只用普通 compile/lower 代替注册门禁。

## 正式独立包与当前支持范围

- `PTO_CSA_VARIANT=tnd_precision` 选择新包；与性能包并存。
- HC pre/input norm/O-proj/HC post 复用当前 Native 层；Q/KV、两路 Compressor、Indexer、sparse attention 由 PTO TND 执行。
- 使用真实 Native metadata/cache/slot mapping；保持 C4、TP1、WIN128、每请求1…6、最大B64的当前契约。
- 固定规约，显式 atomic1 会被拒绝；仅支持已验证的 ringbuffer runtime，HBG 不宣称已支持。
- 24项CPU回归及增量pre-commit已通过；变长及128K已执行，差异见下表。

## v9/v10：WO-B 单因素与 WO-A 首差定位

v9 将 WO-B 两个 scale 先合并后乘整数累加值；完整门禁通过，但50个最终输出差异没有变化。
该候选未合入。没有直接复制 opus55 的 combined-scale 结论到当前环境。

v10 在不改变计算的诊断副本中导出 WO-A 的 BF16 输出、INT8 codes 和 FP32 scale：

| 比较对象 | 元素数 | 不同元素数 | 结果 |
| --- | --- | --- | --- |
| WO-A BF16 vs Native helper | 196608 | 31 | 最大绝对差1.52587890625e-5 |
| factual INT8 | 196608 | 1 | token21、col1051，差1 |
| 同 PTO WO-A 的 Native 动态量化 codes | 196608 | 0 | 逐 bit 一致 |
| factual / 相同 WO-A 的 FP32 scale | 各24 | 0 | 逐 bit 一致 |
| Native helper完整O-proj vs真实SK输出 | 98304 | 0 | 证明该助手输出边界保真 |
| PTO HC前O-proj vs真实SK输出 | 98304 | 463 | 最大311 ULP；不能用最终HC后的75 ULP代替此值 |

诊断臂与正式候选的所有输出/状态逐位一致；Native观察臂与未修改Native亦一致。
实际SK的heads与O-proj输出均验证compiled/raw、地址稳定、hidden扰动及恢复初态，完整replay门禁通过。
WO-A中间值来自Native eager助手，不冒充SK内部直接导出；该助手的完整O-proj输出已与实际SK逐bit一致。

首次分歧在 WO-A。唯一 codes 差异对应 OA=[21,1051]：PTO=0.0034027099609375，Native=0.00341796875。
相同 OA 下量化全部一致，故不能以再次修改量化规则解决该输入差异。
安装的 Native NZ transpose-batch-matmul 路径使用 `MM_CFG_K_SHIFT`，当前PTO自然K序不同；
真正复刻仍需获得对应形状的实际tiling，不能只根据宏名称猜K顺序。

v11精度候选让PTO发布真实TND heads，由同一Native层的 `_forward_o_proj` 消费，
PTO仍执行Q/KV、Compressor、Indexer与attention。该边界复用方案是混合精度实现，
不能写成“纯PTO O-proj已逐bit对齐”。性能版23份源码仍保持原SHA。

## v11：Native O-proj 边界复用，8K等长初次结果

v11移除融合入口中PTO O-proj的执行，由PTO以零数值转换的copy将group-major heads发布为
外部持久buffer `[T,64×512]`；service再调用同一原生impl的 `_forward_o_proj`，然后HC post。
root返回该external heads alias，adapter返回相同buffer。不会整CSA回退，也不会拿Native历史cache替换PTO写入。

8K/B4/T24的393216个最终输出、完整TopK、全部六类cache/state均逐bit一致：
relative L2=0、max_abs=0、maxULP=0。编译图、raw replay、100次连续replay、恢复初态重放、
保护区与Native实际SK profile门禁全通过。

同轮100次replay轮均值：Native429.26μs，性能control481.02μs，精度734.76μs。
精度版比Native慢71.17%，比性能版慢52.75%。本轮优先精度，不能写成保留了性能版的速度。
性能版的源码及备份均保留；此前性能表对应该性能包，不对应本精度包。

该结果仅为8K等长；真正变长T11、B16/T60以及128K尚在后续验证中。
切换示例：`PTO_CSA_VARIANT=tnd_precision VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_RUNTIME=tensormap_and_ringbuffer`。
通过正常模型service入口加载，precision依赖Native decoder layer。
旧 `dsv4_csa_single_layer.py` 的裸 `NativeCSACall` CLI缺少precision私有norm/headsbuffer，不能原样当精度版测试命令；
本轮使用service入口的受控ACLGraph对拍驱动。

## v11扩展验证：真正变长与128K

以下均完整执行Native实际SK、独立cache、无PTO回退、ACLGraph及连续/恢复重放门禁：

| history / requests | 实际T | 最终BF16不同数 | output relative L2 | max ULP | cache/state/TopK |
| --- | --- | --- | --- | --- | --- |
| 8K/B4，6/6/6/6 | 24 | 0 | 0 | 0 | 全部逐bit一致 |
| 8K/B4，6/3/1/1 | 11 | 89 | 5.329045244e-5 | 19 | KV与TopK一致；两类FP32 state不同 |
| 8K/B16，1/2/3/4/5/6/6/6/1/2/3/4/5/6/4/2 | 60 | 1172 | 8.326564486e-5 | 106 | TopK/raw/indexer一致；compressed有1个1ULP差，FP32 state不同 |
| 128K/B4，6/6/6/6 | 24 | 224 | 5.552246232e-5 | 55 | 全部cache/state/TopK逐bit一致 |

T11并未补到24，T60也未补到96。运行通过证明真实TND可执行，不代表精度目标全部达到。
两类FP32 state的relative L2约3e-7；接近零处ULP可很大，不能仅凭小L2称它们逐bit一致。
128K等长cache已全一致仍有最终差，不能简单把state差归因成当前输出误差。
因此继续真实heads阶段定位，并对Native形状相关compressor列分块做独立候选验证。

原扩展任务因CANN静态编译把长目录编码进文件名，超过255字节，在Native阶段失败；
改短输出目录后原门禁全部保留，重新执行成功。没有降成eager或取消SK检查。

## v12/v13：形状相关 Compressor K-shift 修正（已加入正式包）

参考本地 `dev/pypto-dsv4-csa-tnd-opus55-20260924` 的形状相关列分块思路，
再逐行核对当前Native `CompressorKernelPerf::SetBaseSize`：非等长默认dBase64；
等长且总token数不超过 `128*(24/(head_dim/64))` 时，main512采用32、indexer128采用16。
这些列块决定K循环起点。精度包此前将32/16固定用于所有形状，改为读取live设备query bounds判断等长，
避免把判定固化进capture；没有复制opus55整个kernel，也没有改变pool/量化/布局。
当前Native校准限A3/AIC24、C4、T≤384；正式runtime对未校准的非A3设备明确拒绝。

| 单因素 | T11主FP32 state不同数 | T11 indexer FP32 state不同数 | 最终输出不同数 |
| --- | --- | --- | --- |
| v11固定列分块 | 18890 | 4387 | 89 |
| v12仅main按形状选择列分块 | 0 | 4387 | 89 |
| v13再处理indexer列分块 | 0 | 0 | 89 |

v13完整NPU复核：T11、B16/T60的全部六类cache/state、TopK逐bit一致；
T60先前compressed cache的1个1ULP差亦消除。T60最终输出仍1172不同/max106ULP。
8K等长B4重测的整层输出及全部cache/state仍逐bit一致。
这些结果证明本轮state差异不是对应的当前step最终输出差异来源，不能将两者混为同一根因。

## v14：真实Q与heads阶段定位

观察Native实际 `npu_sparse_attn_sharedkv` 的Q输入，未增加Native tensor计算；
PTO独立诊断副本导出相同Q。Native Q及heads引用均通过compiled/raw、hidden扰动和A/B/A恢复检查，
Native观察臂/未修改Native、PTO导出臂/未导出候选的全部八类输出/状态均逐bit一致。

T11实际Q的360448元素全部逐bit一致；heads同样360448元素，仅 `[token10,head63,dim215]`
一处差1ULP（原始位型差3bit），绝对差3.0517578125e-5。该维度位于NOPE区，不参与inverse RoPE。
两边相同实现的Native O-proj helper分别逐bit复现各自捕获的输出边界，
一个heads舍入边界差通过动态量化放大成528个HC前投影差、89个最终输出差。
因此本例末位差定位到attention本体，不再继续改Q路径或猜inverse RoPE。
这不能证明其他128K/B16用例也只有同一处差异。

Q导出预检最初因inline reshape无法推断参数metadata失败；改为显式SSA变量后完整ABI/lower/C++通过，
未把该预检问题归为NPU数值错误。

## attention后续候选：生成指令门禁

- v15把PV的常数改写成literal128，复核才发现原值就是128，是语义no-op。
- v16把QK的L1加载显式分K128，生成指令核对显示基线本来已是K128 MAD，不能称为改变硬件累加K。
- 二者合并任务仍pending时已撤销，未运行，不把它们记为有效AB或优化。
- 基线PV的N512使编译器实际生成Left64×32/Right32×512；Native MM2使用N128/K128。
- v17只拆PV输出为4段N128，生成CPP已确认Left64×128/Right128×128，
  每段FP32 accumulator跨4个K块保留，全部输出写回后才发PV_READY；完整ABI/C++和独立审查通过。
  实测已完成，T11 全部 cache/state、TopK 逐bit一致，最终输出仍为89元素不同、
  最大19ULP、relative L2 5.329045244e-5，与v13相同。该单因素没有精度收益，不合入正式包。
  同轮均值：Native397.16μs、性能对照438.09μs、精度候选656.62μs；
  不将跨轮基线漂移解释为该修改的性能收益。

正式精度包仍执行PTO attention、Native O-proj；性能包与原SHA备份保持独立。

## 结构化历史证据

[history_20261009.json](evidence_tnd_precision/history_20261009.json)记录37组已完成阶段：
真实T、请求长度、Native对照的逐状态精度、100次重放检查、延迟统计、SK设备事件计数、
各臂源码SHA及原始报告SHA。省略重复的设备事件长名称和计时样本数组，原始报告仍保留在实验目录。
正式精度包源码SHA和性能包23文件SHA同时归档；后续结果追加新stage，不改写旧stage数字。


## v18：softmax 分层求和单因素与四档回归

只修改 FP32 block sum：连续8元素求和得到64个partial，再分组求和得到8个partial，最后求和。
max、exp、alpha更新、BF16 probability舍入、PV与最终division全部保持v13不变。
生成CPP确认三层reshape无padding混入、head内flatten顺序正确、workspace不覆盖尚需读取的exp。
这是有源码依据的经验性数值对齐；Native raw128与compressed512的实际归约边界不同，
不能宣称该统一树与所有Native指令逐位等价。

| 用例 | v13 final差异数 / 最大ULP | v18 final差异数 / 最大ULP | v18 heads差异数 / 最大ULP | v18 final relative L2 |
| --- | --- | --- | --- | --- |
| 8K/B4/T11，长度6,3,1,1 | 89 / 19 | 0 / 0 | 0 / 0 | 0 |
| 8K/B4/T24，等长6 | 0 / 0 | 0 / 0 | 0 / 0 | 0 |
| 8K/B16/T60，非等长 | 1172 / 106 | 55 / 30 | 1 / 1 | 1.824205835e-5 |
| 128K/B4/T24，等长6 | 224 / 55 | 57 / 138 | 1 / 1 | 1.911754277e-5 |

全部四档完成ACLGraph、compiled/raw、A/B/A、100连续replay、reset、guards及实际Native SK profile。
Native观察臂与未修改Native八类状态逐bit一致；同组heads经Native O-proj helper逐bit复现实际captured输出。
四档TopK及六类cache/state均与Native全tensor逐bit一致。
T11实际heads360448元素、等长B4实际heads786432元素全部逐bit一致。

仍未达标的两点已从保存的BF16原始位型定位：

- T60：`[token14,head38,dim500]`，PTO=-0.0108642578125，Native=-0.01092529296875。
- 128K：`[token8,head18,dim501]`，PTO=-0.0198974609375，Native=-0.019775390625。

两点都在ROPE区域，heads各差1ULP/1bit，经动态O-proj量化后被放大。
128K的final不同元素更少，但最大ULP由55升到138；不能仅按relative L2下降宣布全面精度收益。
**v18尚未合入正式精度包**，接下来导出inverse RoPE之前的实际BF16边界继续归因。

同轮延迟均值（μs），precision包含Native HC/norm/O-proj，当前精度优先，不表示速度收益：

| 用例 | Native | 性能对照 | v18 precision |
| --- | --- | --- | --- |
| 8K/B4/T11 | 399.56 | 449.48 | 681.25 |
| 8K/B4/T24 | 415.56 | 478.82 | 748.61 |
| 8K/B16/T60 | 584.76 | 698.85 | 1136.21 |
| 128K/B4/T24 | 645.99 | 577.06 | 839.38 |

测试条件仍是单个真实C4层和合成输入，不能外推整模型或DP/EP16吞吐。
正收益首项runner的归档SHA为74e1f44f00f50ed9238c028f6b93423f31ce43360c9bda9d4adedf5c2400f1d1；
四档heads runner的实际SHA保存在每份结构化记录，运行脚本、冻结candidate与profile均独立复核。


## v20：实际 inverse RoPE 前边界定位

Native观察臂在真实 `npu_sparse_attn_sharedkv` 返回后，将实际BF16结果立即复制到独立持久缓冲。
PTO诊断副本导出已存在的 `n_bf16`，未改变division、CAST_RINT或inverse算术。
生成CPP确认该原始BF16值在导出前未被inverse写覆盖，MTE3存活期及任务依赖正确。
正式precision/control与v18源码一致，probe仅增加外部Out导出。

Native两臂heads及八类状态逐bit一致；PTO导出臂与候选heads及八类状态逐bit一致。
Native/PTO preinverse引用均通过compiled/raw、hidden A/B/A与100/reset检查，Native probe实际SK保留。
Native preinverse缓冲与postinverse不共址，NOPE部分逐bit相同，ROPE部分实际不同，排除了错取postinverse的引用。

| 用例 | preinverse不同元素 / 最大ULP | 首差坐标 | PTO BF16值 | Native BF16值 |
| --- | --- | --- | --- | --- |
| B16/T60/8K | 1 / 1 | [14,38,500] | -0.0115966796875 | -0.01165771484375 |
| B4/T24/128K | 1 / 1 | [8,18,501] | -0.0030364990234375 | -0.003021240234375 |

preinverse与postinverse均仅一个元素差1ULP/1bit，final结果保持v18的55/57个差异。
因此inverse不是首次误差来源；此证据不单独证明两边inverse的所有算术逐bit等价。
继续检查实际Q与attention内部，暂不修改inverse RoPE。

所选SCFA路径的实际源码 `Init` 与host `SplitBalanced` 都使用 `mBaseSize=gSize`，
每块仅一个query，两个首差点的raw window都是128 key，不存在多query union宽度。
后续只有在这些用例的实际Q逐bit一致、真实op attrs符合该路径后，才测试raw128专用规约因素。


## v21：扩展实际 Q 观察到 B16 与 128K

Native在真实attention调用前复制实际Q到独立持久缓冲，复制前验证实际source为NPU/BF16/[T,64,512]，
避免copy的隐式类型转换或广播伪造对齐。PTO在同一Q根分配上增加只读导出；
生成orchestration确认Q生产task与导出task建立真实RAW且父scope在导出结束前存活。
两侧Q均通过compiled/raw、hidden A/B/A、100/reset更新与恢复检查。
Native观察臂与未修改Native、PTO导出臂与v18的heads及完整八类状态逐bit一致。

| 用例 | 实际Q元素 | Q不同元素 | Q最大ULP | preinverse不同元素/ULP |
| --- | --- | --- | --- | --- |
| B16/T60/8K | 1966080 | 0 | 0 | 1 / 1 |
| B4/T24/128K | 786432 | 0 | 0 | 1 / 1 |

真实调用参数两档一致：`layout_q=TND`、`layout_kv=PA_ND`、`cmp_ratio=4`、
`ori_mask_mode=4`、`cmp_mask_mode=3`、`ori_win_left=127`、`ori_win_right=0`，
没有ori_sparse_indices，softmax_scale=0.04419417382415922。
因此这两档剩余误差位于attention本体，实际Q/cache/state/TopK均已逐bit对齐。

Native设备profile的SK事件为14/15，观察臂为15/16；额外观测改变了SK分组，
不能声称dispatch完全相同。严格heads/全状态fidelity通过，真实static+SK资格仍保留。

下一候选v22仅对raw128分母先逐元素相加两个64列半区，再做两级8元素规约；
compressed512以及max/exp/alpha/PV/probability cast/div全部保持v18。
该阶段当时只完成无设备编译与生成代码门禁，NPU结果尚待完成、尚未进入正式包；
它当前只针对已核对的长context raw128资格，不泛化短context或其他attention模式。


## v22/v23：按 Native raw128 与 compressed512 分别规约

前置实测确认Q、cache/state、TopK全部逐bit一致，真实参数满足SCFA路径。
v22仅把raw128的分母改为：先逐元素Add两段64列，再规约为8元素partial和最终sum。
compressed512保留v18的三层8元素规约；max、exp、alpha、probability舍入、PV、division均不改。
T60与128K的actual heads、最终输出及其余状态全部逐bit一致，原来的两处1ULP差消失。

v23增加读取本次真实position的运行时保护：position>=127才使用校准规约；
短窗口的raw与compressed都保持原正式v13的generic row_sum，不把未测路径自动迁移。
这项条件不是按B、S6或capture时常量选择，继续支持真实非等长TND。

### 带保护版本的完整 NPU 回归

| 用例 | 实际heads元素 | heads不同元素 / 最大ULP | final不同元素 / 最大ULP | cache/state与TopK | 门禁 |
| --- | --- | --- | --- | --- | --- |
| 8K/B4/T11，6,3,1,1 | 360448 | 0 / 0 | 0 / 0 | 全逐bit | 全通过 |
| 8K/B4/T24，等长6 | 786432 | 0 / 0 | 0 / 0 | 全逐bit | 全通过 |
| 8K/B16/T60，非等长 | 1966080 | 0 / 0 | 0 / 0 | 全逐bit | 全通过 |
| 128K/B4/T24，等长6 | 786432 | 0 / 0 | 0 / 0 | 全逐bit | 全通过 |

四档使用hard assertion要求precision与Native全八状态及actual heads逐bit一致。
另测history96/B4/T24，hard assertion验证v23与原v13完整八状态及各自独立heads逐bit一致，
compiled/raw、100连续重放与reset后owned heads均一致。该短case的actual Native heads也一致，
但不能据单点推广所有短context精度。
所有case均保留真实Native static/SK、观察臂fidelity、保护区与非有限值检查。

### 同轮 Graph Replay 延迟均值

单位μs，precision显式复用Native HC/norm/O-proj；性能对照为未修改的性能包。

| 用例 | Native | 性能对照 | 带保护精度版 |
| --- | --- | --- | --- |
| 8K/B4/T11 | 402.04 | 439.68 | 672.99 |
| 8K/B4/T24 | 415.66 | 483.57 | 753.26 |
| 8K/B16/T60 | 615.48 | 689.81 | 1140.94 |
| 128K/B4/T24 | 618.94 | 575.16 | 834.49 |

当前目标为精度，精度版比Native慢，不能把性能包优化成果误称为精度版加速。
这仍是单个真实C4层权重+合成输入对拍，未验收整模型、多层误差累积或DP/EP16/EPLB吞吐。
正式precision的数值源码AST与通过硬件回归的v23一致，性能包23文件SHA仍与98eadc516一致。

### 预检失败记录与修复

最初的保护候选生成器命中较早的重复 `if qk_sb==0`，误将gather/同步包含在窗口条件内，
短分支引用未定义sm_exp。独立审查与完整无设备lowering均阻止了该候选上卡，正式源码未被修改。
改用唯一sm_exp锚点后，断言求和区域前后源码保持原样；重新编译与生成CPP检查通过才上卡。
回归driver最初对control访问native_heads，独立审查发现其不存在；改为只检查precision/probe。
上述失败未作为NPU数值结果，不跳过或弱化验证门禁。

正式precision增量UT共26项通过（227选定框架、CPU测试，无新增NPU分配），
新增测试在同一输入表上更新position 126→127→126，执行真实guard AST验证两类块的分支切换与覆盖。
该CPU测试不代替FP32硬件数值对拍，后者由上述五档完整Graph回归证明。
