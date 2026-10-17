# 首次FULL6同输入内部定位（2026-10-17）

## 当前结论

**已定位WKV首差；单层单因素修正及原服务轨迹的首次FULL6均已逐bit通过。** 原失败轨迹的首差是forward index74、首次FULL6、L20 heads。
补齐与旧整模型Native相同的13包静态装载和真实reselect后，在相同输入、metadata及pre-cache下，
Native组件heads已逐bit复现旧整模型gold；首个已观测内部差异定位为WKV投影的一个BF16元素：
`[token=2, channel=204]`，静态Native为`-0.162109375`，当前PTO为`-0.1630859375`。
Q_A、QR、QR scale和RoPE后Q在该门禁之前全bit0。

此前未加载13包的Native组件、eager/rawgraph、历史cache重建与PTO一致，
其适用范围现明确限定为“装包前运行条件”，不能代替旧整模型Native的静态选核条件。
仅改变WKV的K256累加次序后，独立候选已通过真实输入A及Graph A/B/A；
回到原root graph后，相关cache、TopK及旧整模型Native heads全部逐bit一致。
单因素证明L20首差的因果修正；后续原服务轨迹已验证rank0全部43层观测边界及outer前75轮逐bit一致，
但未继续完整192 token验收。

数学提交仍为`c1ef141ae`，原基准及性能版未修改；当前分支为
`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
不新跑整模型Native2，不扩展配置；每个NPU任务均设置`--max-time 780`。

## 原轨迹证据

沿用[原128K/B24失败记录](DSV4_FLASH_CSA_TND_FULL_PTO_128K_20261017.md)的真实权重、bank、
step09冻结源码、确定性、入场参考与FULL `[6,144]`。只新增私有诊断helper。
所有新增图内观测只复制实际输出，不注入Native值或重算值。

| 检查 | 结果 |
| --- | --- |
| 全部16rank实际输入/入场 | 前75轮与保存Native参考完全相同 |
| 诊断观测保真 | 两次hook的outer前75轮均与原失败candidate相同 |
| 内部阶段 | index0～73所有43层已观测边界相同 |
| index74首差 | L20/heads；L0～19及L20输入、positions、Norm相同 |
| 首差请求 | q21，query bounds `[0,6]`，位置131524～131529 |
| heads差异 | 1011/196608元素；max_abs=0.015625，max BF16 ULP=237 |
| 每行heads差异数 | `[0,0,357,176,225,253]` |
| 完整192 token或接受事件 | 首差即退出，未完成全量验收 |

上述误差来自实际graph的heads，与后续O-proj放大的误差分开统计，不能把O-proj当作最早根因。
aux在本模型中的真实采样层通过保存Native值的摘要核验为L40、L41、L42，未使用EAGLE3默认层号推测。

## 已完成的小任务

所有模块测试使用真实L20或对应层的Native step074输入与原模型权重；不是合成权重。

| 任务/目的 | 时间 | 结果 | 适用范围 |
| --- | --- | --- | --- |
| modules1：21层HC-pre、Norm、WKV、Indexer投影 | 2m2s | 483项全bit0 | eager与同A rawgraph；不含CSA/cache |
| hcpost1：21层HC-post | 1m50s | 84项全bit0 | 保存Native O-proj输入、Native gates；不含当前误差heads |
| shared1：共享地址的大/小图切换 | 1m53s | 861项全bit0 | 144行由6行重复构成；不含CSA/cache及真实全轨迹 |
| prefix1：L20真实前75帧producer扫描 | 1m33s | 675项全bit0 | 原NONE21/FULL144/FULL6档位；每帧插入Native primitive作gold |
| hook1：原16卡轨迹内部边界 | 约9分钟占卡 | 原L20/heads首差 | 列表21m39含约13分钟排队，不能当占卡时间 |
| hook2：原首差前后cache与producer导出 | 8m27s | 同一首差、导出qualified | 保留原捕获tensor绑定、物理页号、读写页覆盖及outer保真 |

prefix1逐帧会执行Native gold，不能称为无干扰cache轨迹重放；shared1是布局上下文代理，
不能凭其通过排除整模型生命周期问题。Norm rstd未作为数值gold验收。
真实hook2导出的Norm、WKV和Indexer raw投影，再对相同输入的Native primitive均逐bit一致。

## 子阶段诊断与未完成项

QKV诊断直接复用当前正式`qkv_proj_rope` inline，预定比较顺序为QR、scale、KV、Q，
eager先通过后再检查同输入rawgraph。它不是保存Native FULL图内Q/KV的直接观测。

| 版本 | 状态 | 解释 |
| --- | --- | --- |
| qkv-v1 | 诊断解析失败 | 新wrapper的`task_dummy`缺少`deps=[]`；无QKV数值结论 |
| qkv-v2 | 诊断守卫错误 | NZ权重直接`view(uint8)`触发Identity/TransData错误；无QKV数值结论 |
| qkv-v3 | exit0，1m29s | QR、scale、RoPE后KV与Q的eager/rawgraph均逐bit一致；守卫通过 |
| core-v1 | 状态观测范围不符，1m44s | heads、TopK一致；发现整模型结束时未访问state页另有写入，不能当独立L20输出 |
| core-v2 | exit0，1m30s | active state与共享alias写入对original post；未访问页对original pre，全部通过 |
| native-core-v2 | exit0，1m29s | 同pre-cache Native组件的有效metadata、cache、TopK、heads均与PTO bit0 |
| native-graph-v1 | exit0，1m31s | 同pre-cache Native eager/rawgraph heads、cache连续两次bit0，仍不同旧整模型gold |
| bank历史CPU核查 | 通过 | 初始history131346之前32836条compressed KV、Indexer key和scale全bit0 |
| raw-history-v1 | exit0，1m6s | 127条live历史raw KV逐bit一致；按每个position的最后一次实际write输入、原M144重放Native WKV/Norm/RoPE |
| compressor-history-v1 | exit0，1m41s | 真实step35～73输入，从bank state重建的45条压缩KV及8条近期state全bit0 |
| indexer-history-v1 | exit0，1m15s | 同一历史输入，Native压缩/Hadamard/动态量化后的45条key、FP16 scale及8条inner state全bit0 |
| layout-v1 | exit0，1m34s | 非空物理页整体反序、全链路同步寻址后，Native/PTO仍bit0；Native自身heads也未变化 |
| native-forward-v1 | exit0，1m30s | saved heads→Oproj与actual heads→Oproj均一致；完整Native forward与组件重放heads一致 |
| static-wkv-order-v1 | exit0，1m55s | 32组均推断自然K256序；真实A及Graph A/B/A全bit0；只换WKV后原core的cache、TopK、旧gold heads全bit0 |
| static-core-v1 | 首差停止 | 13包装载/reselect后Native heads匹配旧gold；Q系列一致，WKV在[2,204]首差1个BF16元素 |

测试工具错误未计为数学精度失败，未通过降低容差或跳过守卫获得成功结论。
core-v1的原失败记录保留；core-v2根据实际读写集合区分原整模型其它阶段的写入，
同时强制未访问页保持original pre，并非忽略这些页。main state的active与alias写入页为
`[5452,5483,10909,12027,12031,15286,18597,23141]`。

### 同输入组件对照的含义

Native与PTO共享同一份实际pre-cache，但使用独立恢复的运行，保留typed alias、页号与stride，
每次运行检查首尾carrier守卫与不可变输入。原PTO root先通过原head/TopK/cache保真门禁。
随后Native组件重新执行Q/KV、压缩、Indexer及attention，得到相同heads。
Native rawgraph也与其eager相同；图内未重新hook Q/TopK，不能把eager的Q/TopK观测扩展称graph也通过。

这些结果只能说明当前给定cache下的组件运算一致，不能证明此前生成cache或原Native图上下文一致。
bank历史核查只覆盖初始prefix；raw-history进一步覆盖了pre74的127条live raw KV，全部一致；Native历史重建又覆盖了bank之后45条压缩KV、45条Indexer key/scale及末8条两类state，均一致。
历史重建使用私有16页ring、真实已存normed、原M144/B24均匀补位；初始bank seed、首step35/start131346、
首次pool所需历史页覆盖和当前compact cos/sin均有硬门禁。
它证明这些逻辑producer重放与PTO缓存一致，仍不能代替旧Native整模型实际物理cache快照。

### 已找到的静态核装载条件差异

旧Native rank0日志明确包含`OFFICIAL_STATIC_CACHE packages=13`、实际reselect成功及installed13；
此前单层helper虽然使用同一环境和配置，却没有调用静态包装载，因此这些组件结果尚未对齐该条件。
TorchNPU通过lazy alias提供内嵌npugraph_ex；导入TorchNPU之前顶层find_spec为None不能证明包不存在。

static-core-v1已只在新任务私有`opp_env/opp`加载相同的13个缓存包，官方bridge拒绝重新编译，
验证CANN/SOC/确定性/module及包摘要，并要求真实reselect成功。
先保存完整产物，再按Q_A、QR/scale、Q、WKV、Indexer与cache/heads的边界停止首差。
该任务在WKV停止，未把后面的cache/Indexer精度记成已验收；Native heads与旧gold一致是已保存的独立观测。
这仍是当前首差的单层诊断，不新跑整模型Native2或后续大配置。

## WKV单因素修正及适用范围

`static-wkv-order-v1`使用与首差任务相同的13包，先确认静态Native WKV的原模块调用和F.linear均复现首差目标，
再使用6组三元K256有限值查询，约束既有44种循环次序假设，32个N16列组均唯一落到自然序
`0,1,...,15`。它复用现有`WKV_COLUMN_ORDERS[1]`及`WKV_K_ORDERS[0]`，不改kernel数学。
有限查询用于限定标定，不是通用矩阵乘累加次序的源码证明。

修正后依次通过：

1. 真实首差输入A的WKV eager输出与静态Native逐bit一致。
2. 真实输入A、另一输入B、再回A，Graph A/B/A逐bit一致，且B确实改变输入与输出。
3. 输入、权重、次序tensor不可变；输出poison、首尾守卫均通过。
4. **仅**把新WKV填入原CSA core的既有输入buffer，恢复同一pre-cache，重放原已捕获图。
5. raw KV、compressed KV、active main/inner state、Indexer cache、TopK及旧整模型Native heads全bit0。

原core的输入、metadata、缓存寻址和其他计算均未改变。此前1011个heads差异由此消失。
这里只证明当前L20/首次FULL6；不把它扩展成完整192 token或所有形状的验收。

独立复核同时发现：`enable_static_kernel=True`并不等于真实装包完成；装包前相同配置下旧表才匹配。
因此未将“配置True就切换mode1”写入正式service。`static-wkv-service-v1`是后续独立服务候选：
先委托原官方bridge完成13包/SDK/确定性摘要验证，外层跟踪真实reselect成功，再启用自然序T6。
捕获T6时若资格未知直接报错；正式源码、原256形状表及性能基准不修改。
保存的实际阶段集合为43层，其中CSA为`2,4,...,42`共21层；CPU审计按精确层号集合核对，
不以旧文档中的HCA数量推断实际执行层数。
原轨迹只复测至forward74：0～73轮仍对原失败candidate做保真门禁；第74轮对旧Native各层及outer验收，
成功即写`FIRST_FULL6_QUALIFIED.json`并主动停止，不继续192 token或其它配置。

| 证据 | SHA256 |
| --- | --- |
| static-core result.json | `790800d2bf297a8f644a7c6fe913f3a128cf8cd010bf6a996781d94f50525603` |
| static-core Native产物 | `5db189411b6da8ade71c5cc30bacead04eb5a4d21de81f2f55ab6cb5ef9fff09` |
| WKV首差快照 | `08a9222cbd7c658fea2876ff74deae39632d76e33891f47fb33c0cc6f1a89fae` |
| static-wkv-order result.json | `c96c3dfb70f295c65a8f649e024ca0d0a0e6f392839fa4a013a8cdb384667a93` |
| 官方静态包metadata | `863a1e5daffc54aec0e9e8afb9f809f1bc6f4b3a2487e440615f3c942f36c602` |

## 实际服务首FULL6修正复测

`task_20261017_163746_897257146`已完成，`FIRST_FULL6_QUALIFIED.json`存在且CPU独立审计通过。
任务exit1来自**通过门禁后主动抛出的停止标记**`INTENTIONAL_FIRST_FULL6_QUALIFIED_STOP`，
不是数值失败；不存在`full-probe/STOP.json`。任务结束后全部16卡均已释放。
列表显示23m27s包含约14分钟排队；最早运行日志16:51:37，成功停止17:01:00，
含退出清理实际占卡约10分钟，低于15分钟，提交仍为max780。

| 审计项目 | 结果 |
| --- | --- |
| 源码/helper身份 | step09原3361文件逐项SHA，helper manifest及bootstrap/witness/full_probe摘要均通过 |
| 原Native/bank身份 | 已存Native源/派生参考摘要、bank plan与token文件门禁通过；不新跑Native2 |
| 静态核条件 | 16rank均有相同13包metadata SHA及真实reselect成功；rank0安装13包核验通过 |
| 捕获范围 | 每rank精确21个CSA层`2,4,...,42`均记录ready=true、capture=true、mode1 |
| 全部16rank实际输入 | 每rank恰好75轮，所有字段除本地execute_index外与Native参考相同 |
| 入场与图 | 每rank首次实际FULL6均为forward74；同一q21/6行；无抢占 |
| 原错误候选保真 | rank0前0～73轮outer仍完全相同 |
| rank0内部边界 | 前75轮、实际43层的全部已观测字段摘要与旧Native内部参考逐bit一致 |
| rank0 outer | target hidden、aux hidden和MTP residual前75轮摘要与Native1逐bit一致 |
| 停止范围 | 首FULL6通过即退出，未执行后续配置；不声称完成192 token或全部rank内部浮点比较 |
| 正式默认与基准 | 数学仍为c1ef141ae；修正在独立qualified候选中，不静默覆盖默认256形状标定 |

此处的“原服务轨迹通过”覆盖保存Native参考的前75轮及首次FULL6，
不是最终生成token序列验收；最后一步通过后主动终止，未取得完整decode结果。
修正没有复制Native的Q/KV/heads注入服务；运行的是原PTO WKV kernel，仅更换其T6 order tensor选择。
单层注入仅用于前文因果定位，实际服务候选从真实Norm输出执行PTO投影及原CSA全链路。

| 服务证据 | SHA256 |
| --- | --- |
| FIRST_FULL6_QUALIFIED.json | `6a616162252737531dcf61d91c35f8deb82391a41cf5fee67545d1e78691f612` |
| CPU独立审计 | `e888cf9dde4f804bec2deb438e60a3b46419ea9cf1e97742b97f5c40fa63365e` |
| 修正后首FULL6原值快照，远端保留 | `0d8fcb61e6e3bd6c7be7d754f467360b07cd069529bc95c9913be8577d9affb9` |

服务输出目录为`/data/pyptouser/yejia/lc128/full6-static-wkv1`；
`audit_static_service.py --label full6-static-wkv1`在CPU重算全部输入、stage/outer、源码与资格门禁。
本轮没有继续后续大配置，也没有重新运行整模型Native2。

## Cache与metadata导出契约

仅rank0、L20、已知index74导出。捕获时保留真实FULL6描述符，禁止被同T的eager调用覆盖。
前置保存相关读表页与写入slot页；结束时再次核验实际写入页全部已包含。
只有原candidate outer保真通过、首差仍是L20/heads时才写入`qualified`后的快照。

实际底层存储有别名：compressed与main state共享32KiB物理页载体，Indexer与inner state共享4160B载体。
后续重放必须保留该关系及原物理页号；保存页之外未持久化，不能声称复制了整块原分配。
物理页包含其他dtype与padding，应逐字节核验，不能要求FP32 reinterpret的所有字节都对应有限浮点数。

真实TopK每行512项唯一且均在该行可见范围内，这只排除明显越界/重复，不证明选点与Native相同。
compact slot中存在`[-1,31]`；负槽存在本身不能证明Native scatter bug，更不能据此模仿潜在非法写入。

## 复现资产与身份

沿用原失败记录中的模型、bank及环境身份。Native内部参考为已保存的step074，未新执行整模型Native2。

| 资产 | SHA256 |
| --- | --- |
| Native step074.pt | `5b873509016a7419abf5639a808da58950eb2d3d3bd332c00ed8625e00174481` |
| Native stages.jsonl | `87fa6b787a138a58cafc9b18ddb10554c8ce08b9064f831ecd0a6a105d8a5945` |
| Native rank0原轨迹 | `c39ed2c21f2a07aae2ca061af037c8614c242450305c281b76c5d3b8badbf09f` |
| hook1 helper manifest | `ea7a4953be5ad379e910cfd25dbd84a3c2413680ea9a3f5ae4b5b3c9c2f4ed04` |
| hook2 helper manifest | `8644200a5eb1a2ca0f71569d058691e8d211ddda89e065bd5fd2df599b3cf61d` |
| hook2 l20-before.pt | `3faf46510b167e259013dddc7a6f59412313b112502ca02a64f796abe4eec04f` |
| hook2 l20-after.pt | `782f75c8ccbce27810d56ba16c9969990f0ab7ccb90bb41953fb255f0a77ea42` |

远端自有实验根为`/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/full6-localization-20261017`。
大快照保存在`/data/pyptouser/yejia/lc128/full6-hook2/full-probe`，不包含在Git中。
本地工作材料保存在workspace的`reports/dsv4-full6-localization-20261017`。

已执行任务对应关系：

- modules1：`task_20261017_142207_17398486912`
- hcpost1：`task_20261017_142936_201817221913`
- shared1：`task_20261017_143400_215743116134`
- prefix1：`task_20261017_150141_277081821021`
- hook1：`task_20261017_142615_185987930096`
- hook2：`task_20261017_150820_28943064164`
- qkv-v1：`task_20261017_151903_315087018314`（1m6s）
- qkv-v2：`task_20261017_152131_32070925168`（1m29s）
- qkv-v3：`task_20261017_152744_333581926588`（1m29s）
- core-v1：`task_20261017_152950_337800222538`（1m44s）
- core-v2：`task_20261017_153908_355084717040`（1m30s）
- native-core-v2：`task_20261017_154203_36070664597`（1m29s）
- native-graph-v1：`task_20261017_155133_377374325645`（1m31s）
- raw-history-v1：`task_20261017_155808_390762910011`（1m6s）
- compressor-history-v1：`task_20261017_160259_407098313477`
- indexer-history-v1：`task_20261017_160722_2143614959`（1m15s）
- layout-v1：`task_20261017_161108_13427332583`（1m34s）
- native-forward-v1：`task_20261017_161506_22796420172`（1m30s）
- static-core-v1：`task_20261017_162129_4471475432`
- static-wkv-order-v1：`task_20261017_162829_62585030363`（1m55s）
- static-wkv-service-v1：`task_20261017_163746_897257146`（16卡、max780；首FULL6资格通过后主动exit1，实际运行约10分钟）
- 调度参数错误：`task_20261017_163643_86278520385`，`--device all`非法，0秒退出；未启动helper、未占卡。
  之后按`--device auto --device-num 16`重提，保留该操作失误记录。

新增helper沿用原预检、源码清单及环境脚本。完整轨迹只提交`precision-witness`，
单卡module helper用独立标签，每次保留历史输出；QKV及core需要传入上表after SHA。
所有提交仍通过`task-submit --max-time 780`，不得绕过调度直接裸跑NPU。

## 归档与复现

小型证据及所有独立helper归档于
[full6-localization-evidence.tar.gz](evidence_tnd_precision/full_pto_20261016/full6-localization-evidence.tar.gz)。
包内`manifest.json`列出全部文件摘要和正式WKV/service/Indexer源码绑定；大tensor及模型不入Git。
归档共186个逐文件摘要条目，1,568,323 bytes，SHA256：
`8ac0a49173b8d9d2967655bc48d5b8563496488493c2b3022d8a93d70cc88e56`。
需要既有冻结step09、真实模型、bank、原Native参考及SDK/runtime，归档并非新机器自包含环境。

单层单因素修正复现：

```bash
# 在227已有同一资产环境，先运行helper的环境/源码预检。
task-submit --device auto --max-time 780 --ptoas 0.63 \
  --env PYTHONNOUSERSITE=1 \
  --env ASCEND_HOME_PATH=<已核对的CANN路径> \
  --env PTOAS_ROOT=<已核对的PTOAS路径> \
  --env CCACHE_DIR=<既有checkout的.ccache> \
  --env CMAKE_BUILD_PARALLEL_LEVEL=4 \
  "bash <实验根>/static-wkv-order-v1/run.sh <新独立标签> 782f75c8ccbce27810d56ba16c9969990f0ab7ccb90bb41953fb255f0a77ea42"
```

实际服务首FULL6复现：

```bash
bash <实验根>/static-wkv-service-v1/preflight.sh
task-submit --device auto --device-num 16 --max-time 780 --ptoas 0.63 \
  --env PYTHONNOUSERSITE=1 \
  --env ASCEND_HOME_PATH=<已核对的CANN路径> \
  --env PTOAS_ROOT=<已核对的PTOAS路径> \
  --env CCACHE_DIR=<既有checkout的.ccache> \
  --env CMAKE_BUILD_PARALLEL_LEVEL=4 \
  "bash <实验根>/static-wkv-service-v1/run-case.sh precision-witness <新独立标签>"
```

实际SDK、PTOAS、模型、bank、资产路径及清单均固化于helper脚本，所有输出标签必须新建。
单层成功要求`result.json`的`complete/pass`均为true且全部比较bit0；服务路径即使exit1，
也必须区分有`FIRST_FULL6_QUALIFIED.json`的主动成功停止与`STOP.json`的首差停止；
不存在成功marker不得计为通过。之后仍需独立审核全部16rank的输入前缀与capture选表日志。

本地既有WKV源码/原256形状标定绑定回归：
`python -m pytest -q tests/pypto_test/test_full_precision_wkv_cpu.py`，3 passed；
未修改数学源码，因此没有重新编译或安装vLLM/PyPTO/Simpler。

### 本地静态检查

增量`pre-commit run --files ...`及新增/关联MD的manual markdownlint均通过。
另依仓库要求运行`bash format.sh ci`，全量检查因既有归档生成代码的长行、历史日志拼写、
旧helper的forbidden imports等失败；未把它计成通过，也未扩展清理历史测试资产。
全量formatter自动修改的346个无关tracked文件均恢复HEAD，正式数学和原始证据保持原样；
本次改动范围仍只有关联文档、新定位文档与新证据包。
