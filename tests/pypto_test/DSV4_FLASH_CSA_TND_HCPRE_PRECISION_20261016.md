# TND完整PTO精度恢复：HC-pre（2026-10-16）

## 范围与状态

分支 `dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`，起点 `99c288652`。
保留原 `tnd_precision` 作为混合精度基准；候选 `tnd_full_precision` 继续分模块恢复。
**nalinaly仅作为HC-pre至HC-post替换范围的参考，实际Native才是数值黄金。**

本轮恢复HC-pre：原始BF16残差在PTO核内精确扩展为FP32，PTO执行HC投影、
RMS统计、门控、Sinkhorn和混合，发布与Native相同的BF16 mixed及FP32 post/comb。
HC投影使用PyPTO已有的 `pl.jit.extern` 接口和PTO C++指令设置本核HF32模式；
没有调用Native HC-pre、修改框架、全局拦截编译器，或预先转换原权重。
外部源码及可解析的include闭包参与现有缓存摘要，仍须同时固定SDK/编译器/ISA身份。

| 模块 | 原混合精度基准 | 当前候选 |
| --- | --- | --- |
| HC-pre | Native | **PTO，含显式外部PTO HF32核** |
| 输入Norm | Native | Native，待单独恢复 |
| Q及KV后处理 | PTO | 原精度PTO |
| WKV投影 | Native | Native，待单独恢复 |
| Indexer权重投影 | Native | Native，待单独恢复 |
| Indexer主体、sparse attention | PTO | 原精度PTO |
| O-proj | Native | Native，待单独恢复 |
| HC-post | Native | 上一检查点已恢复PTO |

当前HC-pre准入限定A3、TMR、`1≤T≤256`，保持原Native metadata准入。
超过256行时Native的mDim/split-K改变，当前候选拒绝，不能误用本轮mDim1内核。
**准入范围不是全范围逐bit证明；本轮实测限定T11、L4。**
真实请求长度 `[1,2,3,5]`，T11，不按请求padding到S6；HC-pre本身按token行处理。

## 如何定位到Native计算边界

黄金是当前安装版本实际执行的 `torch.ops._C_ascend.npu_hc_pre_v2`，
先验证Native重复及原始NPU图A/B/A一致。没有用CPU复算或nalinaly输出来替代黄金。

对照实际A3源码并观测当前Native tiling：

| 边界 | 旧PTO | 当前Native对齐方向 |
| --- | --- | --- |
| HC投影 | 4个4096列分片，普通FP32 MAD | 22个分片，前21个768列、末个256列；K32、HF32 MAD |
| RMS平方和 | Vector按512列归约及高精度rsqrt | Native Cube Gram矩阵对角；同22分片，顺序求和，Sqrt后Div |
| 分片归约 | 旧分组 | 按K分片0至21顺序累加 |
| sigmoid | recip形式 | 显式Div；单因素实测本case输出不变，不能将它认作已证实根因 |
| comb列和、mixed四路和 | 部分成对求和 | 对照Native顺序0、1、2、3相加 |
| mixed发布 | BF16 | 保留最终BF16边界，输入Norm另行验证 |

仅开启HF32时mixed已一致，但FP32 post/comb仍有残余；
进一步对齐上述Native统计和累加顺序后，诊断候选三个输出一致。

### 实际Native中间量的可信观测

私有改名Native副本的15个原数学文件做归一化比对，先用实际Native验证三输出一致。
在第一阶段完成后、第二阶段不再使用的xCast区域写512字节显式header，记录
实际tiling、自身设备地址及当前输入标记；不改AB/A2及其数学。
桥返回并保活实际workspace，分配前后各512字节保护区；读取时检查唯一header、
地址域、容量、标记、保护区，再按Native物理布局取有效AB和A2对角。

其中一次桥误用 `storage().data()`，忽略保护区view的512字节offset，guard失败。
该观测作废；仅修为 `data_ptr()` 后重新验收，未放宽保护区门禁。
有效观测的三输出36项比较、分片A/B/A均通过。
原始workspace未保存到Git；运行时guard资格由审计过的硬门禁确认，不能声称离线重算过guard。

### HF32不能用普通FP32加权重截断替代

对实际硬件做one-hot BF16输入、264个合成FP32权重的单项乘法观测，
使AB首分片直接等于转换后的权重，不受多项求和干扰。
正负、三个指数、halfway两侧和保留位奇偶均覆盖；A2首分片严格为1，其余为0。
64种候选模型中，本组唯一精确模型为“清低12位，最近邻、半程远离零”。
这只描述当前样本和安装A3行为，不外推NaN、subnormal、溢出或所有指数。
不能从SDK中TransMode的名字直接推出向零截断语义。

真实L4权重按该规则预转换，保留HF32 MAD时，5808个AB及242个A2对角值
在eager/图A/B/A下全部逐bit一致。
**相同预转换权重，仅切换普通FP32 MAD，AB出现5133个差异，A2仍全部一致。**
因此还必须保留Native的HF32累加路径，不能只做权重预处理。
正式候选保留原始权重，由显式PTO C++内核在六个MAD位置保存、设置、恢复CTRL。

## 历史对拍结果

均为实际Native黄金；以下HC-pre完整输出均是L4真实HC权重、T11、seed17合成hidden。
表中的数量是不同元素数；失败即停止该次后续阶段，没有改阈值。

| 实现/单因素 | mixed | post | comb | 结论 |
| --- | ---: | ---: | ---: | --- |
| 原有未接入PTO HC-pre | 140 | 33 | 176 | 首差 |
| 仅HC投影MAD设置Native HF32 | 0 | 12 | 129 | 尚未一致 |
| HF32加sigmoid显式Div | 0 | 12 | 129 | 与上一版本逐bit相同，负对照 |
| Native统计/顺序加法加HF32诊断 | 0 | 0 | 0 | eager A通过，尚无该版本图结论 |
| 截断权重加普通FP32 MAD | 328 | 33 | 176 | 首差 |
| 最近邻、半程向零预权重加普通FP32 MAD | 6 | 33 | 161 | 首差 |
| 实测最近邻、半程远离零预权重加普通FP32 MAD | 1 | 6 | 131 | 首差，仍不能替代HF32 MAD |
| 正式extern HF32、原权重、Native顺序 | 0 | 0 | 0 | eager及原始图A/B/A通过 |
| 上项加PTO核内BF16扩展 | 0 | 0 | 0 | eager及原始图A/B/A通过 |

额外两组分片检查：原始权重加诊断HF32、原始权重加正式extern HF32，
均在eager与图A/B/A中逐bit等于实际Native的全部5808个AB及242个A2对角值。

### 模块与服务的验收界限

- 模块：实际Native先重复、图A/B/A，再PTO eager及图A/B/A；各阶段三个输出逐bit，
  输出NaN毒化、前后保护行、输入及原权重不变。没有Torch输入扩展替代正式PTO计算。
- 服务：保留原精度基准，四臂为Native1、Native2、原基准、新候选。
  使用真实L4全部attention权重，hidden/history合成；history8191、T11变长请求。
  各臂独立cache、同初态，真实FULL ACLGraph/static编译、SK关闭、无host updated nodes。
  对比输出、heads、TopK及六类cache/state，A/B/A、即时guards；禁止候选Native HC-pre/HC-post及fallback。
- 七项CPU测试验证保留父类metadata准入，且T257/384不会使用mDim1 HC-pre。
- 不包含整模型、其它CSA层、192 token、DP16、性能、128K结论。

最终服务复测 `task_20261016_175803_390864330684` 已exit0：

- Native2、原精度基准、新候选，对Native1的A/B两组×9类张量，**54项全部逐bit一致**。
- 四张独立真实FULL图各三次原始replay，A/B/A恢复和即时保护区检查通过。
- 候选Native HC-pre/HC-post被替换为抛错函数，确实执行PTO；未fallback。
- 预热后candidate `object_hits +3`，generation/binary/bypass及缓存错误均零增量。
- 当前23个单卡任务均已结束，没有遗留占卡任务；单项上限780秒。没有运行128K。

### 保留的服务验证失败

1. `task_20261016_174739_333095031682`：`--lengths`参数解析退出，未执行数值测试。
   单独source环境未复现位置参数变化，不能断言是环境脚本改写参数。
   后续固定声明的本次小配置，并使用 `--key=value` 和显式配置打印。
2. `task_20261016_175040_347394011254`：54项数值已一致、图A/B/A及guards通过，
   但最终缓存门禁发现generation/bypass各增加1，因此整体仍记失败。
   在私有产物副本中复现 `Unresolved local extern include: tensor.h`：
   外部源码误将runtime SDK头写为local引号include，无法被持久缓存打包。
   仅将 `tensor.h`、`intrinsic.h` 改为系统angle include，
   无设备打包成功后建立step04新源码身份，再复测；没有改runtime或放宽门禁。

## 冻结身份与复现

[完整329文件证据归档](evidence_tnd_precision/full_pto_20261016/hcpre-evidence.tar.gz)，
2,208,973字节，SHA256：
`5b7c8c76dec6363298cc0a602dd95e89f4f2e906aedab315c5ae8de4db0e2928`。
包含逐文件SHA、23项任务历史、成功/失败报告和日志、Native私有观测源码及桥、
生成与显式PTO源码、有效分片原值、one-hot转换原值、去掉权重/hidden的首差输出。
不含模型权重、venv、原208MiB workspace、大层cache快照。
其中 `results/hpreconv2/result/conversion.pt` 是当时序列化失败留下的不完整文件，
按原SHA保留失败历史，**不能加载或用于数值结论**；有效转换原值来自hpreconv3。
其余首差快照仅保留actual/expected，原始完整快照的SHA另存。

| 冻结源 | 文件数 | manifest SHA256 |
| --- | ---: | --- |
| step02：核内BF16 HC-pre模块 | 3355 | `451ca9ff12410c44d35c65d6df67b5cb97ec3063b550c22e2526d719b076a73a` |
| step03：服务数值通过、缓存未通过 | 3355 | `5a64960a423d92d51465bd445947fab53dd53d08edd3203c06f640caa0bb1307` |
| step04：服务完整门禁通过 | 3355 | `c4f53fd7b717ff34664a6b08ad310642b0859ba32d12fbcbc91589d6b6784567` |

当前四项候选文件与实测step04逐SHA一致，原 `tnd_precision` 23文件SHA未变。
实际加载Native扩展SHA为
`8614af9fa51256dd6c4d83f07790f9a88328e331c70fe98a777cc2be772a7a47`，
另记录ATB插件及私有观测bridge/package身份。

1. 先按[前一检查点](DSV4_FLASH_CSA_TND_FULL_PRECISION_20261016.md)恢复step01冻结源和环境；
   完整版本与模型配置见[精度复现手册](DSV4_FLASH_CSA_TND_PRECISION_REPRO_20261016.md)。
   当前Python3.12、Torch2.10.0+cpu、Torch-NPU2.10.0.post2、vLLM0.25.1、
   CANN9.2.0-beta.2、ATB9.2.0-beta.2、PTOAS0.66，私有PyPTO3e87a843基线及Simpler a54c0509。
2. 核对归档SHA，解压后校验 `evidence-manifest.json` 全部329项。
   对step01源另建副本，分别应用 `source-overlay/step02` 和 `source-overlay/step04`；
   逐项核对对应 `frozen/*/frozen-source.json`，不要覆盖旧源。
3. 对照归档的env和脚本，调整新的自有绝对路径，并另存调整后helpers身份。
   先无设备检查imports、编译器/CANN/ATB/ISA路径、冻结源及候选签名。
   不重新安装框架，不使用其它用户代理、目录或设备锁。
4. 先单独运行 `experiments/hcpre-bf16-v1/run-hcpre.sh`，再单独运行
   `experiments/hcpre-layer-v3/run-layer.sh`。后者固定L4、history8191、真实长度 `[1,2,3,5]`。
   输出目录必须不存在并保持短路径；配置、hidden seed、原权重及数学源不得变化。
5. 每项分别使用 `task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 780
   --env PYTHONNOUSERSITE=1 "bash <run-script> <new-label>"`。
   env显式PTOAS_ROOT实际选择0.66，CLI会提示忽略0.63。约5分钟检查，不建自动调度。
6. 模块逐bit、guards、A/B/A和服务所有数值/图/缓存门禁必须全部通过。
   发现首差即停止，不提交更大配置。历史诊断可按任务索引逐项复现；
   若要导Native中间量，先独立构建并验证改名Native/桥fidelity，不能跳过workspace资格检查。

独立审核覆盖正式extern ABI、动态维度、RAW依赖、逐行widen、buffer生命周期、
原基准保留及缓存根因；最终证据另按SHA和JSON逐项复核。

