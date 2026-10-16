# TND完整PTO精度恢复：输入RMSNorm（2026-10-16）

## 范围

分支 `dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`，起点 `19d0b16d1`。
原 `tnd_precision` 混合精度基准保持不变，本轮只推进 `tnd_full_precision` 的输入Norm。
**nalinaly只参考HC-pre至HC-post替换范围，数值黄金是当前安装的实际Native。**

当前候选已恢复PTO HC-pre、输入Norm、HC-post；WKV投影、Indexer权重投影和O-proj仍Native。
所以当前还不能称HC-pre至HC-post全段PTO替换完成。

Norm直接使用原生加载后的BF16 `[4096]` 权重，要求原始连续ND、epsilon=1e-6、未加载bias；
不做权重dtype转换、重排、预量化或额外残差合并。
真实token行数由TND调用传入，不按请求padding到6；统计缓冲每行8个FP32仅为设备32字节对齐。
父类A3/TMR/metadata准入及HC-pre的T≤256限制继续生效。

## 为什么不能只看BF16输出

初始模块测试的BF16输出已逐bit一致，但FP32逆RMS仍有3个元素相差1 ULP。
本轮保留逆RMS与输出双重逐bit门禁，发现首差即退出后续阶段。
没有以CPU复算、nalinaly结果、allclose或放宽阈值代替实际Native。

实际SDK源码参考为 `impl/ops_nn/ascendc/rms_norm/rms_norm.h::Compute` 与
`reduce_common.h::ReduceSumHalfInterval`：4096列先对半相加到64列，最后WholeReduce；
使用标量 `1/sqrt(sum/4096+epsilon)`，输出顺序为 `(x*rstd)*gamma` 再BF16 RINT。
源码参考与当前op具体dispatch是不同证据；本轮没有直接观测Native内部sum或tiling，
最终判定仍以实际 `torch_npu.npu_rms_norm` 为准。

### 定位过程

| 版本/单因素 | BF16输出 | FP32逆RMS | 结论 |
| --- | --- | --- | --- |
| 整行PTO归约、高精度rsqrt | 0差 | 3元素±1ULP | 首差停，未跑PTO图 |
| 仅换向量Sqrt/Div | 0差 | 同样3元素、bit值相同 | 不能只替换表达式 |
| 导出sum/variance/approx sqrt，独立scalarDiv | 与上项逐字节相同 | 同样3元素 | 观测保真、未改变原数学 |
| 半区归约到64，仍向量Sqrt/Div | 0差 | 3元素±1ULP | 仍需对齐标量舍入 |
| 上项加FP32平方根中点舍入修正及设备scalarDiv | 0差 | 0差 | 模块eager、图A/B/A通过 |
| 去掉诊断输出和无用任务的正式Norm源码 | 0差 | 0差 | 独立模块eager、图A/B/A通过 |

半区版本的实际PTO variance经CPU FP32 sqrt/div，可复现本组11个Native逆RMS；
向量sqrt相对CPU sqrt有3个差异，使用未修正root的scalarDiv还剩2个差异，
vectorDiv比该scalarDiv另多1个差异。这只作为定位证据，不是Native内部观测或CPU黄金。

直接在外部PTO核调用 `__builtin_sqrtf` 的方案在无设备链接时出现 `undefined sqrtf`，
未提交NPU，也未改编译器、运行库或链接流程。
正式候选采用本分支QR路径已经使用的INT64平方根中点比较：修正A3向量sqrt候选，
按FP32最近邻/半程偶数舍入，再执行设备scalar除法。
该做法对齐计算边界，不是对最终输出加补偿。
当前使用两步修正；本轮样本和后续门禁不能外推所有FP32输入。

生成C++已确认整数bit临时值为各任务独立UB Tile，标量SetValue/GetValue在S/V依赖保护内；
最终rstd通过32字节向量store写入，避免相邻请求共享cache line的GM标量写。
平方和通过支持的 `pl.jit.extern` 调用显式PTO C++指令，不调用Native Norm、不全局拦截编译器。

## 模块测试方案与结果

使用真实L4 HC参数和Norm权重，seed17合成BF16 hidden `[11,4,4096]`；
实际Native HC-pre产生Norm输入。Norm权重存储为FP32，按实际Native loader复制到BF16。
输入SHA256为 `b847db1573f8bbc47b2dd17f28364019b21b1be91f2604eb6cb6262f46c023b7`。
模型为 `DeepSeek-V4-Flash-0731-w8a8`，75 shard。

1. Native重复和原始NPU图A/B/A先验收，确定性开启。
2. PTO eager先比较BF16输出及FP32逆RMS；首差保存后立刻退出。
3. 通过后才捕获PTO图，用相同输入地址更新A/B/A；B的hidden加0.125后重新生成Norm输入。
4. 输出NaN毒化，检查输入、原权重不变及前后保护行；rstd全部8列必须逐bit相同。
5. Native与PTO的B输出必须变化，A恢复后仍与最初黄金逐bit相同。

诊断修正版 `task_20261016_185029_284628724746` exit0；
正式源码独立模块版 `task_20261016_190408_37494929773` exit0。
**两者各32项比较全部逐bit一致，Native资格、PTO eager和原始图A/B/A全部通过。**
前者约19:03才开始运行，之前排队不占卡。所有本轮任务执行上限780秒；无128K测试。

## 服务FULL四臂

`task_20261016_190819_394225423249` completed(exit=0)。

- L4真实完整attention权重，合成hidden/history8191；B4真实长度 `[1,2,3,5]`、T11，
  各组Native query bounds均为 `[0,1,3,6,11]`，无每请求S6 padding。
- 四臂：Native1、Native2、原混合基准、新候选。各臂独立cache且初始内容相同。
- 四张独立真实FULL ACLGraph，static编译成功，无host updated nodes；SK关闭，非eager。
- 额外记录实际Native/Baseline Norm返回Tensor，保活其图内地址；候选直接观察持久normed缓冲。
  B输入使Norm输出变化，A恢复后逐bit恢复；不是用重新计算的Norm冒充本次中间量。
- Norm输出、最终输出、heads、TopK、六类cache/state，共10类张量；
  Native2/原混合基准/候选，对Native1的A/B共 **60项全部逐bit一致**。
- 候选Native HC-pre、Norm、HC-post均改为抛错函数，且禁止fallback，确认真正执行PTO。
- 编译调用与原始replay相同，各臂三次raw replay，输入A/B/A、即时cache guards通过。
- warm后candidate object_hits增加4，generation/binary/bypass及缓存错误均零增量。

本轮服务保持既有HC-pre验证范围；不能替代其它T/其它层或整模型token验收。

## 固定环境与复现

沿用[HC-pre记录](DSV4_FLASH_CSA_TND_HCPRE_PRECISION_20261016.md)的环境：
Python3.12，Torch2.10.0+cpu、Torch-NPU2.10.0.post2，vLLM0.25.1，CANN9.2.0-beta.2，
相应ATB；PTOAS实际为0.66，命令行0.63被显式PTOAS_ROOT覆盖。
原Native扩展、私有PyPTO及Simpler环境复用，没有重装框架。

服务源快照step05共3357文件，manifest SHA256：
`097ba9612957968e1b4d496b0f502e3bc260712eff155e49d33b57993e99b055`。
step04已验收HC-pre快照保留，原混合基准23个文件SHA不变。

### 证据与复现步骤

仓库证据：

- `evidence_tnd_precision/full_pto_20261016/norm-index.json`：范围、版本、任务、结果索引。
- `evidence_tnd_precision/full_pto_20261016/norm-evidence.tar.gz`：102个成员的SHA清单、
  全部独立诊断源码/生成代码/首差输出/日志、正式源码overlay和服务结果。
- 压缩包802585字节，SHA256：
  `10781cfca5bf59fe354cb52311965930ced1d494ee2f2343fd7a004a352b1012`。
- 已核对全部102个SHA、四个正式源码文件与当前工作树一致、原混合基准23文件不变。
- 首差快照只保留actual/expected，模型权重及原始输入未归档；输入通过seed、模型参数摘要复现。
- debug中的sqrt/scalar_inverse仍是未修正root的独立观测，不能当作v7最终参与数学的root。

沿用已验证的私有环境及目录，在227下依次运行；先核对上一节SDK/模块路径：

```bash
base=/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex
root="$base/full-pto-precision-20261016"
source "$root/step01/env.sh"
# 只做无设备验证，不能在登录shell调用探针NPU主体。
python "$root/step01/check_frozen.py"
SOURCE_ROOT="$root/step05/source" python "$root/norm-layer-v1/check_step05.py"
```

独立模块，诊断修正和精简正式源各跑一项（输出标签需换成未存在的名字）：

```bash
task-submit --device auto --max-time 780 --ptoas 0.63 \
  "bash $root/input-norm-rounded-v7/run-norm.sh norm_round_reproduce"
# 查询前项通过后才执行下一项。
task-submit --device auto --max-time 780 --ptoas 0.63 \
  "bash $root/input-norm-formal-v8/run-norm.sh norm_formal_reproduce"
```

每次用 `task-submit --status <task-id>` 和 `--log <task-id>` 跟进，
首差应保存first-difference并退出，不并行提交后续配置。
两项complete/pass、所有bit_equal及raw_graph_ABA通过后，才运行服务：

```bash
task-submit --device auto --max-time 780 --ptoas 0.63 \
  "bash $root/norm-layer-v1/run-layer.sh norm_layer_reproduce"
```

脚本自身设置精确SDK/ATB/Python/PYTHONPATH、短TMPDIR、独立OPP/static/cache目录，
冻结源码检查在实际NPU操作前完成。服务固定长度、层号、历史长度并显式打印，不依赖位置参数默认值。
模块结果在`/data/pyptouser/yejia/fp16/<label>/result/result.json`，服务结果在同label根的`result.json`。
成功必须同时有数值、真实图、guards、禁止Native调用及warm后无新增编译门禁；不能只看exit0或最终输出。

换机器/目录时先恢复证据中的helper与上述匹配依赖，再显式修改脚本路径、重新冻结源码摘要；
这些操作会产生新的测试身份，不得沿用本轮源SHA或声称输入/运行环境完全相同。

本轮不能证明其它CSA层、其它T档位、真实整模型192 token、DP16、128K或性能。
