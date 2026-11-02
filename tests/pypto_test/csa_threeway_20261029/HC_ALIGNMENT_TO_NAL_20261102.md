# HC对照nalinaly效率版：细分测量与单因素验证

## 范围与结论

承接[真实图容量归因](PERFORMANCE_REBASE_20261102.md)，本轮保留默认
`65bebf7f`精度基准，单独对齐HC的搬运和流水。候选互不叠加。
没有修改共享SDK或生产生成代码，算子候选全部使用现有DSL。

本轮有效候选为RMS批量对角线读取：图容量T240/有效T48下降1.09%，
全有效T240下降1.42%，T6下降1.76%；混合T18基本持平。
T240、混合T18、T6的输出及五组完整物理cache/state均与Native FULL逐字节一致。
默认生产算子仍冻结，不把单层验收视为新的整模型逐token验收。

直接同环境对照仍为nalinaly`1574ed59`。
已只读确认其最新公开分支`2d05303d`的
[HC源码](https://github.com/nalinaly/vllm-ascend/blob/2d05303d613a901577a58c24c341e07967444fb7/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd/hc_stages.py)
仍保留4段K4096、内层K256两级流水。最新ACLNN/SuperKernel整网后端
与本轮Python/Simpler单层不同，不跨后端计算收益。

## 两方HC实现差异

| 部分 | nalinaly效率版 | 我方冻结精度版 | 本轮处理 |
| --- | --- | --- | --- |
| 输入加宽 | BF16到FP32，同时算32×K512平方和 | 同一旧加宽/RMS helper | 本轮保留 |
| HC输入布局 | 直接消费FP32输入 | 另做HF32模拟所需的8列重复打包 | 改批量布局做单因素 |
| 权重布局 | 原FP32权重 | HF32位转换、BF16高低位和8行交错 | 本轮保留 |
| 投影 | 4段K4096，K256内层两级流水 | 22段K768，重复输入/高低位乘积，宽M内层普通循环 | 尝试两级流水 |
| 输入RMS | 消费旧Vector平方和/rsqrt | Cube按Native顺序生成22段平方partial，再左折叠、sqrt/div | 数学保持，只改读取 |

因此不能把效率版整个HC数学路径原样替换后声称精度不变。
本轮只借鉴批量搬运和流水，继续保留Native归约和舍入边界。

## HC内部时钟

沿用真实第4层权重、合成共同hidden/cache/history，8K，有效B8/T48，
图容量T240、40请求槽。普通性能仍为完整FULL graph Event计时，
5次热身后30次交替A/B，恢复cache与检查在计时外。

诊断副本复用G6输出组流水下没有消费者的40/41槽：

- AIV：加宽完成、输入打包完成，原INPUT标记表示权重打包完成。
- AIC：HC投影开始、投影完成，原PARTIAL标记表示平方和完成。
- CPU和设备入口均检查`OUTPUT_GROUP_PIPELINE=True`；其他策略不能使用此诊断。
- 所有原计算和生产消费标记保留。诊断eager/FULL及5次重复FULL的
  输出有效行和全部物理cache均与各自未插桩版本逐字节相同。

下表是5个窗口中每个实际参与核完成区间的中位数，单位μs；
加宽只统计30个有M8块的AIV，打包统计48个AIV，投影/平方统计24个AIC。

| 区间 | 我方 | nalinaly |
| --- | ---: | ---: |
| AIV输入加宽及旧RMS | 23.93 | 20.44 |
| AIV输入打包 | 51.74 | 无；相邻标记0.46 |
| AIV权重打包 | 11.90 | 无；相邻标记0.48 |
| AIC HC线性投影 | 14.62 | 16.62 |
| AIC Native顺序平方和 | 35.64 | 无；相邻标记0.44 |

这些是**插桩后包含搬运、等待和排空的完成区间**，不是未插桩纯计算时间。
新增publish会fence/cacheinvalid/排空FIX，可能改变原阶段间重叠，不能相加，
也不能把nalinaly相邻标记机械当作可扣除常数。

本形状下HC线性本体并不比nalinaly慢；较大额外工作在输入打包和Native平方和。
这解释了为什么继续只调投影流水不是优先方向。

任务`task_20261102_231254_34725759713`完成，物理8卡；
两方各5窗自对拍通过，普通版本不含这些新时钟标记。

## 单因素一：HC投影两级流水，不采用

`hc-linear-pipeline2-20261102-v1`只把宽M分支后续K循环从
`pl.range(1,count)`改成`pl.pipeline(1,count,stage=2)`。
首次matmul、递增K地址、同一FP32 accumulator、22段左折叠、短M路径均不改。
逆AST检查证明整个模块的其他内容不变。

CPU及独立生成物审查：55参数、原AIC/Vector两角色；MMA按k/k+1递增，
同一Acc，尾项与FIX完成依赖保留。L1上界475136B，L0A/B各65536B，
Acc131072B，缓冲区没有越界。

| 实现 | 中位数/μs | 均值/μs |
| --- | ---: | ---: |
| 同组冻结基准 | 1354.61 | 1355.71 |
| HC两级流水 | 1360.44 | 1361.75 |

候选慢0.43%，仅10/30对更快。不继续精度验收，不合入。
任务`task_20261102_231017_341768229436`完成，物理8卡。

## 单因素二：输入reshape/concat，不采用

`hc-input-concat-20261102-v1`保留原FP32往返和尾行清零，
把transpose→32次8行双复制→transpose替换为：
`16×256 → 512×8 → 横向concat自身 → 512×16 → 16×512`。
独立索引枚举证明8192个输出元素与旧布局相同，逆AST保护其余计算。

生成物显示连续RowMajor reshape。FP32输入16KiB位于`[65536,81920)`，
concat输出32KiB位于`[32768,65536)`，互不重叠；Vec上界仍155872B。
concat→cast→MTE3写回和复用等待保留，没有新增核角色。

| 实现 | 中位数/μs | 均值/μs |
| --- | ---: | ---: |
| 同组冻结基准 | 1349.13 | 1349.07 |
| 输入reshape/concat | 1342.53 | 1342.23 |

中位下降0.49%，20/30对更快，未达≥1%且≥24/30的预设稳定门槛。
不合入，未继续跨实现/Native精度验收。
不能用51.74μs的诊断区间推导这项布局改写可省51.74μs。
任务`task_20261102_233014_38169064625`完成，物理8卡。

## 单因素三：RMS批量对角线读取

`hc-rms-batch-gather-20261102-v3`只改变22段对角线的读取方式：
一次Tile MGather按行搬入176×16个FP32值，再在Tile中取176个原对角线。
保持part0到part21的FP32左折叠、sqrt、1.0/div及同域8行写回。
1.0常量和store仅改用同数值Tile形式以满足接口。

行地址为`part*p16+t0+lane`，本地对角线为
`(part*8+lane)*16+(t0%16+lane)`；后者≤2815，不串行。
地址构造经INT32桥接，范围小于2^24，FP32索引构造不改变整数地址。
枚举涵盖p16=16、32、48、64、128、192、240、256及所有M8块。
原PARTIAL等待保证22段物理producer已完成，没有提前读取尚未生成的值。

v1在CPU发现Tensor gather结果不能直接Tile extract；v2遇到INDEX→FP32不支持；
v3改为已验证Tile MGather和INDEX→INT32→FP32。失败均未上卡、SDK未改。

生成物独立审查通过55参数/两角色及Vec上界155872B。
MGather输出`[32768,44032)`，对角线索引`[44032,44736)`，scratch
`[44736,45440)`，TGather值`[45440,46144)`分离；MTE2完成后才复用原索引区。
22段加链、sqrt/div和8×1同域store保留。

### 性能与精度结果

均为8K、同卡串行30对交替A/B；性能来自未插桩完整FULL graph Event计时。
收益定义为`1 - 候选中位数 / 同组冻结中位数`，不同配置不横向相减。

| 有效请求/图容量 | 冻结中位数/μs | 候选中位数/μs | 延迟降低 | 更快对数 | 精度结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| B8/T48，容量T240/40槽 | 1349.93 | 1335.18 | 1.09% | 25/30 | 与冻结输出及五组完整分配逐字节一致 |
| B40/T240，无请求padding | 1963.02 | 1935.22 | 1.42% | 29/30 | 与冻结及Native FULL逐字节一致 |
| B4/T18，query为1,5,6,6 | 729.05 | 729.35 | -0.04% | 18/30 | 与冻结及Native FULL逐字节一致 |
| B1/T6，无请求padding | 653.17 | 641.68 | 1.76% | 28/30 | 与冻结及Native FULL逐字节一致 |

均值分别为1351.60→1337.29、1961.27→1935.87、730.60→725.76、
653.03→641.97μs。混合T18的均值下降没有中位数和胜出对数支持，不认定稳定收益。

Native三项均先验证自身eager/FULL一致；候选输出、SWA、compressed、state、
indexer、indexer_state共六项均相同，五组物理分配的非合法slot写入字节均为0。
这里没有导出内部TopK，也没有测新候选的整模型token；不能宣称任意输入逐bit等价。
有效T48/容量T240只对冻结版验收，未冒用不同容量Native图作为oracle。

### 验收来源绑定

候选`hc_input_rms_stages.py`的SHA256固定为：

```text
73b41dde99ee0d3a43e069fba277715c55092c3cd000bfb53b0a2a943a49bbd7
```

最早T48结果没有记录运行时文件哈希，因此不回填哈希作为后续资格。
T240原始结果在运行时已记录并校验候选文件哈希，且本项自身通过≥1%、≥24/30
的性能门槛及冻结/Native精度验收。后续`qualification-bound.json`仅把该原始结果
已记录的哈希提升为统一字段，并保留原始文件SHA和字段来源；不读取当前manifest补写历史。
T18/T6启动时验证当前文件哈希与该资格记录一致；当前形状自身性能门槛单独记录，
不会因引用T240资格而把T18性能标为通过。

### 任务与原始证据

| 内容 | 任务 | 结果目录名 |
| --- | --- | --- |
| T48/容量T240 | `task_20261102_234837_87314961` | `hc-rms-batch-gather-padded-20261102-v3` |
| T240 Native | `task_20261103_000017_22272922338` | `hc-rms-batch-gather-native240-20261102-v3` |
| 混合T18 Native | `task_20261103_001018_4283938599` | `hc-rms-batch-gather-native18-mixed-20261103-v3` |
| T6 Native | `task_20261103_001333_52648829894` | `hc-rms-batch-gather-native6-20261103-v3` |

上述任务全部exit=0并释放设备，仅用物理8卡。证据包含原始结果、冻结runner、
候选manifest、时钟原始窗口和CPU编译记录；不包含模型权重或大tensor快照。

归档：[本轮完整证据](evidence/hc-alignment-20261103-v1-evidence.tar.gz)，
压缩后2649466字节，120个实验文件及一份逐文件SHA清单。归档SHA256：

```text
99042078ce25481f99a9ac908e1b7e74c240edb95fe46569068b060c3934cef6
```

独立审查确认：候选包57个Python文件只有RMS文件变化，运行时哈希资格链有效，
索引/尾M8有界且22段累加顺序保持；未发现阻断。审查结论同样限于上述观测范围。

## 环境与复现

共享工具链仍为`csa-shared-20261024-group-dims`，PyPTO d44925c、
PTOAS914863ed、Simpler57e499、CANN9.2 beta2、Python3.12、
Torch2.10/Torch-NPU2.10.0.post2、vLLM0.25.1，NZ2和确定性环境。
CPU构建用ccache及四路并发，不占卡。每个硬件任务外层300秒、内层240秒。

```bash
# HC细分时钟，CPU先分别对ours/nalinaly运行：
python phase_clock_compile.py "$r" ours "$cpu_out" --batch 40 --hc-detail
# 按项目共同env和bootstrap包装后，使用task-submit：
python paired_single_layer.py "$r" "$out" --batch 8 \
  --token-capacity 240 --request-capacity 40 --clock --hc-detail \
  --clock-cpu-label 20261102-hc-detail-v1
# 未插桩A/B，source由相应*_candidate.py生成，候选互不叠加：
python paired_single_layer.py "$r" "$out" --batch 8 \
  --token-capacity 240 --request-capacity 40 --candidate "$candidate/source"
# T240自身通过性能门槛后验收Native FULL：
python paired_single_layer.py "$r" "$out240" --batch 40 \
  --candidate "$candidate/source" --native-check
# 以绑定原始运行时哈希的T240结果为资格，继续其他形状：
python paired_single_layer.py "$r" "$out18" --batch 4 --query-lengths 1,5,6,6 \
  --candidate "$candidate/source" --native-check \
  --qualification-reference "$qualification"
python paired_single_layer.py "$r" "$out6" --batch 1 \
  --candidate "$candidate/source" --native-check \
  --qualification-reference "$qualification"
```

只允许物理8–13、15，14与0–7禁用；全部通过task-submit。
任何单层结果都不冒充新整模型逐token资格，不重跑Native2或128K。
只有普通性能稳定胜出，才扩展冻结版/Native精度和不同T验证。

## 下一步

不采用投影两级流水和concat布局候选；保留RMS批量读取为独立待整模型资格候选。
8K/192 token复测快照已准备，复用已有Native结果，不跑Native2或128K。
现有DP=EP=8需要八卡；默认允许的8–13、15只有七卡，未获本轮14卡例外授权前不提交。
无设备预编译任务`task_20261103_002328_76665812565`已exit=0；
使用整模型模块命名空间、table_columns=2048、max_seq_len=9000、task_window=2048，
T240用时219.65秒、T6用时189.09秒，两项均`pass=true`且`npu_initialized=false`。
凭据为[evidence/T240](evidence/hc-rms-model-t240-cpu-20261103-v1.json)及
[evidence/T6](evidence/hc-rms-model-t6-cpu-20261103-v1.json)；这不构成设备或token验证。
全模型通过后再考虑生产入口接入；尚不能用本轮约1%的单层收益宣称追平nalinaly。
后续重点仍是避免输入重复打包、复用布局；HC投影本体不是当前首要差距。

一个尚未实现的单因素方向是分离“加宽已完成”和“全部INPUT已完成”：
目前AIC先等待AIV完成加宽、输入打包和权重打包，再依次执行HC投影及Native平方和。
平方和只依赖加宽后的FP32输入，可以考虑提前到打包期间执行，再等待INPUT后做HC投影。
两者独立，必须保留各自内部归约、同一PARTIAL完成条件和代际语义。
这需要独立新阶段槽、同步区容量/初始化和生成物审查，尚无性能或精度实测；
不能把插桩的35.64μs平方区间直接算成预期净收益。
源级独立审查未见等待环，但必须48个AIV均发布完成，并沿用现有payload排空/获取协议。
如果增加阶段60并将COUNT改为61，信号区应增加36KiB，每AIV清零Tile增加768B；
所有入口、清零视图与诊断工具需同步更新，不能复用旧artifact或旧signal分配。
新增发布本身会排空流水，仍需先生成物审查、再做未插桩性能A/B，不预设有收益。
