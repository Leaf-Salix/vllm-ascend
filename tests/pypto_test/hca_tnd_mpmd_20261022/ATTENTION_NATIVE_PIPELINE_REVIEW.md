# HCA attention：参考recipes与A3核内实现优化PV流水

2026-10-23。用户明确本轮主要参考attention。候选先在冻结包测试，再按区间集成，
用完整HCA直接调用的L1首尾判定，不使用ACLGraph或重新比较Host/JIT。

## 已实际读到的来源

GitCode网页限流，git请求返回Empty reply。原本地recipes checkout是部分克隆，
attention文件只有目录项、缺少blob，不能把ls-tree结果当成已读过源码。
随后从`hicann/cann-recipes-infer` GitHub镜像按**相同提交**
`6d060311ae75f6105fe4216e20566875da5a8838`取得源码，没有冒称最新master：

- `ops/pypto_python/impl/compressed_flash_attention_pypto.py`：组合raw滑窗与压缩KV、
  sink及softmax/PV计算，配置Cube/Vector nbuffer。该版接口要求KV页128，计算体只拼
  四个压缩页，并通过`T//B`推导统一S。不能直接搬入当前Native页32、128K以上历史、
  S1～S6混长的TND算子，也不能因此删掉本仓device Q/L及边界处理。
- `ops/pypto_python/impl/sliding_window_attention_pypto.py`：明确使用TND及cu_seqlens_q，
  可参考有效窗口和按形状组织计算；其动态API不是当前hw-native-sys PyPTO语法。
- `ops/tilelang/ds_v4/sparse_flash_attention.py`：QK→online softmax→PV→输出归约，
  有UB持久累积和局部KV缓存；这是稀疏Top-K示例，不是C128直接替代品。
- `ops/ascendc/docs/custom-npu_sparse_attn_sharedkv.md`：接口与边界说明，
  不将接口说明当作全部核内调度的证据。

完整只读文件、原始URL、提交和许可保留在`references/recipes_attention_20261023/`。
没有安装或执行这些参考代码。实际A3 C128路径继续核对本仓
`csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/`的SWA/CFA实现：

- `sparse_attn_sharedkv_swa_kernel.h`：PRELOAD_NUM=2，三份RunInfo跨query衔接。
- `sparse_attn_sharedkv_swa_block_cube.h`：L0A/L0B各两个32KiB槽，L0C两个64KiB槽；
  QK沿D拆L1 K256、L0 K128；PV同样通过乒乓槽和事件交叠搬运/计算/写回。
- `sparse_attn_sharedkv_swa_block_vector.h`：按实际列数做softmax，输入缓冲交替，
  新旧PV缩放/相加及最终除法。它也存在GM中间累积，不能笼统说Native完全驻留UB。

## 先排除已经存在和已经失败的方向

本仓已有QK两步提前执行、三个KV槽、跨query连续流水，生成码也已有QK L0双缓冲。
不把再次写这些选项算成新增优化。旧double-query、K512归约、分页直读、full-mask快路径
均有失败/回退记录；本轮没有原样重复那些试验。
旧K512候选还有KV重载、不同L1微块和softmax舍入变化，不能把它的结论等同于Native结构。

## 本次选择的缺口：PV供数与结果要同时双缓冲

直接检查当前提交`4a65d78cd`的AIC生成C++，PV的四个128列输出块将Right均分配在L0B地址0。
相邻`TEXTRACT`之前存在`wait_flag(PIPE_M, PIPE_MTE1, …)`，由同地址复用引入串行约束。
原L0C则已有0/32768两个结果槽。每块仍是64×128×128，Right32KiB、Acc32KiB。

第一版改成`pl.pipeline(4, stage=2)`，CPU实际编译通过，生成码Right分为0/32768，
但Acc又复用成地址0。只改善供数、牺牲FIX与MAD交叠，并不是完整双缓冲方案，未发起设备测试。

第二版显式保留相邻两块Right及两个结果的同时存活：先提取0/1列块并各自matmul，
写回第0块后提前准备2/3块；第2块matmul后写回第1块，再算第3块并收尾。
生成码Right与Acc均为0/32768，完整编排和两个根入口的incore C++编译通过。
数学点积、K128分块、块内softmax、跨块alpha/beta更新、FFTS事件和TND几何完全保持。
没有新跨核同步，没有修改共享SDK；是否更快以完整层配对结果决定，不靠地址表预记收益。

| 实现 | PV L0B地址 | PV L0C地址 | 设备验证 |
|---|---|---|---|
| 当前正式基线 | 0 | 0 / 32768 | 已有基线 |
| pipeline-only候选 | 0 / 32768 | 0 | 不单独测量 |
| 显式双Right/双Acc候选 | 0 / 32768 | 0 / 32768 | B28改善、B16回退，不整体采用 |

地址证据与带行号C++片段：`evidence/ATTENTION_PV_CODEGEN.json`及同目录`attention_pv_*_codegen.txt`。
完整候选为`candidates/attention_pv_l0_double_buffer.py`、`attention_pv_l0_dual_buffers.py`；
运行身份分别冻结在`experiments/attention_pv_db_v1`、`attention_pv_dual_v2`。
控制为`attention_resume_base_v1`，两侧使用公共`csa-shared-20261022-logical-codegen`。
任务及参数在`ATTENTION_PV_TASKS.json`，不能将Native严格storage旧差异改写成PASS。

## 小粒度两项结果与转向大块的原因

同任务、每侧七窗，均为完整HCA的L1 incore首尾，单位μs：

| 候选 | 128K/B16 | 128K/B28 | 决策 |
|---|---:|---:|---|
| PV双Right/双Acc | 499.04→506.34，1/7更快 | 766.96→758.38，6/7更快 | 不全局采用 |
| Q的后三个K128块驻留L0A | 514.90→520.02，1/7更快 | 765.52→768.08，3/7更快 | 不采用 |

第二项保留QK的K0与PV共用16KiB，其余48KiB保存Q的K1/2/3，减少反复L1→L0A提取。
两项都保持输出/state/cache精确，但仅减局部搬运不能稳定缩短完整层，继续堆这些选项意义有限。

## 大块候选：K512归约、D256供数，保留三槽与两步提前执行

候选`candidates/attention_macro512_d256.py`由`build_attention_macro.py`生成。
以下先记录独立候选，最终集成见末节。不同于旧K512试验的两槽/一步提前执行，本版保留三份CV交接槽、
QK提前两步以及跨query连续调度，保留原device Q/L、页表、非法页与未初始化行屏蔽。

- QK将N512拆成四个N128，每个块按D256供L1、K128供L0，FP32累加。
- PV在四份64×128 FP32的L0C中完成宏块K归约，四份共128KiB；之后才写回D512结果。
- Softmax每个AIV先处理H16再处理H16；在线max/sum及左右D256累积仍在本AIV的UB中。
- KV不再能把三份完整K512×D512常驻L1；PV会重载KV。这项成本明确计入比较。
- Cube延迟PV的描述符采用本核Scalar FIFO，Vector采用原UB描述符环；跨核交接仍是FFTS事件。
  没有GM Scalar轮询，没有新增Host取值或AICPU metadata算子。

128K且有效压缩历史1024行时，块数从`8个K128+1个raw`变为`2个K512+1个raw`。
QK/PV有效MAC数量没有减少，但跨块更新和PV FP32中间结果少了六份。
以H64、D512计，每query的PV写回加Vector读回逻辑字节少1.5MiB，PV重载KV增加约1.125MiB，
净减少约0.375MiB；B16/S6约36MiB、B28/S6约63MiB。这只是源码逻辑GM字节估算，
没有据此声称HBM实测流量，也不能忽略L2命中、有效尾块及调度影响。

### 精度与性能证据

独立attention探针用实际候选函数，生成分页fixture并比较密集FP64。两次eager执行精确，
输出尾保护区不变，未发现非有限值；边界覆盖负页、未初始化NaN、部分历史、零有效长度。

| 合成场景 | 原版对FP64 max_abs / RMSE | K512对FP64 max_abs / RMSE |
|---|---:|---:|
| history=131072，B5/S6 | 0.00111895 / 0.000126769 | 0.00110998 / 0.000129745 |
| history=16507，边界组合 | 0.00237789 / 0.000206907 | 0.00237789 / 0.000206933 |

两场景候选对原版均有最大0.001953125差异。扩大归约块会改变舍入，**不是逐bit调度优化**。
探针只报告误差，不设更宽的验收阈值。完整层继续用原Native输出0.01/0.01容差，
cache/state必须与原版精确，重复执行/写保护也必须通过。

| 完整层首轮 | 原版L1 | K512 L1 | 性能提升 control/candidate−1 | 更快配对 |
|---|---:|---:|---:|---:|
| 128K/B16/S6 | 517.60 | 497.80 | 3.98% | 7/7 |
| 128K/B28/S6 | 769.46 | 748.48 | 2.80% | 7/7 |

Native输出通过原容差，cache/state相对控制精确。Native严格storage原差异仍在，不能写成整层
所有检查或整网token通过。attention AIC平均核耗时分别约147.81→140.38、230.73→221.16μs；
完整层还受到任务起止位置变化影响，不把约20μs全归因于Cube算术提速。
其余档位、S1/S5/混长与padding验证见持续更新的`ATTENTION_SUMMARY.md/json`。

### 编译尝试、负结果与新增对照

1. v1/v2的条件yield/FIFO偏移触发未绑定symbol；v3将QK load的地址直接写成当前request/tick
   表达式，并用分支无关的Scalar FIFO描述符排空，完整两个JIT根及incore C++编译通过。
   排空拍描述符夹到raw块；只有已生产的前两拍才消费，不越界读取未来项。
2. 四组纯CPU最小复现确认是内层pipeline克隆外层Scalar IterArg的问题，
   `LowerPipelineLoops`首次出现`old__FREE_VAR`，普通range控制通过。
   已提交[PyPTO #3130](https://github.com/hw-native-sys/pypto/issues/3130)，含源码定位、
   建议修改和完整可运行复现，未改共享SDK、未提PR。证据在`evidence/PYPTO_PIPELINE_EXTERNAL_ITERARG_ISSUE.*`。
3. 独立探针v3遗漏bootstrap传入的`--device`参数，运行器在设备算子执行前退出2；
   v4只补参数接收，未覆盖失败日志。
4. Softmax每AIV一次H32的候选需要202080B Vec，超过188416B上限，CPU阶段停止，未上卡。
   不通过删缓存保护或改硬件限额强行编译。
5. QK相邻N128块由range改为pipeline深度2，保持每点积K归约顺序。完整编译通过，
   生成码的前两个QK Acc地址为0/32768，PV仍为0/32768/65536/98304。
   B16/B28增量对照的控制**已含K512**；若无收益不叠加采用，数据见同名任务记录。

完整CPU结果、失败日志、冻结包路径、原始L1路径和大小保留在本目录的evidence及
`ATTENTION_ARTIFACTS.json`；无权重、二进制或原始大trace入Git，不做hash扫描。

## 用户允许多区间选择后的正式集成

用户明确“不用全局替换，可以多一点档位选择”。`attention.py`现在由初始化时的工厂绑定
K128/K512，分页聚合、mask、FFTS、PV输出归约和inverse RoPE共用，运行期不执行生成器。
新增约200行计算策略，没有再复制整份attention；完整层三个JIT入口仍共用36项device Tensor ABI。

| 已有Host字段与shape | attention | 投影 |
|---|---|---|
| 非NZ、max_seq_lens缺失/小于131072，或T≤32 | K128，KV三槽驻留L1 | 原策略 |
| NZ、max_seq_lens≥131072、33≤T≤128 | K512，D256供数 | 原策略 |
| NZ、max_seq_lens≥131072、129≤T≤192 | K512，D256供数 | 已有Q_B/O_B M192/N128 |
| NZ、max_seq_lens≥131072、T>192且未超过原容量 | K512，D256供数 | 原大T通用策略，尚无新增性能数据 |

`max_seq_lens`是Native已有的Host最大KV长度；T是输入token容量，不通过设备回读获得。
每个请求的实际长度、query边界、位置和有效页仍逐次读取设备Tensor。分档不是B×6静态化。

### 最终版本对本轮原版的配对结果

同任务双侧完整层L1首尾。短历史/小T只是核对共用模板没有退步，不把其采样差当作K512收益。

| 场景 | 控制μs | 最终μs | 性能提升 | 更快配对 | 实际策略 |
|---|---:|---:|---:|---:|---|
| 128K/B4/S6 | 273.22 | 266.30 | +2.60% | 见JSON | K128 |
| 128K/B8/S6 | 351.50 | 347.58 | +1.13% | 5/5 | K512 |
| 128K/B16/S6 | 537.84 | 518.42 | +3.75% | 7/7 | K512 |
| 128K/B24/S6 | 687.20 | 676.86 | +1.53% | 4/5 | K512+大T投影 |
| 128K/B28/S6 | 771.66 | 765.88 | +0.75% | 6/7 | K512+大T投影 |
| 8K/B32/S6 | 603.94 | 598.44 | +0.92% | 见JSON | K128 |

B28最终收益小于首轮，所以单独用“原始K512候选”作控制检查共用模板：
`attention_policy_refactor_b28_v4`为762.66→748.12μs，7/7更快，输出/cache/state精确。
这不支持“共用模板导致回退”的判断；也不能把它与771.66的另一组控制相拼来报收益。
所有正式结果均是Native输出原容差通过、cache/state相对各自控制精确、重复和写保护通过。

### 动态与边界验证及明确遗留

- S1/B16真实custom op三次调用：default；S5/B28：tall_projection；
  B6、Q=[3,6,6,6,6,6]、T=33：long_attention。均通过接管/KV钩子顺序、
  Native输出及与对应冻结数学版本的精确比较。
- 256K/B16：long_attention，输出与独立K512版本精确，Native输出通过原容差。
- 正式K512在独立混长Q=[1,5,6,1,2]下对FP64：128K max_abs=0.00110998、
  RMSE=0.000129802；16507边界组合max_abs=0.00237789、RMSE=0.000217861。
  重复、非有限值、失活请求、尾部保护检查通过。
- **整层混长跨C128边界未通过Native输出**：B5、Q=[1,5,6,1,2]、
  history=[127,16383,131073,0,16507]、T容量24、请求容量8。
  原K128与候选K512对Native的max_abs都为0.0810546875；
  原版25702个元素超差，候选11332个。候选cache/state与原版精确，保护区/重复通过。
  这是新增场景暴露的既有整层问题，不能写成新attention已经解决，也不能绕过门禁采性能。
  S1/S5及上述独立attention通过不代表这项整层验收通过。

### 共用模板的CPU尝试也保留

- policy_v1误把`pl.constexpr`当函数调用，属于本地DSL写法错误；改成编译期常量条件。
- policy_v2两分支的归约环分别16/32列，前端检查未选分支时不匹配；统一环布局。
- policy_v3把H16结果插到H32环列子视图，生成TMOV后被A3拒绝。
  独立两布局CPU复现确认：3×32列插入失败、6×16整行插入通过。
  已补充到既有[PyPTO #2717](https://github.com/hw-native-sys/pypto/issues/2717#issuecomment-6363110842)，
  含完整缩减源码、PTO片段和subview/物理stride修复建议，不重复新建issue。
- policy_v4统一6×16环：K128一次写两行、K512分两次各写一行，逻辑次序和384B容量相同。
  三个完整JIT根、编排C++和所有incore C++均编译通过，随后进行上述真机验证。
- QK相邻N128深度2的增量候选：B16 535.06→545.94μs、B28 757.44→759.36μs，
  输出精确但性能回退，不进入正式分档。
