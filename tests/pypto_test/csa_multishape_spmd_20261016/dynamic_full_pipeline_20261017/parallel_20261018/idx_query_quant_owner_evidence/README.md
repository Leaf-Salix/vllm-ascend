# Indexer Query HAD→Quant：单producer获取候选

**源级审阅与有限owner模型通过；候选已冻结，尚未CPU lowering或上卡。** 从正式`66882634`逐文件复制33源到`source_reference`和`source_candidate`，仅修改候选`vector_schedule.py`原697行等待。没有合入上游Cube的RoPE→Hadamard 16-owner候选。

候选保留真实`tokens=pl.tensor.dim(x,0)`、pl.dynamic接口及device Q/L/positions。固定24 AIC、48 AIV和64 heads来自当前算法/硬件合同，不按B/S/七个测试点建白名单。

## 等待可以收窄的完整范围

参考`indexer_query_stages.py:201–254`：QH_MM_TILE=QH_QUANT_TILE=64；模型FLASH的index_n_heads=64、index_head_dim=128。HAD生产者p处理token=`p+24*w`，一次完整写64×128 FP32；量化AIV v处理token=`v+48*w`，唯一外部业务读取是`qh_acc_gm[t*64:(t+1)*64,0:128]`。每个token的完整读跨度为32768 B，没有跨token行、额外valid尾或pad行读取。

因为48是24的整数倍，所有wave均满足`(v+48*w)%24=v%24`。每个活跃AIV只需获取这个AIC；AIC在其全部HAD wave完成后才发布一次phase，所以一次获取覆盖本AIV所有wave，不把阶段通知误当逐tile通知。实际AIV编号为`block_idx*2+subblock_idx`（`vector_schedule.py:368`），生产者为`v%24`，**不是v//2，也不加AIV的24编号偏移**；例如T25时v24必须等AIC0。

量化的两次64列amax归约只访问已经加载的64×128 tile，后续缩放与INT32→FP16→INT8顺序不变。`v>=T`时外层量化range为空，函数没有Had业务读取，可跳过获取；原PHASE_IDX_QUERY_QUANT发布完整保留，包括空AIV。

```python
if worker < tokens:
    signal = wait_producer(signal, PHASE_IDX_QUERY_HAD, worker % GROUPS, CORES, generation)
```

## 别名、前驱与发布/获取

- `binding.py:193–197,224–240`分别分配Had FP32、query INT8及FP32 scale。每token写区分别为INT8的8192 B和scale的256 B，与Had输入独立；唯一相关view为scale `[T*64,1]→[T,64]`和上游RoPE连续view。`entry.py:36–46`要求这些明确view之外的缓冲不得覆盖；不为自定义allocator任意重叠提供额外保证。Had全T工作区没有跨阶段复用、环槽覆写或待删ACK。
- `cube_schedule.py:303–311`在本核全部HAD写回之后发布。`publish_phase`和`csa_signal_notify`保持fence/cache维护与PIPE_ALL后发信号；`wait_producer`（`phase_sync.py:103–117`）保持同代GE等待、不清空信号，以及获取后的fence/cacheinvalid/fence。没有以读到标记替代业务缓存维护。
- 从被改wait到下一次全QUANT获取之间，只运行本AIV query量化、原QUANT发布和WEIGHTS获取；没有读取其它Had owner的业务数据。系数阶段仍在`vector_schedule.py:710–711`原位置获取全部有效WEIGHTS和QUERY_QUANT owner。
- 全有效QUANT owner为`v=0..min(T,48)-1`，这些v的`v%24`并集恰为全部有效Had owner `0..min(T,24)-1`。因此进入系数前，原全Had前驱完成关系通过QUANT发布/获取重新全部建立，业务缓存也再次维护。**不能随本候选继续收窄这条全QUANT等待。** Score AIC自己的全有效QUERY_QUANT等待（`cube_schedule.py:350`）也未动。
- 后续coeff读取其它token scale/weights、compact Had/key/scale以及主QB/QUERY等数据，仍有各自原有获取；META此前已获取。这次仅移除“本AIV量化不读取”的Had owner依赖，不删除后继屏障，不新加等待边，也不改变代际或发布集合。

这条证明独立于请求是否S1/S5/S6、mixed Q或dummy：HAD与query量化本来按物理T的全部token执行，device有效性过滤仍由原数学/后继计划负责。它不构成这些形状的新增设备精度通过声明。

## 有限模型与完整源码审计

`audit_source_and_owners.py`只用标准库读取AST、git对象和数学常量，没有导入PyPTO/Torch或设备库。先核对源码的实际range表达式与唯一Had外部load，再枚举原Had和Quant全部token波。

- T=0..4096共4097个T、196656个AIV/T组合、8390656个token读写块；每个读块的实际producer集合精确等于新等待集合，无漏/多owner。
- 每token全64头的Had写者及Quant写者均唯一，完整32768 B读取只属于对应token且位于`[0,T*64)`行；验证所有wave和最后一个有效token。
- 每个T均验证后继全QUANT的Had前驱并集完整恢复。包含T25/v24、T48/v47、T49跨第二波及T4096多波边界。
- `source_reference`33源逐文本与git668相同；candidate仅`vector_schedule.py`改变。只将新增If节点恢复为原wait Assign后，33个文件完整AST逐项相等，保留docstring、类型、导入、数学与其余控制流；Cube源码逐字未动。
- T0仅证明本局部空循环/空集合。正式Host `policy.py:77–79`拒绝非正tokens，不能据模型宣称整链已支持T0。

| T | 原GM signal_wait调用数（全48 AIV） | 新调用数 | 原/新业务获取次数 |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 / 0 |
| 1 | 48 | 1 | 48 / 1 |
| 16 | 768 | 16 | 48 / 16 |
| 24 | 1152 | 24 | 48 / 24 |
| 25 | 1152 | 25 | 48 / 25 |
| 48及更大 | 1152 | 48 | 48 / 48 |

原wait实际等`min(T,24)`个owner，不是所有小T都等24个。表中只数helper调用，**不是Scalar轮询次数、硬件周期或收益预测**；保留后继全QUANT可能仍由慢Had核约束关键路径。代码可落地不等于完整L1会改善。

## 历史排重与未验证项

MASTER `DSV4_CSA_SPMD_OPTIMIZATION_LOG.md:17222`及主PAR `idx_had_owner_evidence`是**RoPE→HAD上游**16-AIV owner等待，显式保留本次要改的下游全Had获取；不是同一个候选。`compact_priority_evidence/review.md:17`中的Had→量化是compressed key的PHASE_IDX_HADAMARD，也不是本次PHASE_IDX_QUERY_HAD。检索MASTER及CSA测试说明/审计记录未找到本次一处动态wait换为`v%24`的已完成同方案试验。

本次独立复审见`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_idx_hadamard_quant_wait_review_20261018/README.md`，其结论与本报告一致。源码、owner集合和别名已审阅；生成分配、真实事件、分支中的获取顺序及全链精度/性能均待后续授权，当前未做CPU lowering或NPU任务。

## 复现与归档

`prepare_source.py`只从git668建立两份33源，目录存在时拒绝覆盖；现在不要重跑。`audit_source_and_owners.py`可复算纯文本/整数模型，不编译。`SOURCE_PREPARATION.json`、`SOURCE_AUDIT.json`、`owner_model.json`、`candidate.diff`和`LOCAL_ARTIFACTS.json`保存精确路径、全部4097行模型与文件大小，无校验摘要。两份源码从准备后保持冻结，可整体复制本目录。

## 2026-10-18 06:44 后续状态

单次完整CPU编译与生成代码审阅通过，详见`CPU_REVIEW.json`和`generated_review`。130 roots、58动态轴、单MIX保持，AIC全文及AIV除唯一等待块外的归一化PTO相同；分配与局部事件相同。准备独立提交128K/B16与8K/B32两档；本文前面的“未编译”是冻结候选时的历史状态，不代表当前进度。尚无本候选设备收益结论。
