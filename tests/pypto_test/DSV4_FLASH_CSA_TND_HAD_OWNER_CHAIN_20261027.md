# nalinaly Hadamard直接移植与HC owner链单因素验证（2026-10-27）

## 目标和基准

正式精度基准为11b380e3f，当前分支dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex，保留不改。nalinaly最新1685da69，本轮GitHub核对无新提交。直接取其SPMD TND projection_overlap_stages.py::_query_hadamard_four_tokens主体，仅改变量名并放进我方原Hadamard任务；T<4保留原M64路径，其余用M256。

不是固定S6逻辑：每个token有64个独立head row，合并四token只改变矩阵M维。块可跨请求边界，但没有跨M归约；每行仍完整K128。末块以min(256,T*64-first)限制真实读写，没有为请求补行。

## 保留的数值和调度边界

- 原Indexer投影/RoPE、Q_A/Q_B、WKV、压缩器、打分/TopK、attention和O-proj/HC-post均不改。
- Hadamard之后的BF16舍入、归一化乘法后的BF16舍入、FP16 scale及INT8量化顺序逐AST相同。
- 原24个Hadamard worker、依赖、量化任务和allocator scope不变；未组合此前失败候选。
- 单入口、纯DSL、真实TND；不修改SDK或生成C++。

## 无设备审查

双臂完整64参数CPU编译、artifact.load与NPUfalse通过。63个叶子中仅func31 qr_hadamard_matmul变化，其余62个binary SHA完全相同；控制的63个叶子还与上一轮控制一致。

生成M256输入和输出均携带真实valid行；仅一个K128/FP32 TMATMUL。L1共96KiB，L0A64KiB、L0B32KiB、L0C128KiB，无新增UB或GM容量。T1..256行所有权和尾行边界通过独立检查。源级和生成物独立审查均通过；这些不代表设备精度已通过。

## 观测修复历史

首轮task145540在计时前失败：实际READY artifact命中持久缓存，generated目录只有artifact_manifest和源码，没有binary_manifest。没有性能/Native精度结论，也没有算子报错证据。

独立observer-v2保留父helper/CPU数学资格，只改只读身份门禁：从READY registration读取实际loaded callable，显式验证对象身份、PID/generation/owner及loaded identity；按63个child ID顺序比较实际serialized child与CPU期望。CPU期望也通过固定SDK构造parent再提取child，包含合法尾padding，全部字节都核对；不编译、不注册、不执行期望对象。实际编排与CPU继续做strictalpha、meta字节及config AST检查。

第一次无设备正例抓到了直接CoreCallable没有parent尾padding的观测误拒绝；修正后两臂各两次确定性序列化通过，实际child单字节篡改反例均被拒绝，NPU仍未初始化。没有删除缓存、重装软件或改SDK源文件。固定Simpler57e499官方源码及SHA一并保留供复核。

## 性能与精度结果

首轮任务151419完成exit0，但没有通过性能采用门槛：event中位1609.417→1600.299us（快0.567%，12/12窗胜）；独立L1中位1513.84→1521.74us（慢0.522%，仅2/9窗胜）。实际READY注册和63个child逐字节见证、44次guard/readonly全部通过。没有执行Native精度，不能把流程exit0视为精度验收。

反向任务153101完成exit0：event1588.442→1605.645us（慢1.083%，0/12胜），L11511.82→1514.42us（慢0.172%）。显式Native A/B各八边界、三臂ABA全部逐位一致；44次guard/readonly及实际63子项注册见证通过。

结论：单层此输入未引入数值误差，但双向性能不稳定，不合入，不扩短T/混长/128K或整模型。不能把一次B32逐位资格泛化成全shape或整模型token资格。verify_had_measurements.py从两份原始归档复算event、全部18窗L1、实际执行身份和精度范围，HAD_MEASUREMENT_RECHECK.json通过。

8K/B32/S6/T192、真实第4层权重+合成hidden/history，完整HC-pre至HC-post FULL graph。默认先12x100完整event和独立L1，>=0.5%中位收益、>=10/12配对胜且L1更低才做Native A/B八边界和ABA；本轮反向/共享通过--qualify-boundary显式做精度回归，即使性能失败也记录数值结果，未放宽采用门槛。guard/readonly始终执行。必须反向捕获重复稳定；通过后才扩T21/T126、短T、formal95路由和8K/128K真实整模型token，不将单层资格外推。

所有自有NPU任务outer780/inner750秒，排队不计设备执行时长，原基准不受失败候选影响。

## HC owner链单因素

HC前缀原linearsum按M16分工，紧接pre/post按M8分工，因owner不同需要H4全员会合。借鉴nalinaly按实际owner衔接，只将linearsum逐元素加法改M8，使rms/linearsum/pre_post同属floor(row/8)%48；原22份FP32部分和的相加次序完全保留。不复制nalinaly不同数值的HC实现，不组合Had或此前Query岛。

保留H3的Cube linear生产者屏障、H5的pre到跨owner mix屏障，以及其它所有必要屏障；H4替换为本核bar_all/cacheinvalid/fence，Cube删对应空H4。生成代码实际是TSTORE后MTE3→MTE2 EVENT7等待、PIPE_ALL/DCCI/DSB，再pre_post读取。两角色各9→8次syncall，尾行只写P16内raw0:32，不碰p16起点的tail scratch；UB静态最高end141056字节两臂一致。

双臂64tensor CPU编译/load、NPUfalse通过。63个子项只有prefix Cube/Vector变化，其余61个binary SHA相同；root编排严格alpha等价。源码和实际生成物经过独立审查。v1缺PARTIAL_ROWS导入仅CPU失败，v2独立修复后重新资格，无NPU失败或环境修改。

| 8K/B32/12×100event/9窗L1 | Direct event us | HC owner event us | 耗时变化 | 胜窗 | Direct L1 us | owner L1 us | Native精度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 正向 | 1613.909 | 1606.380 | -0.467% | 12/12 | 1518.62 | 1511.58 | 未执行 |
| 反向 | 1583.499 | 1597.600 | +0.890% | 0/12 | 1493.80 | 1512.18 | A/B8与ABA逐位一致 |

两轮实际注册见证及各44次guard/readonly均通过，但完整性能方向反转，不采用，不扩128K或整模型。正向未执行精度用null记录，不能写为失败。数据由verify_owner_measurements.py从原始归档复算。

Had和HC都出现方向反转，尚不能归因某一个物理原因。因此补测共享同一套64个物理实参的直接root FULL诊断，先与各自原服务图做same-arm八边界等价，再普通event/独立L1和Native A/B8+ABA；结果见下节，与原服务图分表，不修改或覆盖原数据，也不能直接外推整模型。

## 共享64个实际捕获实参的诊断

独立helper不修改parent数学/CPU资格。原Native和两臂服务图仍先完成初始化、eligible、实际注册资格；再以控制的64个真实实参构建各自root FULL图。显式forward替换x_hc(0)、position_ids(40)、x_out(53)，其余实参以及cache/output/topk/workspace全部共用。计时前，新eager/FULL各自与自己的原服务图八边界逐位一致，poison和guard通过。

只在新图捕获时临时包装SDK现有_enqueue_frame作只读观测：从实际将原样传给native.prepare的CallFrame核对64个Tensor的pointer、shape、stride、storage offset、dtype和storage alias；要求实际capture_id非0，原提交函数不变，finally恢复且所有计时/回放窗之前已无hook。再次检查新图真实READY注册的63个子项/编排/meta/config与CPU一致。没有修改SDK文件或生成C++。

| 共享实参直接root FULL，8K/B32 | Direct | HC owner候选 | 候选耗时变化 |
| --- | ---: | ---: | ---: |
| 普通event中位us，12×100 | 1575.633 | 1577.533 | +0.121% |
| 独立L1中位us，9窗 | 1517.16 | 1510.02 | -0.471% |
| 对Native A/B八边界 | 逐位一致 | 逐位一致 | 无首差 |
| Native与两臂ABA | 通过 | 通过 | 无首差 |

完整event仅2/12窗领先，44次guard/readonly通过。虽然L1降低7.14us，完整回放没有收益，仍不采用，不继续扩128K/整模型。原服务图双向结果完整保留、分表记录；本项不能证明之前方向反转必由某个地址因素造成，也不能直接外推整模型。

实测链已核验，可以将后续精力转到更大独立Indexer相位：复用nalinaly相位/owner协议，保留我方Native数值体；不再以几个us的不稳定变化作为合入依据。

## 当前可以直接借用和需要保留的部分

| nalinaly实现部分 | 本轮做法/资格 | 采用状态 |
| --- | --- | --- |
| Hadamard四token M256 | 直接复制主体，保留K128/所有Native量化；本轮B32 A/B8+ABA逐位一致 | 完整性能两向不稳定，不采用 |
| HC生产消费owner衔接 | 保留我方数学，重排M8行owner、局部排空，省一次全体会合 | 反向和共享诊断逐位通过；完整性能未获益，不采用 |
| 独立Indexer相位/Score链 | 下一候选设计，允许复制相位和信号骨架，主Q_B不进入Score前驱 | 尚无编译/设备结果 |
| scale相乘、K顺序、Norm/softmax舍入 | 我方已验Native数值顺序保留，不为复制速度路径改写 | 原精度基准保持 |

不要将前一轮c346的约1088us同SDK单层性能冒称最新1685实测。本轮没有重新测完整nalinaly或Native速度；用同一Direct控制评估单因素，不能从它们的L1中位差直接推断纯AICPU开销。

## 环境与复现

当前分支`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`；正式运行基准11b380e3f保留，文档HEAD可更新。新SDK：PyPTO d44925c17d9ff7d7fa8e5067e41c4b4f80bf3063、PTOAS914863ed3d0eb3e92a4433a4a706a86cabf4c91f（native0.69）、Simpler57e499238fd4ee3548947ad860be456aaeeb16b5。Python3.12.14、Torch2.10.0+cpu/Torch-NPU2.10.0.post2、vLLM0.25.1、CANN9.2.0-beta.2，静态13包metadata SHA863a1e5daffc54aec0e9e8afb9f809f1bc6f4b3a2487e440615f3c942f36c602。完整导入路径和.so SHA见每次preflight，已有SDK未改动。

证据目录`evidence_tnd_precision/had_owner_chain_20261027/`包含执行过的helper/manifest、CPU生成文本和收据、原始event/L1、失败历史、实际注册见证和离线复算脚本。按该目录README复现；文件字节清单见FILES.json。原始完整CPU档不发布二进制，SHA及大小在RAW_CPU_INPUTS.json；重新编译必须再次通过身份门禁。更大独立Indexer方案见NAL_INDEXER_OWNER_CHAIN_NEXT_PLAN.md，尚不能声称有收益或精度资格。

## 本轮结束状态

所有自有设备任务已完成并释放设备，均设置outer780/inner750秒；本轮没有新SDK或候选的整模型8K/128K token验收，正式11b精度基准没有改动。16:46再次通过GitHub核对nalinaly HEAD仍为1685da69，未把旧c346性能改标为最新版本。测试源、实际生成及数字经过独立审查，新的更大Indexer链还只有方案。
