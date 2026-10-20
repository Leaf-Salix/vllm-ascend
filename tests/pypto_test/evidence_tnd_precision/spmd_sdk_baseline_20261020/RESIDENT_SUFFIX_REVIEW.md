# Resident O-proj / HC-post 独立审查

## 结论

**源码/CPU映射审查通过，允许继续独立 CPU lowering；未获得设备同步/精度/性能资格。** 未修改候选或环境、未提交设备。当前仅适用于显式 NZ2 + CANN9.2 布局，不能直接用于9.0的三维ND WO-A。

本次审查最终绑定 **v8** 候选 SHA：`a31a78f89e1ed71c007de2b692434cc0a0f1bc35626b22188418b3e715edd022`。此前flat root版本SHA为`0e78686e...`，独立CPU结果保留；v8另包含六处同址EvalStmt GM写回及仅一个packet loop去pipeline。旧v1/v2/v3的CPU审查结果保留为 `resident-suffix-review/results-v1.json` / `results-v2.json` / `results-v3.json`；二者随后在specialize阶段因helper/reshape元数据失败，不是设备数值失败。v3将原HCpost helper数学body按当前唯一调用实参直接展开，不改SDK；v3已过specialize，随后由InOutUseDiscipline拒绝Cube未返回的InOut；这仍是编译门禁失败，并非设备数值失败。v4将cube/vector全部8个InOut原序返回，group/root按同序重新绑定；数学body、worker、phase均未变。独立检查各call实参16个按形参序、返回/左值8个按同序匹配。随后只将residual/output根Tensor改为 `[T,16384]`，vector直接别名 `residual_flat=residual,y_flat=output`，不再核内reshape；实际数学body不变。独立检查四层函数注解和alias均匹配。最新CPU lowering结果尚待主线程确认。

## 1. 独立验证及源生成器

重新读取实际候选AST，而非只相信生成器的证明文字，独立验证：

- WO-A large/small、WO-B narrow/wide、BF16、整行量化、dequant、HCpost，共8处每logical-worker数学body与源逐AST相同。
- dequant唯一期望改动为最终store的目标 `output→attended`；反向重命名后AST相同。HCpost从 `attended` 读，最终写独立 `output`，没有把Oproj结果误作为HC输出。
- v3已无 `hc_post_block` helper及调用。独立从冻结源按字面实参映射 `t0→t,n0→0,valid_rows→1,ROWS→1,COLS→D` 展开，不调用生成器自己的ExpandHC；实际整个HC循环与该预期AST相同。Mul/Add次序、BF16输入cast、最终BF16 rint及输出切片保留。
- v4 cube/vector/group/root全部8个InOut均返回并匹配重绑定；这是满足数据所有权约束，不能代替生成物中cluster并发/地址别名的验证。
- cube/vector内部没有残余 `pl.spmd`、`pl.parallel`、`pl.at`；外层只有一次24组spmd和一个cluster，角色函数均incore。
- 全T=1..256验证每类逻辑grid恰覆盖一次、HCpost真token恰一次、WOA/WOB尾部容量界、16×8×8 start表rb界、signal槽唯一性。

检查脚本/结果：`resident-suffix-review/check_cpu.py`、`results.json`。原源SHA：Oproj `0528cd10...`、HCpost `31619000...`，与SOURCE_PROOF一致。

`ExpandHC` 当前仅命中这一条赋值调用，严格zip全部10个形参/实参；helper最后return是同一 `y_flat`，展开后显式subscript写同一Out，去掉`y_flat = return_same_alias`不改变数据目标。绑定形参在原helper仅处于Load位置；局部res_0..3/single_y/out_h等不与此前vector计算碰撞。当前变换不是可任意复用的通用inliner：未来实参带副作用、形参被赋值或存在局部名捕获时不可照用。

生成器通过固定body索引提取源码，有结构assert和源SHA记录，当前抽取正确；将来复用时建议直接断言这两个已批准SHA，避免正式源更新后“重新生成相同证明”被误当原冻结数学。当前无此漂移。

## 2. worker和K序

WO-A64逻辑项按`g=logical//8`、`block=logical%8`，由24AIC轮转领取：0..15号物理核各3项、16..23号各2项，64项无重复。WO-B原逻辑grid按24轮转；AIV三类grid和HCpost按48轮转。源中每输出group/block/panel/validrow/K-start均未改。

rows>=192的seedK256、低T原seedK16+15packet+15tail、不等start回退、高start wrap、matmul seed/init_cond和pipeline都保留。WO-B仍16个K512；整行8192 amax/scale、rint→FP16→INT8顺序保留。没有改为Nal分组scale或重新选tile。

## 3. phase DAG

| 阶段 | 发布者 | 等待者 | payload |
| --- | --- | --- | --- |
| 0 begin | 全72各自推进代际 | 不等待别人 | 本core代际 |
| 1 | 24AIC全部WO-A完成 | 48AIV全部 | FP32 accum |
| 2 | 48AIV完成BF16边界 | 48AIV全部 | projected BF16 |
| 3 | 48AIV完成整行量化 | 24AIC全部 | quantized、token_scale |
| 4 | 24AIC完成WO-B | 48AIV全部 | integer_accum |
| 5 | 48AIV完成dequant | 48AIV全部 | attended BF16 |
| 完成 | AIV各自HCpost完成 | runtime整个MIX完成 | 最终HC output |

源码等待有向图无环，每core每阶段即使没有数学项也发布，不会因小T零工作量跳过通知。AIC在phase4后可退出；不能因此让下一次invocation提前运行——依赖runtime必须等整个MIX包括HCpost AIV完成，这要在生成orchestration/运行生命周期核对。

6×72×128=**55296 INT32（216KiB）**。每(phase,core)独占128个word，只使用首word；最大首槽55168 <55296。signal wait是GE且不消费，单调generation可区分重复graph调用。

发布前bar_all/fence/cacheinvalid、等待后fence/cacheinvalid具有正确意图，但**CPU图无环不等于硬件无死锁或业务缓存已可见**。尤其新PTOAS signal lowering、MTE3/FIX drain、MIX两个AIV实际core ID以及角色是否真的同时常驻，都必须用生成物与小设备门禁验证。当前未使用旧FFTS event，没有旧event namespace重用问题。

## 4. 外部工作区与ABI合同

- root Python签名16 Tensor，明显低于新SDK256上限；**实际生成core参数/动态scalar数仍须检查**，不能仅Python16证明整个ABI。
- 外部FP32 accum至少 `ceil(T/128)*128 ×8192`，原 `T+127` 行分配始终充分；外部INT32 integer_accum至少按对应WO-B的16/64行store整块分配，原 `T+63` 始终充分。T240实际最大写到256行，不能仅传240行。
- T限制1..256，start表第0维16足够。外部工作区大小参数ACC_ROWS/INT_ROWS和T彼此独立，候选本体没有容量assert，因此必须由host/harness在启动前硬验证。
- projected/quantized/scale/attended公开shape均是真T；本实验root的residual/output改为 `[T,16384]`，host负责与模型 `[T,4,4096]` 同storage的零拷贝view。必须检查contiguous/stride、data_ptr、storage_offset和覆盖元素数相同，不能只调用可能复制的reshape就宣称零拷贝。线性索引 `t*16384+h*4096+d` 与旧rank3 flatten完全相同，尾行仅临时scratch；未引入request的S6padding。
- 除两份外部GM accum外，候选仍有三处局部`pl.create_tensor`形式的narrow累加器声明。数学源使用了相同声明；必须查看新SDK实际lowering，证明它们落在合法L0C/局部对象，而不是新增GM allocation/runtime任务。不能在看到IR前宣称“只有两份GM、没有任何内部allocation”。
- WO-A和WO-B本地资源可能在单函数里被错误拉长live range，应审实际L0C/L1/UB地址复用及compactvalid尾部；不因body AST一致跳过资源检查。

## 5. 硬件前必须落实的合同

1. baseline旧root先在新SDK完成同输入位精度门禁，排除SDK切换改变数学。
2. NZ2/CANN9.2明确配置，并断言WO-A/WO-B实际format等于根layout要求；禁止ND误绑定。
3. 外部容量、dtype、trueT及合法不重叠关系通过host检查；scratch/signal有首尾guard、输入只读快照和毒化恢复。
4. signals首次完整清零、所有72core每次恰begin一次、单图/顺序流独占，generation在INT32溢出前重建。失败后不得直接复用部分推进状态；不同图并发不得共用signal/temporaries。
5. specialize→IR→实际C++/ISA→完整编译，核验只有一个MIX、24AIC/48AIV、无额外GM分配、无嵌套任务、全部notify/wait地址正确且payload写完才通知。
6. 小T（含T1/6、混合T21）先测，逐阶段比较OA BF16、INT8/scale、最终Oproj、HCpost；T144/T240再复核start分支，强制不等start和high249；A/B/A、zero/tiny/guard、repeat均须通过。首差停止，不借“调度改动不动数学”推定数值一致。

当前没有发现数学提取、逻辑worker覆盖或源码phase DAG的阻断。上述layout/容量/生命周期合同已由主线程确认将进入harness；编译/硬件同步证据尚未产生，应保持独立候选而不接服务。

## 6. v7阻断定位与v8实际生成物复核

v7中六处 `pl.assemble` 作为EvalStmt均在ConvertTensorToTile后生成真实 `pl.tile.store`，不是忽略返回值就丢写回。v7失败点是MemoryReuse的Acc→Acc move拒绝；其36pass可见v4_wide_packet的stage2展开为两packet主循环加末packet，内部又有K256自动K128分块与wrap16×K16分支，造成控制流L0C alias无法合并。该编译失败不能归为数值或硬件失败。

最小修正只将 **v4_wide_packet** 的`pipeline(15,stage=2)`改为`range(15)`：移除prefetch双缓冲控制流，保持seed、15packet、wrap与tail顺序及所有MMA参数不变。没有改tile大小、K256包、舍入或SDK；不能因此推测性能不变。v8已实际通过完整符号pass并生成PTO，原失败保留。

### 精确文件绑定

| 文件 | SHA256 |
| --- | --- |
| v8 source | `a31a78f89e1ed71c007de2b692434cc0a0f1bc35626b22188418b3e715edd022` |
| result.json | `4624dafe5e5846380d4d14d9da49bfa3b408121bc324afe89d195392f1c3a8bb` |
| compiled_meta.json | `186c799737fc6989d9ffad6c77046eeb8195ba267c310371efbf6376dbd60850` |
| orchestration/resident_suffix.cpp | `5f0c471f599af019183af9adccbc07b9fa597514fd17f8481b54daea0657e4db` |
| kernels/resident_precision_suffix.pto | `82d8f2758b90bfc9ced756739606613be716c7801d33515e82b7f10ff32c06e4` |
| 37_after_MemoryReuse.py | `d6fc8b1c0246b5b9d553cb367824c7eb98d4c5a4cb42b3950ce5ce741183b6ee` |

生成物只读审计脚本及完整窄证据：`resident-suffix-review/review_v8_generated.py` / `generated-v8-results.json`。没有拉取模型、Tensor或大型dump。

### 实际编排 / 资源

编排实际仅一个 `rt_submit_task`，`MixedKernels{0,1,1}`，24 block且sync_start=true，16个Tensor参数，无rt_alloc。PTO两个角色各有16指针+3动态index+2个SPMD标量，不能将Python16误作所有机器标量均为0。meta的8个InOut返回和NZ布局匹配预期。

六处workspace写回在最终PTO全保留：前四TSTORE的输出view递归指向 **%arg13=accum**，后两指向 **%arg14=integer_accum**，没有写到新缓冲或错root；具体source位置120/142/201/223/237/251，receipt列在JSON。

全部局部create_tensor已在后端pass变成片上tile，37pass无create_tensor；最终PTO无除alloc_tile以外的alloc，编排无GM分配。实际物理tile静态最高地址端（**不是把每个逻辑memref初始0偏移当成全局峰值**）：

| 空间 | v8 PTO地址最高端 | A3名义容量 | 结论 |
| --- | ---: | ---: | --- |
| AIC L1/Mat | 256KiB | 512KiB | 有裕量 |
| L0A/Left | 64KiB | 64KiB | 刚满，必须后续PTOAS/实际编译 |
| L0B/Right | 64KiB | 64KiB | 刚满，必须后续PTOAS/实际编译 |
| L0C/Acc | 64KiB | 128KiB | 有裕量 |
| AIV UB/Vec | 128KiB | 192KiB | 有裕量 |

多个alloc_tile声明可以共享这些地址，不得把199个UB view尺寸相加作为峰值。当前仅符号/后端IR的静态配置，不是最终机器码资源或runtime额外保留空间验收。

### 同步降低

最终PTO确有 `tnotify(set)`、`twait(ge)`、`PIPE_ALL`、GM fence和cacheinvalid。AIC静态notify3处（begin/phase1/phase4）、wait1处循环；AIV notify4处（begin/phase2/phase3/phase5）、wait4处循环；与源码DAG对应。发布前本核barrier与fence/cache维护没有被DCE删除，等待后获取fence/cache维护也保留。

**本轮PASS限定为实际生成IR/PTO及编排审查**。它不证明PTOAS/CCEC已经成功、不证明GM payload在设备可见、不证明无死锁或逐bit正确。仍须原精度root在新SDK先过门禁，以及完整编译和小T/强制分支/重复代际设备对拍。
