# Indexer cache 源头分离与 Native 兼容性评估

2026-09-27，基于当前保留算子 3d1f0f65。这里只讨论 A3 的 C4 Indexer INT8 key / FP16 scale；
不改变 SWA、压缩 attention cache、两个 Compressor state 的格式或算术。

## 结论

**key/scale 在初始化时分成连续的两个视图，可以做成局部修改；但这不等于每个请求的历史连续。**
现有桥接还按 block table 重排历史页，所以不能只修改分配布局就删除 load/commit 并保持当前 Score 寻址。
此前“确认本体收益后，分配时分离即可完全去掉适配成本”的说法缺少这个前提，应撤回该简化结论。

Native 的 QLI 和 scatter 原本就分别接收 key/scale，并使用各自 stride；源码没有要求二者在每页内相邻。
分离本身不要求重写 Native 算术内核。生产兼容还包括 Native prefill/回退、PTO 精度版适配、
图捕获前的地址绑定、共享缓存视图和 connector 注册，不能仅凭 QLI 接口就声称全主流程已通过。

## 1. 两种连续性及实际代码

当前 Native 每个物理页共 4160 B：32×128 INT8 key（4096 B）＋32 FP16 scale（64 B）。

```text
当前物理布局：[K页0 S页0][K页1 S页1][K页2 S页2]...
源头分离布局：[K页0 K页1 K页2 ...][S页0 S页1 S页2 ...]
某请求页表：  [9, 2, 17, ...]    ← 分离布局不会改变这些物理页号
当前PTO桥接：[请求0逻辑页0、1、2 ...][请求1逻辑页0、1、2 ...]
```

- [runner `_adjust_kv_layout`](../../vllm_ascend/worker/model_runner_v1.py) 用 `as_strided` 创建每页交错的两个 view。
  当前 key 的页 stride=4160 INT8 元素，scale 的页 stride=2080 FP16 元素。
- [SplitIndexerCache.load](../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/indexer_cache.py)
  先用页表取得物理页号，再用两次 `index_select` 搬成按请求连续的历史；额外复制尾部保护页。
- [Score](../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_indexer.py)
  直接用 `request * request_rows + logical_row`，一次取384/512等连续行。
  它不能把任意物理页池误当作这种按请求排列的输入。
- [Compressor写入](../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_indexer_compressor.py)
  当前也写请求逻辑行；桥接再根据 Native slot 取出新增行并 scatter 回物理页。
- [CompressAttentionManager](../../vllm_ascend/core/single_type_kv_cache_manager.py)
  从 block pool 增量取页，也支持已计算块复用；请求换位、释放再分配和前缀共享均不能假设物理页连续。

## 2. Native 的影响范围

| 位置 | 源码行为 | 源头分离的影响 |
| --- | --- | --- |
| QLI Python/C++入口 | `key`、`key_dequant_scale` 两个入参；绑定读取各自 `stride(0)` | 可描述分离布局；连续 key页stride=4096，scale页stride=32 |
| QLI AIC读key | `block_table[b,p] * stride + offset * 128` | 数学及页号不变，按新stride寻址 |
| QLI AIV读scale | `block_id * scaleStride + offset` | 同上，不要求与key共用页内偏移 |
| Native量化更新 | 对key/scale各调用一次scatter；binding传入目标的完整strides | 可原地更新新视图，不需要额外拼回 |
| Native prefill / decode回退 | 消费相同的key/scale tuple、页表、slot | 调用接口可保留，仍需串联验证 |
| PTO精度版 | `indexer_storage()` 强制同storage、scale偏移4096及交错页stride | **不能直接复用新布局**；需要适配或先保持该入口使用旧布局 |
| PTO性能版 | 同样调用旧布局校验，且Score/Compressor使用请求连续布局 | **必须修改**校验、读取及更新寻址；不能只删除桥接 |
| KV管理 | 按spec的page_size_bytes、block ID管理，不直接消费浮点数据 | 保持页数、每页总字节、共享关系可避免改调度器 |
| Mooncake当前实现 | 分tensor记录base、block_len和block_stride；部分注册分支依赖整块raw allocation | 优先保留同一底层分配的两个大区间；需验证实际部署分支，不能扩大为所有connector兼容 |
| ACL Graph | 捕获tensor地址和描述符 | 在初始化/捕获前定布局，不在已有图重放中切换地址 |

证据：
[C++ QLI/scatter binding](../../csrc/torch_binding.cpp)，
[Native QLI AIC](../../csrc/attention/vllm_quant_lightning_indexer/op_kernel/arch32/quant_lightning_indexer_service_cube.h)，
[Native QLI AIV](../../csrc/attention/vllm_quant_lightning_indexer/op_kernel/arch32/quant_lightning_indexer_service_vector.h)，
[Native量化/更新入口](../../vllm_ascend/device/device_op.py)，
[PTO旧布局合同](../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/native_storage.py)，
[Mooncake注册和步长](../../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py)。

## 3. 局部落地方案

**用户最新限定：只针对PTO改造vllm-ascend，粒度最小，不改Native流程。**
当前采用原Native分配与视图，先在PTO内部直接写物理slot，再评估PTO按页读取；
下面源头分离方式是兼容性评估备选，不作为当前生产修改方案。

若采用源头分离，优先保持现有的一块 raw allocation，把前 `num_pages×4096` 字节视为 key，
后 `num_pages×64` 字节视为 FP16 scale。总字节和物理页数不变，两者共享一份所有权和释放周期。
不必新增两块独立内存分配，也不必新增 cache group。实现需只命中 A3 C4 Indexer，
不能全局更改 `_adjust_kv_layout`，也不能套用 A5 的 full-cache ABI。
仅开启性能版时选择新布局，可让默认 Native 和精度版继续原布局；同一性能实例里的 Native回退应消费新布局。
若以后两版统一布局，共用数值中性适配，精度版算术保持原样。

**要消除每步完整桥接，PTO必须直接消费物理页表并按物理slot写入。**
Score保留现有Cube/FP16规约/双query/1024或768策略，加载时按32行页分段，
连续物理页可以合并DMA，非连续页必须按页定位，不能依赖单卡合成页表的排列。
这会将部分加载开销移回核内；需要比较完整CSA收益，不能承诺仍保持当前请求连续输入的本体时间。
新路径不再越过有效物理页做整块读取，因此现有尾部padding读法也必须同步调整。

不建议为维持请求连续读取强制每请求预留整段物理cache：它会扩大到调度、扩容、前缀复用和显存利用率，
不符合“简单且不影响其他主流程”的要求。

近期顺序：

1. 用无PyPTO的小型Native case验证交错与分离布局的scatter、QLI、图重放；结果见下面的证据目录。
2. 先让CSA内部将新增key/scale直接写入Native物理slot，消除外部Torch的取行、定位及scatter链；
   可以先保留原交错分配和入口load，独立验证并取得完整区间收益，不必等待源头分离。
3. 独立评估PTO按页直接加载；复用当前已保留的算术及incore优化，8K/B16与128K/B16共同判断。
   源头分离可以与这一项结合，不能把“key/scale已分离”误报为“请求历史已连续”。
4. 定向覆盖页不连续、页边界、padding和Native→PTO→Native状态交接；通过后再扩到真实权重16卡。
   不重跑已无关的精度/性能矩阵；前缀缓存和connector仅在涉及其部署路径时补对应验证。

## 4. 当前开销与验收边界

当前3d1f0f65、B16、同一套正式第4层权重及合成历史、第二CSA metadata复用，单位μs：

| 上下文 | 入口load | CSA本体 | 出口commit | 完整PTO | Native |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K | 91.77 | 790.78 | 216.26 | 1079.16 | 927.07 |
| 128K | 187.52 | 1234.06 | 220.43 | 1645.76 | 1310.21 |

阶段值来自独立捕获/计时，不能强求它们之和严格等于完整PTO，也不能直接按加减预测改造后耗时。
但外部commit约216～220μs是明确的优化对象；它只写新增少量行，大部分成本不能解释成全历史写回带宽。
[当前数据](results/csa_incore_20260927/indexer_score_panel1024/README.md)。

Native定向验证的脚本、命令及结果目录：
[cache源头分离验证](results/csa_cache_source_split_20260927/)。
任务task_20260927_204637_408988615371退出0：B2/S6、8K/4K混合历史、非连续物理页，
eager及两次修改更新值后的图重放，key、scale位模式、Top-K均精确一致，保护区完整。
[Native-only结果](results/csa_cache_source_split_20260927/native_check.json)。
本次不修改生产分配器，不把单算子通过等同于完整prefill/回退链或整模型已验收。

## 5. 已完成的最小改动：PTO内直接提交slot

仅修改4个PTO性能版文件，复用原Native物理页；新增一个InOut描述符，
在已有key和串行scale任务中写回相同量化结果，删除外部Torch定位/取行/scatter链。
Native分配、算子、页表、slot与调度代码均未修改，precision入口未修改。
两档完整PTO分别降到888.64/1382.39μs，比改造前快17.65%/16.00%；长档仍慢于Native6.23%。
入口完整历史复制仍存在（98.35/184.49μs），下一项继续只在PTO侧处理。
[完整性能、核内变化、保护区和补位图验证](results/csa_cache_direct_commit_20260927/README.md)。

## 6. 内部GM视图直读：可保留Native布局，长档收益尚未取得

历史§116已经通过0/64B两个GM视图直接读L1；旧8K性能差而撤回，不是布局无法表达。
§161无128K实测便关闭路线的外推已更正。本轮复用该方法，一个可写Native根入参，
视图在Score调度函数内派生，避免外部部分重叠alias被PyPTO拒绝；没有修改工具链或Native流程。

8K/B16完整PTO888.64→807.47μs，128K/B16 1382.39→1396.31μs；
长档Score AIC从312.97～323.83增至537.43～565.10μs，完整均值未改善、p95变差。
两档保护区/索引结构/有限值、B4/H4095同图4→3→1→4通过；数值及整模型边界未放宽。
已保存候选补丁并恢复bad985a9，入口复制仍在。下一步优先在PTO内部改善分页DMA/流水，
或短档直读、长档内部紧凑化，继续共用原Native分配；不把分配布局改造作为必要前提。
[本轮结果与全部泳道](results/csa_cache_direct_read_20260927/README.md)。
