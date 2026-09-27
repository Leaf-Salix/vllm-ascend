# Indexer cache：源头分离与原布局的性能对照

更新：2026-09-27。范围仅为A3 C4 Indexer的INT8 key / FP16 scale。
SWA、压缩Attention cache、Compressor state和算术保持不变；历史尝试见验证日志与Git。

用户要求重新实测源头分离，并按性能选择分离策略；与原布局完整CSA差异在5%以内，优先保留原布局由PTO内部处理。
比较完整CSA均值、P95及缓存受扰动时的表现，最终仍以整模型forward验收；不能只比较若干核内任务之和。

## 当前基线与实验范围

正式保留源码2a740c1f：Native原交错布局，PTO内部0/64B GM视图，N128加载交错QK/WS，等待Cube前加载scale。
不含入口历史复制或外部写回；仅长档Score整组准入并禁止提前释放。
2a740c1f单卡8K/128K B16均值776.94/1242.22μs；七档模型数据仍属于f76b3ad4。
[scale提前读取](results/csa_incore_20260927/indexer_scale_prefetch/README.md)。

新的初始化分离实验在独立工作树`.cache/csa-source-split-2a740c1f`：

- 仍使用同一块raw allocation，前`pages×4096`字节为key，后`pages×64`字节为scale。
- `model_runner_v1._reshape_kv_cache_tensors`仅在PTO模型、performance、A3、C4 Indexer时选择两个连续view。
  Native模型、precision及其他cache继续原分支；没有全局修改`_adjust_kv_layout`。
- PTO继续使用一个可写根描述符，内部派生不重叠key/scale区域；无需每步copy或commit桥接。
  Score去掉交错页的0/64B相位判断，Compressor原地更新相同物理slot。
- 页数、总字节、页表、请求生命周期不变。Native prefill/回退消费同一组带stride的key/scale view。
  Native完整prefill链和16卡模型还未验证，不能把单卡通过称作全主流程通过。

CPU完整PTOAS/AICPU编译已通过。CPU直接调用runner初始化函数确认：Native模型和PTO precision页stride仍为4160/2080；
PTO performance为4096/32；共享所有权、偏移和保护区检查通过，实验ABI拒收旧交错布局。
[实验补丁、编译及检查](results/csa_source_split_ab_20260927/)。

## 必须区分两种连续性

```text
Native交错物理页：[K0 S0][K1 S1][K2 S2]...
源头key/scale分离：[K0 K1 K2 ...][S0 S1 S2 ...]
请求逻辑页表：   [9, 2, 17, ...]    ← 仅分离key/scale不会改变这些页号
旧入口重排结果： [请求0逻辑页0、1、2 ...][请求1逻辑页0、1、2 ...]
```

旧`SplitIndexerCache.load`不仅分离key/scale，还通过`index_select`按页表重排到请求连续的临时缓存。
因此“分配时拆开便自然得到旧连续请求输入”不成立。本次最小分离实验仍按物理页读取，不能冒称完全取消页表寻址。

若要合并连续读取，可在算子确认相邻逻辑页映射到连续物理页后合并DMA；非连续页需要回退。
若要无条件取消PTO内部分页，则须从分配策略保证请求历史连续，覆盖增量扩容、请求换位、释放与复用；
前缀共享和connector还要遵守相同映射，不能只让单卡fixture连续而声称生产方案完成。
用户已允许按性能收益选择策略，不把“改动最少”作为拒绝更高收益方案的理由。

## Native兼容依据

| 位置 | 实际契约 | 分离后的处理 |
| --- | --- | --- |
| QLI Python/C++入口 | key/scale独立入参，绑定读取各自stride(0) | 传4096/32的新stride |
| QLI AIC | `block_table * key_stride + row * head_dim` | 页号不变，无须相邻scale |
| QLI AIV | `block_id * scaleStride + row` | 直接读连续scale池 |
| Native量化写入 | key/scale各一次scatter，binding传入目标strides | 原地更新，无需拼回 |
| PTO精度版 | 仍强制同storage、scale偏移4096、交错页stride | 初始化保留原布局 |
| KV管理 | 依赖spec页数/总字节与block ID | 最小分离实验保持全部不变 |
| ACL Graph | 捕获固定地址/描述符 | 初始化定布局，重放期间不切换 |
| connector | 按tensor登记步长，但有分支假设原raw布局 | 当前只分析已使用的逻辑bank恢复；不外推所有connector |

源码依据：[runner](../../vllm_ascend/worker/model_runner_v1.py)、[QLI/scatter绑定](../../csrc/torch_binding.cpp)、
[QLI AIC](../../csrc/attention/vllm_quant_lightning_indexer/op_kernel/arch32/quant_lightning_indexer_service_cube.h)、
[QLI AIV](../../csrc/attention/vllm_quant_lightning_indexer/op_kernel/arch32/quant_lightning_indexer_service_vector.h)、
[Native更新](../../vllm_ascend/device/device_op.py)。实验补丁单独保存，不用主工作树源码链接假冒已合入的新实现。

既有Native-only任务task_20260927_204637_408988615371：B2/S6，8K/4K混合、非连续页，scatter＋QLI、eager＋两次修改值后的图重放，
key/scale位模式与Top-K精确一致、保护区通过。[已有结果](results/csa_cache_source_split_20260927/native_check.json)。不重复这一项。

## 当前验收和待办

1. 原布局与初始化分离固定相同基底2a740c1f，同一任务、同一张卡对照。
2. 单卡B4的8K/128K，固定规约、逐物理行变化的251种scale；8类逻辑输出/cache/state及A→B→A图重放。
3. 两档通过后，B16各5预热20次计时；另在开始事件前写入384MiB独立缓冲，压力时间不计入CSA，报告P95/max。
4. 分别呈现纯物理分离、可合并连续页、请求历史连续的适用条件和收益，避免把不同修改混为一个结果。
5. 只有候选收益值得保留时，再补Native→PTO→Native状态交接及真实权重16卡token/DSpark和forward；先不重跑七档模型。

模型差距的现有证据：f76b3ad4在128K/B16单层CSA快3.34%，模型rank0独立trace中的CSA反慢3.50%；
缓存压力使PTO增加40.96μs、Native仅增加1.50μs，消除了约3%的局部优势。
这支持内存访问状态敏感，尚未闭合正式forward全部差距，也未测得硬件缓存命中率。
[证据与边界](results/csa_model_forward_f76b3ad4_20260927/MODEL_GAP.md)。
