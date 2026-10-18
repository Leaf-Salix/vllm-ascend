## 需求：显式 stride 的零拷贝 GM view，用于 A2/A3 多 ND 加载

这是 #1964 中明确延后的非 canonical stride 能力的具体使用场景，**不是把当前
入口正确拒绝非连续 Tensor 说成编译器误算**。不请求默认复制/contiguous。

DeepSeek V4 的 Native Indexer cache 每页包含4096B INT8 Key和64B FP16 scale，
常见根 Tensor 是连续 INT8 `[pages,4160]`。一个 N64 面板需要两个32行 Key 页。
当设备页表确认两个物理页相邻时，希望保留原 storage，用下面的 view 一次
ND→NZ 读取两页Key，跳过各自scale字节：

```text
shape   = [pages, 32, 128]
strides = [page_bytes, 128, 1]  # 元素单位，page_bytes来自原Tensor的dim
load    = origin [first_page,0,0], shape [2,32,128]
Mat     = [64,128] NZ
```

不相邻、非法页、最后一页重复clamp等情况仍保留原逐页gather。这里没有提出
改变Native KV布局、重排候选、添加host tiling回读或额外Key暂存。

## 当前已核实的行为

2026-10-18读取main的 `python/pypto/language/op/tensor_ops.py::view`，签名仍为
`view(tensor, shape=None, valid_shape=None, *, layout=None)`，只派生canonical
strides；#1964 的decision也明确当时延后stride参数。

本地独立PyPTO工作树 `d01de8523`（含与此接口无关的单SPMD本地修复）做了
完整符号编译和一次设备入口验证。CPU编译通过，但该形状按canonical语义
折叠为连续二维 `[P*32,128]`；这不能用于上面的带页尾间隙布局。

把Native根Tensor用Torch零拷贝描述符传入：

```python
raw = torch.empty((9, 4160), dtype=torch.int8, device=device)
keys = raw.as_strided((9, 32, 128), (4160, 128, 1))
assert keys.data_ptr() == raw.data_ptr()
# entry的参数声明：native_keys: pl.Tensor[[P,32,128], pl.INT8]
entry(keys, query, page_id, output, config=config)
```

入口正确报错：

```text
Parameter 'native_keys' requires a contiguous strided tensor; no copy is made
```

任务在校验阶段退出，未执行候选搬运，因此没有设备数值或性能成功结论。
这条保护应保留，直到输入契约与编译器真的支持stride；不要只删除校验。

当前实际pinned PTO-ISA的 `include/pto/common/arch/memory/tload_common.hpp`
已包含 `TLoadGm2L1Nd2nz` 的multi-ND分支，A2/A3 capability打开：GlobalTensor
Shape2映射ndNum，Stride2映射srcNdMatrixStride。对于两页，期望值为
`ndNum=2,nValue=32,dValue=128,srcNdMatrixStride=page_bytes`，目标行按
两个32行块堆叠。页stride须符合指令范围；超范围应明确拒绝或保留通用回退。

## 建议修改方向（不是PR）

优先扩展显式GM view，避免要求所有根入参立刻支持任意stride。示意API：

```python
key_view = pl.tensor.view(
    native_cache,
    shape=[pl.tensor.dim(native_cache, 0), 32, 128],
    strides=[pl.tensor.dim(native_cache, 1), 128, 1],
)
```

以上是建议接口，当前不可运行。也可采用独立的`strided_view`，由组件决定命名。

1. view仍共享原MemRef、偏移与依赖，不能产生隐式Tile load或数据复制。
2. 校验正stride、dtype/对齐、可访问字节范围；这个view含间隙，不能要求
   逻辑元素数等于原storage元素数，需按最后可访问元素验证边界。
3. `FlattenTileNdTo2D`不能把非canonical页stride压成连续`[128,1]`；若平台
   能使用multi-ND，应保留GlobalTensor的3维形状/stride给TLOAD。
4. 可暂只支持orch建立view并作为incore内部别名传递，不要求本需求新增root。
5. 同步和alias分析必须认识同一Native cache；不能以只读view掩盖原根的写入依赖。

建议验证：page_bytes=4160/4224/8192，偶/奇起页、最后合法两页及A→B→A同图
页号更新；输出可用INT8点积逐元素验证，另查Native scale尾字节不被当成Key、
输入不变及输出保护区。还应检查生成物确为多ND一次加载，并验证非法stride
仍被拒绝。完整CSA性能应另行实测，不能由DMA发射次数直接断言收益。
