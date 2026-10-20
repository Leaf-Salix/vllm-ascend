## 现象

用现有 `load → set_validshape → add → store`，将 `tensor.read`/`cast` 嵌套在有效长度参数中，会在 `ConvertTensorToTileOps` 后生成 `narrowed__ssa_v0_1__FREE_VAR`，随后 `InitMemRef` 报 tile 没有 memory space。把相同 read/cast 先绑定为具名 Scalar 后，完整 CPU lowering 通过。

本例不使用本地新增 compact_compare、SDMA 或点对点同步，不需要 NPU；首次失败在 PyPTO pass 内，未进入 PTOAS。没有提交修改 PR。

## 最小复现

```python
import pypto
import pypto.language as pl
from pypto.runtime import CompileOptions

@pl.jit.incore
def calculate(src: pl.Tensor[[1, 2048], pl.FP32],
              valid: pl.Tensor[[1], pl.INT32],
              out: pl.Out[pl.Tensor[[1, 2048], pl.FP32]]):
    tile = pl.load(src, [0, 0], [1, 2048])
    narrowed = pl.set_validshape(tile, 1, pl.cast(pl.read(valid, [0]), pl.INDEX))
    result = pl.add(narrowed, 1.0)
    pl.store(result, [0, 0], out)
    return out

@pl.jit
def entry(src: pl.Tensor[[1, 2048], pl.FP32],
          valid: pl.Tensor[[1], pl.INT32],
          out: pl.Out[pl.Tensor[[1, 2048], pl.FP32]]):
    with pl.spmd(1):
        out = calculate(src, valid, out)
    return out

pypto.configure_cache(pypto.CacheConfig(enabled=False))
config = CompileOptions(platform="a2a3", output_dir="repro_nested_validshape", dump_passes=True)
entry._resolve_kernel_artifact((), {"config": config}, allow_signature_mode=True)
```

通过的对照只把对应语句改成：

```python
n = pl.read(valid, [0])
n_index = pl.cast(n, pl.INDEX)
narrowed = pl.set_validshape(tile, 1, n_index)
```

这仍然从 device Tensor 读取实际有效长度，没有新增 Host 输入或把有效长度静态化。

## 首个失败 pass 与具体 IR

`11_after_OutlineClusterScopes.py` 尚无 FREE_VAR，但类型和 RHS 已分别持有物化出来的 `t__tmp_v0` / `t__tmp_v1`：

```python
t__tmp_v0 = pl.tensor.read(valid__ssa_v0, [0])
t__tmp_v1 = pl.tensor.read(valid__ssa_v0, [0])
# narrowed 的 TileView.valid_shape=[1, pl.cast(t__tmp_v0, pl.INDEX)]
narrowed__ssa_v0 = pl.tile.set_validshape(tile__ssa_v0, 1, pl.cast(t__tmp_v1, pl.INDEX))
```

`12_after_ConvertTensorToTileOps.py` 首次出现自由变量：

```python
t__tile = pl.tensor.read(valid__ssa_v0, [0])
t__tile_1 = pl.tensor.read(valid__ssa_v0, [0])
# narrowed 类型仍引用旧 t__tmp_v0，RHS 已使用 t__tile_1。
narrowed__ssa_v0 = pl.tile.set_validshape(tile__ssa_v0, 1, pl.cast(t__tile_1, pl.INDEX))
result__ssa_v0 = pl.tile.adds(narrowed__ssa_v0_1__FREE_VAR, 1.0)
pl.tile.store(result__ssa_v0_1__FREE_VAR, [0, 0], out__ssa_v0)
```

后续报错为 `tile 'narrowed__ssa_v0' reached InitMemRef with no memory space`，因此不能把问题简单修成无条件给缺失 Tile 填 Vec；需要先消除定义/使用身份和动态类型引用的不一致。

## 环境与核对边界

- 实测工作区基于 CSA 工具链，本地 PyPTO HEAD `8c360367e0239779a93b495ecf7910e26a9f1fc6`，包含其他独立修复和 compact_compare 原型；上面的复现不调用该原型。
- CPU 对照：嵌套版失败；具名 Scalar 版完整 lowering 通过。没有使用设备执行结果替代编译器复现。
- 只读核对 main `70da9baed1edd18308420d0ac2d5b9f4ede4bb90` 的 `TypePropagatingMutator::HandlePassThroughAssign`，该函数与本地逐字相同；没有宣称已在全新 main 构建中复现。
- [对应源码](https://github.com/hw-native-sys/pypto/blob/70da9baed1edd18308420d0ac2d5b9f4ede4bb90/src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp)。
- 与 #3046 属于相关的动态类型元数据/定义身份问题，但 #3046 首次出现在 FlattenTileNdTo2D 的 GM view；本例更早发生于 ConvertTensorToTileOps，且不使用 GM view。

## 修改建议

1. FlattenCallExpr 物化表达式时，检查同一表达式出现在参数与 TileView.valid_shape 中时，是否应复用同一已物化 Scalar；类型元数据不应引入重复求值的 device read。
2. ConvertTensorToTileOps 对 passthrough Tile 操作，除 Call 参数外，同时用当前 Var 映射重写返回 TileType/TileView 的 valid_shape 等动态表达；一旦类型改变，定义 Var、Call 类型及所有消费者必须绑定到同一个新 Var。
3. 不能让消费者独自重建一个有新类型但没有定义的 Var；也不应把这些遗留临时变量提升成新的 shape 参数。
4. 增加定义身份回归：nested read/cast、named Scalar 对照、连续 set_validshape/view 链和 loop 内使用；验证 pass 后没有 FREE_VAR，valid_shape 中的 Scalar 都有支配定义，再验证完整 lowering。

建议先在主线运行以上最小例确认现状，再选择修复位置。当前本地只用具名 Scalar 写法继续验证算子，没有为这个问题改编译器。
