### Component

EmitC / Codegen：`PTOToEmitC/Tensor/TGather.cpp::emitCompareGather`

### Description

A3 compare-form `pto.tgather` 的 scalar kValue 能通过 MLIR verifier，但生成的
TGATHER 模板和运行参数不匹配当前 PTO-ISA。CPU 上生成 C++ 成功，随后 CCEC
编译失败，尚未运行 NPU。

与已关闭 #654 的 offset 缺失不同：本次生成代码已经有 runtime offset；剩余
问题是模板参数数量/顺序、cdst/tmp 顺序，以及标量阈值和 ISA tile 阈值的接口差异。

### Reproduction (minimal)

下面是独立的已生成 `.pto`，不需要 vLLM 或权重。使用当前 PTOAS 的 A3
默认路径输出 C++，再包含当前 PTO-ISA 编译（需要普通 A3 CCEC 编译环境）：

```mlir
module attributes {pto.target_arch = "a2a3"} {
  func.func @compare_kernel(%arg0: !pto.ptr<f32>, %arg1: !pto.ptr<i32>, %arg2: !pto.ptr<i32>) attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
  %c0_i64 = arith.constant 0 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c6144_i64 = arith.constant 6144 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c8_index = arith.constant 8 : index
  %c64_index = arith.constant 64 : index
  %c1_index = arith.constant 1 : index
  %c0_index = arith.constant 0 : index
  %c4096_index = arith.constant 4096 : index
  %cst_9 = arith.constant 5.00000000000000000e-01 : f32
  %src__ssa_v0_view = pto.make_tensor_view %arg0, shape = [%c8_index, %c64_index], strides = [%c64_index, %c1_index] {layout = #pto.layout<nd>} : !pto.tensor_view<?x?xf32>
  %selected__ssa_v0_view = pto.make_tensor_view %arg1, shape = [%c8_index, %c64_index], strides = [%c64_index, %c1_index] {layout = #pto.layout<nd>} : !pto.tensor_view<?x?xi32>
  %counts__ssa_v0_view = pto.make_tensor_view %arg2, shape = [%c1_index, %c8_index], strides = [%c8_index, %c1_index] {layout = #pto.layout<nd>} : !pto.tensor_view<?x?xi32>
  %values__ssa_v0 = pto.alloc_tile addr = %c0_i64 valid_row = %c8_index valid_col = %c64_index : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
  %src__ssa_v0_pview = pto.partition_view %src__ssa_v0_view, offsets = [%c0_index, %c0_index], sizes = [%c8_index, %c64_index] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<8x64xf32>
  pto.tload ins(%src__ssa_v0_pview : !pto.partition_tensor_view<8x64xf32>) outs(%values__ssa_v0 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
  %tmp__ssa_v0 = pto.alloc_tile addr = %c2048_i64 valid_row = %c1_index valid_col = %c4096_index : !pto.tile_buf<loc=vec, dtype=ui8, rows=1, cols=4096, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
  %ids__ssa_v0 = pto.alloc_tile addr = %c6144_i64 valid_row = %c8_index valid_col = %c64_index : !pto.tile_buf<loc=vec, dtype=i32, rows=8, cols=64, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
  %count__ssa_v0 = pto.alloc_tile addr = %c8192_i64 valid_row = %c1_index valid_col = %c8_index : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=8, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
  pto.tgather ins(%values__ssa_v0, %cst_9, %tmp__ssa_v0 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>, f32, !pto.tile_buf<loc=vec, dtype=ui8, rows=1, cols=4096, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%ids__ssa_v0, %count__ssa_v0 : !pto.tile_buf<loc=vec, dtype=i32, rows=8, cols=64, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=8, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>) {cmpMode = #pto<cmp gt>, offset = 0 : i32}
  %selected__ssa_v0_pview = pto.partition_view %selected__ssa_v0_view, offsets = [%c0_index, %c0_index], sizes = [%c8_index, %c64_index] : !pto.tensor_view<?x?xi32> -> !pto.partition_tensor_view<8x64xi32>
  pto.tstore ins(%ids__ssa_v0 : !pto.tile_buf<loc=vec, dtype=i32, rows=8, cols=64, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%selected__ssa_v0_pview : !pto.partition_tensor_view<8x64xi32>)
  %counts__ssa_v0_pview = pto.partition_view %counts__ssa_v0_view, offsets = [%c0_index, %c0_index], sizes = [%c1_index, %c8_index] : !pto.tensor_view<?x?xi32> -> !pto.partition_tensor_view<1x8xi32>
  pto.tstore ins(%count__ssa_v0 : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=8, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%counts__ssa_v0_pview : !pto.partition_tensor_view<1x8xi32>)
  return
  }
}

```

```bash
ptoas gather_compare_aligned.pto -o gather_compare_aligned.cpp --pto-arch=a3 --pto-level=level3 --enable-insert-sync
```

复现使用 8 行，计数 Tile 的 32B 对齐已经满足，排除了 PyPTO 单行计数未对齐
这个独立问题。源码输入 FP32 阈值为 0.5，不允许无损语义被整数截断替代。

### Actual behavior / error logs

简写生成代码的 Tile 类型别名后：

```cpp
TGATHER<DstTile, SrcTile, TmpTile, CountTile, CmpMode::GT>(
    dst, src, scalar_f32_0_5, tmp, count, 0);
```

```text
error: no matching function for call to 'TGATHER'
candidate template ignored: invalid explicitly-specified argument
for template parameter 'TileDataTmp'
```

当前 PTO-ISA `include/pto/common/pto_instr.hpp` 的 compare overload 是：

```cpp
template <typename D, typename S, typename K, typename C, typename Tmp,
          CmpMode mode, typename... WaitEvents>
RecordEvent TGATHER(D& dst, S& src, K& k_value, C& count, Tmp& tmp,
                   int offset, WaitEvents&... events);
```

即 5 个 Tile 类型再加 CmpMode，调用顺序是 `dst, src, k_tile, count, tmp, offset`。
PTOAS 当前只给 4 个 Tile 类型，而且接受/输出的是 scalar kValue。

### Expected behavior / 修改建议

1. 明确 compare-form 的 scalar 阈值合同，并与 ISA/辅助 lowering 一致；修正
   template 数量/顺序及 cdst/tmp 参数顺序。
2. 当前 ISA 路径的 k_tile 会做 INT→float 转换，不能直接把任意 FP32 阈值
   塞入该旧接口就宣称支持。建议为真正同类型 scalar 阈值提供 lowering，
   或建立支持该语义的 ISA overload。不要仅把 0.5 转整数作为适配。
3. 若提供 scalar lowering，可用 TCMPS 产生 mask、Vector 生成位置，再用
   硬件压缩收集，避免逐候选 Scalar 写 index。此条是性能建议，区别于 ABI bug。
4. 补实际 CCEC 编译测试，覆盖 FP32/FP16 的非整数阈值、INT32 EQ、offset、
   计数/容量边界。仅检查生成 C++ 中出现 TGATHER 无法发现本次问题。
5. 核对输出 count 的有效形状，以及 scratch 的真实访问范围和 Vector→Scalar
   count 可见性后再上卡；本文没有把这些未测试项预记为 PASS。

### Git commit

PTOAS `e5c1c432af8249bac2f905c7648ebafefec3c20a`（0.67 路线）；2026-10-20
只读核对 GitHub master 同函数仍输出相同参数形式，未重新构建整个 master。
PyPTO `2dfa7f120ba5e577566046d920845aa241bfa7bb`。

PTO-ISA revision 见下；它是当前 Simpler 管理且正式 CSA 使用的固定依赖，
不存在修改共享安装或运行时加载临时头文件的情况。

### Host platform / Target Ascend arch

Linux aarch64，A3 / Ascend910_9392，CANN 9.2.0-beta.2。
只有 CPU lowering 与 CCEC 失败复现，没有设备运行或性能结论。

PTO-ISA: `327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d`。
