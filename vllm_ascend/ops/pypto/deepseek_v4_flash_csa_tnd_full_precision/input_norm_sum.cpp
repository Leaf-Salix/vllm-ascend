// SPDX-License-Identifier: Apache-2.0
// PTO half-interval square sum, matching the Native 4096-column order.
#include <cstdint>
#include <pto/pto-inst.hpp>
#include <tensor.h>
#include <intrinsic.h>
using namespace pto;
// Match the bundled Native ReduceSumHalfInterval: add opposite halves to 64,
// then WholeReduceSum (PTO TROWSUM on exactly 64 elements).
template <int N>
__aicore__ __attribute__((always_inline)) void fold_half() {
  Tile<TileType::Vec, float, 1, N, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
      low(1, N), high(1, N);
  TASSIGN(low, 24576);
  TASSIGN(high, 24576 + N * 4);
  TADD(low, low, high);
  pipe_barrier(PIPE_V);
  if constexpr (N > 64) fold_half<N / 2>();
}
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
  set_atomic_none();
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int row = get_block_idx(args);
  auto x_desc = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto out_desc = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto x = reinterpret_cast<__gm__ bfloat16_t*>(x_desc->buffer.addr) + x_desc->start_offset;
  auto out = reinterpret_cast<__gm__ float*>(out_desc->buffer.addr) + out_desc->start_offset;
  using InShape = Shape<1, 1, 1, 1, 4096>;
  using InStride = Stride<4096, 4096, 4096, 4096, 1>;
  using OutShape = Shape<1, 1, 1, 1, 8>;
  using OutStride = Stride<8, 8, 8, 8, 1>;
  GlobalTensor<bfloat16_t, InShape, InStride, Layout::ND> input(x + row * 4096, InShape(), InStride());
  GlobalTensor<float, OutShape, OutStride, Layout::ND> output(out + row * 8, OutShape(), OutStride());
  Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
       CompactMode::Null>
      raw(1, 4096);
  Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
       CompactMode::Null>
      wide(1, 4096), square(1, 4096);
  // A padded row-major result has the required 32B allocation and validRow=1.
  // TROWSUM writes column0; the other seven diagnostic columns remain zero.
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
      sum(1, 8);
  TASSIGN(raw, 0);
  TASSIGN(wide, 8192);
  TASSIGN(square, 24576);
  TASSIGN(sum, 57344);
  TLOAD(raw, input);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TCVT(wide, raw, RoundMode::CAST_ROUND, SaturationMode::ON);
  pipe_barrier(PIPE_V);
  TMUL(square, wide, wide);
  TEXPANDS(sum, 0.0f);
  pipe_barrier(PIPE_V);
  fold_half<2048>();
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
      last(1, 64), work(1, 64);
  TASSIGN(last, 24576);
  TASSIGN(work, 40960);
  TROWSUM(sum, last, work);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(output, sum);
  set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  pipe_barrier(PIPE_ALL);
#endif
}
