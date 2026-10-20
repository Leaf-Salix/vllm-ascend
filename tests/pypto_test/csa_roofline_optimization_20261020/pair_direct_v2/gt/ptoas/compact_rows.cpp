#include "pto/pto-inst.hpp"
using namespace pto;

template <typename Tensor>
static AICORE inline auto PTOAS__GLOBAL_TENSOR_DATA(Tensor &tensor)
    -> decltype(tensor.data()) {
  return tensor.data();
}


template <unsigned indexCols>
AICORE inline void PTOAS__MAKE_LINEAR_INDICES(
    __ubuf__ int32_t *indices, __ubuf__ uint8_t *work, int32_t offset) {
  auto seed0 = reinterpret_cast<__ubuf__ float*>(work);
  auto seed1 = seed0 + 128;
  // 与 ISA 的带 scratch TCI 同一 Vector 构造方式；没有按候选逐项 Scalar 写入。
  set_mask_count();
  set_vector_mask(0, 8);
  #pragma unroll
  for (int i = 0; i < 8; ++i)
    vector_dup(seed0 + i * 8, static_cast<float>(i) * 0.125f, 1, 1, 1, 1, (int64_t)0);
  pipe_barrier(PIPE_V);
  set_vector_mask(0, 64);
  vcgadd(seed1, seed0, 1, 1, 1, 8);
  pipe_barrier(PIPE_V);
  set_vector_mask(0, 8);
  vmuls(seed0, seed1, 8.0f, 1, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  set_mask_norm();
  set_vector_mask(-1, -1);
  vbrcb(reinterpret_cast<__ubuf__ uint32_t*>(seed0),
        reinterpret_cast<__ubuf__ uint32_t*>(seed0), 1, 8, 1);
  pipe_barrier(PIPE_V);
  vadd(seed1, seed1, seed0, 1, 1, 0, 1, 8, 0, 8);
  pipe_barrier(PIPE_V);
  vconv_f322s32r(reinterpret_cast<__ubuf__ int32_t*>(seed1), seed1, 1, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  #pragma unroll
  for (int i = 0; i < indexCols / 64; ++i)
    vadds(indices + i * 64, reinterpret_cast<__ubuf__ int32_t*>(seed1),
          offset + i * 64, 1, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);

}

template <typename D, typename S, typename Tmp, typename C, CmpMode Mode>
__tf__ AICORE void PTOAS__GATHER_COMPARE_SCALAR_RAW(
    typename D::TileDType __out__ dst,
    typename S::TileDType __in__ src,
    typename Tmp::TileDType __out__ tmp,
    typename C::TileDType __out__ count,
    typename S::DType threshold, int32_t offset, unsigned rows, unsigned cols) {
  // index 按 64 个 INT32 对齐；后区先生成 Vector 索引种子，随后复用为
  // bit mask 和完整压缩暂存，不能再从 tmp 的末尾继续越界分配。
  constexpr unsigned indexCols = ((S::Cols + 63) / 64) * 64;
  constexpr unsigned indexBytes = indexCols * 4;
  constexpr unsigned maskBytes = ((S::Cols + 255) / 256) * 32;
  constexpr unsigned collectBytes = maskBytes + indexBytes;
  constexpr unsigned scratchBytes = collectBytes > 1024 ? collectBytes : 1024;
  static_assert(Tmp::Rows * Tmp::Cols * sizeof(typename Tmp::DType) >=
                indexBytes + scratchBytes, "compare gather tmp is too small");
  static_assert(Mode == CmpMode::EQ || Mode == CmpMode::GT || Mode == CmpMode::GE,
                "A3 scalar compare gather currently supports eq/gt/ge");
  auto out = reinterpret_cast<__ubuf__ int32_t*>(__cce_get_tile_ptr(dst));
  auto input = reinterpret_cast<__ubuf__ typename S::DType*>(__cce_get_tile_ptr(src));
  auto base = reinterpret_cast<__ubuf__ uint8_t*>(__cce_get_tile_ptr(tmp));
  auto counts = reinterpret_cast<__ubuf__ typename C::DType*>(__cce_get_tile_ptr(count));
  auto indices = reinterpret_cast<__ubuf__ int32_t*>(base);
  auto work = base + indexBytes;

  PTOAS__MAKE_LINEAR_INDICES<indexCols>(indices, work, offset);

  auto mask = work;
  auto packed = reinterpret_cast<__ubuf__ uint32_t*>(work + maskBytes);
  for (unsigned row = 0; row < rows; ++row) {
    // 比较保留原 dtype 的阈值，尤其不能把 0.5 等 FP32 值转成整数。
    constexpr unsigned perRepeat = 256 / sizeof(typename S::DType);
    const unsigned repeats = (cols + perRepeat - 1) / perRepeat;
    set_mask_norm();
    set_vector_mask(-1, -1);
    for (unsigned rep = 0; rep < repeats; rep += 240) {
      const unsigned remain = repeats - rep;
      const unsigned num = remain < 240 ? remain : 240;
      if constexpr (Mode == CmpMode::GT) {
        vcmpvs_gt(mask + rep * perRepeat / 8,
                  input + row * S::RowStride + rep * perRepeat,
                  threshold, num, 1, 1, 8, 8);
      } else if constexpr (Mode == CmpMode::GE) {
        vcmpvs_ge(mask + rep * perRepeat / 8,
                  input + row * S::RowStride + rep * perRepeat,
                  threshold, num, 1, 1, 8, 8);
      } else {
        vcmpvs_eq(mask + rep * perRepeat / 8,
                  input + row * S::RowStride + rep * perRepeat,
                  threshold, num, 1, 1, 8, 8);
      }
    }
    pipe_barrier(PIPE_V);
    set_mask_count();
    set_vector_mask(0, cols);
    vreducev2(packed, reinterpret_cast<__ubuf__ uint32_t*>(indices),
              reinterpret_cast<__ubuf__ uint32_t*>(mask), 1, 1, 0, 8, 0);
    // CANN A3 GatherMask 同样在 vreducev2 后读取该硬件计数寄存器。
    const unsigned kept = get_rsvd_cnt();
    counts[row] = kept;
    pipe_barrier(PIPE_V);
    // 输出不足时只写容量内的前缀，同时返回真实 count；调用方须据 count
    // 识别容量不足，任何情况下均不能覆盖输出或 scratch 的保护区。
    const unsigned copied = kept < D::Cols ? kept : D::Cols;
    if (copied != 0) {
      set_vector_mask(0, copied);
      vcopy(reinterpret_cast<__ubuf__ uint32_t*>(out + row * D::RowStride),
            packed, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
    }
  }
  set_mask_norm();
  set_vector_mask(-1, -1);
}

template <typename D, typename S, typename Tmp, typename C, CmpMode Mode>
AICORE inline void PTOAS__GATHER_COMPARE_SCALAR(
    D &dst, S &src, typename S::DType threshold, Tmp &tmp, C &count,
    int32_t offset) {
  PTOAS__GATHER_COMPARE_SCALAR_RAW<D, S, Tmp, C, Mode>(
      dst.data(), src.data(), tmp.data(), count.data(),
      threshold, offset, src.GetValidRow(), src.GetValidCol());
}

// 成对压缩共用比较掩码。所有索引始终是整数，FP32 score只做位搬运。
// 未命中的尾部不定义；输出写入严格限定在capacity内，count始终返回真实值。
template <typename D, typename S, typename I, typename Tmp, typename C, CmpMode Mode>
__tf__ AICORE void PTOAS__COMPACT_COMPARE_RAW(
    typename D::TileDType __out__ dst,
    typename S::TileDType __in__ src,
    typename I::TileDType __out__ ids,
    typename Tmp::TileDType __out__ tmp,
    typename C::TileDType __out__ count,
    float threshold, int32_t offset, unsigned rows, unsigned cols) {
  constexpr unsigned indexCols = ((S::Cols + 63) / 64) * 64;
  constexpr unsigned indexBytes = indexCols * 4;
  constexpr unsigned maskBytes = ((S::Cols + 255) / 256) * 32;
  constexpr unsigned collectBytes = maskBytes + indexBytes;
  constexpr unsigned scratchBytes = collectBytes > 1024 ? collectBytes : 1024;
  static_assert(Tmp::Rows * Tmp::Cols * sizeof(typename Tmp::DType) >=
                indexBytes + scratchBytes, "compact scratch is too small");
  static_assert(D::Cols == I::Cols, "compact output capacities must match");
  auto values = reinterpret_cast<__ubuf__ uint32_t*>(__cce_get_tile_ptr(dst));
  auto outIds = reinterpret_cast<__ubuf__ uint32_t*>(__cce_get_tile_ptr(ids));
  auto input = reinterpret_cast<__ubuf__ float*>(__cce_get_tile_ptr(src));
  auto counts = reinterpret_cast<__ubuf__ int32_t*>(__cce_get_tile_ptr(count));
  auto base = reinterpret_cast<__ubuf__ uint8_t*>(__cce_get_tile_ptr(tmp));
  auto indices = reinterpret_cast<__ubuf__ int32_t*>(base);
  auto work = base + indexBytes;
  PTOAS__MAKE_LINEAR_INDICES<indexCols>(indices, work, offset);
  auto mask = work;
  auto packed = reinterpret_cast<__ubuf__ uint32_t*>(work + maskBytes);
  for (unsigned row = 0; row < rows; ++row) {
    if (cols == 0) {
      counts[row] = 0;
      continue;
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
    const unsigned repeats = (cols + 63) / 64;
    for (unsigned rep = 0; rep < repeats; rep += 240) {
      const unsigned num = repeats - rep < 240 ? repeats - rep : 240;
      if constexpr (Mode == CmpMode::GT)
        vcmpvs_gt(mask + rep * 8, input + row * S::RowStride + rep * 64,
                  threshold, num, 1, 1, 8, 8);
      else if constexpr (Mode == CmpMode::GE)
        vcmpvs_ge(mask + rep * 8, input + row * S::RowStride + rep * 64,
                  threshold, num, 1, 1, 8, 8);
      else
        vcmpvs_eq(mask + rep * 8, input + row * S::RowStride + rep * 64,
                  threshold, num, 1, 1, 8, 8);
    }
    pipe_barrier(PIPE_V);
    set_mask_count();
    set_vector_mask(0, cols);
    vreducev2(packed, reinterpret_cast<__ubuf__ uint32_t*>(indices),
              reinterpret_cast<__ubuf__ uint32_t*>(mask), 1, 1, 0, 8, 0);
    const unsigned kept = get_rsvd_cnt();
    counts[row] = kept;
    pipe_barrier(PIPE_V);
    const unsigned copied = kept < D::Cols ? kept : D::Cols;
    if (copied != 0) {
      set_vector_mask(0, copied);
      vcopy(outIds + row * I::RowStride, packed, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      // 使用相同mask压缩score的32位表示，不执行浮点转换或重新计算。
      set_vector_mask(0, cols);
      vreducev2(packed,
                reinterpret_cast<__ubuf__ uint32_t*>(input + row * S::RowStride),
                reinterpret_cast<__ubuf__ uint32_t*>(mask), 1, 1, 0, 8, 0);
      pipe_barrier(PIPE_V);
      set_vector_mask(0, copied);
      vcopy(values + row * D::RowStride, packed, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
    }
  }
  set_mask_norm();
  set_vector_mask(-1, -1);
}

template <typename D, typename S, typename I, typename Tmp, typename C, CmpMode Mode>
AICORE inline void PTOAS__COMPACT_COMPARE(
    D &dst, S &src, I &indices, float threshold, Tmp &tmp, C &count, int32_t offset) {
  PTOAS__COMPACT_COMPARE_RAW<D, S, I, Tmp, C, Mode>(
      dst.data(), src.data(), indices.data(), tmp.data(), count.data(),
      threshold, offset, src.GetValidRow(), src.GetValidCol());
}


enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}

template <typename Ptr>
static AICORE inline void PTOAS__DCCI_SINGLE_CACHE_LINE(Ptr ptr) {
  dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);
}

AICORE void compact_rows(__gm__ float* v1, __gm__ int32_t* v2, __gm__ float* v3, __gm__ int32_t* v4, __gm__ float* v5, __gm__ int32_t* v6, __gm__ int32_t* v7, int64_t v8) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c4128_i64
  const int64_t v9 = 4128;
  // pto: %c12320_i64
  const int64_t v10 = 12320;
  // pto: %c30752_i64
  const int64_t v11 = 30752;
  // pto: %c0_i64
  const int64_t v12 = 0;
  // pto: %c4096_i64
  const int64_t v13 = 4096;
  // pto: %c2048_index
  const int64_t v14 = 2048;
  // pto: %c1_index
  const int64_t v15 = 1;
  // pto: %c1040_index
  const int64_t v16 = 1040;
  // pto: %c16_index
  const int64_t v17 = 16;
  // pto: %c0_index
  const int64_t v18 = 0;
  // pto: %c18432_index
  const int64_t v19 = 18432;
  // pto: %c1024_index
  const int64_t v20 = 1024;
  // pto: %c8_index
  const int64_t v21 = 8;
  // pto: %src__ssa_v0_view
  const int64_t v22 = 1;
  // pto: %src__ssa_v0_view
  const int64_t v23 = 1;
  // pto: %src__ssa_v0_view
  const int64_t v24 = 1;
  // pto: %src__ssa_v0_view
  int64_t v25 = (int64_t) v8;
  // pto: %src__ssa_v0_view
  int64_t v26 = v25 * v14;
  // pto: %src__ssa_v0_view
  int64_t v27 = v24 * v26;
  // pto: %src__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v28 = pto::Shape<1, 1, 1, -1, -1>(v22, v23, v24, v25, v14);
  // pto: %src__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v29 = pto::Stride<-1, -1, -1, -1, -1>(v23 * v27, v27, v26, v14, v15);
  // pto: %src__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v30 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v28, v29);
  // pto: %values_out__ssa_v0_view
  const int64_t v31 = 1;
  // pto: %values_out__ssa_v0_view
  const int64_t v32 = 1;
  // pto: %values_out__ssa_v0_view
  const int64_t v33 = 1;
  // pto: %values_out__ssa_v0_view
  int64_t v34 = (int64_t) v8;
  // pto: %values_out__ssa_v0_view
  int64_t v35 = v34 * v16;
  // pto: %values_out__ssa_v0_view
  int64_t v36 = v33 * v35;
  // pto: %values_out__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v37 = pto::Shape<1, 1, 1, -1, -1>(v31, v32, v33, v34, v16);
  // pto: %values_out__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v38 = pto::Stride<-1, -1, -1, -1, -1>(v32 * v36, v36, v35, v16, v15);
  // pto: %values_out__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v39 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v5, v37, v38);
  // pto: %ids_out__ssa_v0_view
  const int64_t v40 = 1;
  // pto: %ids_out__ssa_v0_view
  const int64_t v41 = 1;
  // pto: %ids_out__ssa_v0_view
  const int64_t v42 = 1;
  // pto: %ids_out__ssa_v0_view
  int64_t v43 = (int64_t) v8;
  // pto: %ids_out__ssa_v0_view
  int64_t v44 = v43 * v16;
  // pto: %ids_out__ssa_v0_view
  int64_t v45 = v42 * v44;
  // pto: %ids_out__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v46 = pto::Shape<1, 1, 1, -1, -1>(v40, v41, v42, v43, v16);
  // pto: %ids_out__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v47 = pto::Stride<-1, -1, -1, -1, -1>(v41 * v45, v45, v44, v16, v15);
  // pto: %ids_out__ssa_v0_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v48 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v46, v47);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  for (int64_t v49 = v18; v49 < v8; v49 += v15) {
    // pto: %values__ssa_v0
    ;
    Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v50 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v14);
    // pto: %values__ssa_v0
    ;
    uint64_t v51 = (uint64_t) v9;
    TASSIGN(v50, v51);
    // pto: %0
    ;
    int64_t v52 = v49 < v18 ? v18 : v49;
    // pto: %src__ssa_v0_pview
    ;
    __gm__ float* v53 = PTOAS__GLOBAL_TENSOR_DATA(v30);
    // pto: %src__ssa_v0_pview
    ;
    const int64_t v54 = 0;
    // pto: %src__ssa_v0_pview
    ;
    const int64_t v55 = 2048;
    // pto: %src__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, 2048> v56 = pto::Shape<1, 1, 1, 1, 2048>();
    // pto: %src__ssa_v0_pview
    ;
    pto::Stride<2048, 2048, 2048, 2048, 1> v57 = pto::Stride<2048, 2048, 2048, 2048, 1>();
    // pto: %src__ssa_v0_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 2048>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND> v58 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 2048>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND>(v53 + (v54 + v52 * v55), v56, v57);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v50, v58);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %n__tile
    ;
    int32_t v59 = (v2)[v49];
    // pto: %1
    ;
    int64_t v60 = (int64_t) v59;
    v50.SetValidShape(v15, v60);
    // pto: %threshold__tile
    ;
    float v61 = (v3)[v49];
    // pto: %offset__tile
    ;
    int32_t v62 = (v4)[v49];
    // pto: %scratch__ssa_v0
    ;
    Tile<TileType::Vec, uint8_t, 1, 18432, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v63 = Tile<TileType::Vec, uint8_t, 1, 18432, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v19);
    // pto: %scratch__ssa_v0
    ;
    uint64_t v64 = (uint64_t) v10;
    TASSIGN(v63, v64);
    // pto: %packed__ssa_v0
    ;
    Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v65 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v20);
    // pto: %packed__ssa_v0
    ;
    uint64_t v66 = (uint64_t) v11;
    TASSIGN(v65, v66);
    // pto: %ids__ssa_v0
    ;
    Tile<TileType::Vec, int32_t, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v67 = Tile<TileType::Vec, int32_t, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v20);
    // pto: %ids__ssa_v0
    ;
    uint64_t v68 = (uint64_t) v12;
    TASSIGN(v67, v68);
    // pto: %count__ssa_v0
    ;
    Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v69 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v15);
    // pto: %count__ssa_v0
    ;
    uint64_t v70 = (uint64_t) v13;
    TASSIGN(v69, v70);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    PTOAS__COMPACT_COMPARE<Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, int32_t, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, uint8_t, 1, 18432, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, CmpMode::GT>(v65, v50, v67, v61, v63, v69, v62);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %values_out__ssa_v0_pview
    ;
    __gm__ float* v71 = PTOAS__GLOBAL_TENSOR_DATA(v39);
    // pto: %values_out__ssa_v0_pview
    ;
    const int64_t v72 = 8;
    // pto: %values_out__ssa_v0_pview
    ;
    const int64_t v73 = 1040;
    // pto: %values_out__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, 1024> v74 = pto::Shape<1, 1, 1, 1, 1024>();
    // pto: %values_out__ssa_v0_pview
    ;
    pto::Stride<1040, 1040, 1040, 1040, 1> v75 = pto::Stride<1040, 1040, 1040, 1040, 1>();
    // pto: %values_out__ssa_v0_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1040, 1040, 1040, 1040, 1>, pto::Layout::ND> v76 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1040, 1040, 1040, 1040, 1>, pto::Layout::ND>(v71 + (v72 + v52 * v73), v74, v75);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v76, v65);
    // pto: %ids_out__ssa_v0_pview
    ;
    __gm__ int32_t* v77 = PTOAS__GLOBAL_TENSOR_DATA(v48);
    // pto: %ids_out__ssa_v0_pview
    ;
    const int64_t v78 = 8;
    // pto: %ids_out__ssa_v0_pview
    ;
    const int64_t v79 = 1040;
    // pto: %ids_out__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, 1024> v80 = pto::Shape<1, 1, 1, 1, 1024>();
    // pto: %ids_out__ssa_v0_pview
    ;
    pto::Stride<1040, 1040, 1040, 1040, 1> v81 = pto::Stride<1040, 1040, 1040, 1040, 1>();
    // pto: %ids_out__ssa_v0_pview
    ;
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1040, 1040, 1040, 1040, 1>, pto::Layout::ND> v82 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1040, 1040, 1040, 1040, 1>, pto::Layout::ND>(v77 + (v78 + v52 * v79), v80, v81);
    TSTORE(v82, v67);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    // pto: %t__tmp_v1
    ;
    int32_t v83 = v69.GetValue(v18);
    // pto: %flat_offset_mul, %flat_offset
    ;
    int64_t v84 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v49 * (uint64_t) v17) + (uint64_t) v21);
    (v7)[v84] = v83;
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
  #endif // __DAV_VEC__

  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}