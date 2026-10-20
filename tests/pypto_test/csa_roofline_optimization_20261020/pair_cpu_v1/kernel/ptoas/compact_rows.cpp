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

AICORE void compact_rows(__gm__ float* v1, __gm__ float* v2, __gm__ int32_t* v3, __gm__ float* v4, __gm__ float* v5, int64_t v6) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  pto::MrgSortExecutedNumList v7 = pto::MrgSortExecutedNumList{0, 0, 0, 0};
  // pto: %c55296_i64
  const int64_t v8 = 55296;
  // pto: %c63488_i64
  const int64_t v9 = 63488;
  // pto: %c20480_i64
  const int64_t v10 = 20480;
  // pto: %c36864_i64
  const int64_t v11 = 36864;
  // pto: %c4096_i64
  const int64_t v12 = 4096;
  // pto: %c0_i64
  const int64_t v13 = 0;
  const int64_t v14 = 40960;
  // pto: %c12288_i64
  const int64_t v15 = 12288;
  // pto: %c2048_index
  const int64_t v16 = 2048;
  // pto: %c1_index
  const int64_t v17 = 1;
  // pto: %c1024_index
  const int64_t v18 = 1024;
  // pto: %c1040_index
  const int64_t v19 = 1040;
  // pto: %c0_index
  const int64_t v20 = 0;
  // pto: %c192_index
  const int64_t v21 = 192;
  // pto: %c0_i32
  const int32_t v22 = 0;
  const int32_t v23 = 8192;
  // pto: %c512_index
  const int64_t v24 = 512;
  // pto: %cst_17
  const float v25 = -3.40282347E+38f;
  // pto: %c64_i32
  const int32_t v26 = 64;
  // pto: %c256_i32
  const int32_t v27 = 256;
  // pto: %c4096_index
  const int64_t v28 = 4096;
  // pto: %c1024_i32
  const int32_t v29 = 1024;
  // pto: %c1022_index
  const int64_t v30 = 1022;
  // pto: %c18432_index
  const int64_t v31 = 18432;
  // pto: %cst_26
  const float v32 = 0.0f;
  // pto: %src__ssa_v0_view
  const int64_t v33 = 1;
  // pto: %src__ssa_v0_view
  const int64_t v34 = 1;
  // pto: %src__ssa_v0_view
  const int64_t v35 = 1;
  // pto: %src__ssa_v0_view
  int64_t v36 = (int64_t) v6;
  // pto: %src__ssa_v0_view
  int64_t v37 = v36 * v16;
  // pto: %src__ssa_v0_view
  int64_t v38 = v35 * v37;
  // pto: %src__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v39 = pto::Shape<1, 1, 1, -1, -1>(v33, v34, v35, v36, v16);
  // pto: %src__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v40 = pto::Stride<-1, -1, -1, -1, -1>(v34 * v38, v38, v37, v16, v17);
  // pto: %src__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v41 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v39, v40);
  // pto: %old__ssa_v0_view
  const int64_t v42 = 1;
  // pto: %old__ssa_v0_view
  const int64_t v43 = 1;
  // pto: %old__ssa_v0_view
  const int64_t v44 = 1;
  // pto: %old__ssa_v0_view
  int64_t v45 = (int64_t) v6;
  // pto: %old__ssa_v0_view
  int64_t v46 = v45 * v18;
  // pto: %old__ssa_v0_view
  int64_t v47 = v44 * v46;
  // pto: %old__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v48 = pto::Shape<1, 1, 1, -1, -1>(v42, v43, v44, v45, v18);
  // pto: %old__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v49 = pto::Stride<-1, -1, -1, -1, -1>(v43 * v47, v47, v46, v18, v17);
  // pto: %old__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v50 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v48, v49);
  // pto: %ref__ssa_v0_view
  const int64_t v51 = 1;
  // pto: %ref__ssa_v0_view
  const int64_t v52 = 1;
  // pto: %ref__ssa_v0_view
  const int64_t v53 = 1;
  // pto: %ref__ssa_v0_view
  int64_t v54 = (int64_t) v6;
  // pto: %ref__ssa_v0_view
  int64_t v55 = v54 * v19;
  // pto: %ref__ssa_v0_view
  int64_t v56 = v53 * v55;
  // pto: %ref__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v57 = pto::Shape<1, 1, 1, -1, -1>(v51, v52, v53, v54, v19);
  // pto: %ref__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v58 = pto::Stride<-1, -1, -1, -1, -1>(v52 * v56, v56, v55, v19, v17);
  // pto: %ref__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v59 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v57, v58);
  // pto: %candidate__ssa_v0_view
  const int64_t v60 = 1;
  // pto: %candidate__ssa_v0_view
  const int64_t v61 = 1;
  // pto: %candidate__ssa_v0_view
  const int64_t v62 = 1;
  // pto: %candidate__ssa_v0_view
  int64_t v63 = (int64_t) v6;
  // pto: %candidate__ssa_v0_view
  int64_t v64 = v63 * v19;
  // pto: %candidate__ssa_v0_view
  int64_t v65 = v62 * v64;
  // pto: %candidate__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v66 = pto::Shape<1, 1, 1, -1, -1>(v60, v61, v62, v63, v19);
  // pto: %candidate__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v67 = pto::Stride<-1, -1, -1, -1, -1>(v61 * v65, v65, v64, v19, v17);
  // pto: %candidate__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v68 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v5, v66, v67);
  // pto: %root_inline56__phi_v4
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v69 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
  // pto: %root_inline56__phi_v4
  uint64_t v70 = (uint64_t) v13;
  TASSIGN(v69, v70);
  // pto: %root_inline56__phi_v3
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v71 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
  // pto: %root_inline56__phi_v3
  uint64_t v72 = (uint64_t) v13;
  TASSIGN(v71, v72);
  // pto: %packed_values_inline73__phi_v2
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v73 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
  // pto: %packed_values_inline73__phi_v2
  uint64_t v74 = (uint64_t) v10;
  TASSIGN(v73, v74);
  // pto: %selected_inline80__phi_v2
  Tile<TileType::Vec, int32_t, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v75 = Tile<TileType::Vec, int32_t, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
  // pto: %selected_inline80__phi_v2
  uint64_t v76 = (uint64_t) v12;
  TASSIGN(v75, v76);
  // pto: %next_root_inline71__phi_v2
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v77 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
  // pto: %next_root_inline71__phi_v2
  uint64_t v78 = (uint64_t) v9;
  TASSIGN(v77, v78);
  // pto: %new_root_inline79__phi_v2
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v79 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
  // pto: %new_root_inline79__phi_v2
  uint64_t v80 = (uint64_t) v13;
  TASSIGN(v79, v80);
  // pto: %root_inline84__phi_v2
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v81 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
  // pto: %root_inline84__phi_v2
  uint64_t v82 = (uint64_t) v13;
  TASSIGN(v81, v82);
  // pto: %root_inline112__phi_v4
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v83 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
  // pto: %root_inline112__phi_v4
  uint64_t v84 = (uint64_t) v13;
  TASSIGN(v83, v84);
  // pto: %root_inline112__phi_v3
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v85 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
  // pto: %root_inline112__phi_v3
  uint64_t v86 = (uint64_t) v12;
  TASSIGN(v85, v86);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  for (int64_t v87 = v20; v87 < v6; v87 += v17) {
    // pto: %values__ssa_v0
    ;
    Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v88 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v16);
    // pto: %values__ssa_v0
    ;
    uint64_t v89 = (uint64_t) v8;
    TASSIGN(v88, v89);
    // pto: %0
    ;
    int64_t v90 = v87 < v20 ? v20 : v87;
    // pto: %src__ssa_v0_pview
    ;
    __gm__ float* v91 = PTOAS__GLOBAL_TENSOR_DATA(v41);
    // pto: %src__ssa_v0_pview
    ;
    const int64_t v92 = 0;
    // pto: %src__ssa_v0_pview
    ;
    const int64_t v93 = 2048;
    // pto: %src__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, 2048> v94 = pto::Shape<1, 1, 1, 1, 2048>();
    // pto: %src__ssa_v0_pview
    ;
    pto::Stride<2048, 2048, 2048, 2048, 1> v95 = pto::Stride<2048, 2048, 2048, 2048, 1>();
    // pto: %src__ssa_v0_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 2048>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND> v96 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 2048>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND>(v91 + (v92 + v90 * v93), v94, v95);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
    TLOAD(v88, v96);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %root__ssa_v0
    ;
    Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v97 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
    // pto: %root__ssa_v0
    ;
    uint64_t v98 = (uint64_t) v9;
    TASSIGN(v97, v98);
    // pto: %old__ssa_v0_pview
    ;
    __gm__ float* v99 = PTOAS__GLOBAL_TENSOR_DATA(v50);
    // pto: %old__ssa_v0_pview
    ;
    const int64_t v100 = 0;
    // pto: %old__ssa_v0_pview
    ;
    const int64_t v101 = 1024;
    // pto: %old__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, 1024> v102 = pto::Shape<1, 1, 1, 1, 1024>();
    // pto: %old__ssa_v0_pview
    ;
    pto::Stride<1024, 1024, 1024, 1024, 1> v103 = pto::Stride<1024, 1024, 1024, 1024, 1>();
    // pto: %old__ssa_v0_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v104 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v99 + (v100 + v90 * v101), v102, v103);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    TLOAD(v97, v104);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    // pto: %n__tile
    ;
    int32_t v105 = (v3)[v87];
    // pto: %t__ci_tmp_v0
    ;
    Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v106 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v21);
    // pto: %t__ci_tmp_v0
    ;
    uint64_t v107 = (uint64_t) v10;
    TASSIGN(v106, v107);
    // pto: %t__tmp_v1
    ;
    Tile<TileType::Vec, int32_t, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v108 = Tile<TileType::Vec, int32_t, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v16);
    // pto: %t__tmp_v1
    ;
    uint64_t v109 = (uint64_t) v11;
    TASSIGN(v108, v109);
    // pto: %ci_tmp_view
    ;
    Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v110;
    TRESHAPE(v110, v106);
    // pto: %ci_dst_view
    ;
    Tile<TileType::Vec, int32_t, 1, 2048, BLayout::RowMajor, 1, 2048, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v111;
    TRESHAPE(v111, v108);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TCI<Tile<TileType::Vec, int32_t, 1, 2048, BLayout::RowMajor, 1, 2048, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v111, v22, v110);
    // pto: %indices__ssa_v0
    ;
    Tile<TileType::Vec, int32_t, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v112 = Tile<TileType::Vec, int32_t, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v16);
    // pto: %indices__ssa_v0
    ;
    uint64_t v113 = (uint64_t) v10;
    TASSIGN(v112, v113);
    pipe_barrier(PIPE_V);
    TADDS(v112, v108, v23);
    // pto: %2
    ;
    int64_t v114 = (int64_t) v105;
    // pto: %4
    ;
    bool v115 = v114 <= v24;
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    if (v115) {
      // pto: %tiny_v_inline54__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v116 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v24);
      // pto: %tiny_v_inline54__ssa_v0
      ;
      uint64_t v117 = (uint64_t) v11;
      TASSIGN(v116, v117);
      pipe_barrier(PIPE_V);
      uint16_t v118 = static_cast<uint16_t>(v20);
      uint16_t v119 = static_cast<uint16_t>(v20);
      TEXTRACT(v116, v88, v118, v119);
      v116.SetValidShape(v17, v114);
      // pto: %t__tmp_v2
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v120 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v24);
      // pto: %t__tmp_v2
      ;
      uint64_t v121 = (uint64_t) v11;
      TASSIGN(v120, v121);
      pipe_barrier(PIPE_V);
      TFILLPAD<pto::TFillPadMode::InPlace>(v120, v116);
      // pto: %tiny_v_v2_inline60__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v122 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v24);
      // pto: %tiny_v_v2_inline60__ssa_v0
      ;
      uint64_t v123 = (uint64_t) v12;
      TASSIGN(v122, v123);
      pipe_barrier(PIPE_V);
      TMAXS(v122, v120, v25);
      // pto: %tiny_i_inline57__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v124 = Tile<TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v24);
      // pto: %tiny_i_inline57__ssa_v0
      ;
      uint64_t v125 = (uint64_t) v10;
      TASSIGN(v124, v125);
      // pto: %t__tmp_v3
      ;
      Tile<TileType::Vec, uint32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v126 = Tile<TileType::Vec, uint32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v24);
      // pto: %t__tmp_v3
      ;
      uint64_t v127 = (uint64_t) v10;
      TASSIGN(v126, v127);
      // pto: %tiny_pairs_inline61__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v128 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v18);
      // pto: %tiny_pairs_inline61__ssa_v0
      ;
      uint64_t v129 = (uint64_t) v11;
      TASSIGN(v128, v129);
      // pto: %sort32_src_view
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v130;
      TRESHAPE(v130, v122);
      // pto: %sort32_idx_view
      ;
      Tile<TileType::Vec, uint32_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v131;
      TRESHAPE(v131, v126);
      // pto: %sort32_dst_view
      ;
      Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v132;
      TRESHAPE(v132, v128);
      pipe_barrier(PIPE_V);
      TSORT32(v132, v130, v131);
      // pto: %tiny_pairs_v1_inline52__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v133 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v18);
      // pto: %tiny_pairs_v1_inline52__ssa_v0
      ;
      uint64_t v134 = (uint64_t) v12;
      TASSIGN(v133, v134);
      pipe_barrier(PIPE_V);
      TMRGSORT(v133, v128, v26);
      // pto: %tiny_pairs_v2_inline62__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v135 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v18);
      // pto: %tiny_pairs_v2_inline62__ssa_v0
      ;
      uint64_t v136 = (uint64_t) v11;
      TASSIGN(v135, v136);
      pipe_barrier(PIPE_V);
      TMRGSORT(v135, v133, v27);
      // pto: %root_inline56__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v137 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
      // pto: %root_inline56__ssa_v0
      ;
      uint64_t v138 = (uint64_t) v13;
      TASSIGN(v137, v138);
      pipe_barrier(PIPE_V);
      uint16_t v139 = static_cast<uint16_t>(v20);
      uint16_t v140 = static_cast<uint16_t>(v20);
      TEXTRACT(v137, v135, v139, v140);
    } else {
      // pto: %6
      ;
      if (v114 <= v18) {
        // pto: %small_v_inline64__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v141 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
        // pto: %small_v_inline64__ssa_v0
        ;
        uint64_t v142 = (uint64_t) v11;
        TASSIGN(v141, v142);
        pipe_barrier(PIPE_V);
        uint16_t v143 = static_cast<uint16_t>(v20);
        uint16_t v144 = static_cast<uint16_t>(v20);
        TEXTRACT(v141, v88, v143, v144);
        v141.SetValidShape(v17, v114);
        // pto: %t__tmp_v4
        ;
        Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v145 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v18);
        // pto: %t__tmp_v4
        ;
        uint64_t v146 = (uint64_t) v11;
        TASSIGN(v145, v146);
        pipe_barrier(PIPE_V);
        TFILLPAD<pto::TFillPadMode::InPlace>(v145, v141);
        // pto: %small_v_v2_inline66__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v147 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v18);
        // pto: %small_v_v2_inline66__ssa_v0
        ;
        uint64_t v148 = (uint64_t) v12;
        TASSIGN(v147, v148);
        pipe_barrier(PIPE_V);
        TMAXS(v147, v145, v25);
        // pto: %small_i_inline63__ssa_v0
        ;
        Tile<TileType::Vec, int32_t, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v149 = Tile<TileType::Vec, int32_t, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
        // pto: %small_i_inline63__ssa_v0
        ;
        uint64_t v150 = (uint64_t) v10;
        TASSIGN(v149, v150);
        // pto: %t__tmp_v5
        ;
        Tile<TileType::Vec, uint32_t, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v151 = Tile<TileType::Vec, uint32_t, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
        // pto: %t__tmp_v5
        ;
        uint64_t v152 = (uint64_t) v10;
        TASSIGN(v151, v152);
        // pto: %small_pairs_inline51__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v153 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v16);
        // pto: %small_pairs_inline51__ssa_v0
        ;
        uint64_t v154 = (uint64_t) v11;
        TASSIGN(v153, v154);
        // pto: %8
        ;
        Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v155;
        TRESHAPE(v155, v147);
        // pto: %9
        ;
        Tile<TileType::Vec, uint32_t, 1, 1024, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v156;
        TRESHAPE(v156, v151);
        // pto: %10
        ;
        Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, 2048, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v157;
        TRESHAPE(v157, v153);
        pipe_barrier(PIPE_V);
        TSORT32(v157, v155, v156);
        // pto: %small_pairs_v1_inline49__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v158 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v16);
        // pto: %small_pairs_v1_inline49__ssa_v0
        ;
        uint64_t v159 = (uint64_t) v12;
        TASSIGN(v158, v159);
        pipe_barrier(PIPE_V);
        TMRGSORT(v158, v153, v26);
        // pto: %small_pairs_v2_inline48__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v160 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v16);
        // pto: %small_pairs_v2_inline48__ssa_v0
        ;
        uint64_t v161 = (uint64_t) v11;
        TASSIGN(v160, v161);
        pipe_barrier(PIPE_V);
        TMRGSORT(v160, v158, v27);
        // pto: %small_left_inline65__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v162 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v18);
        // pto: %small_left_inline65__ssa_v0
        ;
        uint64_t v163 = (uint64_t) v11;
        TASSIGN(v162, v163);
        // pto: %11
        ;
        Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v164;
        // pto: %11
        ;
        uint64_t v165 = (uint64_t) v11;
        TASSIGN(v164, v165);
        // pto: %small_right_inline46__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v166 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v18);
        // pto: %small_right_inline46__ssa_v0
        ;
        uint64_t v167 = (uint64_t) v14;
        TASSIGN(v166, v167);
        // pto: %12
        ;
        Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v168;
        // pto: %12
        ;
        uint64_t v169 = (uint64_t) v14;
        TASSIGN(v168, v169);
        // pto: %small_tmp_inline44__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v170 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v16);
        // pto: %small_tmp_inline44__ssa_v0
        ;
        uint64_t v171 = (uint64_t) v12;
        TASSIGN(v170, v171);
        // pto: %small_merged_inline50__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v172 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v16);
        // pto: %small_merged_inline50__ssa_v0
        ;
        uint64_t v173 = (uint64_t) v15;
        TASSIGN(v172, v173);
        pipe_barrier(PIPE_V);
        TMRGSORT<Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>, Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>, Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>, false>(v172, v7, v170, v164, v168);
        // pto: %root_inline56__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v174 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
        // pto: %root_inline56__ssa_v1
        ;
        uint64_t v175 = (uint64_t) v13;
        TASSIGN(v174, v175);
        pipe_barrier(PIPE_V);
        uint16_t v176 = static_cast<uint16_t>(v20);
        uint16_t v177 = static_cast<uint16_t>(v20);
        TEXTRACT(v174, v172, v176, v177);
      } else {
        v88.SetValidShape(v17, v114);
        // pto: %t__tmp_v6
        ;
        Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v178 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v16);
        // pto: %t__tmp_v6
        ;
        uint64_t v179 = (uint64_t) v11;
        TASSIGN(v178, v179);
        pipe_barrier(PIPE_V);
        TFILLPAD(v178, v88);
        // pto: %large_v_v1_inline53__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v180 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v16);
        // pto: %large_v_v1_inline53__ssa_v0
        ;
        uint64_t v181 = (uint64_t) v12;
        TASSIGN(v180, v181);
        pipe_barrier(PIPE_V);
        TMAXS(v180, v178, v25);
        // pto: %t__tmp_v7
        ;
        Tile<TileType::Vec, uint32_t, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v182 = Tile<TileType::Vec, uint32_t, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v16);
        // pto: %t__tmp_v7
        ;
        uint64_t v183 = (uint64_t) v10;
        TASSIGN(v182, v183);
        // pto: %large_pairs_inline47__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v184 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v28);
        // pto: %large_pairs_inline47__ssa_v0
        ;
        uint64_t v185 = (uint64_t) v11;
        TASSIGN(v184, v185);
        // pto: %14
        ;
        Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, 2048, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v186;
        TRESHAPE(v186, v180);
        // pto: %15
        ;
        Tile<TileType::Vec, uint32_t, 1, 2048, BLayout::RowMajor, 1, 2048, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v187;
        TRESHAPE(v187, v182);
        // pto: %16
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, 1, 4096, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v188;
        TRESHAPE(v188, v184);
        pipe_barrier(PIPE_V);
        TSORT32(v188, v186, v187);
        // pto: %large_pairs_v1_inline67__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v189 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v28);
        // pto: %large_pairs_v1_inline67__ssa_v0
        ;
        uint64_t v190 = (uint64_t) v10;
        TASSIGN(v189, v190);
        pipe_barrier(PIPE_V);
        TMRGSORT(v189, v184, v26);
        // pto: %large_pairs_v2_inline59__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v191 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v28);
        // pto: %large_pairs_v2_inline59__ssa_v0
        ;
        uint64_t v192 = (uint64_t) v11;
        TASSIGN(v191, v192);
        pipe_barrier(PIPE_V);
        TMRGSORT(v191, v189, v27);
        // pto: %large_pairs_v3_inline43__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v193 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v28);
        // pto: %large_pairs_v3_inline43__ssa_v0
        ;
        uint64_t v194 = (uint64_t) v10;
        TASSIGN(v193, v194);
        pipe_barrier(PIPE_V);
        TMRGSORT(v193, v191, v29);
        // pto: %root_inline56__ssa_v2
        ;
        Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v195 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
        // pto: %root_inline56__ssa_v2
        ;
        uint64_t v196 = (uint64_t) v11;
        TASSIGN(v195, v196);
        pipe_barrier(PIPE_V);
        uint16_t v197 = static_cast<uint16_t>(v20);
        uint16_t v198 = static_cast<uint16_t>(v20);
        TEXTRACT(v195, v193, v197, v198);
        // pto: %root_inline56__ssa_v2_mv
        ;
        Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v199 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
        // pto: %root_inline56__ssa_v2_mv
        ;
        uint64_t v200 = (uint64_t) v13;
        TASSIGN(v199, v200);
        pipe_barrier(PIPE_V);
        TMOV(v199, v195);
      };
    };
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
    // pto: %full_sorted__ssa_v0
    ;
    Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v201 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
    // pto: %full_sorted__ssa_v0
    ;
    uint64_t v202 = (uint64_t) v13;
    TASSIGN(v201, v202);
    // pto: %tmp__ssa_v0
    ;
    Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v203 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v16);
    // pto: %tmp__ssa_v0
    ;
    uint64_t v204 = (uint64_t) v11;
    TASSIGN(v203, v204);
    // pto: %full_merge__ssa_v0
    ;
    Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v205 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v16);
    // pto: %full_merge__ssa_v0
    ;
    uint64_t v206 = (uint64_t) v10;
    TASSIGN(v205, v206);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    TMRGSORT<Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, false>(v205, v7, v203, v201, v97);
    // pto: %full_top__ssa_v0
    ;
    Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v207 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
    // pto: %full_top__ssa_v0
    ;
    uint64_t v208 = (uint64_t) v11;
    TASSIGN(v207, v208);
    pipe_barrier(PIPE_V);
    uint16_t v209 = static_cast<uint16_t>(v20);
    uint16_t v210 = static_cast<uint16_t>(v20);
    TEXTRACT(v207, v205, v209, v210);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %ref__ssa_v0_pview
    ;
    __gm__ float* v211 = PTOAS__GLOBAL_TENSOR_DATA(v59);
    // pto: %ref__ssa_v0_pview
    ;
    const int64_t v212 = 8;
    // pto: %ref__ssa_v0_pview
    ;
    const int64_t v213 = 1040;
    // pto: %ref__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, 1024> v214 = pto::Shape<1, 1, 1, 1, 1024>();
    // pto: %ref__ssa_v0_pview
    ;
    pto::Stride<1040, 1040, 1040, 1040, 1> v215 = pto::Stride<1040, 1040, 1040, 1040, 1>();
    // pto: %ref__ssa_v0_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1040, 1040, 1040, 1040, 1>, pto::Layout::ND> v216 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1040, 1040, 1040, 1040, 1>, pto::Layout::ND>(v211 + (v212 + v90 * v213), v214, v215);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v216, v207);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    // pto: %threshold_inline81__ssa_v0
    ;
    float v217 = v97.GetValue(v30);
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_S, PIPE_V, EVENT_ID3);
    // pto: %19
    ;
    bool v218 = v25 != v25 || v217 != v217;
    // pto: %19
    ;
    bool v219 = v25 == v25 && v217 == v217;
    // pto: %19
    ;
    bool v220 = v219 && v25 < v217;
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    // pto: %kept_inline78__phi_v2
    ;
    int32_t v221;
    if (v220) {
      v88.SetValidShape(v17, v114);
      // pto: %compare_tmp_inline76__ssa_v0
      ;
      Tile<TileType::Vec, uint8_t, 1, 18432, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v222 = Tile<TileType::Vec, uint8_t, 1, 18432, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v31);
      // pto: %compare_tmp_inline76__ssa_v0
      ;
      uint64_t v223 = (uint64_t) v11;
      TASSIGN(v222, v223);
      // pto: %packed_values_inline73__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v224 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
      // pto: %packed_values_inline73__ssa_v0
      ;
      uint64_t v225 = (uint64_t) v10;
      TASSIGN(v224, v225);
      // pto: %selected_inline80__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v226 = Tile<TileType::Vec, int32_t, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
      // pto: %selected_inline80__ssa_v0
      ;
      uint64_t v227 = (uint64_t) v12;
      TASSIGN(v226, v227);
      // pto: %count_inline72__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v228 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v17);
      // pto: %count_inline72__ssa_v0
      ;
      uint64_t v229 = (uint64_t) v15;
      TASSIGN(v228, v229);
      pipe_barrier(PIPE_V);
      PTOAS__COMPACT_COMPARE<Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, int32_t, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, uint8_t, 1, 18432, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, CmpMode::GE>(v224, v88, v226, v217, v222, v228, v23);
      // pto: %kept_inline78__ssa_v1
      ;
      int32_t v230 = v228.GetValue(v20);
      v221 = v230;
    } else {
      // pto: %selected_inline80__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v231 = Tile<TileType::Vec, int32_t, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
      // pto: %selected_inline80__ssa_v1
      ;
      uint64_t v232 = (uint64_t) v11;
      TASSIGN(v231, v232);
      TEXPANDS(v231, v22);
      // pto: %packed_values_inline73__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v233 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
      // pto: %packed_values_inline73__ssa_v1
      ;
      uint64_t v234 = (uint64_t) v10;
      TASSIGN(v233, v234);
      TEXPANDS(v233, v32);
      // pto: %selected_inline80__ssa_v1_mv
      ;
      Tile<TileType::Vec, int32_t, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v235 = Tile<TileType::Vec, int32_t, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
      // pto: %selected_inline80__ssa_v1_mv
      ;
      uint64_t v236 = (uint64_t) v12;
      TASSIGN(v235, v236);
      pipe_barrier(PIPE_V);
      TMOV(v235, v231);
      v221 = v105;
    };
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    set_flag(PIPE_S, PIPE_V, EVENT_ID2);
    set_flag(PIPE_S, PIPE_V, EVENT_ID1);
    // pto: %22
    ;
    int64_t v237 = (int64_t) v221;
    // pto: %23
    ;
    wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID3);
    if (v237 == v20) {
      // pto: %next_root_inline71__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v238 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
      // pto: %next_root_inline71__ssa_v0
      ;
      uint64_t v239 = (uint64_t) v9;
      TASSIGN(v238, v239);
    } else {
      // pto: %24
      ;
      bool v240 = v221 < v105;
      // pto: %22
      ;
      int64_t v241 = (int64_t) v221;
      // pto: %22
      ;
      int64_t v242 = (int64_t) v221;
      // pto: %29
      ;
      bool v243 = v242 <= v24;
      // pto: %31, %32
      ;
      bool v244 = v243 | v114 > v18;
      // pto: %26, %27, %33, %35
      ;
      if ((v240 & v241 <= v18 & v244) & v220) {
        if (v243) {
          // pto: %tiny_v_inline97__ssa_v0
          ;
          Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v245 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v24);
          // pto: %tiny_v_inline97__ssa_v0
          ;
          uint64_t v246 = (uint64_t) v15;
          TASSIGN(v245, v246);
          pipe_barrier(PIPE_V);
          uint16_t v247 = static_cast<uint16_t>(v20);
          uint16_t v248 = static_cast<uint16_t>(v20);
          TEXTRACT(v245, v73, v247, v248);
          // pto: %22
          ;
          int64_t v249 = (int64_t) v221;
          v245.SetValidShape(v17, v249);
          // pto: %t__tmp_v8
          ;
          Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v250 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v24);
          // pto: %t__tmp_v8
          ;
          uint64_t v251 = (uint64_t) v15;
          TASSIGN(v250, v251);
          pipe_barrier(PIPE_V);
          TFILLPAD<pto::TFillPadMode::InPlace>(v250, v245);
          // pto: %tiny_v_v2_inline92__ssa_v0
          ;
          Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v252 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v24);
          // pto: %tiny_v_v2_inline92__ssa_v0
          ;
          uint64_t v253 = (uint64_t) v13;
          TASSIGN(v252, v253);
          pipe_barrier(PIPE_V);
          TMAXS(v252, v250, v25);
          // pto: %tiny_i_inline88__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v254 = Tile<TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v24);
          // pto: %tiny_i_inline88__ssa_v0
          ;
          uint64_t v255 = (uint64_t) v12;
          TASSIGN(v254, v255);
          // pto: %t__tmp_v9
          ;
          Tile<TileType::Vec, uint32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v256 = Tile<TileType::Vec, uint32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v24);
          // pto: %t__tmp_v9
          ;
          uint64_t v257 = (uint64_t) v12;
          TASSIGN(v256, v257);
          // pto: %tiny_pairs_inline89__ssa_v0
          ;
          Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v258 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v18);
          // pto: %tiny_pairs_inline89__ssa_v0
          ;
          uint64_t v259 = (uint64_t) v15;
          TASSIGN(v258, v259);
          // pto: %40
          ;
          Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v260;
          TRESHAPE(v260, v252);
          // pto: %41
          ;
          Tile<TileType::Vec, uint32_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v261;
          TRESHAPE(v261, v256);
          // pto: %42
          ;
          Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v262;
          TRESHAPE(v262, v258);
          pipe_barrier(PIPE_V);
          TSORT32(v262, v260, v261);
          // pto: %tiny_pairs_v1_inline90__ssa_v0
          ;
          Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v263 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v18);
          // pto: %tiny_pairs_v1_inline90__ssa_v0
          ;
          uint64_t v264 = (uint64_t) v13;
          TASSIGN(v263, v264);
          pipe_barrier(PIPE_V);
          TMRGSORT(v263, v258, v26);
          // pto: %tiny_pairs_v2_inline94__ssa_v0
          ;
          Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v265 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v18);
          // pto: %tiny_pairs_v2_inline94__ssa_v0
          ;
          uint64_t v266 = (uint64_t) v15;
          TASSIGN(v265, v266);
          pipe_barrier(PIPE_V);
          TMRGSORT(v265, v263, v27);
          // pto: %root_inline84__ssa_v0
          ;
          Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v267 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
          // pto: %root_inline84__ssa_v0
          ;
          uint64_t v268 = (uint64_t) v13;
          TASSIGN(v267, v268);
          pipe_barrier(PIPE_V);
          uint16_t v269 = static_cast<uint16_t>(v20);
          uint16_t v270 = static_cast<uint16_t>(v20);
          TEXTRACT(v267, v265, v269, v270);
        } else {
          // pto: %22
          ;
          int64_t v271 = (int64_t) v221;
          v73.SetValidShape(v17, v271);
          // pto: %t__tmp_v10
          ;
          Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v272 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v18);
          // pto: %t__tmp_v10
          ;
          uint64_t v273 = (uint64_t) v10;
          TASSIGN(v272, v273);
          pipe_barrier(PIPE_V);
          TFILLPAD<pto::TFillPadMode::InPlace>(v272, v73);
          // pto: %small_v_v1_inline86__ssa_v0
          ;
          Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v274 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v18);
          // pto: %small_v_v1_inline86__ssa_v0
          ;
          uint64_t v275 = (uint64_t) v10;
          TASSIGN(v274, v275);
          pipe_barrier(PIPE_V);
          TMAXS(v274, v272, v25);
          // pto: %t__tmp_v11
          ;
          Tile<TileType::Vec, uint32_t, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v276 = Tile<TileType::Vec, uint32_t, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
          // pto: %t__tmp_v11
          ;
          uint64_t v277 = (uint64_t) v12;
          TASSIGN(v276, v277);
          // pto: %small_pairs_inline98__ssa_v0
          ;
          Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v278 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v16);
          // pto: %small_pairs_inline98__ssa_v0
          ;
          uint64_t v279 = (uint64_t) v11;
          TASSIGN(v278, v279);
          // pto: %44
          ;
          Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v280;
          TRESHAPE(v280, v274);
          // pto: %45
          ;
          Tile<TileType::Vec, uint32_t, 1, 1024, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v281;
          TRESHAPE(v281, v276);
          // pto: %46
          ;
          Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, 2048, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v282;
          TRESHAPE(v282, v278);
          pipe_barrier(PIPE_V);
          TSORT32(v282, v280, v281);
          // pto: %small_pairs_v1_inline91__ssa_v0
          ;
          Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v283 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v16);
          // pto: %small_pairs_v1_inline91__ssa_v0
          ;
          uint64_t v284 = (uint64_t) v10;
          TASSIGN(v283, v284);
          pipe_barrier(PIPE_V);
          TMRGSORT(v283, v278, v26);
          // pto: %small_pairs_v2_inline96__ssa_v0
          ;
          Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v285 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v16);
          // pto: %small_pairs_v2_inline96__ssa_v0
          ;
          uint64_t v286 = (uint64_t) v11;
          TASSIGN(v285, v286);
          pipe_barrier(PIPE_V);
          TMRGSORT(v285, v283, v27);
          // pto: %left_inline87__ssa_v0
          ;
          Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v287 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v18);
          // pto: %left_inline87__ssa_v0
          ;
          uint64_t v288 = (uint64_t) v11;
          TASSIGN(v287, v288);
          // pto: %47
          ;
          Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v289;
          // pto: %47
          ;
          uint64_t v290 = (uint64_t) v11;
          TASSIGN(v289, v290);
          // pto: %right_inline95__ssa_v0
          ;
          Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v291 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v18);
          // pto: %right_inline95__ssa_v0
          ;
          uint64_t v292 = (uint64_t) v14;
          TASSIGN(v291, v292);
          // pto: %48
          ;
          Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v293;
          // pto: %48
          ;
          uint64_t v294 = (uint64_t) v14;
          TASSIGN(v293, v294);
          // pto: %tmp_inline82__ssa_v0
          ;
          Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v295 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v16);
          // pto: %tmp_inline82__ssa_v0
          ;
          uint64_t v296 = (uint64_t) v10;
          TASSIGN(v295, v296);
          // pto: %merged_inline85__ssa_v0
          ;
          Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v297 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v16);
          // pto: %merged_inline85__ssa_v0
          ;
          uint64_t v298 = (uint64_t) v12;
          TASSIGN(v297, v298);
          pipe_barrier(PIPE_V);
          TMRGSORT<Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>, Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>, Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>, false>(v297, v7, v295, v289, v293);
          // pto: %root_inline84__ssa_v1
          ;
          Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v299 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
          // pto: %root_inline84__ssa_v1
          ;
          uint64_t v300 = (uint64_t) v11;
          TASSIGN(v299, v300);
          pipe_barrier(PIPE_V);
          uint16_t v301 = static_cast<uint16_t>(v20);
          uint16_t v302 = static_cast<uint16_t>(v20);
          TEXTRACT(v299, v297, v301, v302);
          // pto: %root_inline84__ssa_v1_mv
          ;
          Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v303 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
          // pto: %root_inline84__ssa_v1_mv
          ;
          uint64_t v304 = (uint64_t) v13;
          TASSIGN(v303, v304);
          pipe_barrier(PIPE_V);
          TMOV(v303, v299);
        };
        // pto: %new_root_inline79__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v305 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
        // pto: %new_root_inline79__ssa_v0
        ;
        uint64_t v306 = (uint64_t) v13;
        TASSIGN(v305, v306);
      } else {
        // pto: %t__ci_tmp_v1
        ;
        Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v307 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v21);
        // pto: %t__ci_tmp_v1
        ;
        uint64_t v308 = (uint64_t) v10;
        TASSIGN(v307, v308);
        // pto: %t__tmp_v12
        ;
        Tile<TileType::Vec, int32_t, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v309 = Tile<TileType::Vec, int32_t, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v16);
        // pto: %t__tmp_v12
        ;
        uint64_t v310 = (uint64_t) v11;
        TASSIGN(v309, v310);
        // pto: %50
        ;
        Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v311;
        TRESHAPE(v311, v307);
        // pto: %51
        ;
        Tile<TileType::Vec, int32_t, 1, 2048, BLayout::RowMajor, 1, 2048, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v312;
        TRESHAPE(v312, v309);
        pipe_barrier(PIPE_V);
        TCI<Tile<TileType::Vec, int32_t, 1, 2048, BLayout::RowMajor, 1, 2048, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v312, v22, v311);
        // pto: %original_indices_inline70__ssa_v0
        ;
        Tile<TileType::Vec, int32_t, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v313 = Tile<TileType::Vec, int32_t, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v16);
        // pto: %original_indices_inline70__ssa_v0
        ;
        uint64_t v314 = (uint64_t) v10;
        TASSIGN(v313, v314);
        pipe_barrier(PIPE_V);
        TADDS(v313, v309, v23);
        if (v115) {
          // pto: %tiny_v_inline110__ssa_v0
          ;
          Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v315 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v24);
          // pto: %tiny_v_inline110__ssa_v0
          ;
          uint64_t v316 = (uint64_t) v11;
          TASSIGN(v315, v316);
          pipe_barrier(PIPE_V);
          uint16_t v317 = static_cast<uint16_t>(v20);
          uint16_t v318 = static_cast<uint16_t>(v20);
          TEXTRACT(v315, v88, v317, v318);
          v315.SetValidShape(v17, v114);
          // pto: %t__tmp_v13
          ;
          Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v319 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v24);
          // pto: %t__tmp_v13
          ;
          uint64_t v320 = (uint64_t) v11;
          TASSIGN(v319, v320);
          pipe_barrier(PIPE_V);
          TFILLPAD<pto::TFillPadMode::InPlace>(v319, v315);
          // pto: %tiny_v_v2_inline116__ssa_v0
          ;
          Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v321 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v24);
          // pto: %tiny_v_v2_inline116__ssa_v0
          ;
          uint64_t v322 = (uint64_t) v12;
          TASSIGN(v321, v322);
          pipe_barrier(PIPE_V);
          TMAXS(v321, v319, v25);
          // pto: %tiny_i_inline113__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v323 = Tile<TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v24);
          // pto: %tiny_i_inline113__ssa_v0
          ;
          uint64_t v324 = (uint64_t) v10;
          TASSIGN(v323, v324);
          // pto: %t__tmp_v14
          ;
          Tile<TileType::Vec, uint32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v325 = Tile<TileType::Vec, uint32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v24);
          // pto: %t__tmp_v14
          ;
          uint64_t v326 = (uint64_t) v10;
          TASSIGN(v325, v326);
          // pto: %tiny_pairs_inline117__ssa_v0
          ;
          Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v327 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v18);
          // pto: %tiny_pairs_inline117__ssa_v0
          ;
          uint64_t v328 = (uint64_t) v11;
          TASSIGN(v327, v328);
          // pto: %57
          ;
          Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v329;
          TRESHAPE(v329, v321);
          // pto: %58
          ;
          Tile<TileType::Vec, uint32_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v330;
          TRESHAPE(v330, v325);
          // pto: %59
          ;
          Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v331;
          TRESHAPE(v331, v327);
          pipe_barrier(PIPE_V);
          TSORT32(v331, v329, v330);
          // pto: %tiny_pairs_v1_inline108__ssa_v0
          ;
          Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v332 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v18);
          // pto: %tiny_pairs_v1_inline108__ssa_v0
          ;
          uint64_t v333 = (uint64_t) v12;
          TASSIGN(v332, v333);
          pipe_barrier(PIPE_V);
          TMRGSORT(v332, v327, v26);
          // pto: %tiny_pairs_v2_inline118__ssa_v0
          ;
          Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v334 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v18);
          // pto: %tiny_pairs_v2_inline118__ssa_v0
          ;
          uint64_t v335 = (uint64_t) v11;
          TASSIGN(v334, v335);
          pipe_barrier(PIPE_V);
          TMRGSORT(v334, v332, v27);
          // pto: %root_inline112__ssa_v0
          ;
          Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v336 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
          // pto: %root_inline112__ssa_v0
          ;
          uint64_t v337 = (uint64_t) v13;
          TASSIGN(v336, v337);
          pipe_barrier(PIPE_V);
          uint16_t v338 = static_cast<uint16_t>(v20);
          uint16_t v339 = static_cast<uint16_t>(v20);
          TEXTRACT(v336, v334, v338, v339);
        } else {
          // pto: %61
          ;
          if (v114 <= v18) {
            // pto: %small_v_inline120__ssa_v0
            ;
            Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v340 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
            // pto: %small_v_inline120__ssa_v0
            ;
            uint64_t v341 = (uint64_t) v11;
            TASSIGN(v340, v341);
            pipe_barrier(PIPE_V);
            uint16_t v342 = static_cast<uint16_t>(v20);
            uint16_t v343 = static_cast<uint16_t>(v20);
            TEXTRACT(v340, v88, v342, v343);
            v340.SetValidShape(v17, v114);
            // pto: %t__tmp_v15
            ;
            Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v344 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v18);
            // pto: %t__tmp_v15
            ;
            uint64_t v345 = (uint64_t) v11;
            TASSIGN(v344, v345);
            pipe_barrier(PIPE_V);
            TFILLPAD<pto::TFillPadMode::InPlace>(v344, v340);
            // pto: %small_v_v2_inline122__ssa_v0
            ;
            Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v346 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v18);
            // pto: %small_v_v2_inline122__ssa_v0
            ;
            uint64_t v347 = (uint64_t) v12;
            TASSIGN(v346, v347);
            pipe_barrier(PIPE_V);
            TMAXS(v346, v344, v25);
            // pto: %small_i_inline119__ssa_v0
            ;
            Tile<TileType::Vec, int32_t, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v348 = Tile<TileType::Vec, int32_t, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
            // pto: %small_i_inline119__ssa_v0
            ;
            uint64_t v349 = (uint64_t) v10;
            TASSIGN(v348, v349);
            // pto: %t__tmp_v16
            ;
            Tile<TileType::Vec, uint32_t, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v350 = Tile<TileType::Vec, uint32_t, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
            // pto: %t__tmp_v16
            ;
            uint64_t v351 = (uint64_t) v10;
            TASSIGN(v350, v351);
            // pto: %small_pairs_inline107__ssa_v0
            ;
            Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v352 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v16);
            // pto: %small_pairs_inline107__ssa_v0
            ;
            uint64_t v353 = (uint64_t) v11;
            TASSIGN(v352, v353);
            // pto: %63
            ;
            Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v354;
            TRESHAPE(v354, v346);
            // pto: %64
            ;
            Tile<TileType::Vec, uint32_t, 1, 1024, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v355;
            TRESHAPE(v355, v350);
            // pto: %65
            ;
            Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, 2048, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v356;
            TRESHAPE(v356, v352);
            pipe_barrier(PIPE_V);
            TSORT32(v356, v354, v355);
            // pto: %small_pairs_v1_inline105__ssa_v0
            ;
            Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v357 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v16);
            // pto: %small_pairs_v1_inline105__ssa_v0
            ;
            uint64_t v358 = (uint64_t) v12;
            TASSIGN(v357, v358);
            pipe_barrier(PIPE_V);
            TMRGSORT(v357, v352, v26);
            // pto: %small_pairs_v2_inline104__ssa_v0
            ;
            Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v359 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v16);
            // pto: %small_pairs_v2_inline104__ssa_v0
            ;
            uint64_t v360 = (uint64_t) v11;
            TASSIGN(v359, v360);
            pipe_barrier(PIPE_V);
            TMRGSORT(v359, v357, v27);
            // pto: %small_left_inline121__ssa_v0
            ;
            Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v361 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v18);
            // pto: %small_left_inline121__ssa_v0
            ;
            uint64_t v362 = (uint64_t) v11;
            TASSIGN(v361, v362);
            // pto: %66
            ;
            Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v363;
            // pto: %66
            ;
            uint64_t v364 = (uint64_t) v11;
            TASSIGN(v363, v364);
            // pto: %small_right_inline102__ssa_v0
            ;
            Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v365 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v18);
            // pto: %small_right_inline102__ssa_v0
            ;
            uint64_t v366 = (uint64_t) v14;
            TASSIGN(v365, v366);
            // pto: %67
            ;
            Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v367;
            // pto: %67
            ;
            uint64_t v368 = (uint64_t) v14;
            TASSIGN(v367, v368);
            // pto: %small_tmp_inline100__ssa_v0
            ;
            Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v369 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v16);
            // pto: %small_tmp_inline100__ssa_v0
            ;
            uint64_t v370 = (uint64_t) v12;
            TASSIGN(v369, v370);
            // pto: %small_merged_inline106__ssa_v0
            ;
            Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v371 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v16);
            // pto: %small_merged_inline106__ssa_v0
            ;
            uint64_t v372 = (uint64_t) v15;
            TASSIGN(v371, v372);
            pipe_barrier(PIPE_V);
            TMRGSORT<Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>, Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>, Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>, false>(v371, v7, v369, v363, v367);
            // pto: %root_inline112__ssa_v1
            ;
            Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v373 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
            // pto: %root_inline112__ssa_v1
            ;
            uint64_t v374 = (uint64_t) v12;
            TASSIGN(v373, v374);
            pipe_barrier(PIPE_V);
            uint16_t v375 = static_cast<uint16_t>(v20);
            uint16_t v376 = static_cast<uint16_t>(v20);
            TEXTRACT(v373, v371, v375, v376);
          } else {
            v88.SetValidShape(v17, v114);
            // pto: %t__tmp_v17
            ;
            Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v377 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v16);
            // pto: %t__tmp_v17
            ;
            uint64_t v378 = (uint64_t) v11;
            TASSIGN(v377, v378);
            pipe_barrier(PIPE_V);
            TFILLPAD(v377, v88);
            // pto: %large_v_v1_inline109__ssa_v0
            ;
            Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v379 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v16);
            // pto: %large_v_v1_inline109__ssa_v0
            ;
            uint64_t v380 = (uint64_t) v8;
            TASSIGN(v379, v380);
            pipe_barrier(PIPE_V);
            TMAXS(v379, v377, v25);
            // pto: %t__tmp_v18
            ;
            Tile<TileType::Vec, uint32_t, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v381 = Tile<TileType::Vec, uint32_t, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v16);
            // pto: %t__tmp_v18
            ;
            uint64_t v382 = (uint64_t) v10;
            TASSIGN(v381, v382);
            // pto: %large_pairs_inline103__ssa_v0
            ;
            Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v383 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v28);
            // pto: %large_pairs_inline103__ssa_v0
            ;
            uint64_t v384 = (uint64_t) v11;
            TASSIGN(v383, v384);
            // pto: %70
            ;
            Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, 2048, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v385;
            TRESHAPE(v385, v379);
            // pto: %71
            ;
            Tile<TileType::Vec, uint32_t, 1, 2048, BLayout::RowMajor, 1, 2048, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v386;
            TRESHAPE(v386, v381);
            // pto: %72
            ;
            Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, 1, 4096, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v387;
            TRESHAPE(v387, v383);
            pipe_barrier(PIPE_V);
            TSORT32(v387, v385, v386);
            // pto: %large_pairs_v1_inline123__ssa_v0
            ;
            Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v388 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v28);
            // pto: %large_pairs_v1_inline123__ssa_v0
            ;
            uint64_t v389 = (uint64_t) v10;
            TASSIGN(v388, v389);
            pipe_barrier(PIPE_V);
            TMRGSORT(v388, v383, v26);
            // pto: %large_pairs_v2_inline115__ssa_v0
            ;
            Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v390 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v28);
            // pto: %large_pairs_v2_inline115__ssa_v0
            ;
            uint64_t v391 = (uint64_t) v11;
            TASSIGN(v390, v391);
            pipe_barrier(PIPE_V);
            TMRGSORT(v390, v388, v27);
            // pto: %large_pairs_v3_inline99__ssa_v0
            ;
            Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v392 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v17, v28);
            // pto: %large_pairs_v3_inline99__ssa_v0
            ;
            uint64_t v393 = (uint64_t) v10;
            TASSIGN(v392, v393);
            pipe_barrier(PIPE_V);
            TMRGSORT(v392, v390, v29);
            // pto: %root_inline112__ssa_v2
            ;
            Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v394 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
            // pto: %root_inline112__ssa_v2
            ;
            uint64_t v395 = (uint64_t) v11;
            TASSIGN(v394, v395);
            pipe_barrier(PIPE_V);
            uint16_t v396 = static_cast<uint16_t>(v20);
            uint16_t v397 = static_cast<uint16_t>(v20);
            TEXTRACT(v394, v392, v396, v397);
            // pto: %root_inline112__ssa_v2_mv
            ;
            Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v398 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
            // pto: %root_inline112__ssa_v2_mv
            ;
            uint64_t v399 = (uint64_t) v12;
            TASSIGN(v398, v399);
            pipe_barrier(PIPE_V);
            TMOV(v398, v394);
          };
          // pto: %root_inline112__phi_v3_mv
          ;
          Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v400 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
          // pto: %root_inline112__phi_v3_mv
          ;
          uint64_t v401 = (uint64_t) v13;
          TASSIGN(v400, v401);
          pipe_barrier(PIPE_V);
          TMOV(v400, v85);
        };
        // pto: %new_root_inline79__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v402 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
        // pto: %new_root_inline79__ssa_v1
        ;
        uint64_t v403 = (uint64_t) v13;
        TASSIGN(v402, v403);
      };
      // pto: %merge_tmp_inline69__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v404 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v16);
      // pto: %merge_tmp_inline69__ssa_v0
      ;
      uint64_t v405 = (uint64_t) v11;
      TASSIGN(v404, v405);
      // pto: %merged_inline75__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v406 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v16);
      // pto: %merged_inline75__ssa_v0
      ;
      uint64_t v407 = (uint64_t) v10;
      TASSIGN(v406, v407);
      pipe_barrier(PIPE_V);
      TMRGSORT<Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, false>(v406, v7, v404, v79, v97);
      // pto: %next_root_inline71__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v408 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
      // pto: %next_root_inline71__ssa_v1
      ;
      uint64_t v409 = (uint64_t) v11;
      TASSIGN(v408, v409);
      pipe_barrier(PIPE_V);
      uint16_t v410 = static_cast<uint16_t>(v20);
      uint16_t v411 = static_cast<uint16_t>(v20);
      TEXTRACT(v408, v406, v410, v411);
      // pto: %next_root_inline71__ssa_v1_mv
      ;
      Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v412 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
      // pto: %next_root_inline71__ssa_v1_mv
      ;
      uint64_t v413 = (uint64_t) v9;
      TASSIGN(v412, v413);
      pipe_barrier(PIPE_V);
      TMOV(v412, v408);
    };
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // pto: %compact_top__ssa_v0
    ;
    Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v414 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
    // pto: %compact_top__ssa_v0
    ;
    uint64_t v415 = (uint64_t) v9;
    TASSIGN(v414, v415);
    // pto: %candidate__ssa_v0_pview
    ;
    __gm__ float* v416 = PTOAS__GLOBAL_TENSOR_DATA(v68);
    // pto: %candidate__ssa_v0_pview
    ;
    const int64_t v417 = 8;
    // pto: %candidate__ssa_v0_pview
    ;
    const int64_t v418 = 1040;
    // pto: %candidate__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, 1024> v419 = pto::Shape<1, 1, 1, 1, 1024>();
    // pto: %candidate__ssa_v0_pview
    ;
    pto::Stride<1040, 1040, 1040, 1040, 1> v420 = pto::Stride<1040, 1040, 1040, 1040, 1>();
    // pto: %candidate__ssa_v0_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1040, 1040, 1040, 1040, 1>, pto::Layout::ND> v421 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1040, 1040, 1040, 1040, 1>, pto::Layout::ND>(v416 + (v417 + v90 * v418), v419, v420);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(v421, v414);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}