// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_GATHER_COMPARE_SCALAR_SUPPORT_H
#define PTO_GATHER_COMPARE_SCALAR_SUPPORT_H

// A3 的旧 ISA compare 重载接收阈值 Tile，不能承接 IR 的真实浮点 Scalar。
// 此辅助实现保持 Scalar 类型，使用硬件压缩；所有临时字节都位于调用方 tmp。
// 只在 A3 的 compare-form 实际出现时发出，不改变索引或固定 pattern gather。
inline constexpr const char *kGatherCompareScalarSupport = R"cpp(
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
)cpp";

#endif
