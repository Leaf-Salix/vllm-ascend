#include <cstdint>
#define __gm__
#define __DAV_VEC__ 1
#define PIPE_ALL 0
#define DSB_DDR 0
enum class cache_line_t {ENTIRE_DATA_CACHE, SINGLE_CACHE_LINE};
using mem_dsb_t = int;
inline void set_mask_norm() {}
inline void set_vector_mask(int64_t, int64_t) {}
inline void pipe_barrier(int) {}
inline void dcci(void*, cache_line_t) {}
inline void dsb(int) {}
extern "C" void kernel_entry(__gm__ int32_t* v1, __gm__ int32_t* v2) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int32_t v3 = -1;
  // pto: %zero
  const int64_t v4 = 0;
  // pto: %one
  const int64_t v5 = 1;
  // pto: %zero_i32
  const int32_t v6 = 0;
  // pto: %input
  int32_t v7 = (v1)[v4];
  // pto: %condition, %b, %runtime_value
  int32_t v8 = (int32_t) ((int64_t) v7 > v6);
  (v2)[v4] = v3;
  (v2)[v5] = v8;
  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
  #endif // __DAV_VEC__

  return;
}