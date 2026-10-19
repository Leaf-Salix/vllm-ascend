#include <cstdint>
#define __gm__
#define PIPE_ALL 0
enum class cache_line_t {ENTIRE_DATA_CACHE, SINGLE_CACHE_LINE};
using mem_dsb_t = int;
inline void pipe_barrier(int) {}
inline void dcci(void*, cache_line_t) {}
inline void dsb(mem_dsb_t) {}
extern "C" void bool_cast(__gm__ int32_t* v1, bool v2) {
  using T = float;
  // pto: %c0_index
  const int64_t v3 = 0;
  // pto: %0
  const int32_t v4 = 0;
  // pto: %0
  int32_t v5 = v4 - (int32_t) v2;
  (v1)[v3] = v5;
  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
  return;
}