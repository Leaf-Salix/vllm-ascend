// Copyright (c) PyPTO Contributors.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include <cassert>
#include <cstdio>
#include "aicpu/device_time.h"

int main() {
    const auto freq = device_time_frequency_hz();
    assert(freq > 0);
    for (int i = 0; i < 8; ++i) {
        const uint64_t before = device_time_now_ticks();
        const uint64_t scaled = sys_cnt_now_ticks();
        const uint64_t after = device_time_now_ticks();
        const uint64_t low = static_cast<__uint128_t>(before) * PLATFORM_PROF_SYS_CNT_FREQ / freq;
        const uint64_t high = static_cast<__uint128_t>(after) * PLATFORM_PROF_SYS_CNT_FREQ / freq;
        assert(low <= scaled && scaled <= high);
    }
    std::printf("host frequency=%llu, profile frequency=%llu, branch=%s: PASS\n",
        static_cast<unsigned long long>(freq),
        static_cast<unsigned long long>(PLATFORM_PROF_SYS_CNT_FREQ),
        freq == PLATFORM_PROF_SYS_CNT_FREQ ? "identity" : "rescale");
}
