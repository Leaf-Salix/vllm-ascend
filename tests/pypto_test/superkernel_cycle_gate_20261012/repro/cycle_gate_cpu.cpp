// Copyright (c) PyPTO Contributors.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include <cassert>
#include "scheduler/scheduler_types.h"

static uint64_t reads = 0;
uint64_t get_sys_cnt_aicpu() { return ++reads * 10; }

int main() {
    uint64_t accumulated = 17;
    {
        CYCLE_COUNT_START(false);
        CYCLE_COUNT_LAP(accumulated);
        CYCLE_COUNT_LAP(accumulated);
        assert(reads == 0 && accumulated == 17 && _t0 == 0 && _t1 == 0);
    }
    {
        CYCLE_COUNT_START(true);
        assert(reads == 1 && _t0 == 10);
        CYCLE_COUNT_LAP(accumulated);
        CYCLE_COUNT_LAP(accumulated);
        assert(reads == 3 && accumulated == 37 && _t0 == 30 && _t1 == 30);
    }
}
