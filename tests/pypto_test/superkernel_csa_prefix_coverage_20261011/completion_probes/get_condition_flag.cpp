#include <cstdint>
extern "C" __aicore__ void probe(__gm__ int64_t *out) { out[0] = get_condition_flag(); }
