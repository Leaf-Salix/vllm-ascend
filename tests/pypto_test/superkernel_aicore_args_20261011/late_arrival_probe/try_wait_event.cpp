#include <cstdint>
extern "C" __aicore__ void probe(__gm__ int64_t *out) {
    out[0] = try_wait(PIPE_V, PIPE_S, EVENT_ID7);
}
