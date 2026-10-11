#include <cstdint>
extern "C" __aicore__ void probe(__gm__ uint16_t *out, __gm__ uint16_t *token) {
    auto *scratch = reinterpret_cast<__cbuf__ uint16_t*>(524224);
    copy_gm_to_cbuf(scratch, token, 0, 1, 2, 0, 0, (pad_t)0);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID7);
    set_flag(PIPE_M, PIPE_MTE3, EVENT_ID7);
    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_M, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID7);
    pipe_barrier(PIPE_MTE3);
    copy_cbuf_to_gm(out, scratch, 0, 1, 2, 0, 0);
    pipe_barrier(PIPE_ALL);
}
