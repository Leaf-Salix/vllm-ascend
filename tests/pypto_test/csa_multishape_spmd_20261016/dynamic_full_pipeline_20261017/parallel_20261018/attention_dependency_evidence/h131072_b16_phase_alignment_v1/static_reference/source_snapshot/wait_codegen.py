# SPDX-License-Identifier: Apache-2.0
"""已选 GM 信号等待策略的纯源码变换，不修改进程全局 PyPTO backend。

编译集成应在 PTOAS 返回源码后、CCEC 编译前调用，并核对返回的替换数量。
当前 PyPTO 尚无公开的 per-kernel 源码回调；临时安装编译回调仍由隔离验证
进程负责。仅搬迁本模块不表示生产 JIT 已自动启用此策略。
"""

WAIT_BACKOFF_TICKS = 100

WAIT_HELPER = r"""
template <typename Signal>
static AICORE inline void CSA_WAIT_LOCAL_GM(Signal& signal, int32_t expected, pto::comm::WaitCmp cmp) {
  // 单写者单调进度。写侧仍由 TNOTIFY 清理/失效并 DSB，payload 由调用方维护。
  PTO_ASSERT(cmp == pto::comm::WaitCmp::GE, "CSA local wait only supports GE");
  auto ptr = reinterpret_cast<volatile __gm__ int32_t*>(signal.data());
  uint32_t spin = 0;
  while (true) {
    __asm__ __volatile__("" ::: "memory");
    dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);
    __asm__ __volatile__("" ::: "memory");
    if (*ptr >= expected) break;
    // 首次读取不延迟；只有尚未就绪时，短暂只读本核计时寄存器。
    const uint64_t csa_wait_start = get_sys_cnt();
    while ((uint64_t)(get_sys_cnt() - csa_wait_start) < 100) {
      __asm__ __volatile__("" ::: "memory");
    }
    PTO_ASSERT(spin < 100000000, "CSA local wait spin limit");
    if ((++spin & 63) == 0) {
      pipe_barrier(PIPE_ALL);
    }
  }
}
"""


def rewrite_wait_source(source):
    """只替换本次完整 MIX SPMD 的等待；保留发布和业务数据维护。"""
    needle = "pto::comm::TWAIT("
    marker = "using namespace pto;"
    count = source.count(needle)
    if count <= 0 or source.count(marker) != 1 or "CSA_WAIT_LOCAL_GM(" in source:
        raise ValueError("融合源码的等待/命名空间合同改变或已经执行过替换")
    source = source.replace(needle, "CSA_WAIT_LOCAL_GM(")
    source = source.replace(marker, marker + "\n" + WAIT_HELPER)
    return source, {
        "wait_sites": count,
        "reader": "dcci_then_volatile_load",
        "backoff_syscnt_ticks": WAIT_BACKOFF_TICKS,
        "ramp_backoff": False,
        "initial_backoff_syscnt_ticks": WAIT_BACKOFF_TICKS,
        "writer_unchanged": True,
        "payload_maintenance_unchanged": True,
        "device_validated": False,
    }
