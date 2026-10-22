# SPDX-License-Identifier: Apache-2.0
"""Diagnostic phase endpoints for the shared SDK's native150 wait lowering.

Uses reserved signal words 16..19 of each role's private 512-byte phase slot.
It does not alter native wait, notify, FIFO or mathematical work. The extra
Scalar stores/cache maintenance perturb timing: never eligible for L1 gains.
"""

import re

CLOCK_HELPERS = r"""
template <typename Signal>
static AICORE inline void CSA_PROFILE_WAIT(Signal& signal, int32_t expected,
    uint64_t& total_ticks, uint32_t& count) {
  const uint64_t start = get_sys_cnt();
  PTOAS__WAIT_GE_BACKOFF<150>(signal, expected);
  total_ticks += get_sys_cnt() - start;
  count += 1;
}
template <typename Signal>
static AICORE inline void CSA_PROFILE_NOTIFY(Signal& signal, int32_t value,
    pto::comm::NotifyOp op, uint64_t total_ticks, uint32_t count) {
  const uint64_t start = get_sys_cnt();
  pto::comm::TNOTIFY(signal, value, op);
  const uint64_t end = get_sys_cnt();
  auto ptr = reinterpret_cast<volatile __gm__ uint32_t*>(signal.data());
  ptr[16] = static_cast<uint32_t>(start);
  ptr[17] = static_cast<uint32_t>(end);
  ptr[18] = static_cast<uint32_t>(total_ticks);
  ptr[19] = count;
  dcci((__gm__ void*)(ptr + 16), cache_line_t::SINGLE_CACHE_LINE);
  dsb(DSB_DDR);
}
"""


def instrument(source):
    if set(re.findall(r"PTOAS__WAIT_GE_BACKOFF<(\d+)>", source)) != {"150"}:
        raise ValueError("Phase probe requires native150 lowering")
    if "CSA_PROFILE_WAIT" in source or "pto::comm::TWAIT(" in source:
        raise ValueError("Refuse double instrumentation or legacy waits")
    audit = {"eligible_for_performance": False, "reserved_signal_words": [16, 17, 18, 19]}
    for original, replacement in (
        ("PTOAS__WAIT_GE_BACKOFF<150>", "CSA_PROFILE_WAIT"),
        ("pto::comm::TNOTIFY", "CSA_PROFILE_NOTIFY"),
    ):
        pattern = re.compile(r"^(\s*)" + re.escape(original) + r"\(([^\n]+)\);$", re.M)
        source, count = pattern.subn(
            lambda m, replacement=replacement: f"{m[1]}{replacement}({m[2]}, csa_wait_ticks, csa_wait_count);", source
        )
        if not count:
            raise ValueError(f"No instrumentation site: {original}")
        audit[original] = count
    for guard in ("  #if defined(__DAV_CUBE__)\n", "  #if defined(__DAV_VEC__)\n"):
        if source.count(guard) != 1:
            raise ValueError("Expected exactly one AIC and one AIV body")
        source = source.replace(guard, guard + "  uint64_t csa_wait_ticks = 0;\n  uint32_t csa_wait_count = 0;\n")
    marker = "AICORE void cube_schedule("
    if source.count(marker) != 1:
        raise ValueError("Expected the original single MIX module")
    return source.replace(marker, CLOCK_HELPERS + "\n" + marker), audit


def export_entries(candidate, baseline, output):
    import importlib

    from compile_diagnostics import export_clock

    abi = importlib.import_module(type(candidate).__module__.rsplit(".", 1)[0] + ".runtime_abi")
    phases = {
        value: key.removeprefix("PHASE_")
        for key, value in vars(abi).items()
        if key.startswith("PHASE_") and key != "PHASE_COUNT"
    }
    for name in ("OA_GROUP", "O_QUANT_GROUP"):
        base = getattr(abi, "PHASE_" + name + "_BASE")
        phases.update({base + i: f"{name}_{i}" for i in range(8)})
    data = {"eligible_for_performance": False, "scope": "Phase endpoints and accumulated GM waits; includes probe cost"}
    for name, entry in (("candidate", candidate), ("reference_v122", baseline)):
        signal = entry.buffers["signal"].cpu().view(abi.PHASE_COUNT, 72, 128)
        data[name] = export_clock(signal, output / f"phase_clock_{name}", phases)
    return data
