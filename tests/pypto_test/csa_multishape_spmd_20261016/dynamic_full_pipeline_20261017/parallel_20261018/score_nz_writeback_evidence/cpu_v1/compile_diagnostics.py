# SPDX-License-Identifier: Apache-2.0
"""完整动态CSA的编译期诊断：等待退避A/B及核内阶段时间戳。

只作用于dynamic_csa_full，不改原动态PTO。时戳占每核每阶段512B槽的第2条
64B行（16..19项），第一条信号行及其他核的槽保持不变。读写与DCCI开销使
插桩时间只能用于归因，不能作为性能收益证据。可选退避复用正式目录已有实现。
"""

import contextlib
import json
import re
from pathlib import Path


CLOCK_HELPERS = r"""
template <typename Signal>
static AICORE inline void CSA_PROFILE_WAIT(Signal& signal, int32_t expected,
    pto::comm::WaitCmp cmp, uint64_t& total_ticks, uint32_t& count) {
  const uint64_t start = get_sys_cnt();
  CSA_UNDERLYING_WAIT(signal, expected, cmp);
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
  // 本核独占该phase槽；写第二条cache行，不读改写信号所在行。
  ptr[16] = static_cast<uint32_t>(start);
  ptr[17] = static_cast<uint32_t>(end);
  ptr[18] = static_cast<uint32_t>(total_ticks);
  ptr[19] = count;
  dcci((__gm__ void*)(ptr + 16), cache_line_t::SINGLE_CACHE_LINE);
  dsb(DSB_DDR);
}
"""


def instrument_clock(source, *, wait_backoff):
    wait_name = "CSA_WAIT_LOCAL_GM" if wait_backoff else "pto::comm::TWAIT"
    audit = {}
    for symbol, replacement in ((wait_name, "CSA_PROFILE_WAIT"), ("pto::comm::TNOTIFY", "CSA_PROFILE_NOTIFY")):
        pattern = re.compile(r"^(\s*)" + re.escape(symbol) + r"\(([^\n]+)\);$", re.M)
        source, count = pattern.subn(lambda m: f"{m[1]}{replacement}({m[2]}, csa_wait_ticks, csa_wait_count);", source)
        if count <= 0:
            raise ValueError(f"没有可插桩调用：{symbol}")
        audit[symbol] = count
    for guard in ("  #if defined(__DAV_CUBE__)\n", "  #if defined(__DAV_VEC__)\n"):
        if source.count(guard) != 1:
            raise ValueError("无法唯一定位AIC/AIV函数内核区")
        source = source.replace(guard, guard + "  uint64_t csa_wait_ticks = 0;\n  uint32_t csa_wait_count = 0;\n")
    helper = CLOCK_HELPERS.replace("CSA_UNDERLYING_WAIT", wait_name)
    # 退避helper需先定义，故插在两个实际核函数之前。
    marker = "AICORE void cube_schedule("
    if source.count(marker) != 1:
        raise ValueError("完整CSA AIC定义不唯一")
    source = source.replace(marker, helper + "\n" + marker)
    return source, audit


@contextlib.contextmanager
def compilation_probe(directory, *, wait_backoff=False, phase_clock=False):
    from pypto.backend import pto_backend
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.wait_codegen import rewrite_wait_source

    original = pto_backend._compile_pto_module
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)

    def hooked(pto_code, unit_name, output_dir, *args, **kwargs):
        source = original(pto_code, unit_name, output_dir, *args, **kwargs)
        if unit_name != "dynamic_csa_full":
            return source
        audit = {"unit": unit_name, "wait_backoff": wait_backoff, "phase_clock": phase_clock}
        (directory / "original.cpp").write_text(source)
        from nz_store_codegen import rewrite as rewrite_nz_store

        source, audit["nz_store_experiment"] = rewrite_nz_store(source)
        if wait_backoff:
            source, audit["wait_change"] = rewrite_wait_source(source)
        if phase_clock:
            source, audit["clock_change"] = instrument_clock(source, wait_backoff=wait_backoff)
        (directory / "modified.cpp").write_text(source)
        (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
        return source

    pto_backend._compile_pto_module = hooked
    try:
        yield
    finally:
        pto_backend._compile_pto_module = original


def export_clock(signal, destination, phases):
    """用已发布阶段端点及累计TWAIT时间区分阶段间等待和其他工作。"""
    rows = []
    slots = signal.tolist()
    for phase, cores in enumerate(slots):
        for core, values in enumerate(cores):
            if values[0]:
                rows.append(
                    {
                        "phase": phase,
                        "name": phases.get(phase, "BEGIN"),
                        "core": core,
                        "ticks": [v & 0xFFFFFFFF for v in values[16:20]],
                    }
                )
    if not rows or any(r["ticks"][0] == 0 for r in rows):
        raise ValueError("已发布阶段缺少时戳，不能使用未插桩产物")
    anchor = rows[0]["ticks"][0]
    events = []
    for core in range(72):
        steps = [r for r in rows if r["core"] == core]
        for row in steps:
            row["notify_start_us"] = ((row["ticks"][0] - anchor + 2**31) % 2**32 - 2**31) / 50
            row["notify_end_us"] = ((row["ticks"][1] - anchor + 2**31) % 2**32 - 2**31) / 50
            row["cumulative_wait_us"] = row["ticks"][2] / 50
            row["cumulative_wait_calls"] = row["ticks"][3]
        steps.sort(key=lambda row: row["notify_start_us"])
        for before, row in zip(steps, steps[1:]):
            start, end = before["notify_end_us"], row["notify_start_us"]
            wait = row["cumulative_wait_us"] - before["cumulative_wait_us"]
            row.update(interval_us=end - start, wait_us=wait, other_us=end - start - wait)
            events.append(
                {
                    "name": row["name"],
                    "ph": "X",
                    "pid": 0,
                    "tid": core,
                    "ts": start,
                    "dur": end - start,
                    "args": {"wait_us": wait, "other_us": end - start - wait, "includes_probe_cost": True},
                }
            )
    destination.mkdir(exist_ok=True)
    (destination / "phase_clock.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    (destination / "phase_clock_trace.json").write_text(json.dumps({"traceEvents": events}, ensure_ascii=False) + "\n")
    return {
        "clock_freq_hz": 50000000,
        "eligible_for_performance": False,
        "records": len(rows),
        "raw": str(destination / "phase_clock.json"),
        "trace": str(destination / "phase_clock_trace.json"),
    }
