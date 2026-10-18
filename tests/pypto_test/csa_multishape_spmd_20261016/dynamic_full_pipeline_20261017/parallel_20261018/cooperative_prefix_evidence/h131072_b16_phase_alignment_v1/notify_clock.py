# SPDX-License-Identifier: Apache-2.0
"""静态/动态完整CSA的同口径发布端点诊断；不作为性能验收。

两侧都只包裹TNOTIFY，等待、FFTS、业务fence和数学保持原样。每个端点写四个
INT32（开始、结束、两个零保留字段），并发布独占cache行。动态使用原每phase
512B槽的16..19字；静态使用每核512B槽的16+4*dispatch_seq处。静态B16最大
seq为22，最末字107，仍在原隔离区内。word0和第一条cache行不改；只豁免实际
会写的时钟字，其余padding、工作区边界、records/waits继续逐项检查。

本模块不计算或伪造等待耗时；端点区间包含原计算/等待/搬运/缓存维护及插桩。
"""

import contextlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

from static_baseline_adapter import PreparedStaticBaseline, expected_progress


def instrument_notify(source, *, static=False, max_sequence=22):
    pattern = re.compile(r"^(\s*)pto::comm::TNOTIFY\(([^\n]+)\);$", re.M)
    rewritten, count = pattern.subn(lambda m: f"{m[1]}CSA_NOTIFY_CLOCK({m[2]});", source)
    if count <= 0:
        raise ValueError("目标核没有TNOTIFY，拒绝产生空诊断")
    offset = "16"
    check = ""
    if static:
        if 16 + 4 * (max_sequence + 1) > 128:
            raise ValueError("静态时钟超过本核隔离区")
        offset = "16 + 4 * (static_cast<uint32_t>(value) % 1024)"
        check = f'PTO_ASSERT((static_cast<uint32_t>(value) % 1024) <= {max_sequence}, "clock seq overflow");'
    helper = r"""
template <typename Signal>
static AICORE inline void CSA_NOTIFY_CLOCK(Signal& signal, int32_t value,
    pto::comm::NotifyOp op) {
  CSA_SEQUENCE_CHECK
  const uint64_t start = get_sys_cnt();
  pto::comm::TNOTIFY(signal, value, op);
  const uint64_t end = get_sys_cnt();
  auto ptr = reinterpret_cast<volatile __gm__ uint32_t*>(signal.data());
  auto clock = ptr + CSA_CLOCK_OFFSET;
  clock[0] = static_cast<uint32_t>(start);
  clock[1] = static_cast<uint32_t>(end);
  clock[2] = 0;
  clock[3] = 0;
  dcci((__gm__ void*)clock, cache_line_t::SINGLE_CACHE_LINE);
  dsb(DSB_DDR);
}
""".replace("CSA_SEQUENCE_CHECK", check).replace("CSA_CLOCK_OFFSET", offset)
    marker = "AICORE void csa_static_aic(" if static else "AICORE void cube_schedule("
    if rewritten.count(marker) != 1:
        raise ValueError("AIC定义不唯一，不能可靠插入时钟helper")
    return rewritten.replace(marker, helper + "\n" + marker), {
        "notify_sites": count,
        "wait_and_ffts_unchanged": True,
        "clock_words": 4,
        "static_max_sequence": max_sequence if static else None,
        "eligible_for_performance": False,
    }


@contextlib.contextmanager
def compilation_probe(directory):
    from pypto.backend import pto_backend
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.wait_codegen import rewrite_wait_source

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    original = pto_backend._compile_pto_module

    def wrapped(pto_code, unit_name, output_dir, *args, **kwargs):
        source = original(pto_code, unit_name, output_dir, *args, **kwargs)
        if unit_name != "dynamic_csa_full":
            return source
        (directory / "original.cpp").write_text(source)
        source, waits = rewrite_wait_source(source)
        source, clock = instrument_notify(source)
        (directory / "modified.cpp").write_text(source)
        (directory / "audit.json").write_text(json.dumps({"waits": waits, "clock": clock}, indent=2) + "\n")
        return source

    pto_backend._compile_pto_module = wrapped
    try:
        yield
    finally:
        pto_backend._compile_pto_module = original


def static_steps(layout):
    steps = {(core, 0): {"name": "BEGIN", "task": None} for core in range(72)}
    max_sequence = 0
    for batches in layout["batches"].values():
        for batch in batches:
            for member in batch["members"]:
                seq = member["dispatch_seq"]
                max_sequence = max(max_sequence, seq)
                if member["publish_completion"]:
                    key = (member["core"], seq)
                    if key in steps:
                        raise ValueError("同核同seq存在两个发布者")
                    steps[key] = {"name": member["kernel_name"], "task": member["task"]}
    return steps, max_sequence


def write_clock(rows, destination):
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    begin = next(row for row in rows if row["name"] == "BEGIN")
    anchor = begin["ticks"][0]
    for row in rows:
        start, end = row["ticks"]
        if not (start or end):
            raise AssertionError("实际发布的端点缺少时戳")
        row["notify_start_us"] = ((start - anchor + 2**31) % 2**32 - 2**31) / 50
        row["notify_end_us"] = ((end - anchor + 2**31) % 2**32 - 2**31) / 50
    origin = min(row["notify_start_us"] for row in rows if row["name"] == "BEGIN")
    for row in rows:
        row["notify_start_us"] -= origin
        row["notify_end_us"] -= origin
    events = []
    for core in range(72):
        ordered = sorted((row for row in rows if row["core"] == core), key=lambda row: row["notify_start_us"])
        for before, row in zip(ordered, ordered[1:]):
            start = before["notify_end_us"]
            events.append({"name": row["name"], "ph": "X", "pid": 0, "tid": core,
                           "ts": start, "dur": row["notify_start_us"] - start,
                           "args": {"includes_compute_wait_transfer_and_probe": True}})
    (destination / "phase_clock.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    (destination / "phase_clock_trace.json").write_text(json.dumps({"traceEvents": events}, ensure_ascii=False) + "\n")
    return {"records": len(rows), "clock_freq_hz": 50000000, "eligible_for_performance": False,
            "raw": str(destination / "phase_clock.json"), "trace": str(destination / "phase_clock_trace.json")}


def export_clock(signal, destination, phases):
    rows = []
    for phase, cores in enumerate(signal.tolist()):
        for core, values in enumerate(cores):
            if values[0]:
                rows.append({"phase": phase, "name": phases.get(phase, "BEGIN"), "core": core,
                             "ticks": [int(value) & 0xFFFFFFFF for value in values[16:18]]})
    return write_clock(rows, destination)


class ClockedStaticBaseline(PreparedStaticBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps, max_sequence = static_steps(self.layout)
        original = self.wait_module.rewrite_wait_source

        def rewrite(source):
            source, changes = original(source)
            source, clock = instrument_notify(source, static=True, max_sequence=max_sequence)
            changes["clock"] = clock
            return source, changes

        self.wait_module.rewrite_wait_source = rewrite
        self.policy = SimpleNamespace(constexprs=lambda: {"historical_static": True})
        self.allocations = {}
        self.buffers = {"signal": self.prepared.sections["progress"]}

    def readback(self, generation):
        import torch

        progress = self.buffers["signal"].cpu().view(72, 128)
        expected = torch.tensor(expected_progress(self.layout, generation), dtype=progress.dtype)
        allowed = torch.zeros((72, 128), dtype=torch.bool)
        allowed[:, 0] = True
        for core, sequence in self.steps:
            start = 16 + 4 * sequence
            allowed[core, start:start + 4] = True
            if not bool((progress[core, start:start + 2] != 0).any()):
                raise AssertionError("静态已发布seq缺失时钟")
            if not bool((progress[core, start + 2:start + 4] == 0).all()):
                raise AssertionError("静态时钟保留字段被非预期写入")
        guard = self.prepared.workspace_guard_bytes
        storage = self.prepared.workspace_allocation
        checks = {
            "progress_passed": torch.equal(progress[:, 0], expected),
            "remaining_padding_zero": bool((progress[~allowed] == 0).all()),
            "workspace_guards_passed": bool((storage[:guard].cpu() == 37).all() and (storage[-guard:].cpu() == 37).all()),
            "control_tables_unchanged": all(torch.equal(self.prepared.sections[key].cpu(), value)
                                             for key, value in self._controls.items()),
        }
        checks["passed"] = all(checks.values())
        if not checks["passed"]:
            raise AssertionError(f"静态诊断保护失败：{checks}")
        return checks

    def export_clock(self, signal, destination):
        rows = []
        for (core, sequence), metadata in self.steps.items():
            start = 16 + 4 * sequence
            rows.append({**metadata, "core": core, "dispatch_seq": sequence,
                         "ticks": [int(value) & 0xFFFFFFFF for value in signal[core, start:start + 2]]})
        return write_clock(rows, destination)
