# SPDX-License-Identifier: Apache-2.0
"""Correlate bounded AICPU publication intervals with private CSA prefix traces.

The dedicated runtime embeds one process-local trace address. Source files are
restored after its build, and the ordinary runtime binaries are never replaced.
Neither these binaries nor the generated CSA may be reused in another process.
"""

import difflib
import json
import os
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median

import tmr_csa_full_128k_b16_compile as compile_demo
import tmr_csa_prefix_coverage as demo
import tmr_csa_prefix_coverage_instrument as coverage
import torch
from simpler_setup.runtime_builder import RuntimeBuilder
from tmr_csa_score_fair_compile import PACKAGES

CORE_WORDS = 24
CPU_WORDS = 8


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


@contextmanager
def instrument_scheduler(root, address, output):
    """Build-only source instrumentation; no dispatch, ACK or queue decisions change."""
    scheduler = root / "src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler"
    header, impl = scheduler / "scheduler_context.h", scheduler / "scheduler_dispatch.cpp"
    originals = {path: path.read_text() for path in (header, impl)}
    h = replace_once(
        originals[header],
        '#include "aicpu/device_phase_aicpu.h"',
        '#include "aicpu/cache_maintenance.h"\n#include "aicpu/device_phase_aicpu.h"',
    )
    h = replace_once(
        h, "        bool valid{false};\n", "        bool valid{false};\n        uint64_t *publication_trace{nullptr};\n"
    )
    publication = "        write_reg(h.reg_addr, RegId::DATA_MAIN_BASE, static_cast<uint64_t>(h.reg_task_id));\n"
    h = replace_once(
        h,
        publication,
        """        const uint64_t publication_begin = get_sys_cnt_aicpu();
"""
        + publication
        + """        wmb();  // Bound completion of the existing MMIO publication.
        const uint64_t publication_end = get_sys_cnt_aicpu();
        if (h.publication_trace != nullptr) {
            h.publication_trace[2] = publication_begin;
            h.publication_trace[3] = publication_end;
            cache_flush_range(h.publication_trace, 64);
        }
""",
    )
    marker = "    return PublishHandle{core_exec_state.reg_addr,\n"
    setup = f"""    uint64_t *publication_trace = nullptr;
    if (core_id >= 0 && core_id < {coverage.MAX_RAW_CORES}) {{
        const uint32_t record = reg_task_id < {coverage.RECORDS_PER_CORE} ? reg_task_id : 0;
        publication_trace = reinterpret_cast<uint64_t *>({address}ULL)
            + (core_id * {coverage.RECORDS_PER_CORE} + record) * {CPU_WORDS};
        publication_trace[0] = reinterpret_cast<uint64_t>(&payload.global_context);
        publication_trace[1] = reg_task_id;
        publication_trace[4] = core_id;
        publication_trace[5] = payload.src_payload != 0;
        publication_trace[6] = overlap.predecessor_reg_task_id;
        publication_trace[7] = overlap.scalar_prefix;
    }}
"""
    cpp = replace_once(originals[impl], marker, setup + marker)
    cpp = replace_once(
        cpp,
        "                         true};\n",
        "                         true,\n                         publication_trace};\n",
    )
    updates = {header: h, impl: cpp}
    patch = []
    for path, source in updates.items():
        relative = path.relative_to(root)
        patch.extend(
            difflib.unified_diff(
                originals[path].splitlines(keepends=True),
                source.splitlines(keepends=True),
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    (output / "scheduler_instrumentation.patch").write_text("".join(patch))
    try:
        for path, source in updates.items():
            path.write_text(source)
        yield
    finally:
        for path, source in originals.items():
            assert path.read_text() == updates[path], f"concurrent edit: {path}"
            path.write_text(source)


class PublicationProbe:
    def __init__(self, args, report):
        self.output = args.output.resolve()
        self.report = report
        self.cpu = None
        self.round = 0
        self.original_builder_dirs = RuntimeBuilder._CACHE_DIR, RuntimeBuilder._LIB_DIR
        self.original_source = coverage.instrument_source
        self.original_instrument = coverage.instrument_wrappers
        self.original_analyze = coverage.analyze_trace

    @contextmanager
    def instrument(self, address, functions):
        self.cpu = torch.full(
            (coverage.MAX_RAW_CORES, coverage.RECORDS_PER_CORE, CPU_WORDS),
            -1,
            dtype=torch.int64,
            device=f"npu:{self.report['device']}",
        )
        root = Path(os.environ["SCALAR_OVERLAP_ROOT"]) / "simpler"
        RuntimeBuilder._CACHE_DIR = self.output / "runtime/cache"
        RuntimeBuilder._LIB_DIR = self.output / "runtime/lib"
        print("STAGE: build isolated publication-trace runtime", flush=True)
        with instrument_scheduler(root, self.cpu.data_ptr(), self.output):
            binaries = RuntimeBuilder(platform="a2a3").get_binaries("tensormap_and_ringbuffer", build=True)
        self.report["publication_probe"] = {
            "binaries": {key: str(value) for key, value in asdict(binaries).items()},
            "normal_runtime": str(self.original_builder_dirs[1]),
            "source_restored_after_build": True,
            "clock_frequency_hz": 50_000_000,
            "publication_interval": "counter before existing MMIO write through counter after wmb",
            "not_a_performance_measurement": True,
        }

        def source_with_context(source, function_id, address, has_tail):
            source = self.original_source(source, function_id, address, has_tail)
            marker = "        coverage_row[0] = get_coreid();\n"
            return replace_once(
                source,
                marker,
                marker + "        coverage_row[16] = reinterpret_cast<uint64_t>(coverage_ctx);\n"
                "        dcci(coverage_row + 16, SINGLE_CACHE_LINE, CACHELINE_OUT);\n",
            )

        old_words = coverage.WORDS_PER_RECORD
        coverage.WORDS_PER_RECORD = CORE_WORDS
        coverage.instrument_source = source_with_context
        try:
            with self.original_instrument(address, functions):
                yield
        finally:
            coverage.instrument_source = self.original_source
            coverage.WORDS_PER_RECORD = old_words

    def reset(self):
        assert self.cpu is not None
        self.cpu.fill_(-1)

    def analyze(self, tensor, functions):
        cpu = self.cpu.cpu()
        rows = [cpu[int(core), int(seq)].tolist() for core, seq in (cpu[:, :, 1] >= 0).nonzero()]
        (self.output / f"publication_raw_round{self.round}.json").write_text(json.dumps(rows) + "\n")
        assert (cpu[:, 0] == -1).all(), "publication trace overflow"
        result = self.original_analyze(tensor[:, :, :16].contiguous(), functions)
        publications = {(row[0], row[1]): row for row in rows}
        assert len(publications) == len(rows)
        records = [tensor[int(core), int(seq)].tolist() for core, seq in (tensor[:, :, 8] >= 0).nonzero()]
        assert len(records) == len(rows), (len(records), len(rows))
        previous = {(row[0], row[9]): row for row in records}
        names = {row["id"]: row["name"] for row in functions}
        counts, details = Counter(), []
        pickup_delays = []
        for row in records:
            publication = publications[row[16], row[9]]
            _, sequence, lower, upper, logical, gated, prior, flags = publication
            assert 0 < lower <= upper and lower <= row[1], (publication, row[:16])
            assert (sequence, prior, flags) == (row[9], row[11], row[10]), (publication, row)
            counts["matched"] += 1
            counts["gated"] += bool(gated)
            pickup_delays.append(row[1] - upper)
            if not flags & 1:
                continue
            producer = previous[row[0], prior]
            handoff = bool(row[13] & 2)
            if upper < producer[2]:
                phase = "published_before_predecessor_return"
            elif lower > producer[3]:
                phase = "published_after_predecessor_full_finish"
            elif upper < producer[5]:
                phase = "published_before_predecessor_drain"
            else:
                phase = "publication_interval_crosses_tail"
            counts[f"{'handoff' if handoff else 'miss'}:{phase}"] += 1
            details.append(
                {
                    "core": row[0],
                    "logical_core": logical,
                    "sequence": sequence,
                    "function": names[row[8]],
                    "predecessor": names[producer[8]],
                    "handoff": handoff,
                    "phase": phase,
                    "predecessor_has_tail": bool(producer[13] & 4),
                    "published_lo": lower,
                    "published_hi": upper,
                    "producer_return": producer[2],
                    "producer_drain": producer[5],
                    "producer_full_finish": producer[3],
                    "consumer_entry": row[1],
                    "consumer_prefix_end": row[14],
                    "consumer_wait_end": row[15],
                }
            )
        result["publication"] = {
            "counts": dict(counts),
            "details": details,
            "raw_records": rows,
            "entry_minus_publication_upper_ticks": {
                "min": min(pickup_delays),
                "mean": mean(pickup_delays),
                "median": median(pickup_delays),
                "max": max(pickup_delays),
            },
        }
        (self.output / f"publication_round{self.round}.json").write_text(
            json.dumps(result["publication"], indent=2) + "\n"
        )
        self.round += 1
        print("PUBLICATION", dict(counts), flush=True)
        return result

    def restore(self):
        RuntimeBuilder._CACHE_DIR, RuntimeBuilder._LIB_DIR = self.original_builder_dirs


def run(args, report):
    probe = PublicationProbe(args, report)
    old = demo.WORDS_PER_RECORD, demo.instrument_wrappers, demo.analyze_trace
    demo.WORDS_PER_RECORD = CORE_WORDS
    demo.instrument_wrappers = probe.instrument
    demo.analyze_trace = probe.analyze
    try:
        original_run(args, report, reset_extra_trace=probe.reset)
    finally:
        demo.WORDS_PER_RECORD, demo.instrument_wrappers, demo.analyze_trace = old
        probe.restore()


if __name__ == "__main__":
    compile_demo.PACKAGES.update(baseline=PACKAGES["plain_score"], mixed=PACKAGES["early_score"])
    original_run = demo.run
    demo.run = run
    demo.main()
