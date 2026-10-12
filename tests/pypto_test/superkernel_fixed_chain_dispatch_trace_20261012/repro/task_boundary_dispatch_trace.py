# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Correlate fixed-chain dispatch and AICore pickup without enabling serial DFX."""

import argparse
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import task_boundary_acceptance as acceptance
import task_boundary_chain_trace as chain
import task_boundary_normal_state_probe as probe

PROGRAM = chain._TRACE_PROGRAM.replace("[48,8]", "[96,8]").replace("[817,8]", "[1633,8]")
PROGRAM = PROGRAM.replace("[stage * 48,0]", "[stage * 96,0]")


def changes(root):
    """The second half of each diagnostic trace view belongs only to AICPU."""
    result = {}
    # Apply the retained implementation as a temporary source overlay so the
    # already-queued artifacts keep their original SDK/native ABI revision.
    retained = "1c56bcd3e1dd1266cdaa5147cdc25f1d63fef34e"
    prefix = "src/a2a3/runtime/tensormap_and_ringbuffer/"
    for relative in (
        "aicore/aicore_executor.cpp",
        "runtime/dispatch_payload.h",
        "runtime/orchestrator.cpp",
        "runtime/runtime_types.h",
        "runtime/scheduler/scheduler_dispatch.cpp",
    ):
        path = root / (prefix + relative)
        old = path.read_text()
        new = subprocess.check_output(["git", "show", f"{retained}:{prefix}{relative}"], cwd=root, text=True)
        if old != new:
            result[path] = old, new
    base = root / "src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler"
    path = base / "scheduler_context.h"
    old = path.read_text()
    new = old.replace(
        '#include "scheduler_types.h"',
        '#include "scheduler_types.h"\n#include "aicpu/device_time.h"\n#include "aicpu/cache_maintenance.h"',
    )
    anchor = "        bool valid{false};\n    };"
    assert new.count(anchor) == 1
    new = new.replace(
        anchor, "        bool valid{false};\n        uint64_t *boundary_trace{nullptr};\n    };"
    )
    anchor = "        write_reg(h.reg_addr, RegId::DATA_MAIN_BASE, static_cast<uint64_t>(h.reg_task_id));"
    assert new.count(anchor) == 1
    new = new.replace(
        anchor,
        """        const uint64_t publish_before = get_sys_cnt_aicpu();
"""
        + anchor
        + """
        wmb();
        const uint64_t publish_after = get_sys_cnt_aicpu();
        if (h.boundary_trace != nullptr) {
            h.boundary_trace[2] = publish_before;
            h.boundary_trace[3] = publish_after;
            cache_flush_range(h.boundary_trace, 64);
        }""",
    )
    result[path] = old, new
    path = base / "scheduler_dispatch.cpp"
    old, new = result.get(path, (path.read_text(), path.read_text()))
    begin = new.index("SchedulerContext::PublishHandle SchedulerContext::prepare_subtask_to_core(")
    end = new.index("\nint SchedulerContext::prepare_block_for_dispatch(", begin)
    section = new[begin:end]
    anchor = "    CoreTracker &tracker = core_trackers_[thread_idx];"
    assert section.count(anchor) == 1
    section = section.replace(anchor, "    const uint64_t prepare_begin = get_sys_cnt_aicpu();\n" + anchor)
    anchor = "    return PublishHandle{core_exec_state.reg_addr,"
    assert section.count(anchor) == 1
    section = section.replace(
        anchor,
        """    const uint64_t prepare_end = get_sys_cnt_aicpu();
    uint64_t *trace = nullptr;
    // This isolated runtime only records the diagnostic program's INT64 [96,8] view.
    if (slot_state.payload->tensor_count == 2 && block_idx >= 0 && block_idx < 48) {
        const auto &tensor = slot_state.payload->tensors[1];
        if (tensor.ndims == 2 && tensor.shapes[0] == 96 && tensor.shapes[1] == 8) {
            trace = reinterpret_cast<uint64_t *>(tensor.buffer.addr) + tensor.start_offset +
                    (48 + block_idx) * 8;
            trace[0] = prepare_begin;
            trace[1] = prepare_end;
            trace[4] = reinterpret_cast<uint64_t>(&payload.global_context);
            trace[5] = reg_task_id;
            trace[6] = core_id;
            trace[7] = payload.global_context.scalar_prefix | (uint64_t{to_pending} << 32) |
                       (uint64_t{payload.src_payload != 0} << 33) | (static_cast<uint64_t>(thread_idx) << 40);
        }
    }
"""
        + anchor,
    )
    anchor = "                         true};"
    assert section.count(anchor) == 1
    section = section.replace(anchor, "                         true, trace};")
    result[path] = old, new[:begin] + section + new[end:]
    return result


def make_program(directory, mode):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "program.py"
    if path.exists():
        assert path.read_text() == PROGRAM
    else:
        path.write_text(PROGRAM)
    spec = importlib.util.spec_from_file_location("boundary_dispatch_trace_program", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TraceProgram


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "compile", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    acceptance.make_program = make_program
    if args.action == "compile":
        acceptance.artifact(root / "compiled", "early", compile_now=True)
        return
    from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

    RuntimeBuilder._CACHE_DIR = root / "runtime/cache"
    RuntimeBuilder._LIB_DIR = root / "runtime/lib"
    if args.action == "build":
        probe.changes = changes
        probe.build(root)
        return
    assert os.environ.get("TASK_DEVICE") == str(args.device)
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415
    from pypto.torch import init  # noqa: PLC0415
    from pypto.torch.launch import enqueue  # noqa: PLC0415

    torch.set_num_threads(2)
    torch_npu.npu.set_device(args.device)
    init(device=args.device, runtime="tensormap_and_ringbuffer", ring_task_window=128)
    runtime = get_process_kernel_state()
    acceptance.SCALAR_STEPS = 256
    runs = []
    try:
        kernel = runtime.ensure_callable(
            acceptance.artifact(root / "compiled", "early"), runtime.require_config()
        )
        data = torch.empty((acceptance.ROWS, acceptance.COLS), device=f"npu:{args.device}")
        trace = torch.empty((1633, 8), dtype=torch.int64, device=data.device)
        initial = (torch.arange(48 * 32 * 512, dtype=torch.float32) % 31).reshape(-1, 512)
        for work in (5, 167):
            print(json.dumps({"phase": "enqueue", "work": work}), flush=True)
            data.fill_(float("nan"))
            data[:1536].copy_(initial)
            trace.fill_(-1)
            enqueue(kernel, (data, trace, 17, work, 256, 7))
            torch_npu.npu.synchronize()
            acceptance.verify(data, initial, work, 17)
            values = trace.cpu().tolist()
            assert values[-1] == [-1] * 8
            entries = []
            mapping = {}
            for stage in range(17):
                for block in range(48):
                    core = values[stage * 96 + block]
                    cpu = values[stage * 96 + 48 + block]
                    assert core[4] & 255 == 7, (stage, block, core)
                    assert 0 < cpu[0] <= cpu[1] <= cpu[2] <= cpu[3], (stage, block, cpu)
                    assert cpu[4] > 0 and cpu[5] > 0 and cpu[6] >= 0
                    assert mapping.setdefault(cpu[6], core[0]) == core[0]
                    entries.append(
                        dict(
                            stage=stage,
                            block=block,
                            core=core[0],
                            start=core[1],
                            returned=core[2],
                            end=core[3],
                            flags=core[4],
                            wait_start=core[5],
                            wait_ticks=core[6],
                            compute_end=core[7],
                            prepare_start=cpu[0],
                            prepare_end=cpu[1],
                            publish_before=cpu[2],
                            publish_after=cpu[3],
                            context=cpu[4],
                            dispatch_id=cpu[5],
                            logical_core=cpu[6],
                            prefix=cpu[7] & 0xFFFFFFFF,
                            pending=bool(cpu[7] & (1 << 32)),
                            gated=bool(cpu[7] & (1 << 33)),
                            scheduler_thread=cpu[7] >> 40,
                        )
                    )
            assert 0 < len(mapping) <= 48 and len(set(mapping.values())) == len(mapping)
            runs.append(dict(work=work, device=args.device, entries=entries, logical_to_physical=mapping))
    finally:
        runtime.drain()
        runtime.close()
    report = dict(diagnostic_only=True, output_and_guards_exact=True, runs=runs)
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "records_per_run": [len(r["entries"]) for r in runs]}))


if __name__ == "__main__":
    main()
