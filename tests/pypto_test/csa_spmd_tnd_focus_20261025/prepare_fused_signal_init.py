# SPDX-License-Identifier: Apache-2.0
"""Prepare a private, device-capability-guarded signal initialization candidate."""

import ast
import difflib
import json
import shutil
import textwrap
from pathlib import Path

from prepare import replace_once


def add_parameter(source, function):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == function)
    lines = source.splitlines(True)
    end = next(i for i in range(node.lineno, node.body[0].lineno) if lines[i].startswith(")"))
    lines.insert(end, "    FUSED_SIGNAL_INIT: pl.constexpr = False,\n")
    return "".join(lines)


def patch_entry(source):
    source = add_parameter(source, "_csa_core_group")
    source = replace_once(
        source,
        "    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],\n",
        (
            "    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],\n"
            "    signal_init_matrix: pl.InOut[pl.Tensor[[VECTORS, PHASE_COUNT * CORES * SIGNA"
            "L_SLOT_WORDS // VECTORS], pl.INT32]],\n"
        ),
    )
    # These are the two member calls in the explicit Group, not its entry callers.
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_csa_core_group")
    lines = source.splitlines(True)
    calls = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id in ("cube_schedule", "vector_schedule")
    ]
    assert len(calls) == 2
    for call in sorted(calls, key=lambda n: n.end_lineno, reverse=True):
        lines.insert(call.end_lineno - 1, "            FUSED_SIGNAL_INIT=FUSED_SIGNAL_INIT,\n")
        if call.func.id == "vector_schedule":
            signal_arg = next(arg for arg in call.args if isinstance(arg, ast.Name) and arg.id == "signal")
            lines.insert(signal_arg.end_lineno, "            signal_init_matrix,\n")
    source = "".join(lines)
    for function in ("decode_csa_spmd_debug", "decode_csa_spmd"):
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == function)
        lines = source.splitlines(True)
        body = "".join(lines[node.lineno - 1 : node.end_lineno])
        clear = (
            "    signal_matrix = pl.reshape(signal, [48, (PHASE_COUNT * CORES * SIGNAL_SLOT_WORDS // VECTORS)])\n"
            "    with pl.spmd(48, sync_start=True):\n"
            "        signal_matrix = _clear_internal_signal(signal_matrix)\n"
            "    signal = pl.reshape(signal_matrix, [PHASE_COUNT * CORES * SIGNAL_SLOT_WORDS])\n"
        )
        body = replace_once(body, clear, "")
        begin = body.index("    with pl.spmd(GROUPS, sync_start=True):\n")
        end = body.index("    return (", begin)
        group = body[begin:end]
        group = replace_once(
            group,
            "            signal,\n            QUERY_GROUP,\n",
            "            signal,\n            signal_init_matrix,\n            QUERY_GROUP,\n",
        )
        fused = replace_once(
            group, "pl.spmd(GROUPS, sync_start=True)", "pl.spmd(pl.system.available_cluster_count(), sync_start=True)"
        )
        fused = replace_once(
            fused,
            "            PLAN_TABLE_COLUMNS=PLAN_TABLE_COLUMNS,\n",
            "            PLAN_TABLE_COLUMNS=PLAN_TABLE_COLUMNS,\n            FUSED_SIGNAL_INIT=True,\n",
        )
        fallback = replace_once(
            group,
            "            PLAN_TABLE_COLUMNS=PLAN_TABLE_COLUMNS,\n",
            "            PLAN_TABLE_COLUMNS=PLAN_TABLE_COLUMNS,\n            FUSED_SIGNAL_INIT=False,\n",
        )
        guarded = (
            (
                "    # Fixed CSA ownership is valid for exactly 24 MIX groups. The query\n"
                "    # also proves full occupancy to the hard-syncall compiler verifier.\n"
                "    # Other device widths retain the original separate initialization.\n"
                "    signal_init_matrix = pl.reshape(signal, [VECTORS, PHASE_COUNT * CORES * SIGN"
                "AL_SLOT_WORDS // VECTORS])\n"
                "    if pl.system.available_cluster_count() == GROUPS:\n"
            )
            + textwrap.indent(fused, "    ")
            + "    else:\n"
            + textwrap.indent(clear + fallback, "    ")
        )
        body = body[:begin] + guarded + body[end:]
        lines[node.lineno - 1 : node.end_lineno] = [body]
        source = "".join(lines)
    source = source.replace(
        "    每次调用均通过设备任务清零，再由依赖启动业务SPMD。图重放包含同样的\n",
        "    24组全占用时每次调用在主MIX内清零并硬屏障，其余设备保留独立清零。图重放包含同样的\n",
    )
    return source


def patch_schedule(source, role):
    function = f"{role}_schedule"
    source = add_parameter(source, function)
    if role == "vector":
        source = replace_once(
            source,
            "from .phase_sync import begin_invocation, publish_phase, wait_row_owners\n",
            "from .phase_sync import SIGNAL_SLOT_WORDS, begin_invocation, publish_phase, wait_row_owners\n",
        )
        source = replace_once(
            source, "    COMPRESSED_COLS,\n    CORES,\n", "    COMPRESSED_COLS,\n    CORES,\n    PHASE_COUNT,\n"
        )
        source = replace_once(
            source,
            "    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],\n",
            (
                "    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],\n"
                "    signal_init_matrix: pl.InOut[pl.Tensor[[VECTORS, PHASE_COUNT * CORES * SIGNA"
                "L_SLOT_WORDS // VECTORS], pl.INT32]],\n"
            ),
        )
        prior = (
            "    core = GROUPS + worker\n"
            "    signal, generation = begin_invocation(signal, core)\n"
            "    pl.system.set_ffts(ffts)\n"
        )
        init = (
            "    core = GROUPS + worker\n"
            "    pl.system.set_ffts(ffts)\n"
            "    if FUSED_SIGNAL_INIT:\n"
            "        # Exactly the original 48 contiguous, disjoint 11520-word ranges.\n"
            "        # Preserve the full 512-byte phase-slot stride and all padding.\n"
            "        zero = pl.tile.full([1, PHASE_COUNT * CORES * SIGNAL_SLOT_WORDS // VECTO"
            "RS], dtype=pl.INT32, value=0)\n"
            "        pl.store(zero, [worker, 0], signal_init_matrix)\n"
        )
        suffix = "    signal, generation = begin_invocation(signal, core)\n"
    else:
        prior = (
            "    worker = pl.tile.get_block_idx()\n"
            "    signal, generation = begin_invocation(signal, worker)\n"
            "    pl.system.set_ffts(ffts)\n"
        )
        init = "    worker = pl.tile.get_block_idx()\n    pl.system.set_ffts(ffts)\n    if FUSED_SIGNAL_INIT:\n"
        suffix = "    signal, generation = begin_invocation(signal, worker)\n"
    barrier = (
        "        # Both explicit Group members rendezvous before any signal read/write.\n"
        "        # This is hardware FFTS (events 11/12/13), never a GM boot barrier.\n"
        "        pl.system.cacheinvalid()\n"
        "        pl.system.fence()\n"
        "        pl.system.syncall(core_type=pl.KernelType.MIX)\n"
        "        pl.system.cacheinvalid()\n"
    )
    return replace_once(source, prior, init + barrier + suffix)


def main():
    here = Path(__file__).resolve().parent
    parent = here / "frozen/pypto/deepseek_v4_flash_csa_spmd_tnd"
    target = here / "candidates/fused_signal_init_v3"
    shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    patches = []
    for name, transform in (
        ("entry.py", patch_entry),
        ("cube_schedule.py", lambda text: patch_schedule(text, "cube")),
        ("vector_schedule.py", lambda text: patch_schedule(text, "vector")),
    ):
        path = target / name
        before = path.read_text()
        after = transform(before)
        ast.parse(after)
        path.write_text(after)
        patches.extend(
            difflib.unified_diff(
                before.splitlines(True), after.splitlines(True), fromfile=f"a/{name}", tofile=f"b/{name}"
            )
        )
    (here / "fused_signal_init_v3.patch").write_text("".join(patches))
    report = {
        "status": "CPU_gate_pending",
        "source": str(target),
        "parent": str(parent),
        "baseline_commit": "4f9b563133989900a5ca492583eb6bb5ba5f51ac",
        "shared_toolchain": "csa-shared-20261024-group-dims",
        "isa_pin": "327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d",
        "capability_guard": "available_cluster_count() == 24; fused launch uses the same full-device query",
        "fallback": "original 48-AIV clear followed by 24-group MIX, FUSED_SIGNAL_INIT=False",
        "signal_words": 60 * 72 * 128,
        "clear_words_per_aiv": 60 * 72 * 128 // 48,
        "signal_slot_bytes": 512,
        "hardware_barrier_events": [11, 12, 13],
        "existing_business_events": [0, 1, 2, 3, 4, 5],
        "measurement": "complete selected branch, including in-kernel clear and barrier; no timing result",
        "device_execution": False,
        "changed_files": ["entry.py", "cube_schedule.py", "vector_schedule.py"],
    }
    (here / "fused_signal_init_v3.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
