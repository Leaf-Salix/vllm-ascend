# SPDX-License-Identifier: Apache-2.0
"""Check that the G6 visibility trial moves checks without changing tile work."""

import collections
import hashlib
import json
import re
from pathlib import Path


def step_body(text):
    matches = list(re.finditer(r"scf.for %single_step\w* = [^\n]+\{", text))
    assert len(matches) == 1
    start = matches[0].end()
    depth = 1
    for end in range(start, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return matches[0].start(), text[start:end]
    raise AssertionError("Unclosed Score step")


def cube_body(text):
    text = text[text.index("AICORE void cube_schedule("):text.index("AICORE void vector_schedule(")]
    return re.sub(r"^\s*// pto:.*\n", "", text, flags=re.M)


def main():
    here = Path(__file__).resolve().parent
    before = here / "runs/current_native_typefix_control_b16_v58/kernel/ptoas"
    after = here / "runs/score_visibility_cpu_b16_v64/kernel/ptoas"
    cpp = [(p / "dynamic_csa_full.cpp").read_text() for p in (before, after)]
    pto = [(p / "dynamic_csa_full.pto").read_text() for p in (before, after)]
    cubes = [cube_body(s) for s in cpp]
    assert cubes[0] == cubes[1], "Cube instructions changed"
    steps = [step_body(s) for s in pto]
    geometry_locations = [s.count("/request_geometry.py") + s.count("/indexer_group_stages.py") for _, s in steps]
    assert geometry_locations[0] > 0 and geometry_locations[1] == 0
    for index in range(3):
        definition = re.search(r"%cached_visible" + str(index) + r"\w* = scf.if", pto[1])
        assert definition and definition.start() < steps[1][0]
        assert re.search(r"%cached_visible" + str(index) + r"\w*", steps[1][1])
    allocations = [collections.Counter(re.findall(
        r"pto.alloc_tile addr = (\S+) [^\n]* : (!pto.tile_buf<[^>]+>)", s,
    )) for s in pto]
    assert allocations[0] == allocations[1], "Tile layout or allocation changed"
    tile_ops = [re.findall(r"\b(pto\.t[a-z0-9_]+)\b", s) for _, s in steps]
    assert tile_ops[0] == tile_ops[1], "Score step tile operation sequence changed"
    synchronization = {}
    for symbol in ("PTOAS__WAIT_GE_BACKOFF<150>(", "pto::comm::TNOTIFY(",
                   "wait_flag_dev(", "ffts_cross_core_sync("):
        counts = [s.count(symbol) for s in cpp]
        assert counts[0] == counts[1]
        synchronization[symbol] = counts
    report = {
        "passed": True, "device_execution": False,
        "cube_executable_text_equal_without_pto_comments": True,
        "cube_sha256": hashlib.sha256(cubes[0].encode()).hexdigest(),
        "geometry_source_locations_inside_step_before_after": geometry_locations,
        "three_guarded_visibility_values_defined_before_step": True,
        "all_tile_layout_allocation_multisets_equal": True,
        "score_step_tile_operation_sequence_equal": True,
        "score_step_tile_operation_sites": len(tile_ops[0]),
        "synchronization_site_counts_before_after": synchronization,
        "limitation": "CPU code generation only; scalar lifetime and machine scheduling may change",
    }
    (after.parents[1] / "codegen_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
