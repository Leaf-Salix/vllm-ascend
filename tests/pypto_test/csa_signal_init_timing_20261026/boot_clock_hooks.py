# SPDX-License-Identifier: Apache-2.0
"""只在测量进程插入四次取时，延迟到业务结束写出，不编辑共享SDK。"""

import contextlib
import hashlib
import json
import re
from pathlib import Path

MAGIC = 741852963000


def instrument(source):
    assert "CSA_INIT_INTERVAL_V1" not in source
    cube = source.index("AICORE void cube_schedule(")
    vector = source.index("AICORE void vector_schedule(")
    assert cube < vector
    parts = [source[:cube], source[cube:vector], source[vector:]]
    counts = {}
    for role, index in (("aic", 1), ("aiv", 2)):
        body = parts[index]
        opening = body.index("{") + 1
        declarations = "\n" + "\n".join(f"  int64_t csa_init_clock_{i};" for i in range(4)) + "\n"
        body = body[:opening] + declarations + body[opening:]
        for slot in range(5):
            match = re.findall(r"const int64_t (v\d+) = " + str(MAGIC + slot) + r";", body)
            assert len(match) == 1, (role, slot, match)
            pattern = re.compile(r"^(\s*)([^\n=]+\[[^\n]+\]) = " + match[0] + r";$", re.M)

            def replacement(found, slot=slot, role=role):
                indent, lhs = found[1], found[2]
                barrier = f'{indent}__asm__ __volatile__("" ::: "memory");\n'
                if slot < 4:
                    return (
                        barrier
                        + f"{indent}csa_init_clock_{slot} = static_cast<int64_t>(get_sys_cnt());\n"
                        + barrier.rstrip("\n")
                    )
                physical = "static_cast<int64_t>(get_block_idx())" if role == "aic" else "static_cast<int64_t>(-1)"
                return (
                    barrier
                    + f"{indent}auto csa_init_dump_slot4 = &({lhs});\n"
                    + "".join(f"{indent}csa_init_dump_slot4[{i - 4}] = csa_init_clock_{i};\n" for i in range(4))
                    + f"{indent}{lhs} = {physical};\n"
                    + barrier.rstrip("\n")
                )

            body, count = pattern.subn(replacement, body)
            assert count == 1, (role, slot, count)
            counts[f"{role}_{slot}"] = count
        assert body.count("SYNCALL<SyncCoreType::Mix>();") == 1
        assert (
            body.index("csa_init_clock_0 =")
            < body.index("csa_init_clock_1 =")
            < body.index("SYNCALL<SyncCoreType::Mix>();")
            < body.index("csa_init_clock_2 =")
            < body.index("csa_init_clock_3 =")
            < body.index("csa_init_dump_slot4")
        )
        assert body.index("csa_init_clock_3 =") < body.index("TNOTIFY(")
        counts[role + "_deferred_write_verified"] = True
        parts[index] = body
    return parts[0] + "// CSA_INIT_INTERVAL_V1\n" + parts[1] + parts[2], counts


@contextlib.contextmanager
def compilation_probe(directory, *, enabled=True):
    if not enabled:
        yield
        return
    from pypto.backend import pto_backend

    original = pto_backend._compile_pto_module
    directory.mkdir(exist_ok=False)
    audits = []

    def hooked(pto_code, unit_name, output_dir, *args, **kwargs):
        source = original(pto_code, unit_name, output_dir, *args, **kwargs)
        if unit_name != "dynamic_csa_full":
            return source
        revised, sites = instrument(source)
        (directory / "original.cpp").write_text(source)
        (directory / "modified.cpp").write_text(revised)
        audits.append(
            {
                "sites": sites,
                "original_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "modified_sha256": hashlib.sha256(revised.encode()).hexdigest(),
            }
        )
        (directory / "audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2) + "\n")
        return revised

    pto_backend._compile_pto_module = hooked
    try:
        yield
        assert len(audits) == 1, "须插桩唯一MIX模块，拒绝复用未插桩缓存"
    finally:
        pto_backend._compile_pto_module = original


def export_entry(entry, window):
    import torch

    storage = entry._diagnostic_clock_storage.cpu()
    path = Path(window["raw"]).with_name("clock_raw.json")
    path.write_text(json.dumps({"storage_int64": storage.tolist(), "guard_words": 8}) + "\n")
    assert bool((storage[:8] == 37).all() and (storage[-8:] == 37).all())
    clocks = storage[8:-8].view(72, 8)
    assert torch.equal(clocks[:, 5], torch.arange(72, dtype=torch.int64))
    assert bool((clocks[:, 6] == 1).all() and (clocks[:, 7] == 37).all())
    assert set(clocks[:24, 4].tolist()) == set(range(24)) and bool((clocks[24:, 4] == -1).all())
    aic = clocks[:24, 4].tolist()
    mapping = aic + [24 + 2 * aic[i // 2] + i % 2 for i in range(48)]
    assert set(mapping) == set(range(72))
    raw = json.loads(Path(window["raw"]).read_text())
    rows = {r[0]: r for r in raw["aicore_tasks"]}
    assert set(rows) == set(range(72)) and len(raw["aicore_tasks"]) == 72
    frequency = raw["metadata"]["clock_freq_hz"]
    scale = 1e6 / frequency
    cores = []
    values = clocks.tolist()
    for logical, physical in enumerate(mapping):
        start, arrived, released, calibration = values[logical][:4]
        row = rows[physical]
        assert row[3] <= start <= arrived <= released <= calibration <= row[4], (
            logical,
            physical,
            values[logical],
            row,
        )
        cores.append(
            {
                "logical_core": logical,
                "physical_core": physical,
                "role": "aic" if logical < 24 else "aiv",
                "ticks": values[logical][:4],
                "pre_sync_us": (arrived - start) * scale,
                "sync_wait_us": (released - arrived) * scale,
                "clear_and_sync_local_us": (released - start) * scale,
                "adjacent_clock_read_us": (calibration - released) * scale,
            }
        )
    first_vector_start = min(v[0] for v in values[24:])
    last_arrival, last_release = max(v[1] for v in values), max(v[2] for v in values)
    assert first_vector_start <= last_arrival <= last_release
    return {
        "guards_passed": True,
        "physical_intervals_validated": 72,
        "physical_mapping": "AIC硬件编号；AIV按Simpler固定1C2V拓扑推导，并逐核检查时间戳包含关系",
        "raw_clock": str(path),
        "clock_freq_hz": frequency,
        "cores": cores,
        "global_clear_and_sync_us": (last_release - first_vector_start) * scale,
        "to_last_participant_arrival_us": (last_arrival - first_vector_start) * scale,
        "last_arrival_to_last_release_us": (last_release - last_arrival) * scale,
        "all_roles_initialization_span_us": (last_release - min(v[0] for v in values)) * scale,
        "L1_start_to_last_release_us": (last_release - min(r[3] for r in rows.values())) * scale,
        "generation_and_business_excluded": True,
    }
