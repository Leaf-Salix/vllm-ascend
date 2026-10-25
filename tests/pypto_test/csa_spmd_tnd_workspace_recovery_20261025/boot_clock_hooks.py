# SPDX-License-Identifier: Apache-2.0
"""进程内PTOAS后处理与只读导出，不编辑共享SDK，不将插桩耗时用于收益。"""

import contextlib
import hashlib
import json
import re
from pathlib import Path

MAGIC = 741852963000


def instrument(source):
    assert "CSA_DIAG_CLOCK_V3" not in source
    cube = source.index("AICORE void cube_schedule(")
    vector = source.index("AICORE void vector_schedule(")
    assert cube < vector
    parts = [source[:cube], source[cube:vector], source[vector:]]
    counts = {}
    for role, part_index in (("aic", 1), ("aiv", 2)):
        body = parts[part_index]
        for slot in range(5):
            match = re.findall(r"const int64_t (v\d+) = " + str(MAGIC + slot) + r";", body)
            assert len(match) == 1, (role, slot, match)
            pattern = re.compile(r"^(\s*)([^\n=]+\[[^\n]+\]) = " + match[0] + r";$", re.M)
            value = "static_cast<int64_t>(get_sys_cnt())"
            if slot == 4:
                # 与Simpler kernel_mode_kernel.cpp使用完全相同的运行时索引公式。
                # 不用名义24/2代替硬件builtin，也不靠插入helper的架构宏选择角色。
                value = (
                    "static_cast<int64_t>(get_block_idx())"
                    if role == "aic"
                    else "static_cast<int64_t>(get_block_idx() * get_subblockdim() "
                    "+ get_subblockid() + get_block_num())"
                )
            body, count = pattern.subn(
                lambda m, value=value: (
                    f'{m[1]}__asm__ __volatile__("" ::: "memory");\n'
                    f"{m[1]}{m[2]} = {value};\n"
                    f'{m[1]}__asm__ __volatile__("" ::: "memory");'
                ),
                body,
            )
            assert count == 1, (role, slot, count)
            counts[f"{role}_{slot}"] = count
        parts[part_index] = body
    return parts[0] + "// CSA_DIAG_CLOCK_V3\n" + parts[1] + parts[2], counts


@contextlib.contextmanager
def compilation_probe(directory):
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
                "scope": "四个硬件时间戳store与物理核编号，数学及同步未改，非性能收益证据",
            }
        )
        (directory / "audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2) + "\n")
        return revised

    pto_backend._compile_pto_module = hooked
    try:
        yield
        assert len(audits) == 1, "须正好插桩一个MIX模块，拒绝缓存复用"
    finally:
        pto_backend._compile_pto_module = original


def export_entry(entry, window):
    import torch

    storage = entry._diagnostic_clock_storage.cpu()
    raw_clock = Path(window["raw"]).with_name("clock_raw.json")
    raw_clock.write_text(json.dumps({"storage_int64": storage.tolist(), "guard_words": 8}) + "\n")
    assert bool((storage[:8] == 37).all() and (storage[-8:] == 37).all())
    clocks = storage[8:-8].view(72, 8)
    assert bool((clocks[:, 7] == 37).all())
    assert torch.equal(clocks[:, 5], torch.arange(72, dtype=torch.int64))
    physical_mapping = set(clocks[:, 4].tolist()) == set(range(72))
    raw = json.loads(Path(window["raw"]).read_text())
    rows = {r[0]: r for r in raw["aicore_tasks"]}
    scale = 1e6 / raw["metadata"]["clock_freq_hz"]
    lower, upper = min(r[3] for r in rows.values()), max(r[4] for r in rows.values())
    if physical_mapping:
        physical_mapping = all(rows[v[4]][3] <= v[0] <= v[3] <= rows[v[4]][4] for v in clocks.tolist())
    result = []
    for logical, values in enumerate(clocks.tolist()):
        start, initialized, first_math, end, physical, _, generation, _ = values
        assert lower <= start <= initialized <= first_math <= end <= upper, (logical, values, lower, upper)
        record = rows[physical] if physical_mapping else None
        result.append(
            {
                "logical_core": logical,
                "physical_core": physical,
                "role": "aic" if logical < 24 else "aiv",
                "generation": generation,
                "ticks": values[:4],
                "wrapper_and_prologue_us": (start - record[3]) * scale if record else None,
                "initialization_and_generation_us": (initialized - start) * scale,
                "first_math_and_input_wait_us": (first_math - initialized) * scale,
                "remaining_business_us": (end - first_math) * scale,
                "tail_including_clock_flush_us": (record[4] - end) * scale if record else None,
            }
        )
    return {
        "performance_eligible": False,
        "guards_passed": True,
        "cores": result,
        "physical_mapping_verified": physical_mapping,
        "raw_clock": str(raw_clock),
        "mapping_limit": None if physical_mapping else "不报告按物理核对齐的解包/尾部时间；只使用同一逻辑核内的时间差",
    }
