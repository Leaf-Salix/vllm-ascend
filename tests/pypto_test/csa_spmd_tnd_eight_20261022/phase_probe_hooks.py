# SPDX-License-Identifier: Apache-2.0
"""Process-local diagnostic compilation hook; does not edit the shared SDK."""

import contextlib
import hashlib
import importlib
import json

from phase_clock import instrument


@contextlib.contextmanager
def compilation_probe(enabled, directory):
    if not enabled:
        yield
        return
    from pypto.backend import pto_backend

    original = pto_backend._compile_pto_module
    directory.mkdir(exist_ok=False)
    records = []

    def hooked(pto_code, unit_name, output_dir, *args, **kwargs):
        source = original(pto_code, unit_name, output_dir, *args, **kwargs)
        if unit_name != "dynamic_csa_full":
            return source
        revised, audit = instrument(source)
        (directory / "original.cpp").write_text(source)
        (directory / "modified.cpp").write_text(revised)
        audit.update(
            original_sha256=hashlib.sha256(source.encode()).hexdigest(),
            modified_sha256=hashlib.sha256(revised.encode()).hexdigest(),
        )
        records.append(audit)
        (directory / "audit.json").write_text(json.dumps(records, indent=2) + "\n")
        return revised

    pto_backend._compile_pto_module = hooked
    try:
        yield
        assert len(records) == 1, "Expected one instrumented MIX module; reject cache reuse"
    finally:
        pto_backend._compile_pto_module = original


def export_entry(entry, directory):
    from compile_diagnostics import export_clock

    abi = importlib.import_module(type(entry).__module__.rsplit(".", 1)[0] + ".runtime_abi")
    phases = {
        value: key.removeprefix("PHASE_")
        for key, value in vars(abi).items()
        if key.startswith("PHASE_") and key != "PHASE_COUNT"
    }
    for name in ("OA_GROUP", "O_QUANT_GROUP"):
        base = getattr(abi, "PHASE_" + name + "_BASE")
        phases.update({base + i: f"{name}_{i}" for i in range(8)})
    signal = entry.buffers["signal"].cpu().view(abi.PHASE_COUNT, 72, 128)
    assert not signal[:, :, 1:16].count_nonzero() and not signal[:, :, 20:].count_nonzero()
    return export_clock(signal, directory, phases)
