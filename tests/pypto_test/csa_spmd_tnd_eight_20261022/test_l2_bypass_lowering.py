# SPDX-License-Identifier: Apache-2.0
"""CPU evidence: current PyPTO's BYPASS attribute has no offset in PTOAS 0.69."""

import hashlib
import json
import subprocess
from pathlib import Path

import pypto.language as pl
from measurement_provenance import capture_toolchain
from pypto import backend, codegen, ir
from pypto.backend import BackendType
from pypto.ir import OptimizationStrategy, PassManager


@pl.program
class Bypass:
    @pl.function
    def main(
        self,
        a: pl.Tensor[[32, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        out: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
    ) -> pl.Tensor[[32, 64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
            pl.set_cache_policy(b, pl.CachePolicy.BYPASS)
            out = pl.assemble(out, pl.matmul(a, b, out_dtype=pl.FP32), [0, 0])
        return out


@pl.program
class Default:
    @pl.function
    def main(
        self,
        a: pl.Tensor[[32, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        out: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
    ) -> pl.Tensor[[32, 64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
            out = pl.assemble(out, pl.matmul(a, b, out_dtype=pl.FP32), [0, 0])
        return out


def test_bypass_currently_lowers_to_identical_default_cpp(tmp_path):
    here = Path(__file__).resolve().parent
    workspace = next(p for p in here.parents if (p / "env-dsv4-0251rc1.sh").is_file())
    manifest = workspace / "vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_shared_toolchain/manifest.json"
    sdk = json.loads(manifest.read_text())
    assert sdk["status"] == "validated"
    binary = workspace / sdk["ptoas"]["build"] / "bin/ptoas"
    observed_sdk = capture_toolchain(sdk)
    output = tmp_path / "l2_bypass_cpu"
    output.mkdir(exist_ok=False)
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    emitted, cpp = {}, {}
    try:
        for name, program in (("default", Default), ("bypass", Bypass)):
            optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
            incore = [f for f in optimized.functions.values() if f.func_type != pl.FunctionType.Orchestration]
            assert len(incore) == 1
            single = ir.Program(incore, incore[0].name, optimized.span)
            result = codegen.PTOCodegen().generate(single, emit_source_loc=False)
            emitted[name] = result if isinstance(result, str) else "".join(result.values())
            pto = output / f"{name}.pto"
            target = output / f"{name}.cpp"
            pto.write_text(emitted[name])
            command = [str(binary), str(pto), "--pto-arch=a3", "--pto-level=level3", "-o", str(target)]
            run = subprocess.run(command, text=True, capture_output=True, check=False)
            (output / f"{name}.log").write_text(run.stdout + run.stderr)
            assert run.returncode == 0, run.stderr
            cpp[name] = target.read_text()
    finally:
        backend.reset_for_testing()
    marker = "cache_policy = #pto.load_cache_policy<l2_bypass>"
    hinted = [line.strip() for line in emitted["bypass"].splitlines() if marker in line]
    assert len(hinted) == 1 and marker not in emitted["default"]
    assert "get_l2_cache_offset" not in emitted["bypass"] and " offset = " not in hinted[0]
    assert cpp["bypass"] == cpp["default"]
    assert "get_l2_cache_offset" not in cpp["bypass"] and "TLoadL2Hint" not in cpp["bypass"]
    report = dict(
        device_execution=False,
        shared_toolchain=sdk,
        observed_sdk=observed_sdk,
        hinted_tload=hinted,
        generated_cpp_identical=True,
        default_cpp_sha256=hashlib.sha256(cpp["default"].encode()).hexdigest(),
        bypass_cpp_sha256=hashlib.sha256(cpp["bypass"].encode()).hexdigest(),
        output=str(output),
        interpretation="Missing PyPTO runtime-offset integration; PTOAS safely emits an ordinary load without offset",
        limitations="CPU lowering evidence only; no NPU correctness failure or measured performance loss claimed",
    )
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "shared_toolchain"}, indent=2))
