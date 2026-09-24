# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Check that the precision experiment does not silently combine QKV factors."""

import ast
import importlib.util
import json
from pathlib import Path


def test_generated_factors_are_independent(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    source = repo / "benchmarks/scripts/dsv4_csa_precision/generate.py"
    spec = importlib.util.spec_from_file_location("precision_generator", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.generate(repo, tmp_path / "variants")
    manifest = json.loads((tmp_path / "variants/manifest.json").read_text())
    baseline = manifest["control"]
    factors = json.loads(source.with_name("factors.json").read_text())
    factor_file = "pto_kernels/dspark/qkv_proj_rope.py"
    for name, edits in factors.items():
        changed = [key for key in baseline if baseline[key] != manifest[name][key]]
        assert changed == [factor_file]
        variant = (tmp_path / "variants" / name / factor_file).read_text()
        for edit in edits:
            assert variant.count(edit["anchor"] + edit["insert"]) == 1
            variant = variant.replace(edit["anchor"] + edit["insert"], edit["anchor"], 1)
        assert variant == (tmp_path / "variants/control" / factor_file).read_text()
    for parent, child, factor in (("q_mm", "q_pair", "q_rms"), ("kv_mm", "kv_pair", "kv_rms")):
        variant = (tmp_path / "variants" / child / factor_file).read_text()
        for edit in factors[factor]:
            variant = variant.replace(edit["anchor"] + edit["insert"], edit["anchor"], 1)
        assert variant == (tmp_path / "variants" / parent / factor_file).read_text()
    # Syntax-check every generated diagnostic entry before any expensive NPU JIT.
    for path in (tmp_path / "variants").rglob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
