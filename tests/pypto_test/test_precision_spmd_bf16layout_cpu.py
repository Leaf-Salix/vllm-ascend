# SPDX-License-Identifier: Apache-2.0
"""Bind the independent full-CSA candidate to its measured one-factor stage."""

import ast
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "vllm_ascend/ops/pypto"
OLD = OPS / "deepseek_v4_flash_csa_tnd_precision_spmd_prepack"
NEW = OPS / "deepseek_v4_flash_csa_tnd_precision_spmd_bf16layout"


def functions(path):
    return {node.name: node for node in ast.parse(path.read_bytes()).body if isinstance(node, ast.FunctionDef)}


def test_complete_root_and_accepted_parent_sources_unchanged():
    assert (OLD / "decode_csa.py").read_bytes() == (NEW / "decode_csa.py").read_bytes()
    evidence = Path(__file__).parent / "evidence_tnd_precision/spmd_prepack_20261020/compile-receipt.json"
    receipt = json.loads(evidence.read_text())
    for name, digest in receipt["sources"].items():
        assert hashlib.sha256((OLD / name).read_bytes()).hexdigest() == digest
    for name, digest in receipt["parent_spmd_sources"].items():
        assert (
            hashlib.sha256((OPS / "deepseek_v4_flash_csa_tnd_precision_spmd" / name).read_bytes()).hexdigest() == digest
        )
    for name, digest in receipt["baseline_sources"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest


def test_prefix_keeps_short_path_and_only_conditions_the_measured_layout_call():
    old, new = functions(OLD / "prefix.py"), functions(NEW / "prefix.py")
    assert old.keys() == new.keys()
    vector = new["prepacked_prefix_indexer_vector"]
    branches = [node for node in vector.body if isinstance(node, ast.If)]
    assert len(branches) == 1
    branch = branches[0]
    assert ast.unparse(branch.test) == "pl.tensor.dim(flat, 0) >= 64"
    assert len(branch.body) == len(branch.orelse) == 1
    assert ast.unparse(branch.body[0]) == "packed_flat = hc_pack_input_bf16(wide, packed_flat, worker)"
    assert ast.unparse(branch.orelse[0]) == "packed_flat = hc_pack_input(wide, packed_flat, worker)"
    vector.body[vector.body.index(branch)] = branch.orelse[0]
    for name in old:
        assert ast.dump(old[name]) == ast.dump(new[name]), name


def test_layout_stage_matches_the_actual_three_shape_hardware_candidate():
    evidence = Path(__file__).parent / "evidence_tnd_precision/spmd_bf16_pack_20261020/raw.tar.gz"
    with tarfile.open(evidence) as archive:
        measured = ast.parse(archive.extractfile("bf16_input_pack_candidate.py").read())
    expected = next(
        node for node in measured.body if isinstance(node, ast.FunctionDef) and node.name == "hc_pack_input_bf16"
    )
    assert ast.dump(functions(NEW / "input_layout.py")[expected.name]) == ast.dump(expected)


def test_loading_and_workspace_contract_reuses_accepted_runtime():
    service = ast.parse((NEW / "service.py").read_bytes())
    classes = [node for node in service.body if isinstance(node, ast.ClassDef)]
    assert len(classes) == 1 and classes[0].name == "CSAServiceRuntime"
    assert [ast.unparse(base) for base in classes[0].bases] == ["PrepackedRuntime"]
    assert not any(isinstance(node, ast.FunctionDef) for node in ast.walk(classes[0]))
    imports = [node for node in service.body if isinstance(node, ast.ImportFrom)]
    assert len(imports) == 1
    assert imports[0].module == "deepseek_v4_flash_csa_tnd_precision_spmd_prepack.service"
    assert imports[0].level == 2
    adapter = (NEW / "native_adapter.py").read_text()
    assert '"dsv4_csa_tnd_precision_spmd_bf16layout::attention"' in adapter
    assert "return cls(attention, PrepackedWorkspace(prepare))" in adapter
