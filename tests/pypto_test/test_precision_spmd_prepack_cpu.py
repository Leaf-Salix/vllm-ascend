# SPDX-License-Identifier: Apache-2.0
"""Arithmetic preservation and immutable-weight lifecycle before NPU validation."""

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2] / "vllm_ascend/ops/pypto"
OLD = ROOT / "deepseek_v4_flash_csa_tnd_precision_spmd"
NEW = ROOT / "deepseek_v4_flash_csa_tnd_precision_spmd_prepack"


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_full_root_preserves_every_statement_except_fixed_weight_preparation():
    old = function(OLD / "decode_csa.py", "_full_csa_tnd_spmd")
    new = function(NEW / "decode_csa.py", "_full_csa_tnd_prepacked")
    assert len(old.args.args) == len(new.args.args) == 64
    old.name = new.name
    old.args.args[1] = new.args.args[1]
    assert old.args.args[1].arg == "hc_packed_weight"
    assert ast.unparse(old.args.args[1].annotation) == "pl.Tensor[[32768, 32], pl.BF16]"
    allocations_removed = 0
    for node in ast.walk(old):
        if isinstance(node, ast.With):
            assignments = [
                n for n in node.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "packed_weight"
            ]
            for assignment in assignments:
                assert ast.unparse(assignment.value) == "pl.create_tensor([32768, 32], dtype=pl.BF16)"
                node.body.remove(assignment)
                allocations_removed += 1
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "packed_prefix_indexer_group":
            node.func.id = "prepacked_prefix_indexer_group"
            assert ast.unparse(node.args[0]) == "hc_attn_fn"
            node.args.pop(0)
            for value in node.args:
                if isinstance(value, ast.Name) and value.id == "packed_weight":
                    value.id = "hc_packed_weight"
    assert allocations_removed == 1
    old.body[0] = new.body[0]  # Descriptive root docstring only.
    assert ast.dump(old) == ast.dump(new)


def test_vector_keeps_all_original_math_and_ten_handoffs():
    old = function(OLD / "hc_prefix_indexer.py", "packed_prefix_indexer_vector")
    new = function(NEW / "prefix.py", "prepacked_prefix_indexer_vector")
    old.name = new.name
    old.args.args = [a for a in old.args.args if a.arg not in ("hc_fn", "packed_weight")]
    assignments = [n for n in old.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "packed_weight"]
    assert len(assignments) == 1
    assert ast.unparse(assignments[0].value) == "hc_pack_weight(hc_fn, packed_weight, worker)"
    old.body.remove(assignments[0])
    old.body[-1].value.elts = [n for n in old.body[-1].value.elts if ast.unparse(n) != "packed_weight"]
    assert ast.dump(old) == ast.dump(new)
    assert sum(isinstance(n, ast.Call) and ast.unparse(n.func) == "handoff" for n in ast.walk(new)) == 10
    source = ast.parse((NEW / "prefix.py").read_text())
    assert any(
        isinstance(n, ast.ImportFrom)
        and n.module.endswith("deepseek_v4_flash_csa_tnd_precision_spmd.hc_prefix_indexer")
        and any(a.name == "packed_prefix_indexer_cube" for a in n.names)
        for n in source.body
    )


def test_operator_wrapper_preserves_shared_preparer():
    @dataclass(frozen=True)
    class Operators:
        attention: object
        workspace: object

    class Pool:
        def __init__(self):
            self.initialized = True

    nodes = [n for n in ast.parse((NEW / "native_adapter.py").read_text()).body if isinstance(n, ast.ClassDef)]
    namespace = {"PrecisionOperators": Operators, "PrecisionWorkspace": Pool}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "prepacked-operators", "exec"), namespace)
    prepare = object()
    pool = namespace["PrepackedWorkspace"](prepare)
    original = namespace["CSAOperators"](object(), pool)
    wrapped = type(original)(object(), original.workspace)
    assert wrapped.workspace is pool and pool.prepare_hc_weight is prepare and pool.initialized


@pytest.fixture
def runtime_factory(monkeypatch):
    import weakref

    calls = []

    class Parent:
        def __init__(self, attention, operators, max_num_seqs, layer):
            assert not torch.npu.is_current_stream_capturing()
            calls.append("parent-qualified")
            self.weights = {"hc_attn_fn": layer.hc_attn_fn}

        def __call__(self, *args):
            calls.append("parent-forward")
            return args

    namespace = {"torch": torch, "weakref": weakref, "PrecisionSPMDRuntime": Parent}
    nodes = [
        n
        for n in ast.parse((NEW / "service.py").read_text()).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Assign))
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "prepacked-service", "exec"), namespace)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            is_current_stream_capturing=lambda: False,
            current_stream=lambda device: SimpleNamespace(synchronize=lambda: calls.append("sync")),
        ),
        raising=False,
    )

    def make(inference=False):
        class Layer:
            pass

        layer = Layer()
        with torch.inference_mode(inference):
            layer.hc_attn_fn = torch.nn.Parameter(torch.ones((24, 16384)), requires_grad=False)

        def prepare(source, packed):
            assert calls[-1] == "parent-qualified"
            assert source is layer.hc_attn_fn and tuple(packed.shape) == (32768, 32)
            calls.append("prepare")
            packed.zero_()  # Lifecycle stub only; arithmetic is tested on NPU.

        ops = SimpleNamespace(workspace=SimpleNamespace(prepare_hc_weight=prepare))
        with torch.inference_mode(inference):
            runtime = namespace["CSAServiceRuntime"](None, ops, 40, layer)
        assert calls[-3:] == ["parent-qualified", "prepare", "sync"]
        return runtime, layer, calls, namespace["tensor_identity"]

    return make


@pytest.mark.parametrize("inference", [False, True])
def test_prepare_once_before_capture_and_no_prepare_on_forward(runtime_factory, inference):
    runtime, layer, calls, identity = runtime_factory(inference)
    assert runtime.weights["hc_packed_weight"] is runtime._hc_packed_weight
    assert runtime._hc_packed_weight.numel() * runtime._hc_packed_weight.element_size() == 2097152
    assert not torch.is_inference(runtime._hc_packed_weight)
    assert (identity(layer.hc_attn_fn)[-1] is None) == inference
    for _ in range(3):
        assert runtime(1, 2, 3, 4, 5) == (1, 2, 3, 4, 5)
    assert calls.count("prepare") == calls.count("sync") == 1


@pytest.mark.parametrize("change", ["source-object", "source-version", "packed-object", "packed-version"])
def test_reject_supported_mutations_before_entering_the_root(runtime_factory, change):
    runtime, layer, calls, _ = runtime_factory()
    if change == "source-object":
        layer.hc_attn_fn = layer.hc_attn_fn.clone()
    elif change == "source-version":
        layer.hc_attn_fn.add_(1)
    elif change == "packed-object":
        runtime.weights["hc_packed_weight"] = runtime._hc_packed_weight.clone()
    else:
        runtime._hc_packed_weight.add_(1)
    with pytest.raises(RuntimeError, match="rebuild.*runtime.*graphs"):
        runtime(1, 2, 3, 4, 5)
    assert "parent-forward" not in calls


def test_preparer_preserves_nested_dsl_contexts_after_formatting():
    prepare = function(NEW / "prefix.py", "prepack_fixed_hc_weight")
    outer = prepare.body[0]
    assert isinstance(outer, ast.With) and len(outer.items) == 1
    assert ast.unparse(outer.items[0].context_expr) == "pl.spmd(GROUPS)"
    inner = outer.body[0]
    assert isinstance(inner, ast.With) and len(inner.items) == 1
    assert ast.unparse(inner.items[0].context_expr) == "pl.cluster(name_hint='fixed_hc_weight')"


def test_model_loader_reduction_metadata_is_reexported_from_parent():
    model = (ROOT.parents[1] / "models/pypto_deepseek_v4.py").read_text()
    assert 'importlib.import_module(f"{package}.qkv_proj_rope")' in model
    module = ast.parse((NEW / "qkv_proj_rope.py").read_text())
    imports = [node for node in module.body if isinstance(node, ast.ImportFrom)]
    assert len(imports) == 1
    assert imports[0].level == 2
    assert imports[0].module == "deepseek_v4_flash_csa_tnd_precision_spmd.qkv_proj_rope"
    assert {alias.name for alias in imports[0].names} == {"KV_OK", "QR_OK"}
