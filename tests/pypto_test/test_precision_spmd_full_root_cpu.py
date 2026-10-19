# SPDX-License-Identifier: Apache-2.0
"""Scope, metadata and model-owned scratch contracts before full-root NPU tests."""

import ast
import importlib.util
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2] / "vllm_ascend/ops/pypto"
PACKAGE = ROOT / "deepseek_v4_flash_csa_tnd_precision_spmd"
BASELINE = ROOT / "deepseek_v4_flash_csa_tnd_full_precision"


def function(path, name):
    return next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_root_retains_business_abi_and_entire_attention_suffix():
    old = function(BASELINE / "decode_csa.py", "_full_csa_tnd")
    new = function(PACKAGE / "decode_csa.py", "_full_csa_tnd_spmd")
    assert len(old.args.args) == 62 and len(new.args.args) == 64
    removed = {
        "legacy_wkv",
        "legacy_weights_proj",
        "gamma_cq",
        "gamma_ckv",
        "cmp_norm_w",
        "inner_norm_w",
        "indexer_order",
        "oproj_start",
    }
    expected_names = [a.arg for a in old.args.args if a.arg not in removed]
    expected_names = [
        "norm_packet" if n == "attn_norm_w" else "order_packet" if n == "wkv_order" else n for n in expected_names
    ]
    assert [a.arg for a in new.args.args[:54]] == expected_names
    old_arguments = {a.arg: ast.dump(a) for a in old.args.args}
    for argument in new.args.args[:54]:
        if argument.arg not in ("norm_packet", "order_packet"):
            assert ast.dump(argument) == old_arguments[argument.arg]

    def suffix(f):
        start = next(
            i
            for i, n in enumerate(f.body)
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "max_seq_len" for t in n.targets)
        )
        body = ast.Module(body=f.body[start:], type_ignores=[])
        for n in ast.walk(body):
            if isinstance(n, ast.Call):
                n.args = [a for a in n.args if not (isinstance(a, ast.Name) and a.id.startswith("legacy_"))]
        return ast.dump(body)

    assert suffix(old) == suffix(new)
    scoped = [
        n
        for n in new.body
        if isinstance(n, ast.With)
        and any(
            isinstance(i.context_expr, ast.Call) and ast.unparse(i.context_expr.func) == "pl.scope" for i in n.items
        )
    ]
    prefix = scoped[0]
    allocations = {
        n.targets[0].id
        for n in prefix.body
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "pl.create_tensor"
    }
    assert allocations == {"wide", "packed_weight", "packed_flat", "mixed_tail"}
    assert any(
        isinstance(n, ast.With) and ast.unparse(n.items[0].context_expr) == "pl.spmd(GROUPS, sync_start=True)"
        for n in prefix.body
    )


def test_service_changes_only_call_local_workspace_binding():
    old = function(BASELINE / "service.py", "__call__")
    new = function(PACKAGE / "service.py", "__call__")
    updates = [
        n
        for n in new.body
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "call_weights.update"
    ]
    assert len(updates) == 1
    assert ast.unparse(updates[0].value.args[0]) == "self.operators.workspace.arguments(hidden)"
    keywords = {k.arg: ast.unparse(k.value) for k in updates[0].value.keywords}
    assert keywords == {
        "norm_packet": "self.norm_packet",
        "order_packet": ("self.operators.workspace.order_packets[mode, indexer_mode, OPROJ_TOKEN_MODES[tokens - 1]]"),
    }
    new.body.remove(updates[0])
    selection = next(n for n in new.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "indexer_mode")
    assert ast.unparse(selection.value) == (
        "select_indexer_mode(tokens, static_qualified=self.wkv_profile.metadata_sha256 is not None)"
    )
    new.body.remove(selection)
    for n in ast.walk(new):
        if isinstance(n, ast.Subscript) and ast.unparse(n) == "self.indexer_orders[indexer_mode]":
            n.slice = ast.parse("INDEXER_TOKEN_MODES[tokens - 1]", mode="eval").body
    fences = [
        n
        for n in new.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "not torch.npu.is_current_stream_capturing()"
    ]
    assert len(fences) == 1
    assert ast.unparse(fences[0].body[0]) == "torch.npu.current_stream(hidden.device).synchronize()"
    new.body.remove(fences[0])
    assert ast.dump(old) == ast.dump(new)


@pytest.mark.parametrize("static_qualified", [False, True])
def test_indexer_static_selection_preserves_all_other_calibrated_shapes(static_qualified):
    import runpy

    data = runpy.run_path(str(BASELINE / "indexer_order_data.py"))
    source = ast.parse((PACKAGE / "service.py").read_text())
    selected = [
        n
        for n in source.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "select_indexer_mode"
        or isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in ("STATIC_INDEXER_TOKENS", "STATIC_INDEXER_MODE") for t in n.targets)
    ]
    namespace = {"INDEXER_TOKEN_MODES": data["INDEXER_TOKEN_MODES"]}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "indexer-profile", "exec"), namespace)
    select = namespace["select_indexer_mode"]
    for tokens, baseline_mode in enumerate(data["INDEXER_TOKEN_MODES"], 1):
        expected = 1 if static_qualified and tokens == 6 else baseline_mode
        assert select(tokens, static_qualified=static_qualified) == expected
    for tokens in (0, 257):
        with pytest.raises(ValueError, match="calibrated range"):
            select(tokens, static_qualified=static_qualified)


@pytest.fixture
def workspace(monkeypatch):
    spec = importlib.util.spec_from_file_location("spmd_workspace_cpu", PACKAGE / "workspace.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = SimpleNamespace(capturing=False, stream=7)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            is_current_stream_capturing=lambda: state.capturing,
            current_stream=lambda device: SimpleNamespace(npu_stream=state.stream),
            synchronize=lambda device: None,
        ),
        raising=False,
    )
    pool = module.PrecisionWorkspace()
    return pool, state


@pytest.mark.parametrize("tokens", [1, 6, 21, 24, 60, 240, 256])
def test_explicit_aliases_and_live_descriptor_shapes(workspace, tokens):
    pool, state = workspace
    pool.prepare("cpu")
    hidden = torch.empty((tokens, 4, 4096), dtype=torch.bfloat16)
    values = pool.arguments(hidden)
    p16 = (tokens + 15) // 16 * 16
    assert values is pool.arguments(hidden)
    for name in ("spmd_square", "spmd_linear", "spmd_accum", "spmd_idx_accum"):
        assert values[name] is values[name + "_read"]
    assert values["spmd_square"].shape == (22 * p16, 16)
    assert values["spmd_linear"].shape == (22 * p16, 32)
    assert values["spmd_accum"].shape == (tokens + 63, 512)
    assert values["spmd_idx_accum"].shape == (p16, 64)
    assert values["vec_ws"].shape == (p16 + 8, 64)
    assert values["spmd_ffts"].dtype == torch.int64 and values["spmd_ffts"].shape == (256,)
    new = function(PACKAGE / "decode_csa.py", "_full_csa_tnd_spmd")
    assert set(values) == {a.arg for a in new.args.args[54:]}
    state.capturing = True
    assert pool.arguments(hidden) is values


def test_no_cold_capture_or_cross_stream_access(workspace):
    pool, state = workspace
    state.capturing = True
    with pytest.raises(RuntimeError, match="before capture"):
        pool.prepare("cpu")
    state.capturing = False
    pool.prepare("cpu")
    hidden = torch.empty((1, 4, 4096), dtype=torch.bfloat16)
    state.capturing = True
    pool.arguments(hidden)
    state.stream = 8
    with pytest.raises(RuntimeError, match="capture stream"):
        pool.arguments(hidden)
    state.capturing = False
    with pytest.raises(RuntimeError, match="capture stream"):
        pool.arguments(hidden)


def test_pool_isolation_and_capacity(workspace):
    pool, state = workspace
    other = type(pool)()
    pool.prepare("cpu")
    other.prepare("cpu")
    assert all(pool.buffers[n].data_ptr() != other.buffers[n].data_ptr() for n in pool.buffers)
    assert sum(t.numel() * t.element_size() for t in pool.buffers.values()) == 1869824
    with pytest.raises(ValueError, match="span devices"):
        pool.prepare("meta")
    with pytest.raises(ValueError, match="real token rows"):
        pool.arguments(torch.empty((257, 4, 4096), dtype=torch.bfloat16))


@pytest.mark.parametrize(
    ("filename", "functions", "dead", "forward"),
    [
        ("qkv_proj_rope.py", ("kv_proj_rope", "qkv_proj_rope"), "wkv", {"kv_proj_rope": 1}),
        ("indexer.py", ("indexer_weights_score", "indexer"), "weights_proj", {"indexer_weights_score": 1}),
    ],
)
def test_only_unused_weight_formals_are_removed(filename, functions, dead, forward):
    original = (
        ROOT / "deepseek_v4_flash_csa_tnd_precision" / ("decode_indexer.py" if filename == "indexer.py" else filename)
    )
    terminal = function(original, functions[0])
    assert not any(
        isinstance(n, ast.Name) and n.id == dead for n in ast.walk(ast.Module(body=terminal.body, type_ignores=[]))
    )
    for name in functions:
        old = function(original, name)
        old.args.args = [a for a in old.args.args if a.arg != dead]
        for n in ast.walk(old):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in forward:
                n.args.pop(forward[n.func.id])
        new = function(PACKAGE / filename, name)
        assert ast.dump(old) == ast.dump(new)


def load_workspace():
    spec = importlib.util.spec_from_file_location("spmd_workspace_packet_cpu", PACKAGE / "workspace.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_norm_packet_preserves_all_original_bits():
    m = load_workspace()
    torch.manual_seed(1)
    weights = {
        name: torch.randint(-32768, 32767, (size,), dtype=torch.int16).view(torch.bfloat16)
        for name, size in m.NORM_PARAMETERS
    }
    snapshots = {name: value.view(torch.uint8).clone() for name, value in weights.items()}
    packet = m.pack_norm_weights(weights)
    assert packet.shape == (6272,) and packet.dtype == torch.bfloat16
    offset = 0
    for name, size in m.NORM_PARAMETERS:
        assert offset * 2 % 32 == 0
        assert torch.equal(packet[offset : offset + size].view(torch.uint8), snapshots[name])
        assert torch.equal(weights[name].view(torch.uint8), snapshots[name])
        offset += size
    weights["gamma_cq"] = weights["gamma_cq"].float()
    with pytest.raises(ValueError, match="original contiguous BF16"):
        m.pack_norm_weights(weights)


def order_data():
    values = {}
    for filename in ("wkv_order_data.py", "indexer_order_data.py", "oproj_order_data.py"):
        for n in ast.parse((BASELINE / filename).read_text()).body:
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
                with suppress(TypeError, ValueError):
                    values[n.targets[0].id] = ast.literal_eval(n.value)
    return values


def test_order_packets_share_only_qualified_byte_identical_tables(workspace):
    pool, state = workspace
    pool.prepare("cpu")
    data = order_data()
    wkv = torch.tensor(
        [[[data["WKV_K_ORDERS"][i] for i in mode]][0] for mode in data["WKV_COLUMN_ORDERS"]], dtype=torch.int32
    )
    indexer = torch.tensor(
        [[[data["INDEXER_K_ORDERS"][i] for i in row] for row in mode] for mode in data["INDEXER_TILE_ORDERS"]],
        dtype=torch.int32,
    )
    oproj = torch.tensor(data["OPROJ_TILE_STARTS"], dtype=torch.int32)
    keys = {
        (
            1 if t == 6 else data["WKV_TOKEN_MODES"][t - 1],
            data["INDEXER_TOKEN_MODES"][t - 1],
            data["OPROJ_TOKEN_MODES"][t - 1],
        )
        for t in range(1, 257)
    }
    assert len(keys) == 33
    original = [t.clone() for t in (wkv, indexer, oproj)]
    pool.prepare_orders(wkv, indexer, oproj, keys)
    packets = dict(pool.order_packets)
    # Static T6 now borrows the already prepared T16 packet. Keep an explicit
    # coverage gate so a future calibration update cannot remove that key.
    for t in range(1, 257):
        selected = (
            1 if t == 6 else data["WKV_TOKEN_MODES"][t - 1],
            1 if t == 6 else data["INDEXER_TOKEN_MODES"][t - 1],
            data["OPROJ_TOKEN_MODES"][t - 1],
        )
        assert selected in packets
    for (w, i, o), packet in packets.items():
        assert packet.shape == (2560,) and packet.dtype == torch.int32
        assert torch.equal(packet[:512], wkv[w].reshape(-1))
        assert torch.equal(packet[512:1536], indexer[i].reshape(-1))
        assert torch.equal(packet[1536:], oproj[o].reshape(-1))
    state.capturing = True
    pool.prepare_orders(wkv, indexer, oproj, keys)
    assert all(pool.order_packets[k] is packet for k, packet in packets.items())
    assert all(torch.equal(a, b) for a, b in zip(original, (wkv, indexer, oproj)))
    with pytest.raises(ValueError, match="profiles changed"):
        pool.prepare_orders(wkv, indexer, oproj, ())
