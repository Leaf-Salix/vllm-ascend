# SPDX-License-Identifier: Apache-2.0
"""CPU 回归：真实别名、初态恢复、ND/NZ 转换与采集时序。"""

import importlib
import json
from types import SimpleNamespace as NS

import pytest
import torch

from dsv4_csa_replay import capture_tensors, convert_weight_layouts, materialize, restore_mutable_storages
from dsv4_csa_replay import capture_written_pages, restore_written_pages


def test_mixed_dtype_alias_stride_offset_and_restore():
    storage = torch.arange(256, dtype=torch.uint8)
    tensors = {"a": storage.view(torch.int32).as_strided((3, 3), (4, 1), 16),
               "b": storage.view(torch.int16).as_strided((4,), (2,), 36)}
    meta, payload = capture_tensors(tensors, {"a": "inout", "b": "in"}, {}, {"state_timing": "before_call"})
    assert len(payload) == 1
    assert meta["tensors"]["a"]["source_storage_offset"] == 16
    values, backings = materialize(meta, payload)
    assert values["a"].stride() == (4, 1)
    assert values["b"].data_ptr() - values["a"].data_ptr() == 8
    assert values["a"].untyped_storage().data_ptr() == values["b"].untyped_storage().data_ptr()
    for _ in range(2):
        restore_mutable_storages(meta, payload, backings)
        assert torch.equal(values["a"], tensors["a"])
        assert torch.equal(values["b"], tensors["b"])
        values["a"].zero_()
        assert not torch.equal(values["b"], tensors["b"])
    assert torch.equal(storage, torch.arange(256, dtype=torch.uint8))


@pytest.mark.parametrize("dtype,shape", [(torch.bfloat16, (2, 32, 64)), (torch.int8, (32, 64))])
def test_layout_round_trip_and_no_double_pack(dtype, shape):
    value = (torch.arange(torch.tensor(shape).prod()).reshape(shape) % 97).to(dtype)
    initial = {"wo_a": value}
    source = {"state_timing": "before_call"}
    meta, payload = capture_tensors(initial, {"wo_a": "in"}, {"wo_a": "ND"}, source)
    values, _ = materialize(meta, payload)
    packed, changed = convert_weight_layouts(values, meta, {"wo_a": "NZ"})
    assert changed == ["wo_a"]
    assert not torch.equal(packed["wo_a"], value)
    nz_meta, nz_payload = capture_tensors(packed, {"wo_a": "in"}, {"wo_a": "NZ"}, source)
    nz_values, _ = materialize(nz_meta, nz_payload)
    same, changed = convert_weight_layouts(nz_values, nz_meta, {"wo_a": "NZ"})
    assert changed == [] and same["wo_a"] is nz_values["wo_a"]
    restored, _ = convert_weight_layouts(nz_values, nz_meta, {"wo_a": "ND"})
    assert torch.equal(restored["wo_a"], value)


def test_unknown_layout_and_alias_breaking_conversion_are_rejected():
    with pytest.raises(ValueError, match="旧快照"):
        materialize({}, {})
    value = torch.arange(32 * 64).reshape(32, 64).to(torch.int8)
    tensors = {"wo_a": value, "alias": value.view(-1)}
    meta, _ = capture_tensors(tensors, {"wo_a": "in", "alias": "in"}, {"wo_a": "ND"},
                              {"state_timing": "before_call"})
    with pytest.raises(ValueError, match="共享存储"):
        convert_weight_layouts(tensors, meta, {"wo_a": "NZ"})
    meta["weight_layouts"].clear()
    with pytest.raises(ValueError, match="未知权重布局"):
        convert_weight_layouts(tensors, meta, {"wo_a": "NZ"})


def test_online_comparison_restores_written_pages_and_outputs():
    pairs = {"kv_cache": "ori_slot_mapping", "cmp_kv": "cmp_slot_mapping",
             "idx_kv_cache": "idx_slot_mapping", "compress_state": "state_slot_mapping",
             "inner_compress_state": "inner_state_slot_mapping"}
    tensors = {name: torch.zeros(4, 8) for name in pairs}
    tensors.update({slot: torch.tensor([[-1, -1], [2, 0], [2, 1]]) for slot in pairs.values()})
    tensors.update({name: torch.zeros(2) for name in ("idx_topk_scores", "idx_topk", "x_out")})
    initial = capture_written_pages(tensors)
    for name in pairs:
        tensors[name][2].fill_(7)
    tensors["x_out"].fill_(11)
    restore_written_pages(initial)
    assert all(torch.count_nonzero(tensors[name]) == 0 for name in (*pairs, "x_out"))


class _Direction:
    class InOut:
        def __class_getitem__(cls, value):
            return value

    class Out:
        def __class_getitem__(cls, value):
            return value


def _test_root(x_hc: int, state: _Direction.InOut[int], out: _Direction.Out[int]):
    pass


def test_argdump_captures_before_call_and_reference_after_call(monkeypatch, tmp_path):
    from offline_pd.observer import OfflineCSAObserver
    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark import nz_mode
    from vllm_ascend.ops.pypto.variant import variant_package

    class Call:
        def __init__(self, values, *, layer_name):
            self.args = values

        def __call__(self):
            self.args["state"].add_(1)
            self.args["out"].copy_(self.args["x_hc"] + self.args["state"])

    roots = NS(_decode_csa_tp1_layer=_test_root,
               decode_csa_tp1_layer_test=NS(param_names=("x_hc", "state", "out"),
                                          output_param_names=("state", "out"), __name__="_test_root"))
    package = variant_package()
    original_import = importlib.import_module

    def selected_module(name, *args, **kwargs):
        if name == f"{package}.native_adapter":
            return NS(NativeCSACall=Call)
        if name == f"{package}.decode_csa":
            return roots
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", selected_module)
    monkeypatch.setattr(nz_mode, "root_weight_layouts", lambda _: {})
    observer = OfflineCSAObserver()
    observer.vllm_config = NS(parallel_config=NS(data_parallel_rank=0))
    attention = NS(dsa_attn=NS(_pto_csa_runtime=NS(layer_name="test")))
    observer.model_runner = NS(get_model=lambda: NS(model=NS(layers=[NS(self_attn=attention)])))
    observer.offline_begin_argdump(0, str(tmp_path), 2)
    values = {"x_hc": torch.ones(2), "state": torch.zeros(2), "out": torch.zeros(2)}
    Call(values, layer_name="test")()
    assert observer.offline_end_argdump()["dumped"] == 1
    meta = json.loads((tmp_path / "csa_args_meta.json").read_text())
    before, _ = materialize(meta, torch.load(tmp_path / "csa_args.pt", weights_only=True))
    reference = torch.load(tmp_path / "csa_reference.pt", weights_only=True)
    assert before["state"].tolist() == [0, 0]
    assert reference["state"].tolist() == [1, 1]
    assert reference["out"].tolist() == [2, 2]
