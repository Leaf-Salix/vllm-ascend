# SPDX-License-Identifier: Apache-2.0
"""TND两种调度的Native NZ权重ABI及零拷贝准备合同；不初始化NPU。"""

import importlib
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_ascend import envs

pytestmark = pytest.mark.skipif(envs.VLLM_ASCEND_ENABLE_NZ != 2, reason="TND完整链路的已支持配置为NZ2")

VARIANTS = ("deepseek_v4_flash_csa_spmd_tnd", "deepseek_v4_flash_csa_tnd")
NATIVE_SHAPES = {
    "wq_a": (1024, 4096),
    "wq_b": (1024, 32768),
    "wo_a": (8, 4096, 1024),
    "wo_b": (8192, 4096),
    "wkv": (512, 4096),
    "idx_wq_b": (1024, 8192),
    "weights_proj": (64, 4096),
}


def native_fixture(monkeypatch):
    import torch_npu

    def weight(shape, dtype=torch.bfloat16, channels=None, actual=False):
        module = NS(weight=torch.empty(shape, dtype=dtype, device="cpu" if actual else "meta"))
        if channels is not None:
            module.weight_scale = torch.empty(channels, device="meta")
        return module

    def compressor(width):
        return NS(
            wkv=weight((width * 2, 4096)),
            wgate=weight((width * 2, 4096)),
            ape=torch.empty((4, width * 2), device="meta"),
            norm=weight((width,)),
        )

    attention = NS(
        compress_ratio=4,
        n_local_heads=64,
        n_local_groups=8,
        wq_a=weight(NATIVE_SHAPES["wq_a"]),
        wq_b=weight(NATIVE_SHAPES["wq_b"], torch.int8, 32768),
        wo_a=weight(NATIVE_SHAPES["wo_a"]),
        wo_b=weight(NATIVE_SHAPES["wo_b"], torch.int8, 4096),
        wkv=weight(NATIVE_SHAPES["wkv"], actual=True),
        q_norm=weight((1024,)),
        kv_norm=weight((512,)),
        compressor=compressor(512),
        attn_sink=torch.empty(64, device="meta"),
        indexer=NS(
            wq_b=weight(NATIVE_SHAPES["idx_wq_b"], torch.int8, 8192, actual=True),
            weights_proj=weight(NATIVE_SHAPES["weights_proj"], actual=True),
            compressor=compressor(128),
        ),
    )
    native = {name: getattr(attention, name).weight for name in ("wq_a", "wq_b", "wo_a", "wo_b", "wkv")}
    native.update(idx_wq_b=attention.indexer.wq_b.weight, weights_proj=attention.indexer.weights_proj.weight)
    nz_storages = {value.untyped_storage()._cdata for value in native.values()}
    monkeypatch.setattr(
        torch_npu, "get_npu_format", lambda value: 29 if value.untyped_storage()._cdata in nz_storages else 2
    )

    def reject_cast(*_args):
        raise AssertionError("Native NZ根权重必须直接借用，不允许ND中转或重新打包")

    monkeypatch.setattr(torch_npu, "npu_format_cast", reject_cast)
    return attention, native


@pytest.mark.parametrize("package", VARIANTS)
def test_service_preparation_borrows_all_seven_native_weight_storages(package, monkeypatch):
    attention, native = native_fixture(monkeypatch)
    service = importlib.import_module("vllm_ascend.ops.pypto." + package + ".service")
    prepared = service.CSAServiceRuntime._prepare_weights(attention, None)
    for name, original in native.items():
        assert tuple(prepared[name].shape) == NATIVE_SHAPES[name]
        assert prepared[name].dtype == original.dtype
        assert prepared[name].untyped_storage()._cdata == original.untyped_storage()._cdata
        assert prepared[name].storage_offset() == original.storage_offset()
        if original.device.type == "cpu":
            assert prepared[name].data_ptr() == original.data_ptr()


@pytest.mark.parametrize("package", VARIANTS)
@pytest.mark.parametrize("debug", [False, True])
def test_normal_and_debug_roots_declare_native_nz_geometry(package, debug):
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa.nz_mode import root_weight_layouts, root_weight_shapes

    module = importlib.import_module("vllm_ascend.ops.pypto." + package + ".entry")
    symbol = "decode_csa_spmd" if package.endswith("spmd_tnd") else "decode_csa_tnd"
    root = getattr(module, symbol + ("_debug" if debug else ""))
    layouts, shapes = root_weight_layouts(root), root_weight_shapes(root)
    for name, shape in NATIVE_SHAPES.items():
        assert layouts[name] == "NZ"
        assert shapes[name] == shape


@pytest.mark.parametrize("package", VARIANTS)
@pytest.mark.parametrize("name", ["wkv", "idx_wq_b", "weights_proj"])
def test_service_rejects_wrong_native_matrix_direction(package, name, monkeypatch):
    attention, _ = native_fixture(monkeypatch)
    module = attention.wkv if name == "wkv" else getattr(attention.indexer, "wq_b" if name == "idx_wq_b" else name)
    rows, cols = module.weight.shape
    module.weight = torch.empty((cols, rows), dtype=module.weight.dtype, device="meta")
    service = importlib.import_module("vllm_ascend.ops.pypto." + package + ".service")
    with pytest.raises(ValueError, match="Unexpected loaded weight"):
        service.CSAServiceRuntime._prepare_weights(attention, None)
