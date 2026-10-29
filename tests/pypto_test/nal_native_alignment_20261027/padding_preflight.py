# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Keep the candidate package separate from the frozen Native control.
"""不初始化设备，核验padding服务入口和冻结Native fixture的兼容性。"""

import importlib
import inspect
import json
import os
import sys
from pathlib import Path

import torch_npu
from bind_package import bind
from padding_fixture import load_fixture_module

bind()
assert not torch_npu.npu.is_initialized()
sys.path.insert(0, str(Path(os.environ["SOURCE_ROOT"]) / "tests/pypto_test"))
frozen_fixture = importlib.import_module("dsv4_csa_single_layer")
fixture_module = load_fixture_module()
service_module = importlib.import_module("nal_alignment.deepseek_v4_flash_csa_spmd_tnd.service")
variant_module = importlib.import_module("vllm_ascend.ops.pypto.variant")
signature = inspect.signature(fixture_module.make_fixture)
for name in ("query_lengths", "token_capacity", "request_capacity"):
    assert name in signature.parameters, name
assert callable(variant_module.csa_runtime)
assert not torch_npu.npu.is_initialized()
out = Path(sys.argv[1])
out.mkdir(exist_ok=False)
record = {
    "pass": True,
    "npu_initialized": False,
    "fixture_source": fixture_module.__file__,
    "fixture_signature": str(signature),
    "frozen_fixture_source": frozen_fixture.__file__,
    "frozen_fixture_signature": str(inspect.signature(frozen_fixture.make_fixture)),
    "service_source": service_module.__file__,
    "variant_source": variant_module.__file__,
    "runtime_class": str(service_module.CSAServiceRuntime),
    "sdk_release": os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"],
}
(out / "receipt.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(record, ensure_ascii=False))
