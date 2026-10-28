# SPDX-License-Identifier: Apache-2.0
"""只读分析同一次实际Native/PTO heads快照；不初始化NPU，不反推旋转前值。"""

import hashlib
import json
import sys
from pathlib import Path

import torch

source, output = map(Path, sys.argv[1:])
result = json.loads((source / "result.json").read_text())
assert result["complete"] and result["first_difference"] == "attention_heads_after_inverse_rope"
assert all(row["bit_equal"] for row in result["observation_gate"].values())
assert all(row["bit_equal"] for row in result["native_qli_hook_control"].values())
assert result["precision"]["qli_topk"]["bit_equal"]
payload = torch.load(source / "observation.pt", map_location="cpu", weights_only=True)
actual = payload["attention_heads_after_inverse_rope_actual"].reshape(24, 64, 512)
stages = payload["native_output_stages"]
before = stages["attention"].reshape_as(actual)
after = stages["heads"].reshape_as(actual)
assert all(value.dtype == torch.bfloat16 for value in (actual, before, after))


def metric(a, b):
    a, b = a.contiguous(), b.contiguous()
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    different = a.view(torch.int16) != b.view(torch.int16)
    return {
        "different_elements": int(different.sum()),
        "max_abs": float((a.float() - b.float()).abs().max()),
        "first_coordinates": different.nonzero()[:8].tolist(),
    }


native_nope = metric(before[..., :448], after[..., :448])
assert native_nope["different_elements"] == 0, "Native inverse unexpectedly changes NoPE"
report = {
    "scope": "same 8K/B4/T24 snapshot; CPU bit comparison only, not a new NPU run",
    "source_result_sha256": hashlib.sha256((source / "result.json").read_bytes()).hexdigest(),
    "native_nope_preserved": native_nope,
    "pto_nope_vs_native_preinverse": metric(actual[..., :448], before[..., :448]),
    "pto_rope_vs_native_postinverse": metric(actual[..., 448:], after[..., 448:]),
}
with output.open("x") as stream:
    stream.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(report, ensure_ascii=False))
