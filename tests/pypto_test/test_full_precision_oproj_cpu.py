# SPDX-License-Identifier: Apache-2.0
"""正式O-proj的T1..256起点表与冻结Native有限查询绑定，CPU无需vLLM。"""

import gzip
import hashlib
import json
import runpy
import tarfile
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_PACKAGE = _ROOT / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd_full_precision"
_EVIDENCE = Path(__file__).parent / "evidence_tnd_precision/full_pto_20261016"


def test_all_oproj_token_orders_match_native_qualification():
    data = runpy.run_path(str(_PACKAGE / "oproj_order_data.py"))
    raw = gzip.decompress((_EVIDENCE / "oproj-native-order-map-v1.json.gz").read_bytes())
    assert hashlib.sha256(raw).hexdigest() == data["NATIVE_OPROJ_ORDER_MAP_SHA256"]
    reference = json.loads(raw)
    profile = runpy.run_path(str(_PACKAGE / "wkv_profile.py"))
    assert reference["native_metadata_sha256"] == profile["QUALIFIED_STATIC_METADATA_SHA256"]
    assert reference["k_block"] == 16 and reference["column_unit"] == 128
    assert reference["order_family"] == "ascending_cyclic_256"
    assert len(data["OPROJ_TOKEN_MODES"]) == 256
    assert len(data["OPROJ_TILE_STARTS"]) == 16
    for tokens in range(1, 257):
        plane = data["OPROJ_TILE_STARTS"][data["OPROJ_TOKEN_MODES"][tokens - 1]]
        native = reference["selected"][str(tokens)]
        assert len(native) == (tokens + 15) // 16
        assert [[list(group) for group in row] for row in plane[: len(native)]] == native
        assert len(plane) == 16 and all(len(row) == 8 for row in plane)
        assert all(
            len(group) == 8 and all(type(k) is int and 0 <= k < 256 for k in group) for row in plane for group in row
        )
        assert all(k == 0 for row in plane[len(native) :] for group in row for k in group)


def test_historical_oproj_program_archive_is_immutable():
    archive = _EVIDENCE / "oproj-final-evidence-v1.tar.gz"
    assert (
        hashlib.sha256(archive.read_bytes()).hexdigest()
        == "87f4e5d3559d413673f0553489d58fb5facf6882238b3ec0977ef623a0e02407"
    )
    with tarfile.open(archive) as bundle:
        source = bundle.extractfile("helpers/oproj-sweep-v1/o_proj.py").read().decode()

    assert (
        hashlib.sha256(source.encode()).hexdigest()
        == "9839f81b42e223737cdf48b115452542f1869588510d883861f5a18767324167"
    )
