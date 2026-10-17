# SPDX-License-Identifier: Apache-2.0
"""静态包配置开关不是已安装资格；capture 后的次序不得变化。"""

import runpy
from pathlib import Path

import pytest

_PACKAGE = Path(__file__).parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd_full_precision"
_PROFILE = runpy.run_path(str(_PACKAGE / "wkv_profile.py"))
WKVProfile = _PROFILE["WKVProfile"]
STATIC_SHA = _PROFILE["QUALIFIED_STATIC_METADATA_SHA256"]
MODES = runpy.run_path(str(_PACKAGE / "wkv_order_data.py"))["WKV_TOKEN_MODES"]


def select(profile, tokens=6, capturing=False, static_requested=True):
    return profile.select(tokens, MODES, capturing=capturing, static_requested=static_requested)


def test_config_alone_retains_original_warmup_and_rejects_static_capture():
    profile = WKVProfile()
    assert select(profile) == MODES[5] == 0
    with pytest.raises(RuntimeError, match="actual reselection"):
        select(profile, capturing=True)
    assert not profile.captured


def test_qualified_static_only_changes_six_rows_and_is_per_runtime():
    profile = WKVProfile()
    profile.qualify_static(metadata_sha256=STATIC_SHA, reselected=True)
    for tokens in range(1, 257):
        assert select(profile, tokens) == (1 if tokens == 6 else MODES[tokens - 1])
    assert select(WKVProfile()) == 0
    assert select(profile, capturing=True) == 1
    profile.qualify_static(metadata_sha256=STATIC_SHA, reselected=True)


@pytest.mark.parametrize("sha,reselected", [("unknown", True), (STATIC_SHA, False), (STATIC_SHA, 1)])
def test_invalid_qualification_is_rejected(sha, reselected):
    profile = WKVProfile()
    with pytest.raises(ValueError):
        profile.qualify_static(metadata_sha256=sha, reselected=reselected)
    assert profile.metadata_sha256 is None


def test_capture_of_any_bucket_locks_profile_even_before_six_rows():
    profile = WKVProfile()
    assert select(profile, 144, capturing=True, static_requested=False) == MODES[143]
    with pytest.raises(RuntimeError, match="after graph capture"):
        profile.qualify_static(metadata_sha256=STATIC_SHA, reselected=True)
    assert profile.metadata_sha256 is None
