# SPDX-License-Identifier: Apache-2.0
"""Measure existing maximum reuse separately from early Scalar planning."""

import json
from contextlib import contextmanager

import tmr_csa_full_128k_b16_demo as demo
from tmr_csa_score_scalar_plan_compile import PACKAGES


@contextmanager
def check_metadata_contract():
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_scalar_overlap_demo import native_adapter

    original = native_adapter.NativeCSACall

    def checked_call(*args, **kwargs):
        call = original(*args, **kwargs)
        lengths = call.args["kv_seq_lens"]
        assert call.args["position_ids"].numel() == lengths.numel() * 6
        # The demo intentionally certifies the fixture before graph capture.
        # Exact aliases are legal in the general ABI, so direction alone is
        # insufficient to establish immutable lengths for this experiment.
        lo = lengths.data_ptr()
        hi = lo + lengths.numel() * lengths.element_size()
        aliases = []
        for name, value in call.args.items():
            if name == "kv_seq_lens" or not hasattr(value, "data_ptr"):
                continue
            other_lo = value.data_ptr()
            other_hi = other_lo + value.numel() * value.element_size()
            if max(lo, other_lo) < min(hi, other_hi):
                aliases.append(name)
        # cmp_seq_lens may share the same immutable Native lengths tensor.
        assert set(aliases) <= {"cmp_seq_lens"}, aliases
        print("SCORE_SCALAR_METADATA_CONTRACT " + json.dumps({
            "requests": lengths.numel(), "queries_per_request": 6,
            "readonly_aliases": aliases, "new_host_scan": False,
        }), flush=True)
        return call

    native_adapter.NativeCSACall = checked_call
    try:
        yield
    finally:
        native_adapter.NativeCSACall = original


if __name__ == "__main__":
    demo.PACKAGES["current"] = demo.PACKAGES["mixed"]
    demo.PACKAGES.update(PACKAGES)
    with check_metadata_contract():
        demo.main()
