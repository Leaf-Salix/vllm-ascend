"""DeepSeek-V4 CSA attention-only kernel for vLLM's native cache pages.

The public entry is ``decode_csa.decode_csa_attn_tp1_test``. Its 40 arguments
bind vLLM's shared main-state/compressed-KV allocation, independent raw-KV
pages, and packed inner-state/index pages directly. The vendored path does not retain
the former 46-argument ring and slot-mapping ABI.
"""

UPSTREAM_SUBDIR = "models/deepseek_v4_flash_dspark"
ENTRY_MODULES = ("decode_csa",)
