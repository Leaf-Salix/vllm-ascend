"""DeepSeek-V4 CSA attention-only kernel for vLLM's native cache pages.

The public entry is ``decode_csa.decode_csa_attn_tp1_test``. Its arguments
bind vLLM's raw-KV pool, shared main-state/compressed-KV pool, and packed
inner-state/index pages directly. Historical reads consume native
block tables; current writes consume native raw/compressed/index slot mappings.
The vendored path does not retain the former private ring/repage ABI.
"""

UPSTREAM_SUBDIR = "models/deepseek_v4_flash_dspark"
ENTRY_MODULES = ("decode_csa",)
