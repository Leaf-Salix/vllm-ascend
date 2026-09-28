# SPDX-License-Identifier: Apache-2.0
"""BSH complete attention-half kernel: HC pre, norm, CSA and HC post.

Vendored from nalinaly/vllm-ascend 71153bb3. See SOURCE.json for provenance.
The root accepts and writes the native [tokens, HC streams, hidden] residual.
Production model integration and validation are separate from this import.
"""
