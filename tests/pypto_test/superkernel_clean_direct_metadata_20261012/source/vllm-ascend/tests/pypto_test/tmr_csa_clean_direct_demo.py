# SPDX-License-Identifier: Apache-2.0
"""Measure direct DMA metadata without changing the CSA kernel or dispatch policy."""

import tmr_csa_full_128k_b16_demo as demo

if __name__ == "__main__":
    for variant in ("direct", "clean_direct"):
        demo.PACKAGES[variant] = demo.PACKAGES["mixed"]
    demo.main()
