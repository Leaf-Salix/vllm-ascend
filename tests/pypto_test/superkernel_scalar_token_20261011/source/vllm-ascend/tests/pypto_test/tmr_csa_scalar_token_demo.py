# SPDX-License-Identifier: Apache-2.0
"""Replay original and Scalar-filled AIV notifications on one private SDK."""

import tmr_csa_full_128k_b16_demo as demo

if __name__ == "__main__":
    for name in ("once", "scalar", "word"):
        demo.PACKAGES[name] = demo.PACKAGES["mixed"]
    demo.main()
