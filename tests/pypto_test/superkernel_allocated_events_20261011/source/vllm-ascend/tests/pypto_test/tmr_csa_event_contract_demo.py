# SPDX-License-Identifier: Apache-2.0
"""Replay current ordinary/MIX CSA and the retained same-SDK notification binary."""

import tmr_csa_full_128k_b16_demo as demo

if __name__ == "__main__":
    demo.PACKAGES["previous"] = demo.PACKAGES["mixed"]
    demo.main()
