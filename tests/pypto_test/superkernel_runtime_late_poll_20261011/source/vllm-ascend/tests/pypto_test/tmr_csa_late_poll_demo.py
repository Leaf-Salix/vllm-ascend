# SPDX-License-Identifier: Apache-2.0
"""Replay the same frozen CSA under ordinary, one-shot and late-poll protocols."""

import tmr_csa_full_128k_b16_demo as demo

if __name__ == "__main__":
    demo.PACKAGES["once"] = demo.PACKAGES["mixed"]
    demo.main()
