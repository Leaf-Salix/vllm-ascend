# SPDX-License-Identifier: Apache-2.0
"""Time the callback router and check full CSA state and changed-input replay."""

import tmr_csa_full_128k_b16_demo as demo

if __name__ == "__main__":
    demo.PACKAGES["thin"] = demo.PACKAGES["mixed"]
    demo.main()
