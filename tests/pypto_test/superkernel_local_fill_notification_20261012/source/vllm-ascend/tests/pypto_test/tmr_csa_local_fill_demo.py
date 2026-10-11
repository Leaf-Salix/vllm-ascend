# SPDX-License-Identifier: Apache-2.0
"""Measure complete CSA with locally filled notification scratch."""

import tmr_csa_full_128k_b16_demo as demo

if __name__ == "__main__":
    demo.PACKAGES["local_fill"] = demo.PACKAGES["mixed"]
    demo.main()
