# SPDX-License-Identifier: Apache-2.0
"""Compare ordinary CSA, complete notification and bounded register polling."""

import tmr_csa_full_128k_b16_demo as demo

if __name__ == "__main__":
    demo.PACKAGES["batched"] = demo.PACKAGES["mixed"]
    demo.main()
