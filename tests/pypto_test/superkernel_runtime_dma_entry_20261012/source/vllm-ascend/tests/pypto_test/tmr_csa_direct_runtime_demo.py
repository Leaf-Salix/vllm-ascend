# SPDX-License-Identifier: Apache-2.0
"""Compare the direct runtime notification with current complete callbacks."""

import tmr_csa_full_128k_b16_demo as demo

if __name__ == "__main__":
    demo.PACKAGES["direct"] = demo.PACKAGES["mixed"]
    demo.main()
