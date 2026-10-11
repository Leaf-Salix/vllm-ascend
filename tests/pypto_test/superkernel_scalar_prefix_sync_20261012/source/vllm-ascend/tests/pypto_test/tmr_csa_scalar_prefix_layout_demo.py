# SPDX-License-Identifier: Apache-2.0
"""Compare manual Scalar-prefix placement with current early-start and ordinary CSA."""

import tmr_csa_full_128k_b16_demo as demo
from tmr_csa_scalar_prefix_layout_compile import PACKAGE

if __name__ == "__main__":
    demo.PACKAGES["current"] = demo.PACKAGES["mixed"]
    demo.PACKAGES["prefix_layout"] = PACKAGE
    demo.main()
