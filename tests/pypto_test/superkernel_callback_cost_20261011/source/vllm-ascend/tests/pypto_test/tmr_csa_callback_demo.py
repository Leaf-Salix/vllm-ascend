# SPDX-License-Identifier: Apache-2.0
"""Compare entry-only diagnostics and inline notification with full tail paths."""

import tmr_csa_full_128k_b16_demo as demo

if __name__ == "__main__":
    for name in ("once", "empty", "inline", "recheck"):
        demo.PACKAGES[name] = demo.PACKAGES["mixed"]
    demo.main()
