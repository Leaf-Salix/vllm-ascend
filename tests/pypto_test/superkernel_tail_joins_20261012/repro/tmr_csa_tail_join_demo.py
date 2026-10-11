# SPDX-License-Identifier: Apache-2.0
"""Compare full and compiler-proved notification joins on identical CSA code."""

import tmr_csa_full_128k_b16_demo as demo
from tmr_csa_score_scalar_plan_demo import check_metadata_contract
from tmr_csa_tail_join_compile import PACKAGES

if __name__ == "__main__":
    demo.PACKAGES.update(PACKAGES)
    with check_metadata_contract():
        demo.main()
