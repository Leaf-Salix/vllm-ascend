# SPDX-License-Identifier: Apache-2.0
"""Time identical Score arithmetic with and without local task handoff."""

import tmr_csa_full_128k_b16_demo as demo
from tmr_csa_score_fair_compile import PACKAGES
from tmr_csa_score_scalar_plan_demo import check_metadata_contract

if __name__ == "__main__":
    demo.PACKAGES.update(PACKAGES)
    with check_metadata_contract():
        demo.main()
