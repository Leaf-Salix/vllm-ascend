# SPDX-License-Identifier: Apache-2.0
"""Compare count-mode and fixed normal-mask notification fills on frozen CSA."""

import tmr_csa_full_128k_b16_demo as demo
from tmr_csa_normal_notice_compile import PACKAGES
from tmr_csa_score_scalar_plan_demo import check_metadata_contract

if __name__ == "__main__":
    demo.PACKAGES.update(PACKAGES)
    with check_metadata_contract():
        demo.main()
