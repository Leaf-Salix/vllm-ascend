# SPDX-License-Identifier: Apache-2.0
"""Observe ready-gate paths using the same retained Score arithmetic."""

import tmr_csa_full_128k_b16_compile as compile_demo
import tmr_csa_prefix_coverage as coverage
from tmr_csa_score_fair_compile import PACKAGES
from tmr_csa_score_scalar_plan_demo import check_metadata_contract

if __name__ == "__main__":
    compile_demo.PACKAGES["baseline"] = PACKAGES["plain_score"]
    compile_demo.PACKAGES["mixed"] = PACKAGES["early_score"]
    with check_metadata_contract():
        coverage.main()
