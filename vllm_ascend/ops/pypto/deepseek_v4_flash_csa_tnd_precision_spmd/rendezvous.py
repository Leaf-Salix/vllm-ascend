# SPDX-License-Identifier: Apache-2.0
"""Conservative full-A3 MIX handoff using the already installed DSL.

All 24 AIC and 48 AIV participants execute every rendezvous, including workers
with zero rows. HARD syncall requires full physical occupancy. This first
precision slice uses no custom signal interface, polling hook or SDK changes.
"""

import pypto.language as pl

GROUPS = 24


@pl.jit.inline(auto_scope=False)
def handoff():
    # Complete FIX/MTE3, publish all payload lines, then rendezvous and acquire.
    pl.system.bar_all()
    pl.system.cacheinvalid()
    pl.system.fence()
    pl.system.syncall(core_type=pl.KernelType.MIX)
    pl.system.cacheinvalid()
    pl.system.fence()
