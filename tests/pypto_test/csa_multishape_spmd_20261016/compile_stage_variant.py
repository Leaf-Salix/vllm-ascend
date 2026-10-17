# SPDX-License-Identifier: Apache-2.0
import json
import runpy
import sys
from pathlib import Path
from shape_specialization import STATIC_ROOT
from stage_codegen import independent_stage_codegen

output = Path(sys.argv[sys.argv.index("--output") + 1])
source = Path(sys.argv[sys.argv.index("--source") + 1])
layout = json.loads((source.parent / "layout.json").read_text())
compact = "--compact-stage-inputs" in sys.argv
if compact:
    sys.argv.remove("--compact-stage-inputs")
selected = ("hc_widen_rms",)
if "--stage-kernels" in sys.argv:
    index = sys.argv.index("--stage-kernels")
    selected = tuple(sys.argv[index + 1].split(","))
    del sys.argv[index:index + 2]
with independent_stage_codegen(layout, output / "independent_stages", selected=selected, compact_params=compact):
    runpy.run_path(str(STATIC_ROOT / "compile_jit_entry.py"), run_name="__main__")
