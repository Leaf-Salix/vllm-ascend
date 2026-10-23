"""从已封存probe派生Host观测；原文件、数学、SDK及编译资格全部保留。"""

import hashlib
import json
import runpy
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
base = Path("/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/threeway-small-20261022-codex/capture-v2")
manifest = json.loads((here / "manifest.json").read_text())
assert all(hashlib.sha256((here / name).read_bytes()).hexdigest() == value for name, value in manifest.items())
source = (base / "probe_common.py").read_text()
expected = json.loads((base / "manifest.json").read_text())["probe_common.py"]
assert hashlib.sha256(source.encode()).hexdigest() == expected
runpy.run_path(str(base / "sdk_bind.py"), run_name="__sdk_preflight__")
marker = "    def restore_readonly(fixture):"
assert source.count(marker) == 1
insertion = """    from stages import analyze
    analyze(globals())
    report['stage_diagnostic_only']=True
    report['timing_executed']=False
    report['complete']=report['pass']=True
    save()
    raise SystemExit(0)
"""
derived = source.replace(marker, insertion + marker)
out = Path(sys.argv[sys.argv.index("--out") + 1])
out.mkdir(parents=True, exist_ok=False)
(out / "HOST_DERIVATION.json").write_text(
    json.dumps(
        {
            "base_probe_sha256": expected,
            "derived_probe_sha256": hashlib.sha256(derived.encode()).hexdigest(),
            "observation_manifest": manifest,
            "insertion_before": marker,
            "base_directory": str(base),
            "math_changed": False,
        },
        indent=2,
    )
    + "\n"
)
sys.path.insert(0, str(base))
sys.argv[0] = str(base / "probe_common.py")
exec(
    compile(derived, str(base / "probe_common.py"), "exec"),
    {"__file__": str(base / "probe_common.py"), "__name__": "__main__"},
)
