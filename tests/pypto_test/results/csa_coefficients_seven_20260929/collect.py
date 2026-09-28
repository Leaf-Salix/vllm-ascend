"""Reuse the established seven-case collector with strict current-package evidence."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_previous(name):
    path = ROOT.parent / "csa_compiled_seven_20260929" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"coefficient_seven_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    return module


def main():
    source = json.loads((ROOT / "source.json").read_text())
    base = load_previous("collect")
    for history, batch in base.CASES:
        for side in ("native", "pto"):
            path = ROOT / f"h{history}_b{batch}" / side / "report.json"
            report = json.loads(path.read_text())
            if report["variant"] != source["variant"]:
                raise ValueError("Current runner must record the exact pkg selector")
            if side == "pto":
                package = source["variant"].removeprefix("pkg:")
                expected = Path(source["source"]) / "vllm_ascend/ops/pypto" / package / "service.py"
                if (report["implementation_source"], report["implementation_package"]) != (
                    str(expected),
                    "vllm_ascend.ops.pypto." + package,
                ):
                    raise ValueError("PTO loaded a different private package")
    base.TASKS = (*base.TASKS, "indexer_head_coefficients")
    base.main()


if __name__ == "__main__":
    main()
