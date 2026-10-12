"""Ensure the source-args experiment changes wrappers and capability only."""
import json
import re
import sys
from pathlib import Path

root, baseline = map(Path, sys.argv[1:])
candidate = root / "compiled/early_score/build"
baseline = baseline / "early_score/build"

def body(text):
    text = text.removeprefix('#include "dispatch_payload.h"\n')
    text = text.split("// --- Kernel entry point ---")[0]
    return re.sub(r"^\s*// pto:.*\n", "", text, flags=re.M)

changed = []
kernels = sorted((candidate / "kernels").rglob("*.cpp"))
for source in kernels:
    relative = source.relative_to(candidate)
    if body(source.read_text()) != body((baseline / relative).read_text()):
        changed.append(str(relative))
orch = next((candidate / "orchestration").glob("*.cpp")).read_text()
masks = list(map(int, re.findall(r"\.set_scalar_prefix_tagged_args_mask\((\d+)\)", orch)))
report = {
    "kernels": len(kernels),
    "normalization": "Remove added dispatch header and // pto: SSA comments only; compare before kernel_entry",
    "changed_pre_entry_body": changed,
    "metadata_setter_count": len(masks),
    "metadata_masks": sorted(set(masks)),
}
(root / "body_equivalence.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
assert len(kernels) == 73 and not changed
assert masks and all(mask & 128 and mask & 7 for mask in masks)
assert ".set_scalar_prefix_tagged_args(true)" not in orch
