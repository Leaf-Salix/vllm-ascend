# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Generate uninstrumented control and QKV-aligned kernels for Graph timing."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def generate(repo, output):
    source = repo / "vllm_ascend/attention"
    factors = json.loads(Path(__file__).with_name("factors.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for name in ("baseline", "aligned"):
        target = output / name
        target.mkdir()
        shutil.copy2(source / "pto_attn.py", target / "pto_attn.py")
        shutil.copytree(source / "pto_kernels", target / "pto_kernels", ignore=shutil.ignore_patterns("__pycache__"))
        if name == "aligned":
            path = target / "pto_kernels/dspark/qkv_proj_rope.py"
            text = path.read_text()
            for changes in factors.values():
                for change in changes:
                    anchor = change["anchor"]
                    assert text.count(anchor) == 1, anchor
                    text = text.replace(anchor, anchor + change["insert"])
            path.write_text(text)
        manifest[name] = {
            str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(target.rglob("*.py"))
        }
    changed = [key for key in manifest["baseline"] if manifest["baseline"][key] != manifest["aligned"][key]]
    assert changed == ["pto_kernels/dspark/qkv_proj_rope.py"], changed
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.repo, args.output)
