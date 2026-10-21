"""Recompute saved whole-model token/protocol evidence on CPU; no NPU replay."""

import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(archive, destination):
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert sum(member.size for member in members) < 100_000_000
        for member in members:
            name = Path(member.name)
            assert member.isfile() and not name.is_absolute() and ".." not in name.parts
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source:
                target.write_bytes(source.read())


def verify():
    for name, expected in json.loads((ROOT / "SHA256.json").read_text()).items():
        assert digest(ROOT / name) == expected, name
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        for name in ("native1-whole-v2-text-evidence", "pto1-whole-v2-text-evidence", "sealed-whole-tools"):
            extract(ROOT / f"{name}.tar.gz", work)
        helper = work / "whole-b28-qa168-v2"
        manifest = json.loads((helper / "helper-manifest.json").read_text())
        assert len(manifest["files"]) == 31
        for name, expected in manifest["files"].items():
            assert digest(helper / name) == expected, name
        summary = json.loads((ROOT / "RESULTS.json").read_text())
        assert digest(helper / "helper-manifest.json") == summary["helper_manifest_sha256"]
        assert digest(helper / "canonical-qkv.py") == summary["canonical_QA_sha256"]
        assert digest(helper / "source-manifest.json") == summary["source_manifest_sha256"]
        native = work / "native1"
        candidate = work / "pto1"
        for arm, original, flags in (
            ("native", native / "READY_GOLD.json", []),
            ("pto", candidate / "audit.json", ["--candidate", str(candidate)]),
        ):
            output = work / f"recomputed-{arm}.json"
            subprocess.run(
                [sys.executable, str(helper / "audit.py"), "--native", str(native), *flags, "--output", str(output)],
                check=True,
                capture_output=True,
                text=True,
            )
            assert json.loads(output.read_text()) == json.loads(original.read_text()), arm
        audit = json.loads((candidate / "audit.json").read_text())
        gold = json.loads((native / "READY_GOLD.json").read_text())
        assert digest(candidate / "audit.json") == summary["audit_sha256"]
        assert digest(native / "READY_GOLD.json") == summary["Native_READY_gold_sha256"]
        assert audit["status"] == "PASS_TOKEN_AND_PROTOCOL" and audit["tokens_checked"] == 172032
        assert gold["unique_native_repeat_tokens_compared"] == 86016
        assert all(audit[key] for key in ("tokens_equal", "events_equal", "full_input_trace_equal"))
        for arm in (native, candidate):
            for iteration in range(2):
                for rank in range(16):
                    trace = arm / "trace" / f"round{iteration}" / f"rank{rank}.trace.jsonl"
                    rows = [json.loads(line) for line in trace.read_text().splitlines()]
                    active = [row for row in rows if row["active"]]
                    assert len(active) == 53
                    first = active[0]
                    assert first["graph_mode"] == "NONE" and first["query_lengths"] == [1] * 28
                    assert first["actual_tokens"] == first["padded_tokens"] == 28
                    assert all(row["graph_mode"] == "FULL" for row in active[1:])
                    assert all(set(row["query_lengths"]) == {6} for row in active[1:])
    return {"status": "PASS_SAVED_REAL_MODEL_8K_B28", "token_pairs": 172032, "NPU_replayed": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
