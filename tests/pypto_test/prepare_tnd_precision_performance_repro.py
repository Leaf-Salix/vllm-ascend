"""Prepare a fresh FormalV4 job using only verified files; never run NPU work."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

SOURCE_SHA = "6107e6ed747925793dcfe0e5188dee8486c3a8b91d0c1b7e073b62d38215bc28"
HELPER_SHA = "b52d07a160c144704e2cf9def48a03ee5706efd9f155a1765515548d3eb1e24a"
OLD_JOB = "/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/precision-overnight-20261019/whole-formal-v4"
OLD_REL = "$base/precision-overnight-20261019/whole-formal-v4"
OLD_OUT = "/data/pyptouser/yejia/lc128/perf-overnight-formal-v4-$case_name-$backend-20261019"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(root, name, digest):
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError(f"unsafe manifest path: {name}")
    path = root / name
    if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        raise ValueError(f"symlink forbidden: {path}")
    if not path.is_file() or sha(path) != digest:
        raise ValueError(f"missing or changed file: {path}")
    return path


def replace(text, old, new, count):
    if text.count(old) != count:
        raise ValueError(f"unexpected relocation count: {old!r}")
    return text.replace(old, new)


def prepare(reference, destination, label, source_sha=SOURCE_SHA, helper_sha=HELPER_SHA):
    reference = reference.resolve(strict=True)
    destination = destination.absolute()
    if destination.resolve() != destination:
        raise ValueError("destination must be canonical without symlinks/..")
    ascii_alnum = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    if not 1 <= len(label) <= 64 or label[0] not in ascii_alnum or any(c not in ascii_alnum + "_-" for c in label):
        raise ValueError("label must be 1..64 ASCII letters/digits/_/-")
    destination_text = str(destination)
    if (
        len(destination_text) < 2
        or not destination_text.startswith("/")
        or any(c not in ascii_alnum + "_./-" for c in destination_text[1:])
    ):
        raise ValueError("destination contains unsafe shell characters")
    if destination == reference or reference in destination.parents or destination in reference.parents:
        raise ValueError("reference/destination must not overlap")
    if destination.exists():
        raise FileExistsError(destination)
    hm = reference / "helper-manifest.json"
    sm = reference / "snapshot/manifest.json"
    if sha(hm) != helper_sha or sha(sm) != source_sha:
        raise ValueError("reference manifest is not the pinned FormalV4 checkpoint")
    helpers = json.loads(hm.read_text())["files"]
    sources = json.loads(sm.read_text())["files"]
    # Validate everything before creating the destination. No recursive copy.
    for name, digest in helpers.items():
        checked(reference, name, digest)
    for name, digest in sources.items():
        checked(reference / "snapshot/source", name, digest)
    texts = {name: (reference / name).read_text() for name in ("env.sh", "preflight.sh", "run-case.sh")}
    texts["env.sh"] = replace(texts["env.sh"], OLD_REL, str(destination), 3)
    texts["preflight.sh"] = replace(texts["preflight.sh"], OLD_JOB, str(destination), 1)
    texts["run-case.sh"] = replace(texts["run-case.sh"], OLD_REL, str(destination), 1)
    out = f"/data/pyptouser/yejia/lc128/perf-{label}-$case_name-$backend"
    texts["run-case.sh"] = replace(texts["run-case.sh"], OLD_OUT, out, 1)
    # Existing output dirs would fail mkdir at runtime; reject them now as well.
    for case in ("8k", "128k"):
        for backend in ("native", "pto"):
            output = Path(out.replace("$case_name", case).replace("$backend", backend))
            if output.exists():
                raise FileExistsError(output)
    destination.mkdir()  # Exclusive; parent must already exist. No overwrite/cleanup.
    for root, names, target in [
        (reference, helpers, destination),
        (reference / "snapshot/source", sources, destination / "snapshot/source"),
    ]:
        for name, digest in names.items():
            src = checked(root, name, digest)
            dst = target / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            dst.chmod(src.stat().st_mode & 0o777)
            if sha(dst) != digest:
                raise ValueError(f"copy mismatch: {dst}")
    shutil.copyfile(sm, destination / "snapshot/manifest.json")
    shutil.copyfile(hm, destination / "reference-helper-manifest.json")
    for name, text in texts.items():
        (destination / name).write_text(text)
    record = {
        "identity": "FormalV4",
        "reference_job": str(reference),
        "destination": str(destination),
        "label": label,
        "source_manifest_sha256": source_sha,
        "reference_helper_manifest_sha256": helper_sha,
        "output_template": out,
        "relocated_helpers": {name: {"before": helpers[name], "after": sha(destination / name)} for name in texts},
        "source_files": len(sources),
        "note": "prepare_snapshot.py is archived only; snapshot is already populated. No NPU executed.",
    }
    (destination / "reproduction.json").write_text(json.dumps(record, indent=2) + "\n")
    new_helpers = {name: sha(destination / name) for name in helpers}
    new_helpers.update(
        {name: sha(destination / name) for name in ("reference-helper-manifest.json", "reproduction.json")}
    )
    (destination / "helper-manifest.json").write_text(json.dumps({"files": new_helpers}, indent=2) + "\n")
    for name, digest in new_helpers.items():
        checked(destination, name, digest)
    for name, digest in sources.items():
        checked(destination / "snapshot/source", name, digest)
    if sha(destination / "snapshot/manifest.json") != source_sha:
        raise ValueError("source manifest changed")
    print(json.dumps(record, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-job", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    prepare(args.reference_job, args.destination, args.label)


if __name__ == "__main__":
    main()
