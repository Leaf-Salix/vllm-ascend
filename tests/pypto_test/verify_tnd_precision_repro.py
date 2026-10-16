# SPDX-License-Identifier: Apache-2.0
"""Verify frozen precision evidence and optionally reconstruct its exact source."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path, PurePosixPath


def checked_members(archive):
    members = archive.getmembers()
    names = set()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.name in names or not member.isfile():
            raise ValueError(f"Unsafe or duplicate archive member: {member.name}")
        if any(part.startswith("._") or part == ".DS_Store" for part in path.parts):
            raise ValueError(f"macOS metadata in archive: {member.name}")
        names.add(member.name)
    return members


def verify_archive(path, expected, destination=None):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
        raise ValueError(f"Archive SHA differs: {path}")
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = checked_members(archive)
        manifest_name = "capsule-manifest.json" if expected["kind"] == "capsule" else "evidence-manifest.json"
        metadata = json.load(archive.extractfile(manifest_name))
        hashes = metadata["files"] if expected["kind"] == "capsule" else metadata
        allowed = set(hashes) | {manifest_name}
        if expected["kind"] != "capsule":
            allowed.add("task-statuses.json")
            statuses = json.load(archive.extractfile("task-statuses.json"))
            if not statuses or any(value.strip() != "completed (exit=0)" for value in statuses.values()):
                raise ValueError("Historical successful cohort has incomplete/failed task status")
        if {member.name for member in members} != allowed or len(hashes) != expected["files"]:
            raise ValueError(f"Unexpected/missing archive members: {path}")
        for name, digest in hashes.items():
            if hashlib.sha256(archive.extractfile(name).read()).hexdigest() != digest:
                raise ValueError(f"Member SHA differs: {name}")
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=False)
            for member in members:
                target = destination / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
    return metadata


def restore_source(repo, capsule, base_commit, destination):
    """Read Git plus the tested overlay; write only the 3337 manifest files."""
    with tarfile.open(capsule, "r:gz") as archive:
        manifest = json.load(archive.extractfile("experiment/root-fix-20261016/manifest.json"))
        expected = manifest["files"]
        destination.mkdir(parents=True, exist_ok=False)
        process = subprocess.Popen(["git", "-C", str(repo), "archive", base_commit], stdout=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as exported:
                for member in exported:
                    if member.name in expected and member.isfile():
                        target = destination / member.name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(exported.extractfile(member).read())
            if process.wait() != 0:
                raise RuntimeError("git archive failed")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()
        for member in archive.getmembers():
            if member.name.startswith("source-overlay/"):
                target = destination / member.name.removeprefix("source-overlay/")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
    actual = {str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file()}
    if actual != set(expected):
        raise ValueError("Reconstructed source contains missing/extra files")
    for name, digest in expected.items():
        if hashlib.sha256((destination / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Reconstructed source SHA differs: {name}")
    return len(expected)


def verify_assets(capsule):
    with tarfile.open(capsule, "r:gz") as archive:
        inventory = json.load(archive.extractfile("external/external-asset-identity.json"))
    groups = [(Path(root), data["files"]) for root, data in inventory["banks"].items()]
    groups.extend((Path(inventory[name]["path"]), inventory[name]["files"]) for name in ("model", "vllm_core"))
    core = Path(inventory["vllm_core"]["path"])
    actual_core = {
        str(path.relative_to(core))
        for path in (core / "vllm").rglob("*")
        if path.is_file() and path.suffix in (".py", ".so") and "__pycache__" not in path.parts
    }
    if actual_core != set(inventory["vllm_core"]["files"]):
        raise ValueError("External vLLM core has missing/extra module files")
    for root, files in groups:
        for name, expected in files.items():
            path = root / name
            stat = path.stat()
            if stat.st_size != expected["bytes"] or stat.st_mtime_ns != expected["mtime_ns"]:
                raise ValueError(f"External asset stat differs: {path}")
            if "sha256" in expected:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    while chunk := stream.read(4 * 1024**2):
                        digest.update(chunk)
                if digest.hexdigest() != expected["sha256"]:
                    raise ValueError(f"External asset SHA differs: {path}")
    print("ASSETS VERIFIED: bank/core hashes; model shard size/mtime only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path, help="New directory for verified evidence; never overwrite")
    parser.add_argument("--restore-source", type=Path, help="New directory for the exact frozen source")
    parser.add_argument("--verify-assets", action="store_true", help="Read/hash existing 227 assets; no NPU")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent / "evidence_tnd_precision/repro_20261016"
    index = json.loads((folder / "index.json").read_text())
    if args.extract is not None:
        args.extract.mkdir(parents=True, exist_ok=False)
    for name, expected in index["archives"].items():
        destination = args.extract / expected["directory"] if args.extract is not None else None
        verify_archive(folder / name, expected, destination)
        print(f"VERIFIED {name}: {expected['files']} files")
    if args.verify_assets:
        verify_assets(folder / "repro-capsule-20261016.tar.gz")
    if args.restore_source is not None:
        count = restore_source(
            Path(__file__).resolve().parents[2],
            folder / "repro-capsule-20261016.tar.gz",
            index["base_git_commit"],
            args.restore_source,
        )
        print(f"RESTORED {count} exact source files; no additional files")


if __name__ == "__main__":
    main()
