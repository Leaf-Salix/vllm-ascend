# SPDX-License-Identifier: Apache-2.0
"""冻结共同框架和两份完整CSA包；不修改已验收checkout或安装环境。"""

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

NAL = "41ba392f482bf88f7b6a797e2542443c81497338"
PACKAGE = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"


def export(repo, revision, paths, target):
    archive = subprocess.check_output(["git", "-C", str(repo), "archive", revision, "--", *paths])
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for item in stream:
            if item.isfile():
                destination = target / item.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(stream.extractfile(item).read())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ours", required=True, help="明确指定我方已提交数学包版本，不能沿用旧基准标签")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    ours = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--verify", args.ours + "^{commit}"], text=True
    ).strip()
    args.output.mkdir(parents=True, exist_ok=False)
    tracked = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", NAL], text=True)
    paths = []
    for name in tracked.splitlines():
        if name.startswith("vllm_ascend/") and name.endswith((".py", ".json", ".yaml")):
            paths.append(name)
        elif name.startswith(("tests/pypto_test/offline_pd/", "tests/pypto_test/low_acceptance_20261001/")):
            if name.endswith((".py", ".sh")):
                paths.append(name)
        elif (
            name.startswith("tests/pypto_test/csa_shared_toolchain/")
            and name.endswith((".py", ".sh", ".json"))
            or name.startswith("tests/pypto_test/dsv4_")
            and name.endswith(".py")
        ):
            paths.append(name)
    paths += [
        "tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py",
        "tests/pypto_test/csa_tnd_model_validation_20261023/observer.py",
    ]
    for arm in ("native", "ours", "nalinaly"):
        source = args.output / arm / "source"
        export(repo, NAL, sorted(set(paths)), source)
        if arm == "ours":
            shutil.rmtree(source / PACKAGE)
            export(repo, ours, [PACKAGE], source)
        helper = source / "tests/pypto_test/csa_threeway_20261029"
        helper.mkdir(parents=True)
        for file in here.iterdir():
            if file.suffix in (".py", ".sh"):
                shutil.copy2(file, helper / file.name)
        runner = source / "tests/pypto_test/offline_pd/run.py"
        text = runner.read_text()
        original = '"worker_extension_cls": "offline_pd.observer.OfflineCSAObserver"'
        assert text.count(original) == 1
        runner.write_text(
            text.replace(original, '"worker_extension_cls": "csa_threeway_20261029.observer.TimedObserver"')
        )
        # The outer bounded launcher owns one session. Keep ranks and their
        # multiprocessing workers inside it so first-difference cleanup reaches all.
        text = runner.read_text()
        assert text.count("start_new_session=True") == 1
        original_cleanup = """        for child in children:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
"""
        cleanup = """        for child in children:
            if child.poll() is None:
                child.terminate()
        deadline = time.monotonic() + 3
        while any(child.poll() is None for child in children) and time.monotonic() < deadline:
            time.sleep(0.1)
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
"""
        assert text.count(original_cleanup) == 1
        runner.write_text(
            text.replace("start_new_session=True", "start_new_session=False").replace(original_cleanup, cleanup)
        )
        text = runner.read_text()
        scheduler = '"scheduler_cls": "low_acceptance_20261001.scheduler.AcceptanceScheduler"'
        memory = "gpu_memory_utilization=args.gpu_memory_utilization, block_size=32,"
        assert text.count(scheduler) == text.count(memory) == 1
        runner.write_text(
            text.replace(scheduler, '"scheduler_cls": "csa_threeway_20261029.scheduler.CheckedScheduler"').replace(
                memory, memory + "\n        num_gpu_blocks_override=22000,"
            )
        )
        files = {
            str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "arm": arm,
            "common_framework": NAL,
            "csa_source": ours if arm == "ours" else NAL,
            "math_overlay": PACKAGE if arm == "ours" else None,
            "backend": "native" if arm == "native" else "pto",
            "pto_variant": "spmd_tnd",
            "pto_backend": "python",
            "HCA": "native",
            "gpu_blocks_override": 22000,
            "files": files,
        }
        (source.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(arm, len(files), source)


if __name__ == "__main__":
    main()
