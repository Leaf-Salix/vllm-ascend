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

COMMON_FRAMEWORK = "41ba392f482bf88f7b6a797e2542443c81497338"
NAL = "79ea6d3c8460338a172cb5570d14999d5a544262"
PACKAGE = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"


def adapt_eight_card_tests(source):
    """只在新冻结测试副本中允许CSA八卡；模型与生产算子保持原样。"""
    runner = source / "tests/pypto_test/offline_pd/run.py"
    text = runner.read_text()
    replacements = [
        (
            'if args.mixed_requests and (args.command != "decode" or args.decode_dp != 16',
            'if args.mixed_requests and (args.command != "decode" or args.decode_dp not in (8, 16)',
        ),
        (
            'args.pto_attention not in ("hca", "both")):\n        parser.error("DP/EP8',
            'args.pto_attention not in ("csa", "hca", "both")):\n        parser.error("DP/EP8',
        ),
    ]
    for original, replacement in replacements:
        assert text.count(original) == 1, original
        text = text.replace(original, replacement)
    runner.write_text(text)
    mixed = source / "tests/pypto_test/low_acceptance_20261001/mixed.py"
    text = mixed.read_text()
    assert text.count("range(16)") == 2
    mixed.write_text(text.replace("range(16)", "range(8)"))


def adapt_precision_observer(source):
    observer = source / "tests/pypto_test/csa_tnd_model_validation_20261023/observer.py"
    text = observer.read_text()
    original = "self._tnd_execution_routes[(*route, replayed, resubmitted)] += 1"
    replacement = """all_submitted = all(runtime.submissions > old[0]
                                    for runtime, old in zip(self._tnd_runtimes, before))
                self._tnd_execution_routes[(*route, replayed, resubmitted, all_submitted)] += 1"""
    assert text.count(original) == 1
    text = text.replace(original, replacement)
    original = 'if not context.capturing and not getattr(context, "is_draft_model", False):'
    assert text.count(original) == 1
    text = text.replace(
        original,
        'if (not context.capturing and not getattr(context, "is_draft_model", False)\n'
        '                    and not getattr(self, "_threeway_in_dummy_batch", False)):',
    )
    original = "python_resubmitted=resubmitted,"
    assert text.count(original) == 1
    text = text.replace(original, original + "\n                all_csa_submitted=all_submitted,")
    original = "for (b, t, capacity, length, mode, replayed, resubmitted), steps"
    assert text.count(original) == 1
    observer.write_text(
        text.replace(original, "for (b, t, capacity, length, mode, replayed, resubmitted, all_submitted), steps")
    )
    runner = source / "tests/pypto_test/offline_pd/run.py"
    text = runner.read_text()
    original = 'if __name__ == "__main__":'
    assert text.count(original) == 1
    runner.write_text(
        text.replace(
            original,
            "from csa_threeway_20261029.runtime import install_capacity_guard\ninstall_capacity_guard()\n\n" + original,
        )
    )


def adapt_request_waves(source, wave_size):
    mixed = source / "tests/pypto_test/low_acceptance_20261001/mixed.py"
    text = mixed.read_text()
    end_observation = "llm.collective_rpc('offline_end_observation')"
    begin_observation = "    llm.collective_rpc('offline_begin_observation')\n"
    replacements = [
        ("        llm.collective_rpc('offline_begin_observation')\n", ""),
        (
            "    for offset in range(0, len(chosen), args.batch):",
            begin_observation + f"    for offset in range(0, len(chosen), {wave_size}):",
        ),
        ("chunk = chosen[offset:offset + args.batch]", f"chunk = chosen[offset:offset + {wave_size}]"),
        (
            "        observation = llm.collective_rpc('offline_end_observation')",
            f"        observation = {end_observation} if offset + {wave_size} >= len(chosen) else []",
        ),
    ]
    for original, replacement in replacements:
        assert text.count(original) == 1, original
        text = text.replace(original, replacement)
    mixed.write_text(text)


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
    parser.add_argument("--decode-dp", type=int, choices=(8, 16), default=16)
    parser.add_argument("--gpu-blocks", type=int, default=22000)
    parser.add_argument("--precision-only", action="store_true")
    parser.add_argument("--request-wave-size", type=int, choices=(8, 40), default=40)
    parser.add_argument(
        "--arms", nargs="+", choices=("native", "ours", "nalinaly"), default=("native", "ours", "nalinaly")
    )
    args = parser.parse_args()
    assert args.gpu_blocks > 0
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    ours = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--verify", args.ours + "^{commit}"], text=True
    ).strip()
    args.output.mkdir(parents=True, exist_ok=False)
    tracked = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", COMMON_FRAMEWORK], text=True
    )
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
    for arm in args.arms:
        source = args.output / arm / "source"
        export(repo, COMMON_FRAMEWORK, sorted(set(paths)), source)
        # Our validated test bootstrap adds explicit Native-source selection;
        # the public Nal bootstrap lacks --native-plugin-root. Share this test
        # helper across all arms without changing the common model framework.
        export(repo, ours, ["tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py"], source)
        if arm == "ours":
            shutil.rmtree(source / PACKAGE)
            export(repo, ours, [PACKAGE], source)
        elif arm == "nalinaly":
            shutil.rmtree(source / PACKAGE)
            export(repo, NAL, [PACKAGE], source)
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
                memory, memory + f"\n        num_gpu_blocks_override={args.gpu_blocks},"
            )
        )
        if args.decode_dp == 8:
            adapt_eight_card_tests(source)
        if args.precision_only:
            adapt_precision_observer(source)
        if args.request_wave_size != 40:
            assert args.precision_only
            adapt_request_waves(source, args.request_wave_size)
        files = {
            str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "arm": arm,
            "common_framework": COMMON_FRAMEWORK,
            "test_bootstrap_source": ours,
            "csa_source": ours if arm == "ours" else NAL if arm == "nalinaly" else COMMON_FRAMEWORK,
            "math_overlay": PACKAGE if arm != "native" else None,
            "nalinaly_reference": NAL,
            "backend": "native" if arm == "native" else "pto",
            "pto_variant": "spmd_tnd",
            "pto_backend": "python",
            "HCA": "native",
            "decode_dp": args.decode_dp,
            "precision_only": args.precision_only,
            "request_wave_size": args.request_wave_size,
            "test_adaptations": ["CSA decode DP8 CLI gate", "eight-rank mixed barrier and analysis"]
            if args.decode_dp == 8
            else [],
            "gpu_blocks_override": args.gpu_blocks,
            "files": files,
        }
        (source.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(arm, len(files), source)


if __name__ == "__main__":
    main()
