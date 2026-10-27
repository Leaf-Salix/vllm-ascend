# SPDX-License-Identifier: Apache-2.0
"""冻结本轮缺项的原算子与测试入口，不修改生产算法或共享SDK。"""

import ast
import difflib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OLD = HERE.parent / "csa_performance_20261022"


def replace(text, old, new, count=1):
    assert text.count(old) == count, (old, text.count(old))
    return text.replace(old, new)


def main():
    source = HERE / "sources"
    source.mkdir(exist_ok=False)
    revision = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    shared = json.loads((HERE.parent / "csa_shared_toolchain/manifest.json").read_text())
    assert shared["status"] == "validated"
    for name in ("cases.json", "outer_profile.py", "bsh_entry.py"):
        shutil.copy2(OLD / name, source / name)
    shutil.copy2(OLD / "cases.json", HERE / "cases.json")
    shutil.copy2(REPO / "vllm_ascend/ops/pypto/CSA_PERFORMANCE_TEST_SPEC.md", HERE / "SPEC.md")
    for name in ("scenarios.py", "csa_measure.py"):
        shutil.copy2(OLD / "sources" / name, source / name)
    prefix = "vllm_ascend/ops/pypto/"
    archive = subprocess.check_output(
        [
            "git",
            "-C",
            str(REPO),
            "archive",
            revision,
            *[
                prefix + name
                for name in ("deepseek_v4_flash_csa", "deepseek_v4_flash_csa_tnd", "deepseek_v4_flash_csa_spmd_tnd")
            ],
        ]
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for item in stream:
            if item.isfile() and item.name.endswith(".py"):
                target = source / "tnd" / "pypto" / item.name.removeprefix(prefix)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(stream.extractfile(item).read())
    # 原BSH已通过的150tick及短档临时Tile兼容版本，不改变算术/分档。
    for name in ("deepseek_v4_flash_csa", "deepseek_v4_flash_csa_spmd", "deepseek_v4_flash_csa_spmd_tnd"):
        shutil.copytree(
            OLD / "bsh_short_sources/pypto" / name,
            source / "bsh/pypto" / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    original = (OLD / "sources/run_pto.py").read_text()
    text = replace(original, "import importlib\n", "import importlib\nimport inspect\nimport platform\n")
    text = replace(
        text,
        'parser.add_argument("--family", choices=("tnd", "bsh"), default="tnd")',
        'parser.add_argument("--family", choices=("tnd", "bsh"), default="tnd")\n'
        '    parser.add_argument("--measure-sides", nargs="+", choices=("spmd", "mpmd"), default=["mpmd"])',
    )
    text = replace(
        text,
        "            entries, guards = {}, {}",
        "            entries, guards = {}, {}\n"
        '            report["shared_toolchain"] = json.loads(Path(os.environ["CSA_SHARED_TOOLCHAIN_MANIFEST"]).read_text())\n'  # noqa: E501
        '            report["sampling"] = dict(warmup=3, l1_windows=10, replays_per_window=1, measured_sides=args.measure_sides)\n'  # noqa: E501
        '            report["environment"] = dict(python=platform.python_version(), torch=torch.__version__, torch_npu=torch_npu.__version__,\n'  # noqa: E501
        '                cann=os.environ.get("ASCEND_HOME_PATH"), device=torch.npu.get_device_name(device_id),\n'
        '                deterministic_level=0, atomic_add=0, nz_mode=2, runtime="tensormap_and_ringbuffer")',
    )
    text = replace(
        text,
        """                    entries[side] = importlib.import_module(package + ".binding").PreparedDynamicCSA(
                        native, max_query_tokens=max_query_tokens,
                        max_seq_len=max(case["seq_lens"]), allocator=allocate,
                    )""",
        """                    prepared = importlib.import_module(package + ".binding").PreparedDynamicCSA
                    extra = {"allocator": allocate} if "allocator" in inspect.signature(prepared).parameters else {}
                    entries[side] = prepared(native, max_query_tokens=max_query_tokens,
                                             max_seq_len=max(case["seq_lens"]), **extra)""",
    )
    text = replace(
        text,
        '                    runtime="tensormap_and_ringbuffer",',
        '                    runtime="tensormap_and_ringbuffer",\n'
        '                    ring_heap=[16 * 1024**2, 256 * 1024**2, 16 * 1024**2, 16 * 1024**2] if args.family == "tnd" else None,\n'  # noqa: E501
        '                    ring_task_window=[4096] * 4 if args.family == "tnd" else None,',
    )
    text = replace(
        text,
        "                for sample in range(3):",
        "                measured_graphs = {side: graphs[side] for side in args.measure_sides}\n"
        "                for sample in range(3):",
    )
    text = replace(
        text,
        "order = list(graphs) if sample % 2 == 0 else list(reversed(graphs))",
        "order = list(measured_graphs) if sample % 2 == 0 else list(reversed(measured_graphs))",
        count=2,
    )
    text = replace(
        text,
        "            for side in entries:\n                samples =",
        "            for side in args.measure_sides:\n                samples =",
    )
    text = replace(
        text,
        '            if "mpmd" in entries:\n                report["ratio"]',
        '            if "mpmd" in report["statistics_us"] and "spmd" in report["statistics_us"]:\n                report["ratio"]',  # noqa: E501
    )
    ast.parse(text)
    (source / "run_pto.py").write_text(text)
    (HERE / "runner.patch").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                text.splitlines(True),
                fromfile="20261022/run_pto.py",
                tofile="20261025/run_pto.py",
                n=0,
            )
        )
    )
    manifest = dict(
        source_commit=revision,
        shared_toolchain=shared,
        no_algorithm_changes=True,
        tnd=dict(
            source=str(source / "tnd/pypto"),
            ring_heap_mb=[16, 256, 16, 16],
            task_window=4096,
            measure=["TND-MPMD-L1", "TND-MPMD-AICPU"],
            reference="f72a1f752;仅正确性，不重复采SPMD性能",
        ),
        bsh=dict(
            source=str(source / "bsh/pypto"),
            parent=str(OLD / "bsh_short_sources/pypto"),
            ring="沿用原默认",
            measure=["BSH-MPMD-L1", "BSH-SPMD"],
            compatibility="已验收150tick及短历史临时Tile声明；使用csa_compile_constants以保留B1分档",
        ),
        native="复用已有CANN9.2/static/SuperKernel/多流/NZ2合格原始Profiler记录",
        tnd_spmd="复用已采用f72a/init_coeff_single_spmd_v2八档完整L1及同配置合格控制组",
        exclusions=[
            "旧PTO 26e94e056 SDK记录不改标d44925c1",
            "纯PyTorch复现入口尚无与Native fixture逐项等价证据，不用于本表",
            "诊断插桩、L2/GM扰动、失败或未采用候选不用于性能表",
        ],
    )
    (HERE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (source / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(source)


if __name__ == "__main__":
    main()
