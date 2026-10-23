# SPDX-License-Identifier: Apache-2.0
"""冻结已验证实现并复用合格历史；不修改生产算子或公共 SDK。"""

import ast
import copy
import io
import json
import shutil
import statistics
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PREVIOUS = HERE.parent / "csa_performance_20261022"
SPMD = REPO.parent / ".worktrees/csa-spmd-tnd-20261020/tests/pypto_test/csa_spmd_tnd_eight_20261022"
MPMD = HERE.parent / "csa_mpmd_aicpu_20261022"
REVISION = "d263bc3f2"


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def replace(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def binding(task):
    assert subprocess.check_output(["task-submit", "--status", task], text=True).strip() == "completed (exit=0)"
    log = subprocess.check_output(["task-submit", "--log", task], text=True)
    path = HERE / "evidence" / (task + ".log")
    path.parent.mkdir(exist_ok=True)
    path.write_text(log)
    records = [
        json.loads(line.split(" ", 1)[1]) for line in log.splitlines() if line.startswith("CSA_SHARED_TOOLCHAIN ")
    ]
    assert len(records) == 1
    assert records[0]["status"] == "validated"
    return records[0], str(path)


def reuse_l1(path, side, task, case, shared, source):
    report = read(path)
    assert report["passed"] and report["stage"] == "complete" and report["case"] == case
    for check in report["checks"].values():
        assert all(check["guards"].values())
        assert all(value["status"] == "PASS" for value in check["numerical"].values())
    windows = [window for window in report["windows"] if window["side"] == side]
    assert len(windows) >= 10
    samples, raw_paths = [], []
    for window in windows:
        raw = read(Path(window["raw"]))
        rows = raw["aicore_tasks"]
        assert raw["chip_swimlane_level"] == 1 and rows
        assert all(row[3] > 0 and row[4] >= row[3] for row in rows)
        value = (max(row[4] for row in rows) - min(row[3] for row in rows)) * 1e6 / raw["metadata"]["clock_freq_hz"]
        assert abs(value - window["incore_span_us"]) < 1e-6
        samples.append(value)
        raw_paths.append(window["raw"])
    actual, log = binding(task)
    assert actual["release"] == shared["release"]
    return dict(
        status="PASS",
        reuse=True,
        samples_us=samples,
        valid_windows=len(samples),
        median_us=statistics.median(samples),
        source_report=str(path),
        source_reports=[str(path)],
        task_id=task,
        source_tasks=[task],
        raw=raw_paths,
        input_contract=case,
        device=report["device_id"],
        actual_toolchain_selection=actual,
        task_log=log,
        source_date="2026-10-22",
        manifest=str(HERE / "manifest.json"),
        source_mapping=source,
        correctness="原始 eager、graph、post-L1 输出/TopK/score/cache/state/保护区全部通过",
        sample_definition="一次完整链路 graph replay 的 L1 首尾跨度；包含全部核及 task 间隙",
    )


def main():
    sources = HERE / "sources"
    sources.mkdir(exist_ok=False)
    shared = read(HERE.parent / "csa_shared_toolchain/manifest.json")
    assert shared["status"] == "validated"
    previous = read(PREVIOUS / "PERFORMANCE.json")
    cases = read(PREVIOUS / "cases.json")
    save(HERE / "cases.json", cases)
    shutil.copyfile(REPO / "vllm_ascend/ops/pypto/CSA_PERFORMANCE_TEST_SPEC.md", HERE / "SPEC.md")

    # 复用已验收 BSH 快照，包含正式 B1 constexpr 与短档临时 Tile 的静态类型修正。
    for package in ("deepseek_v4_flash_csa", "deepseek_v4_flash_csa_spmd"):
        shutil.copytree(PREVIOUS / "bsh_short_sources/pypto" / package, sources / "pypto" / package)
    # TND-MPMD 使用已提交、已通过设备验证的 v9，排除未保留的 v11/v12。
    prefix = "vllm_ascend/ops/pypto/"
    data = subprocess.check_output(["git", "-C", str(REPO), "archive", REVISION, prefix + "deepseek_v4_flash_csa_tnd"])
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for item in archive:
            if item.isfile() and item.name.endswith(".py"):
                dest = sources / "pypto" / item.name.removeprefix(prefix)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(archive.extractfile(item).read())

    # 最新完整设备验证快照：混长组重排 v4；其余七档等价于 combined_v1。
    state = SPMD / "runs/h131072_b16_mixed_score_group_order_v3_state/result.json"
    service = SPMD / "runs/h131072_b16_mixed_score_group_order_v4_service/result.json"
    assert read(state)["passed"] and read(service)["passed"]
    group_audit = read(SPMD / "score_group_order_v3_codegen.json")
    assert group_audit["passed"] and all(v["passed"] for v in group_audit["cases"].values())
    destination = sources / "pypto/deepseek_v4_flash_csa_spmd_tnd"
    destination.mkdir()
    for path in (SPMD / "runs/score_group_order_v4").glob("*.py"):
        shutil.copyfile(path, destination / path.name)

    manifest = dict(
        frozen_at=datetime.now().astimezone().isoformat(),
        source_commit=REVISION,
        spec_source="vllm_ascend/ops/pypto/CSA_PERFORMANCE_TEST_SPEC.md",
        spec_commit="cc9d844a5",
        shared_toolchain=shared,
        sampling=dict(warmup=3, l1_windows_per_side=10, replays_per_window=1, outer_profile_replays=30),
        tnd_mpmd=dict(
            snapshot=REVISION,
            source="sources/pypto/deepseek_v4_flash_csa_tnd",
            architecture="独立 task，AICPU 依赖调度；无算法层 GM 软同步",
            evidence=str(MPMD / "leaf_layout_evidence.json"),
        ),
        tnd_spmd=dict(
            snapshot="score_group_order_v4",
            source=str(SPMD / "runs/score_group_order_v4"),
            state_evidence=str(state),
            service_evidence=str(service),
            codegen_evidence=str(SPMD / "score_group_order_v3_codegen.json"),
            selection="两个原始设备验证报告 complete/PASS；不依赖尚未刷新的 pending 摘要",
        ),
        bsh_mpmd=previous["manifest"]["bsh_mpmd"],
        bsh_spmd=previous["manifest"]["bsh_spmd"],
        bsh_source=str(PREVIOUS / "bsh_short_sources/pypto"),
        native=dict(
            source=str(PREVIOUS / "PERFORMANCE.json"),
            reused_cases=8,
            cann="9.2.0-beta.2",
            sdk_dependency="不依赖 PyPTO/PTOAS/Simpler",
        ),
        scope="仅单卡八档性能规范采集，不修改生产算法，不自动 push",
    )
    save(HERE / "manifest.json", manifest)

    cells = {}
    for case in cases["cases"]:
        old = previous["cells"][case["id"]]["Native"]
        assert old["status"] == "PASS" and old["valid_windows"] >= 10
        row = {"Native": copy.deepcopy(old)}
        row["Native"].update(
            reuse=True,
            historical_manifest=str(PREVIOUS / "manifest.json"),
            reuse_reason=(
                "原 CANN9.2 Native 路径、固定业务输入及原始全设备跨度均符合本规范；公共 PTO SDK 更新不影响 Native"
            ),
        )
        for col in ("BSH-MPMD-L1", "BSH-SPMD", "TND-MPMD-AICPU", "TND-MPMD-L1", "TND-SPMD"):
            row[col] = dict(status="N/A" if case["id"] == "C08" and col.startswith("BSH") else "待采集")
        cells[case["id"]] = row
    c05 = cases["cases"][4]
    a = reuse_l1(
        MPMD / "runs/aicpu_v9_h131072_b24_s6/result.json",
        "mpmd",
        "task_20261022_182109_22496531443",
        c05,
        shared,
        "tnd_mpmd",
    )
    b = reuse_l1(
        MPMD / "runs/aicpu_v11_h131072_b24_s6/result.json",
        "mpmd_control",
        "task_20261022_185532_36059292238",
        c05,
        shared,
        "tnd_mpmd",
    )
    for key in ("samples_us", "raw", "source_reports", "source_tasks"):
        a[key].extend(b[key])
    a.update(
        valid_windows=len(a["samples_us"]),
        median_us=statistics.median(a["samples_us"]),
        additional_task_log=b["task_log"],
        reuse_reason="同一已接受 v9 的两次完整采样，保留全部 20 窗；不使用 v11 候选数据",
    )
    cells["C05"]["TND-MPMD-L1"] = a
    cells["C08"]["TND-SPMD"] = reuse_l1(
        SPMD / "runs/h131072_b16_mixed_score_group_order_v2/result.json",
        "candidate",
        "task_20261022_181543_203120114774",
        cases["cases"][7],
        shared,
        "tnd_spmd",
    )
    cells["C08"]["TND-SPMD"]["reuse_reason"] = (
        "v4 的开启分档与 v2 完整生成代码等价；v3 同址状态及 v4 服务验证通过，原始十窗全部保留"
    )
    save(HERE / "inventory.json", cells)
    save(HERE / "jobs.json", [])
    save(
        HERE / "history_audit.json",
        dict(
            previous_report=str(PREVIOUS / "PERFORMANCE.json"),
            reused="Native 八格；C05 TND-MPMD-L1 二十窗；C08 TND-SPMD 十窗",
            missing="其余 PTO 旧表使用 e891351e SDK，不能改标为 26e94e0；只补缺项",
            rejected="旧 TND-MPMD 含阶段软同步，不是本次选定实现；未接受 v11/v12 不入表",
        ),
    )

    text = (PREVIOUS / "sources/run_pto.py").read_text()
    text = replace(
        text,
        'parser.add_argument("--family", choices=("tnd", "bsh"), default="tnd")',
        'parser.add_argument("--family", choices=("tnd", "bsh"), default="tnd")\n'
        '    parser.add_argument("--skip-l1-side", choices=("mpmd", "spmd"))',
    )
    text = replace(
        text,
        '            save("fixture_ready")',
        '            report["environment"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__,\n'
        '                "cann": os.environ.get("ASCEND_HOME_PATH"), '
        '"hardware": torch.npu.get_device_name(device_id),\n'
        '                "shared_release": os.environ.get("CSA_SHARED_TOOLCHAIN_RELEASE")}\n'
        '            report["skip_l1_side"] = args.skip_l1_side\n'
        '            save("fixture_ready")',
    )
    text = replace(
        text,
        "                for sample in range(10):\n                    order = list(graphs)",
        "                measured_graphs = [side for side in graphs if side != args.skip_l1_side]\n"
        "                for sample in range(10):\n                    order = list(measured_graphs)",
    )
    text = replace(
        text,
        "else list(reversed(graphs))\n                    for side in order:\n"
        "                        restore(fixture)\n                        stream.synchronize()\n"
        "                        pypto.torch.begin_dfx()",
        "else list(reversed(measured_graphs))\n                    for side in order:\n"
        "                        restore(fixture)\n                        stream.synchronize()\n"
        "                        pypto.torch.begin_dfx()",
    )
    text = replace(
        text,
        "            for side in entries:\n                samples =",
        "            for side in measured_graphs:\n                samples =",
    )
    text = replace(
        text,
        '            if "mpmd" in entries:\n                report["ratio"]',
        '            if "mpmd" in measured_graphs and "spmd" in measured_graphs:\n                report["ratio"]',
    )
    ast.parse(text)
    (sources / "run_pto.py").write_text(text)
    for name in ("scenarios.py", "csa_measure.py"):
        shutil.copyfile(PREVIOUS / "sources" / name, sources / name)
    for name in ("bsh_entry.py", "outer_profile.py"):
        shutil.copyfile(PREVIOUS / name, sources / name)
    for name in ("manifest.json", "cases.json"):
        shutil.copyfile(HERE / name, sources / name)
    print(HERE)


if __name__ == "__main__":
    main()
