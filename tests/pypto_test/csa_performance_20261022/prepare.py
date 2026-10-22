# SPDX-License-Identifier: Apache-2.0
"""按用户指定规范冻结八档及已验证实现；不修改生产算子或公共SDK。"""

import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
WORKSPACE = REPO.parent
TND_TREE = WORKSPACE / ".worktrees/csa-spmd-tnd-20261020"
MIXED = [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1, 1, 3, 4, 6]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    source = HERE / "sources"
    if (source / "pypto").exists():
        raise FileExistsError("源码已冻结，禁止覆盖")
    revision = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    packages = ["deepseek_v4_flash_csa", "deepseek_v4_flash_csa_spmd", "deepseek_v4_flash_csa_tnd"]
    prefix = "vllm_ascend/ops/pypto/"
    archive = subprocess.check_output(["git", "-C", str(REPO), "archive", revision, *[prefix + p for p in packages]])
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for item in stream:
            if item.isfile() and item.name.endswith(".py"):
                path = source / "pypto" / item.name.removeprefix(prefix)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(stream.extractfile(item).read())
    # 原BSH正式测试使用150tick的编译期适配；复用已验证的公共SDK原生等价接口。
    # 不把无退避的原始signal_wait误当成已验收的BSH基线。
    signals = source / "pypto/deepseek_v4_flash_csa_spmd/signals.py"
    code = signals.read_text()
    needle = "pl.system.signal_wait(signal, [offset], expected)"
    assert code.count(needle) == 1
    signals.write_text(code.replace(needle, "pl.system.signal_wait(signal, [offset], expected, backoff_ticks=150)"))
    original = TND_TREE / "tests/pypto_test/csa_spmd_tnd_opt_20261022/runs/combined_v1"
    spmd_destination = source / "pypto/deepseek_v4_flash_csa_spmd_tnd"
    spmd_destination.mkdir()
    for path in original.glob("*.py"):
        shutil.copyfile(path, spmd_destination / path.name)
    spec = TND_TREE / "vllm_ascend/ops/pypto/CSA_PERFORMANCE_TEST_SPEC.md"
    shutil.copyfile(spec, HERE / "SPEC.md")
    cases = []
    for index, (history, batch, mixed) in enumerate(
        [
            (131072, 1, False),
            (262144, 16, False),
            (131072, 8, False),
            (131072, 16, False),
            (131072, 24, False),
            (131072, 28, False),
            (8192, 32, False),
            (131072, 16, True),
        ],
        1,
    ):
        q = MIXED if mixed else [6] * batch
        bounds = [0]
        for n in q:
            bounds.append(bounds[-1] + n)
        cases.append(
            dict(
                id=f"C{index:02d}",
                name=f"h{history}_b{batch}" + ("_mixed" if mixed else "_s6"),
                history_label=history,
                active_requests=batch,
                request_capacity=batch,
                query_lengths=q,
                history_lengths=[history] * batch,
                valid_tokens=sum(q),
                token_capacity=sum(q),
                query_start_loc=bounds,
                seq_lens=[history + n for n in q],
                seed=1024,
                table_history=None,
                indexer_page_pattern="original",
                reverse_pages=False,
                dummy_query_lengths=[],
            )
        )
    previous = json.loads((REPO / "tests/pypto_test/csa_mpmd_tnd_20261021/cases.json").read_text())
    save(
        HERE / "cases.json",
        dict(
            model=previous["model"],
            layer=4,
            reference_commit=revision,
            correctness=previous["correctness"],
            cases=cases,
            note="独立性能规范场景；不更改暂停优化目标的八档输入或验收分母",
        ),
    )
    shared = json.loads((REPO / "tests/pypto_test/csa_shared_toolchain/manifest.json").read_text())
    assert shared["status"] == "validated"
    save(
        HERE / "manifest.json",
        dict(
            source_commit=revision,
            spec_source=str(spec),
            spec_commit="16bf3561",
            shared_toolchain=shared,
            tnd_spmd=dict(
                source=str(original),
                snapshot="combined_v1",
                adopted=False,
                evidence=[
                    str(TND_TREE / "tests/pypto_test/csa_spmd_tnd_opt_20261022" / n)
                    for n in ("combined_v1_representative_results.json", "combined_v1_service_summary.json")
                ],
                selection="当前设备完整链路与服务验证最完整的组合；不采用仅CPU通过的combined_v5",
            ),
            tnd_mpmd=dict(snapshot=revision, math_adoption=["275fd6e1", "06dced14"]),
            bsh_spmd=dict(
                snapshot=revision,
                last_kernel_commit="df8d43a81",
                backoff_ticks=150,
                compatibility="已验证native_wait_signal_v1.patch，替代原150tick生成代码适配",
            ),
            bsh_mpmd=dict(snapshot=revision),
            sampling=dict(warmup=3, l1_windows_per_side=10, replays_per_window=1, outer_profile_replays=30),
            scope="单卡单层性能数据采集；不恢复优化Goal、不做16卡、不自动push",
        ),
    )
    from prepare_runner import prepare_runner

    prepare_runner(HERE, REPO)
    print(HERE)


if __name__ == "__main__":
    main()
