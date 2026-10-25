# SPDX-License-Identifier: Apache-2.0
"""隔离初始化成本，并准备只清完整Scalar缓存行的内部单SPMD候选。"""

import ast
import json
import shutil
from pathlib import Path

from prepare import hashes, replace_once


def sparse_headers(source, *, vector=False):
    source = source.replace(
        "pl.Tensor[[VECTORS, PHASE_COUNT * CORES * SIGNAL_SLOT_WORDS // VECTORS], pl.INT32]",
        "pl.Tensor[[PHASE_COUNT * CORES, SIGNAL_SLOT_WORDS], pl.INT32]",
    )
    source = source.replace(
        "pl.reshape(signal, [VECTORS, PHASE_COUNT * CORES * SIGNAL_SLOT_WORDS // VECTORS])",
        "pl.reshape(signal, [PHASE_COUNT * CORES, SIGNAL_SLOT_WORDS])",
    )
    if vector:
        source = replace_once(
            source,
            "from .phase_sync import SIGNAL_SLOT_WORDS, begin_invocation, publish_phase, wait_row_owners",
            "from .phase_sync import SIGNAL_INIT_WORDS, SIGNAL_SLOT_WORDS, "
            "begin_invocation, publish_phase, wait_row_owners",
        )
        source = replace_once(
            source,
            "[1, PHASE_COUNT * CORES * SIGNAL_SLOT_WORDS // VECTORS]",
            "[PHASE_COUNT * CORES // VECTORS, SIGNAL_INIT_WORDS]",
        )
        source = replace_once(
            source,
            "    pl.store(zero, [worker, 0], signal_init_matrix)\n",
            "    pl.store(zero, [worker * (PHASE_COUNT * CORES // VECTORS), 0], signal_init_matrix)\n",
        )
        source = source.replace(
            "    # Exactly the original 48 contiguous, disjoint 11520-word ranges.\n"
            "    # Preserve the full 512-byte phase-slot stride and all padding.\n",
            "    # 每个AIV负责90个互不重叠的512B槽，仅清各槽首个完整64B Scalar缓存行。\n"
            "    # 通知只读取首个INT32；其余448B为隔离填充，保持槽距及全部屏障不变。\n",
        )
    return source


def external_signal_entry(source):
    lines = source.splitlines(True)
    nodes = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name in ("decode_csa_spmd", "decode_csa_spmd_debug")
    ]
    for n in sorted(nodes, key=lambda x: x.lineno, reverse=True):
        lines.insert(n.lineno, "    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],\n")
    source = "".join(lines)
    allocation = "    signal = pl.create_tensor([PHASE_COUNT * CORES * SIGNAL_SLOT_WORDS], dtype=pl.INT32)\n"
    assert source.count(allocation) == 2
    return source.replace(allocation, "")


def external_signal_binding(source):
    source = replace_once(
        source,
        "        policy: CSASPMDPolicy | None = None,\n",
        "        policy: CSASPMDPolicy | None = None,\n        allocator=None,\n",
    )
    marker = "        self.allocations: dict[str, torch.Tensor] = {}  # 外部工作区计数为零。\n"
    replacement = """        # 仅实验控制：signal持久且准备时清零，其他scratch仍由Simpler内部申请。
        from .runtime_abi import CORES, PHASE_COUNT
        from .phase_sync import SIGNAL_SLOT_WORDS

        signal_shape = (PHASE_COUNT * CORES * SIGNAL_SLOT_WORDS,)
        signal = (
            torch.empty(signal_shape, dtype=torch.int32, device=device)
            if allocator is None else allocator("signal", signal_shape, torch.int32, device)
        )
        if device.type != "meta" and signal.data_ptr() % 512:
            raise ValueError("实验signal必须512B对齐")
        signal.zero_()
        self.buffers["signal"] = signal
        self.allocations = {"signal": signal}
"""
    return replace_once(source, marker, replacement)


def main():
    here = Path(__file__).resolve().parent
    parent = here / "candidates/internal_single_f72a"
    sparse = here / "candidates/signal_headers_v1"
    hybrid = here / "candidates/external_signal_diagnostic_v1"
    for target in (sparse, hybrid):
        shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("entry.py", "vector_schedule.py"):
        p = sparse / name
        p.write_text(sparse_headers(p.read_text(), vector=name == "vector_schedule.py"))
    p = sparse / "phase_sync.py"
    p.write_text(
        replace_once(
            p.read_text(),
            "SIGNAL_SLOT_WORDS = 128\n",
            "SIGNAL_SLOT_WORDS = 128\n# 完整64B Scalar DCache行；512B槽中只使用首个INT32。\nSIGNAL_INIT_WORDS = 16\n",
        )
    )
    for name, transform in (("entry.py", external_signal_entry), ("binding.py", external_signal_binding)):
        p = hybrid / name
        p.write_text(transform(p.read_text()))
    for name in ("cube_schedule.py", "vector_schedule.py"):
        p = hybrid / name
        s = p.read_text()
        s = replace_once(s, "    pl.system.syncall(core_type=pl.KernelType.MIX)\n", "")
        if name == "vector_schedule.py":
            begin = s.index("    # Exactly the original 48 contiguous")
            end = s.index("    # Both explicit Group members rendezvous", begin)
            s = s[:begin] + s[end:]
        s = s.replace(
            "    # Both explicit Group members rendezvous before any signal read/write.\n"
            "    # This is hardware FFTS (events 11/12/13), never a GM boot barrier.\n",
            "    # 仅实验控制：持久signal用代际推进；保留缓存维护，删除每调用清零和硬屏障。\n",
        )
        p.write_text(s)
    for target in (sparse, hybrid):
        for file in target.glob("*.py"):
            ast.parse(file.read_text())

    frozen = here / "frozen_optimization"
    shutil.copytree(here / "frozen", frozen, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    source = (frozen / "run_pto.py").read_text()
    source = source.replace("migration_control", "current_control").replace("--migration-control", "--current-control")
    source = source.replace('"migration"', '"current"').replace('"current": "separate"', '"current": "fused"')
    source = source.replace("内部独立初始化", "已保留的内部单SPMD")
    (frozen / "run_pto.py").write_text(source)
    for name in ("launch.sh", "compile.sh"):
        s = (here / name).read_text()
        s = s.replace('"$here/frozen/', '"$here/frozen_optimization/').replace(
            '--frozen "$here/frozen"', '--frozen "$here/frozen_optimization"'
        )
        s = s.replace(
            '--migration-control "$here/candidates/internal_separate_4f9b"',
            '--current-control "$here/candidates/internal_single_f72a"',
        )
        (here / name.replace(".sh", "_optimization.sh")).write_text(s)
    manifest_path = here / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sources"]["current"] = manifest["sources"]["candidate"]
    manifest["new_candidates"] = {
        "signal_headers_v1": {"source": str(sparse), "files": hashes(sparse), "production_eligible_if_validated": True},
        "external_signal_diagnostic_v1": {"source": str(hybrid), "files": hashes(hybrid), "production_eligible": False},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    report = {
        "sparse": {
            "signal_slots": 60 * 72,
            "stride_bytes": 512,
            "initialized_bytes_per_slot": 64,
            "rows_per_AIV": 90,
            "write_bytes": 60 * 72 * 64,
            "prior_write_bytes": 60 * 72 * 512,
            "scope": "初始化所有被读写信号所在完整64B缓存行；未使用的槽填充不再整片写零，数学和全部同步不变",
        },
        "hybrid": {"production_eligible": False, "scope": "仅隔离signal生命周期开销；其他全部scratch依然内部申请"},
        "status": "CPU_gate_pending",
    }
    (here / "init_ablation_candidates.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(sparse, hybrid)


if __name__ == "__main__":
    main()
