# SPDX-License-Identifier: Apache-2.0
"""原正式完整 CSA 的 Indexer READY/ACK 等待诊断，不是优化版或性能采样。"""

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import runpy
from statistics import median
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
sys.path.insert(0, str(BASE))

HELPER = r"""
static AICORE inline void CSA_PROFILE_SCORE_WAIT(int32_t event,
    uint64_t& ready_ticks, uint64_t& ack_ticks,
    uint32_t& ready_count, uint32_t& ack_count) {
  // 只观察原 Indexer 0/1。Attention 2..5 保留原调用，不加计时器。
  if (event < 0 || event > 1) {
    wait_flag_dev(event);
    return;
  }
  const uint64_t before = get_sys_cnt();
  wait_flag_dev(event);
  const uint64_t delta = get_sys_cnt() - before;
  if (event == 0) { ready_ticks += delta; ready_count += 1; }
  if (event == 1) { ack_ticks += delta; ack_count += 1; }
}
"""
DECLARATIONS = """
  uint64_t csa_score_ready_ticks = 0, csa_score_ack_ticks = 0;
  uint32_t csa_score_ready_count = 0, csa_score_ack_count = 0;
  bool csa_score_flushed = false;
"""
ARGS = "csa_score_ready_ticks, csa_score_ack_ticks, csa_score_ready_count, csa_score_ack_count"
STORE = r"""
  // 每角色仅在 Score 后的第一个原 phase 发布点导出一次；独占第三条缓存行。
  // 不把计数写入 READY/ACK 协议，也不在所测 wait 周围增加 barrier。
  if (!score_flushed && (ready_count + ack_count) > 0) {
    ptr[32] = static_cast<uint32_t>(ready_ticks);
    ptr[33] = static_cast<uint32_t>(ready_ticks >> 32);
    ptr[34] = ready_count;
    ptr[35] = static_cast<uint32_t>(ack_ticks);
    ptr[36] = static_cast<uint32_t>(ack_ticks >> 32);
    ptr[37] = ack_count;
    dcci((__gm__ void*)(ptr + 32), cache_line_t::SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
    score_flushed = true;
  }
"""


def install_probe(diagnostic):
    original_instrument = diagnostic.instrument_clock
    original_export = diagnostic.export_clock
    helpers = diagnostic.CLOCK_HELPERS
    signature = "pto::comm::NotifyOp op, uint64_t total_ticks, uint32_t count)"
    assert helpers.count(signature) == 1
    helpers = helpers.replace(signature, """pto::comm::NotifyOp op, uint64_t total_ticks, uint32_t count,
    uint64_t ready_ticks, uint64_t ack_ticks, uint32_t ready_count, uint32_t ack_count,
    bool& score_flushed)""")
    marker = "  dsb(DSB_DDR);\n}"
    assert helpers.count(marker) == 1
    diagnostic.CLOCK_HELPERS = HELPER + helpers.replace(marker, "  dsb(DSB_DDR);\n" + STORE + "}")

    def instrument(source, *, wait_backoff):
        source, audit = original_instrument(source, wait_backoff=wait_backoff)
        declaration = "  uint32_t csa_wait_count = 0;\n"
        assert source.count(declaration) == 2
        source = source.replace(declaration, declaration + DECLARATIONS)
        pattern = re.compile(r"^(\s*)wait_flag_dev\((v\d+)\);$", re.M)
        source, count = pattern.subn(
            lambda m: f"{m[1]}CSA_PROFILE_SCORE_WAIT({m[2]}, {ARGS});", source
        )
        if count != 8:
            raise ValueError(f"原源码 FFTS 调用站点应为 8，实际 {count}，不强行插桩")
        notify = re.compile(r"^(\s*)CSA_PROFILE_NOTIFY\(([^\n]+)\);$", re.M)
        source, notify_count = notify.subn(
            lambda m: f"{m[1]}CSA_PROFILE_NOTIFY({m[2]}, {ARGS}, csa_score_flushed);", source
        )
        assert notify_count == audit["pto::comm::TNOTIFY"]
        audit.update(score_wait_source_sites=count, measured_event_ids=[0, 1],
                     dump_words=[32, 38], dump_once_per_role=True,
                     eligible_for_performance=False)
        return source, audit

    def export(signal, destination, phases, *, mask, tokens):
        summary = original_export(signal, destination, phases)
        rows = []
        for phase, cores in enumerate(signal.tolist()):
            phase_name = phases.get(phase, "")
            for core, values in enumerate(cores):
                # runner 原供数探针允许至第46项，这版额外核对38..46仍为零。
                words = [int(v) & 0xFFFFFFFF for v in values[32:47]]
                expected_phase = "SCORE_CUBE" if core < 24 else "SCORE_VECTOR"
                if phase_name != expected_phase:
                    if any(words):
                        raise AssertionError(f"其它 phase 出现 Score 计数: {phase}/{core}")
                    continue
                if any(words[6:]):
                    raise AssertionError(f"Score 导出越过六项区域: {core}")
                ready_count, ack_count = words[2], words[5]
                if core < 24:
                    if ready_count != 0 or ack_count <= 0:
                        raise AssertionError(f"AIC 应只观察正数次 ACK: {core}")
                elif ack_count != 0 or ready_count <= 0:
                    raise AssertionError(f"AIV 应只观察正数次 READY: {core}")
                rows.append(dict(core=core, group=core if core < 24 else (core - 24) // 2,
                                 role="AIC" if core < 24 else "AIV", phase=phase,
                                 ready_count=ready_count, ack_count=ack_count,
                                 ready_us=((words[1] << 32) | words[0]) / 50,
                                 ack_us=((words[4] << 32) | words[3]) / 50))
        by_core = {row["core"]: row for row in rows}
        if sorted(by_core) != list(range(72)):
            raise AssertionError("代表档须导出全部 72 角色")
        for group in range(24):
            if not (by_core[group]["ack_count"] == by_core[24 + 2 * group]["ready_count"]
                    == by_core[25 + 2 * group]["ready_count"]):
                raise AssertionError(f"同组 READY/ACK 消费次数不符: {group}")
        metrics = {}
        for role, field in (("AIC", "ack_us"), ("AIV", "ready_us")):
            times = [row[field] for row in rows if row["role"] == role]
            metrics[role] = dict(min_us=min(times), median_us=median(times), max_us=max(times))
        result = dict(passed=True, rows=rows, metrics=metrics, clock_freq_hz=50000000,
                      tokens=tokens, event_counts_match_per_group=True,
                      eligible_for_performance=False,
                      scope="原 READY/ACK 的 Scalar 包围区间；含探针，不能作为可直接省掉的时间")
        path = Path(destination) / "score_pressure.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        summary["score_pressure"] = dict(passed=True, path=str(path), metrics=metrics)
        return summary

    diagnostic.instrument_clock = instrument
    diagnostic.export_clock = export
    return instrument


def main():
    diagnostic = importlib.import_module("compile_diagnostics")
    instrument = install_probe(diagnostic)
    if "--audit-only" in sys.argv:
        repo = BASE.parents[3]
        rewrite = runpy.run_path(str(repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/wait_codegen.py"))["rewrite_wait_source"]
        raw = HERE / "diag_h131072_b28_v1/codegen_probe/original.cpp"
        source, _ = rewrite(raw.read_text())
        _, audit = instrument(source, wait_backoff=True)
        report = dict(raw=str(raw), audit=audit, device_validated=False)
        (HERE / "SCORE_PRESSURE_HOOK_AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False))
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    known, _ = parser.parse_known_args()
    os.environ["PYPTO_PROG_BUILD_DIR"] = str(known.output.resolve() / "build_output")
    os.environ["PYPTO_CACHE_DIR"] = str(known.output.resolve() / "cache")
    logs = known.output.resolve().parent / (known.output.name + "_device_logs")
    logs.mkdir(parents=True, exist_ok=True)
    os.environ["ASCEND_PROCESS_LOG_PATH"] = str(logs)
    runpy.run_path(str(HERE / "run_supply_device.py"), run_name="__main__")


if __name__ == "__main__":
    main()
