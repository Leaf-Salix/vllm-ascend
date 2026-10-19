# SPDX-License-Identifier: Apache-2.0
"""完整CSA诊断进程内的FFTS等待计数：不修改共享依赖或正式算子。"""
import argparse
import os
import importlib
import json
import re
import runpy
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
sys.path.insert(0, str(BASE))

FFTS_HELPER = r"""
static AICORE inline void CSA_PROFILE_FFTS_WAIT(int32_t event,
    uint64_t& kv_ticks, uint64_t& score_ticks, uint64_t& prob_ticks, uint64_t& pv_ticks,
    uint32_t& kv_count, uint32_t& score_count, uint32_t& prob_count, uint32_t& pv_count) {
  // 0/1是Indexer；保持原调用，不把它计入Attention。
  if (event < 2 || event > 5) {
    wait_flag_dev(event);
    return;
  }
  const uint64_t before = get_sys_cnt();
  wait_flag_dev(event);
  const uint64_t delta = get_sys_cnt() - before;
  if (event == 2) { kv_ticks += delta; kv_count += 1; }
  if (event == 3) { score_ticks += delta; score_count += 1; }
  if (event == 4) { prob_ticks += delta; prob_count += 1; }
  if (event == 5) { pv_ticks += delta; pv_count += 1; }
}
"""

MGATHER_HELPER = r"""
template <pto::Coalesce CoalesceMode, pto::GatherOOB OobMode,
          typename Dst, typename Src, typename Idx>
static AICORE inline void CSA_PROFILE_MGATHER(Dst& dst, Src& src, Idx& idx,
    uint64_t& ticks, uint32_t& count) {
  // 保留完整 MGATHER 内部同步；不额外插入 barrier，也不改变正常调用协议。
  const uint64_t before = get_sys_cnt();
  MGATHER<CoalesceMode, OobMode>(dst, src, idx);
  ticks += get_sys_cnt() - before;
  count += 1;
}
"""
FFTS_HELPER += MGATHER_HELPER

COUNTER_ARGS = (
    "csa_kv_ticks, csa_score_ticks, csa_prob_ticks, csa_pv_ticks, "
    "csa_kv_count, csa_score_count, csa_prob_count, csa_pv_count, csa_mgather_ticks, csa_mgather_count"
)

FFTS_COUNTER_ARGS = COUNTER_ARGS.rsplit(", ", 2)[0]

COUNTER_DECLARATIONS = """
  uint64_t csa_kv_ticks = 0, csa_score_ticks = 0, csa_prob_ticks = 0, csa_pv_ticks = 0;
  uint32_t csa_kv_count = 0, csa_score_count = 0, csa_prob_count = 0, csa_pv_count = 0;
  uint64_t csa_mgather_ticks = 0;
  uint32_t csa_mgather_count = 0;
  bool csa_ffts_flushed = false;
"""

COUNTER_STORE = r"""
  // Attention内没有GM phase发布，首个非零计数的发布点就是该角色Attention结束。
  // 每核仅写一次独占第3条缓存行，不触碰用于核间通知的第一条缓存行。
  if (!ffts_flushed && (kv_count + score_count + prob_count + pv_count) > 0) {
    ptr[32] = static_cast<uint32_t>(kv_ticks);
    ptr[33] = static_cast<uint32_t>(kv_ticks >> 32);
    ptr[34] = kv_count;
    ptr[35] = static_cast<uint32_t>(score_ticks);
    ptr[36] = static_cast<uint32_t>(score_ticks >> 32);
    ptr[37] = score_count;
    ptr[38] = static_cast<uint32_t>(prob_ticks);
    ptr[39] = static_cast<uint32_t>(prob_ticks >> 32);
    ptr[40] = prob_count;
    ptr[41] = static_cast<uint32_t>(pv_ticks);
    ptr[42] = static_cast<uint32_t>(pv_ticks >> 32);
    ptr[43] = pv_count;
    ptr[44] = static_cast<uint32_t>(mgather_ticks);
    ptr[45] = static_cast<uint32_t>(mgather_ticks >> 32);
    ptr[46] = mgather_count;
    dcci((__gm__ void*)(ptr + 32), cache_line_t::SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
    ffts_flushed = true;
  }
"""


def install_probe(diagnostic):
    """只改当前Python进程导入的诊断模块；其源文件保持原样。"""
    original_instrument = diagnostic.instrument_clock
    original_export = diagnostic.export_clock
    helpers = diagnostic.CLOCK_HELPERS
    signature = "pto::comm::NotifyOp op, uint64_t total_ticks, uint32_t count)"
    assert helpers.count(signature) == 1
    helpers = helpers.replace(signature, """pto::comm::NotifyOp op, uint64_t total_ticks, uint32_t count,
    uint64_t kv_ticks, uint64_t score_ticks, uint64_t prob_ticks, uint64_t pv_ticks,
    uint32_t kv_count, uint32_t score_count, uint32_t prob_count, uint32_t pv_count,
    uint64_t mgather_ticks, uint32_t mgather_count, bool& ffts_flushed)""")
    marker = "  dsb(DSB_DDR);\n}"
    assert helpers.count(marker) == 1
    helpers = helpers.replace(marker, "  dsb(DSB_DDR);\n" + COUNTER_STORE + "}")
    diagnostic.CLOCK_HELPERS = FFTS_HELPER + helpers

    def instrument(source, *, wait_backoff):
        source, audit = original_instrument(source, wait_backoff=wait_backoff)
        declaration = "  uint32_t csa_wait_count = 0;\n"
        assert source.count(declaration) == 2
        source = source.replace(declaration, declaration + COUNTER_DECLARATIONS)
        # 仅匹配生成体的SSA变量实参，避免把helper内部wait再次包裹成递归。
        pattern = re.compile(r"^(\s*)wait_flag_dev\((v\d+)\);$", re.M)
        source, count = pattern.subn(
            lambda m: f"{m[1]}CSA_PROFILE_FFTS_WAIT({m[2]}, {FFTS_COUNTER_ARGS});", source
        )
        if count != 8:
            raise ValueError(f"FFTS源码站点预期8，实际{count}，须核对新源结构而非强行插桩")
        gather_pattern = re.compile(
            r"^(\s*)MGATHER<pto::Coalesce::Row, pto::GatherOOB::Zero>\((v\d+), (v\d+), (v\d+)\);$", re.M
        )
        source, gather_count = gather_pattern.subn(
            lambda m: f"{m[1]}CSA_PROFILE_MGATHER<pto::Coalesce::Row, pto::GatherOOB::Zero>({m[2]}, {m[3]}, {m[4]}, csa_mgather_ticks, csa_mgather_count);", source
        )
        assert gather_count == 1, "只测当前 Attention 唯一物理行 MGATHER 站点"
        pattern = re.compile(r"^(\s*)CSA_PROFILE_NOTIFY\(([^\n]+)\);$", re.M)
        source, notify_count = pattern.subn(
            lambda m: f"{m[1]}CSA_PROFILE_NOTIFY({m[2]}, {COUNTER_ARGS}, csa_ffts_flushed);", source
        )
        assert notify_count == audit["pto::comm::TNOTIFY"]
        audit.update(ffts_wait_source_sites=count, measured_event_ids=[2, 3, 4, 5],
                     dump_words=[32, 47], mgather_source_sites=gather_count, dump_once_per_role=True,
                     scalar_interval_only=True, eligible_for_performance=False)
        return source, audit

    def export(signal, destination, phases, *, mask, tokens):
        summary = original_export(signal, destination, phases)
        slots = signal.tolist()
        mask_rows = mask.tolist()[:tokens]
        expected = [sum(int(value > 0) for row in mask_rows[core::24] for value in row[:5])
                    for core in range(24)]
        rows = []
        for phase, cores in enumerate(slots):
            phase_name = phases.get(phase, "")
            for core, values in enumerate(cores):
                words = [int(v) & 0xFFFFFFFF for v in values[32:47]]
                right_phase = phase_name.endswith("ATTN_CUBE") if core < 24 else phase_name.endswith("ATTN_VECTOR")
                if not right_phase:
                    if any(words):
                        raise AssertionError(f"非本角色Attention phase出现计数: {phase}/{core}")
                    continue
                group = core if core < 24 else (core - 24) // 2
                # AIV编号由vector_schedule的2*core+lane定义；按真实mask算有效块数。
                expected_counts = [expected[group], 0, expected[group], 0] if core < 24 else [0, expected[group], 0, expected[group]]
                counts = [words[i * 3 + 2] for i in range(4)]
                if counts != expected_counts:
                    raise AssertionError(f"FFTS次数不符: core={core}, actual={counts}, expected={expected_counts}")
                expected_gather = (sum(int(value > 0) for row in mask_rows[group::24] for value in row[1:5])
                                   if core >= 24 else 0)
                assert words[14] == expected_gather, (core, words[14], expected_gather)
                gather_ticks = (words[13] << 32) | words[12]
                ticks = [(words[i * 3 + 1] << 32) | words[i * 3] for i in range(4)]
                rows.append(dict(core=core, role="AIC" if core < 24 else "AIV", group=group,
                                 phase=phase, phase_name=phase_name, counts=counts, ticks=ticks,
                                 mgather_count=words[14], mgather_us=gather_ticks / 50,
                                 observed_us=[t / 50 for t in ticks], expected_valid_blocks=expected[group]))
        if sorted(row["core"] for row in rows) != list(range(72)):
            raise AssertionError("没有导出全部72角色")
        metrics = []
        for role in ("AIC", "AIV"):
            for event, name in enumerate(("KV", "score", "probability", "PV")):
                selected = [r for r in rows if r["role"] == role and r["counts"][event]]
                if selected:
                    times = [r["observed_us"][event] for r in selected]
                    metrics.append(dict(role=role, event_id=event + 2, event=name, roles=len(selected),
                                        calls_per_role=[r["counts"][event] for r in selected],
                                        observed_us=dict(min=min(times), median=statistics.median(times), max=max(times))))
        gather_values = [row["mgather_us"] for row in rows if row["role"] == "AIV"]
        metrics.append(dict(role="AIV", event="MGATHER", roles=len(gather_values),
                            observed_us=dict(min=min(gather_values), median=statistics.median(gather_values), max=max(gather_values))))
        result = dict(passed=True, clock_freq_hz=50000000, tokens=tokens, rows=rows, metrics=metrics,
                      eligible_for_performance=False,
                      scope="原 wait_flag_dev 及完整 MGATHER 的 Scalar 包围计时；含探针和内部等待，不等于独占引擎耗时")
        Path(destination, "attention_supply.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        summary["attention_fifo_wait"] = dict(passed=True, roles=len(rows), metrics=metrics,
                                              path=str(Path(destination, "attention_supply.json")))
        return summary

    diagnostic.instrument_clock = instrument
    diagnostic.export_clock = export
    return instrument


def main():
    diagnostic = importlib.import_module("compile_diagnostics")
    instrument = install_probe(diagnostic)
    if "--audit-only" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--audit-only", action="store_true")
        args = parser.parse_args()
        repo = BASE.parents[3]
        rewrite = runpy.run_path(str(repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/wait_codegen.py"))["rewrite_wait_source"]
        reports = {}
        for case in ("diag_h131072_b28_v1",):
            raw = HERE / case / "codegen_probe/original.cpp"
            source, _ = rewrite(raw.read_text())
            modified, audit = instrument(source, wait_backoff=True)
            reports[case] = dict(raw=str(raw), audit=audit, bytes=len(modified), device_validated=False)
        (HERE / "HOOK_CPU_AUDIT.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(reports, ensure_ascii=False))
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
