"""Compile the candidate's CPU checks without editing a live SDK checkout."""

import re
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

output = Path(__file__).resolve().parent
root = output.parents[1] / "simpler"
build = root / "build/scalar-overlap-cpp-ut"
target = build / "CMakeFiles/test_tmr_scheduler_execution_inputs.dir"
scratch = output / "cpu"
runtime = root / "src/a2a3/runtime/tensormap_and_ringbuffer"
shutil.copytree(runtime, scratch, dirs_exist_ok=True)
patch = (output / "candidate.patch").read_text()
patch = "".join(p for p in re.split(r"(?=^--- a/)", patch, flags=re.M) if "/aicore/aicore_executor.cpp" not in p)
patch = patch.replace("src/a2a3/runtime/tensormap_and_ringbuffer/", "")
subprocess.run(["git", "apply", "--unsafe-paths", "--directory=" + str(scratch), "-"],
               input=patch, text=True, cwd=root, check=True)
test = (root / "tests/ut/cpp/common/test_tmr_scheduler_execution_inputs.cpp").read_text()
anchor = "#if defined(TEST_TMR_SCALAR_PREFIX)\n"
assert test.count(anchor) == 1
test = test.replace(anchor, anchor + (output / "transition_tests.inc").read_text() + "\n")
# These fixtures construct TaskPayload directly instead of calling submit.
# Refresh the immutable contracts where a fixture sets the submit metadata.
fields = "scalar_prefix|scalar_prefix_tagged_args|runtime_barrier_tail|runtime_compiled_poll|runtime_direct_poll_scratch"
pattern = r"\b(?P<owner>payload|local_payload)\.(?:" + fields + r")\s*(?:=|\|=|&=)[^;]+;"
test = re.sub(pattern, lambda m: m.group(0) + "\n    " + m.group("owner") + ".prepare_overlap_contracts();", test)
test = test.replace("context->prepare_subtask_to_core(", "prepare_and_pickup(")
anchor = "    void init_slot() {"
assert test.count(anchor) == 1
test = test.replace(anchor, """    SchedulerContext::PublishHandle prepare_and_pickup(
        int32_t thread_idx, int32_t core_offset, ChipTaskSlotState &state, SubtaskSlot subslot,
        bool pending, int32_t block, bool force_gate = false
    ) {
        auto handle = context->prepare_subtask_to_core(thread_idx, core_offset, state, subslot,
                                                       pending, block, force_gate);
        if (handle.valid) {
            auto core_id = context->core_trackers_[thread_idx].get_core_id_by_offset(core_offset);
            auto &dispatch = context->payload_per_core_[core_id][handle.reg_task_id & 1u];
            if (dispatch.local_context.async_ctx.local_predecessor_pending != DISPATCH_ORDINARY_ENTRY)
                initialize_submitted_overlap(&dispatch.global_context, handle.reg_task_id);
        }
        return handle;
    }

""" + anchor)
# These two fixtures dispatch through the staging entrypoints; model the same
# AICore pickup before checking the materialized runtime-owned control state.
for name, slot_expr, token in [
    ("ScalarPrefixWithoutLocalProducerRetainsIdlePreDispatch", "0][core.running_reg_task_id & 1u]", "core.running_reg_task_id"),
    ("GatedPendingPrefixCanOverlapAnUnrelatedLocalTask", "core_offset][core.pending_reg_task_id & 1u]", "core.pending_reg_task_id"),
]:
    begin = test.index("TEST_F(TmrSchedulerExecutionInputsTest, " + name + ")")
    end = test.find("\nTEST", begin + 1)
    if end < 0: end = len(test)
    scope = test[begin:end]
    marker = "const auto &dispatch = context->payload_per_core_[" + slot_expr + ";"
    assert scope.count(marker) == 1, (name, scope.count(marker))
    replacement = marker.replace("const auto &", "auto &")
    replacement += "\n        initialize_submitted_overlap(&dispatch.global_context, " + token + ");"
    test = test[:begin] + scope.replace(marker, replacement) + test[end:]
(scratch / "test.cpp").write_text(test)
flags = dict(re.findall(r"^(CXX_\w+) = (.*)$", (target / "flags.make").read_text(), re.M))
link = shlex.split((target / "link.txt").read_text())
options = [link[0], "-I" + str(scratch / "runtime"), "-I" + str(scratch / "common")]
for name in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
    options.extend(shlex.split(flags[name]))
sources = (scratch / "test.cpp", scratch / "runtime/scheduler/scheduler_dispatch.cpp")

def compile_one(source):
    subprocess.run([*options, "-c", str(source), "-o", str(source.with_suffix(".o"))], check=True)

with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(compile_one, sources))
for i, argument in enumerate(link):
    if argument.endswith("common/test_tmr_scheduler_execution_inputs.cpp.o"):
        link[i] = str(sources[0].with_suffix(".o"))
    elif argument.endswith("scheduler/scheduler_dispatch.cpp.o"):
        link[i] = str(sources[1].with_suffix(".o"))
binary = scratch / "test_candidate"
link[link.index("-o") + 1] = str(binary)
subprocess.run(link, cwd=build, check=True)
subprocess.run([str(binary), "--gtest_output=xml:" + str(scratch / "results.xml")], check=True)
