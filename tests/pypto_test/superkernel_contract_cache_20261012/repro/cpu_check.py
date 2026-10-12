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
runtime = root / "src/a2a3/runtime/tensormap_and_ringbuffer/runtime"
shutil.copytree(runtime, scratch, dirs_exist_ok=True)
patch = (output / "candidate.patch").read_text()
patch = "".join(p for p in re.split(r"(?=^--- a/)", patch, flags=re.M) if "/aicore/aicore_executor.cpp" not in p)
patch = patch.replace("src/a2a3/runtime/tensormap_and_ringbuffer/runtime/", "")
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
(scratch / "test.cpp").write_text(test)
flags = dict(re.findall(r"^(CXX_\w+) = (.*)$", (target / "flags.make").read_text(), re.M))
link = shlex.split((target / "link.txt").read_text())
options = [link[0], "-I" + str(scratch)]
for name in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
    options.extend(shlex.split(flags[name]))
sources = (scratch / "test.cpp", scratch / "scheduler/scheduler_dispatch.cpp")

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
