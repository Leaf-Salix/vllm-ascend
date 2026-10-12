"""Exercise batched notification publication in an isolated CPU overlay."""

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
runtime_rel = Path("src/a2a3/runtime/tensormap_and_ringbuffer/runtime")
include_rel = Path("src/common/platform/include")
for relative in (runtime_rel, include_rel):
    shutil.copytree(root / relative, scratch / relative, dirs_exist_ok=True)
patch = (output / "candidate.patch").read_text()
patch = "".join(
    p for p in re.split(r"(?=^--- a/)", patch, flags=re.M)
    if p.startswith("--- a/" + str(runtime_rel)) or p.startswith("--- a/" + str(include_rel))
)
subprocess.run(["git", "apply", "--unsafe-paths", "--directory=" + str(scratch), "-"],
               input=patch, text=True, cwd=root, check=True)
test = (output.parent / "boundary_contract_cache_20261012/cpu/test.cpp").read_text()
anchor = "namespace {\nint register_writes = 0;"
assert test.count(anchor) == 1
test = test.replace(anchor, """namespace {
size_t flushed_notification_count = 0;
size_t notification_flush_calls = 0;
}
namespace aicpu_cache_maintenance {
void flush_ranges_impl(const void *const *addresses, size_t count, size_t bytes) {
    flushed_notification_count = count;
    ++notification_flush_calls;
    EXPECT_EQ(bytes, sizeof(TaskBoundaryNotification));
    for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(reinterpret_cast<uintptr_t>(addresses[i]) % 64, 0u);
        const auto &notice = *static_cast<const TaskBoundaryNotification *>(addresses[i]);
        for (int word = 0; word < 8; ++word) {
            EXPECT_EQ(notice.value[word], 1u);
            EXPECT_EQ(notice.positive[word], 1u);
            EXPECT_EQ(notice.negative[word], ~uint64_t{0});
        }
        for (size_t j = 0; j < i; ++j) EXPECT_NE(addresses[i], addresses[j]);
    }
}
}
""" + anchor)
anchor = "#if defined(TEST_TMR_SCALAR_PREFIX)\n"
assert test.count(anchor) == 1
test = test.replace(anchor, anchor + """
TEST_F(TmrSchedulerExecutionInputsTest, NotificationPublicationBatchesOnlyInitializedOwnedRanges) {
    notification_flush_calls = 0;
    initialize({table_a.data(), table_a.size()});
    EXPECT_EQ(notification_flush_calls, 1u);
    EXPECT_EQ(flushed_notification_count, RUNTIME_MAX_WORKER);
    notification_flush_calls = 0;
    context->assign_own_clusters(0);
    EXPECT_EQ(notification_flush_calls, 1u);
    EXPECT_EQ(flushed_notification_count, 3u);
}
""")
(scratch / "test.cpp").write_text(test)
flags = dict(re.findall(r"^(CXX_\w+) = (.*)$", (target / "flags.make").read_text(), re.M))
link = shlex.split((target / "link.txt").read_text())
options = [link[0], "-I" + str(scratch / runtime_rel), "-I" + str(scratch / include_rel)]
for name in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
    options.extend(shlex.split(flags[name]))
sources = (
    scratch / "test.cpp",
    scratch / runtime_rel / "scheduler/scheduler_dispatch.cpp",
    scratch / runtime_rel / "scheduler/scheduler_cold_path.cpp",
)


def compile_one(source):
    subprocess.run([*options, "-c", str(source), "-o", str(source.with_suffix(".o"))], check=True)


with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(compile_one, sources))
suffixes = ("common/test_tmr_scheduler_execution_inputs.cpp.o", "scheduler/scheduler_dispatch.cpp.o",
            "scheduler/scheduler_cold_path.cpp.o")
for i, argument in enumerate(link):
    for suffix, source in zip(suffixes, sources):
        if argument.endswith(suffix):
            link[i] = str(source.with_suffix(".o"))
binary = scratch / "test_candidate"
link[link.index("-o") + 1] = str(binary)
subprocess.run(link, cwd=build, check=True)
subprocess.run([str(binary), "--gtest_output=xml:" + str(scratch / "results.xml")], check=True)
