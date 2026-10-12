#!/usr/bin/env bash
set -eo pipefail
p="/data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009"
cd "$p/pypto"
source "$p/env.sh"
a="$p/artifacts/boundary_compact_dispatch_20261012"
cmake -S "$p/simpler-compact-dispatch-20261012/tests/ut/cpp" -B "$a/cpu-build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/data/ci-runner/pypto-toolchain/toolchain/2026.08.5/gcc/bin/g++-15 -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$p/simpler/build/scalar-overlap-cpp-ut/_deps/googletest-src" > "$a/cpu-build.log" 2>&1
cmake --build "$a/cpu-build" --target test_tmr_scheduler_execution_inputs --parallel 8 >> "$a/cpu-build.log" 2>&1
"$a/cpu-build/test_tmr_scheduler_execution_inputs" > "$a/cpu.log" 2>&1
