#!/usr/bin/env bash
# 只运行本次GM子视图与既有slice/view相关CPU测试；必须先完成本树构建。
set -eo pipefail
test_report=${1:-ut_v3}
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(git -C "$here" rev-parse --show-toplevel)
workspace_dir=$(cd -- "$repo_dir/.." && pwd)
source "$workspace_dir/env-dsv4-0251rc1.sh"
cd -- "$workspace_dir/.worktrees/pypto-csa-gm-subview-20261019"
source .claude/skills/testing/load-env.sh
export PYTHONPATH="$PWD/tests/issue_3037_env:$PWD/tests:$PWD/python:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
python -c 'import pathlib, pypto, pypto.pypto_core as core; root=pathlib.Path.cwd(); assert root in pathlib.Path(pypto.__file__).resolve().parents; assert root in pathlib.Path(core.__file__).resolve().parents; print(pypto.__file__, core.__file__)'
python -m pytest \
  tests/ut/ir/operators/test_tensor_ops.py \
  tests/ut/ir/transforms/test_convert_tensor_to_tile_ops.py \
  tests/ut/codegen/test_pto_codegen_ops.py \
  -k 'gm_view or tensor_slice or tensor_view' \
  -n "$PYPTO_TEST_JOBS" -q --junitxml="$here/${test_report}.xml"
