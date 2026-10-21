# SPDX-License-Identifier: Apache-2.0
"""Reject the observed missing page-table capacity before reserving a device."""

import pytest
from submit_probe import validate_fixture_arguments


@pytest.mark.parametrize("b1", [True, False])
@pytest.mark.parametrize("table_history,accepted", [(None, False), (131199, False), (131200, True), (131201, True)])
def test_metadata_matrix_requires_future_history_capacity(b1, table_history, accepted):
    args = ["--batch", "1" if b1 else "6", "--query-tokens", "6", "--history", "131072", "--metadata-matrix"]
    args += ["--b1-metadata-matrix"] if b1 else ["--token-capacity", "41", "--request-capacity", "12"]
    if table_history is not None:
        args += ["--table-history", str(table_history)]
    if accepted:
        validate_fixture_arguments(args)
    else:
        with pytest.raises(ValueError, match="history\\+128"):
            validate_fixture_arguments(args)


def test_ordinary_l1_does_not_require_matrix_capacity():
    validate_fixture_arguments(["--batch", "1", "--history", "131072", "--query-tokens", "6", "--l1-windows", "5"])
