"""复用固定提交的整层用例，只把均匀Indexer scale换成可检测错页的输入。"""

import sys
from pathlib import Path


def main():
    source_repo = Path(sys.argv.pop(1)).resolve()
    sys.path.insert(0, str(source_repo / "tests/pypto_test"))
    import dsv4_csa_single_layer as case
    import torch

    original_fixture = case.make_fixture
    original_write = case.write_json
    pattern = {
        "kind": "per_physical_row",
        "formula": "FP16(0.00390625 + ((physical_row * 37) % 251) / 16384)",
        "page_table": "unchanged fixture: reverse physical pages within each request",
    }

    def make_fixture(*args, **kwargs):
        fixture = original_fixture(*args, **kwargs)
        group = fixture["groups"]["indexer"]
        scale = group["views"][1]
        values = torch.arange(scale.numel(), dtype=torch.int64, device=scale.device)
        values = (0.00390625 + ((values * 37) % 251).float() / 16384).to(scale.dtype)
        scale.copy_(values.reshape(scale.shape))
        group["initial"] = group["allocation"].cpu()
        return fixture

    def write_json(path, data):
        if Path(path).name == "report.json":
            data["accuracy_fixture"] = pattern
        original_write(path, data)

    case.make_fixture = make_fixture
    case.write_json = write_json
    case.main()


if __name__ == "__main__":
    main()
