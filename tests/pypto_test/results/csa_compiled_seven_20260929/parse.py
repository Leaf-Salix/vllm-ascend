"""Parse both roots in the frozen retained package without device execution."""
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = json.loads((ROOT / "source.json").read_text())
sys.path.insert(0, str(Path(source["source"]) / "tests/pypto_test"))
from dsv4_csa_env import activate  # noqa: E402

os.environ.update(PTO_CSA_VARIANT=source["variant"], VLLM_ASCEND_ENABLE_NZ="2", VLLM_ASCEND_PTO_CSA_ATOMIC_ADD="0")
activate()
from vllm_ascend.ops.pypto.variant import variant_package  # noqa: E402

module = importlib.import_module(f"{variant_package()}.decode_csa")
graphs = {name: type(getattr(module, name)._get_dep_graph()).__name__
          for name in ("decode_csa_tp1_layer", "decode_csa_tp1_layer_test")}
report = {"status": "PARSE_PASS", "source": source, "graphs": graphs, "device_execution": False}
(ROOT / "parse.json").write_text(json.dumps(report, indent=2) + "\n")
print("PARSE_PASS", graphs)
