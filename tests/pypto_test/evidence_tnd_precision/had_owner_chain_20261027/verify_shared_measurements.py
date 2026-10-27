"""复核共享实参诊断；原服务图结果保持独立。"""

import argparse
import json
import tarfile
from pathlib import Path

from verify_owner_measurements import read_json, verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    path = args.evidence_dir / "hc-owner-shared-v2-evidence.tar.gz"
    result = verify(path, True)
    with tarfile.open(path) as archive:
        data = read_json(archive, "/result.json")
    assert data["measurement_scope"].startswith("shared-args direct-root FULL")
    shared = data["shared_arguments"]
    assert len(shared["identities"]) == 64
    for arm in ("baseline", "candidate"):
        assert shared["same_arm_checks"][arm] == {"eager_8": True, "FULL_8": True, "poison_written": True}
        captured = shared["actual_capture_enqueue"][arm]
        assert captured and any(row["capture_id"] for row in captured)
        assert all(row["all_64_identity_equal"] for row in captured)
        proof = data["shared_actual_execution_identity"][arm]
        assert proof["CPU_all_registered_child_bytes_equal"] and len(proof["registered_children"]) == 63
        assert proof["orchestration_strict_alpha_equal"]
        assert proof["compiled_meta_byte_equal"] and proof["kernel_config_AST_equal"]
    result.update(
        scope=data["measurement_scope"],
        same_arm_eager_FULL_8=True,
        actual_capture_64_pointer_shape_stride_offset_dtype_alias_equal=True,
        shared_root_actual_registration_63_pass=True,
        helper_manifest_sha256=data["observer_helper_manifest_sha256"],
    )
    args.out.write_text(json.dumps({"pass": True, "run": result}, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
