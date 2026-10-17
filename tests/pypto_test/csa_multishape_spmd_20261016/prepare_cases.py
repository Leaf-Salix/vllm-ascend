# SPDX-License-Identifier: Apache-2.0
"""无 NPU 的形状特化验证：七个历史规格，以及非历史 128K/B12。"""

import argparse
import json
import math
from pathlib import Path

from shape_specialization import STATIC_ROOT, build_variant, descriptors_from_roots


def resize_descriptor_fixture(descriptors, old_batch, new_batch):
    """仅用于 CPU 形状检查；真实运行必须传 Native 的实际描述符。

    输入分配公式来自 make_fixture，不合成任何长度/页表/slot 的设备数值。
    cache 物理页保留原容量也可合法绑定；这里只调整批维及 compact 容量。
    """
    tokens = new_batch * 6
    token_roots = {"x_hc", "x_out", "freqs_cos", "freqs_sin", "ori_slot_mapping",
                   "state_slot_mapping", "inner_state_slot_mapping", "position_ids", "idx_topk_scores", "idx_topk"}
    batch_roots = {"kv_seq_lens", "cmp_seq_lens", "state_block_table", "inner_state_block_table",
                   "cmp_block_table", "idx_block_table", "ori_block_table"}
    compact_roots = {"cmp_freqs_cos", "cmp_freqs_sin", "inner_freqs_cos", "inner_freqs_sin",
                     "cmp_slot_mapping", "idx_slot_mapping"}
    for name, descriptor in descriptors.items():
        shape = list(descriptor["shape"])
        if name in token_roots:
            shape[0] = tokens
        elif name in batch_roots:
            shape[0] = new_batch
        elif name in {"cmp_query_start_loc", "idx_query_start_loc"}:
            shape[0] = new_batch + 1
        elif name in compact_roots:
            shape[0] = math.ceil(shape[0] * new_batch / old_batch)
        else:
            continue
        descriptor.update(shape=shape, stride=[math.prod(shape[i + 1:]) for i in range(len(shape))])
    return descriptors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="*", default=["h131072_b4", "h131072_b8", "h131072_b16",
                         "h131072_b24", "h8192_b16", "h8192_b24", "h8192_b32", "h131072_b12"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"scope": "CPU 形状绑定、分片/依赖/视图检查；不代表真机正确性或性能", "cases": []}
    for name in args.cases:
        reference = name if name != "h131072_b12" else "h131072_b8"
        anchor = json.loads((STATIC_ROOT / "bindings" / f"{reference}_prepared.json").read_text())
        descriptors = descriptors_from_roots(anchor["root_arguments"])
        if name == "h131072_b12":
            resize_descriptor_fixture(descriptors, 8, 12)
        history = int(name.split("_")[0][1:])
        try:
            case, contract = build_variant(descriptors, max_seq_len=history + 6,
                                           output_root=args.output, case_name=name)
            item = {"case": name, "passed": True, "anchor": contract["anchor"], "path": str(case),
                    "wait_plan": contract["wait_plan"], "math_batches": contract["signal_audit"]["roles"]}
        except Exception as error:
            item = {"case": name, "passed": False, "error": repr(error)}
            raise
        finally:
            report["cases"].append(item)
            (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps(item, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
