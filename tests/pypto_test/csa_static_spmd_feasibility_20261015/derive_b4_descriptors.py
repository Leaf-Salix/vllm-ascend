# SPDX-License-Identifier: Apache-2.0
"""按既有 fixture 公式还原 B4 形状，避免为补 Host 描述符重复已完成的上卡验证。"""

import argparse
import ast
import copy
import json
from pathlib import Path

from bind_orchestration import tensor_annotation
from build_ir_catalog import functions_from_ir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--b8-descriptors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.b8_descriptors.read_text())
    if source["x_hc"]["shape"][0] != 48 or source["kv_seq_lens"]["metadata_values"] != [131078] * 8:
        raise ValueError("本推导只适用于已经采集的 B8/S6/H131072 fixture")
    _, functions = functions_from_ir(args.ir)
    root = functions["_decode_csa_tp1_layer"]
    dims = {}
    for arg in root.args.args:
        if arg.arg == "self":
            continue
        annotation = tensor_annotation(arg.annotation)
        if annotation is None:
            continue
        for symbol, size in zip(annotation[0].elts, source[arg.arg.removesuffix("__ssa_v0")]["shape"]):
            if isinstance(symbol, ast.Name):
                dims[symbol.id] = size
    dims.update(T_DYN=24, B_DYN=4, QUERY_BOUNDS_DYN=5)
    for name in ("ORI_BLOCK_NUM_DYN", "MAIN_STATE_BLOCK_NUM_DYN", "INNER_STATE_BLOCK_NUM_DYN"):
        if dims[name] != 73:
            raise ValueError(f"fixture 物理页公式已变化：{name}")
        dims[name] = 37
    for name in ("CMP_BLOCK_NUM_DYN", "IDX_NATIVE_CACHE_BLOCK_NUM_DYN"):
        if (dims[name] - 1) % 8:
            raise ValueError("压缩 cache 页数不满足既有 fixture 公式")
        dims[name] = (dims[name] - 1) // 8 * 4 + 1
    for name in ("COMPRESSED_ROWS_DYN", "INDEXER_ROWS_DYN"):
        if dims[name] != 20:
            raise ValueError(f"Native compact 分配公式已变化：{name}")
        dims[name] = 10
    result = copy.deepcopy(source)
    for arg in root.args.args:
        if arg.arg == "self":
            continue
        name = arg.arg.removesuffix("__ssa_v0")
        annotation = tensor_annotation(arg.annotation)
        if annotation is None:
            continue
        item = result[name]
        item["shape"] = [dims[n.id] if isinstance(n, ast.Name) else ast.literal_eval(n)
                         for n in annotation[0].elts]
        # B8 的设备地址不是 B4 的证据；也不复制不参与编排求值的 metadata 数据。
        for field in ("data_ptr", "storage_ptr", "storage_bytes", "metadata_values"):
            item.pop(field, None)
    result["kv_seq_lens"]["metadata_values"] = [131078] * 4
    result["_provenance"] = {
        "kind": "derived_host_descriptors_not_new_device_capture",
        "source": str(args.b8_descriptors),
        "fixture_formula": "dsv4_csa_single_layer.py:85 make_fixture；B4/S6/H131072",
        "required_check": "必须与已采集 B4 的所有实际调用视图、block_num 和任务选择逐项核对后使用",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
