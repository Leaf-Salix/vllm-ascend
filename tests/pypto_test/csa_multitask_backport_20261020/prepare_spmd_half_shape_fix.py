# SPDX-License-Identifier: Apache-2.0
"""仅修复a241fa61不可达B1分支的静态Tile尺寸，不修改共享单SPMD。"""

import json
import shutil

from prepare import HERE


def main():
    base = HERE / "sources/spmd_a241fa61"
    target = HERE / "sources/spmd_a241fa61_half_shape_fix"
    name = "indexer_score_stages.py"
    source = (base / name).read_text()
    old = "[TOPK_CANDIDATES_PER_LEAF // (2 * score_tile), score_tile]"
    new = "[TOPK_CANDIDATES_PER_LEAF // (2 * BUFFERED_LONG_SCORE_TILE), BUFFERED_LONG_SCORE_TILE]"
    if source.count(old) != 2:
        raise ValueError("必须仅有B1临时Tile的create/full两处尺寸")
    shutil.copytree(base, target)
    (target / name).write_text(source.replace(old, new))
    patch = {
        "base_commit": "a241fa61",
        "changed_file": name,
        "old": old,
        "new": new,
        "occurrences": 2,
        "reason": "短档score_tile=768时旧Tile为5×768，未执行B1分支也被解析并要求reshape至4096",
        "effect": "将B1专用Tile固定为4×1024；实际B1驻留域score_tile=1024，表达式值不变",
        "scope": "私有参考兼容补丁；不冒充原a241fa61完整八档验证，不改正式SPMD或多task",
    }
    (target / "compatibility_patch.json").write_text(json.dumps(patch, ensure_ascii=False, indent=2) + "\n")
    (HERE / "SPMD_HALF_SHAPE_COMPATIBILITY.json").write_text(json.dumps(patch, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
