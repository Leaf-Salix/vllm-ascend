# SPDX-License-Identifier: Apache-2.0
"""Indexer Key页按device字节偏移建立零拷贝视图，消除奇偶页两分支。

Native一个量化页占4160字节，Key是其前4096字节。原核用未偏移和
偏移64B两份视图表达页头；本候选直接slice原缓冲中的4096B再reshape。
不改变Native布局、页号读取、非法页夹到0的规则或QK/WS点积顺序。

当前状态：CPU编译失败，InCore slice会lower成Tile，不能作为GM源。
这是保留的失败候选，不可作为已支持的零拷贝接口或已验证性能版本。
"""

import ast
import copy
import json
from pathlib import Path


def use_page_views(catalog, output):
    module = ast.parse(Path(catalog["body_bundle"]).read_text())
    fn = next(node for node in ast.walk(module)
              if isinstance(node, ast.FunctionDef)
              and node.name == "indexer_score_topk_native_pair_aic")
    if any(isinstance(x, ast.Name) and x.id == "key_bytes_view" for x in ast.walk(fn)):
        raise ValueError("重复应用Indexer页视图变换")
    hits = 0

    class ReplaceParity(ast.NodeTransformer):
        def visit_If(self, node):
            nonlocal hits
            text = ast.unparse(node.test)
            if "native_byte" not in text:
                return self.generic_visit(node)
            if not text.endswith("% 128 == 0") or len(node.body) != 2 or len(node.orelse) != 2:
                raise ValueError("Indexer页奇偶分支结构改变")
            even, even_result = node.body
            odd, odd_result = node.orelse
            if (not isinstance(even, ast.AnnAssign) or not isinstance(odd, ast.AnnAssign)
                    or ast.unparse(even.value.func) != "pl.tile.gather_row"
                    or ast.unparse(odd.value.func) != "pl.tile.gather_row"
                    or ast.unparse(even.value.args[1]) != "idx_kv_cache_inline8854__ssa_v0"
                    or ast.unparse(odd.value.args[1]) != "idx_kv_cache_shift64_inline8790__ssa_v0"
                    or ast.unparse(even_result.target) != ast.unparse(odd_result.target)):
                raise ValueError("Indexer页加载绑定改变")
            # 奇偶分支仅源base相差64B，其余dst/offset/shape/关键字必须一致。
            other = copy.deepcopy(odd.value)
            other.args[1] = copy.deepcopy(even.value.args[1])
            if ast.dump(other) != ast.dump(even.value) or ast.unparse(even.value.args[4]) != "[32, 128]":
                raise ValueError("两分支不是同一32x128 Key页的两种基址表示")
            byte = ast.unparse(node.test.left.left)
            prefix = f"key_page_view_{hits}"
            init = ast.parse(f"""
{prefix}_bytes = pl.tensor.slice(key_bytes_view, [4096], [{byte}])
{prefix} = pl.tensor.reshape({prefix}_bytes, [32, 128])
""").body
            assign = copy.deepcopy(even)
            assign.target = copy.deepcopy(even_result.target)
            assign.value.args[1] = ast.Name(id=prefix, ctx=ast.Load())
            assign.value.args[3] = ast.parse("[0, 0]", mode="eval").body
            hits += 1
            return init + [assign]

    ReplaceParity().visit(fn)
    if hits != 32:
        raise ValueError(f"预期32处Key页加载，实际{hits}")
    # 原视图最后可能少64B，这部分仅为最后一页的scale，Key页仍全部在范围内。
    fn.body[1:1] = ast.parse("""
key_bytes_view = pl.tensor.reshape(idx_kv_cache_inline8854__ssa_v0,
    [pl.tensor.dim(idx_kv_cache_inline8854__ssa_v0, 0) * 128])
""").body
    ast.fix_missing_locations(module)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "core_bodies_indexer_page_views.py"
    path.write_text("# Copyright (c) PyPTO Contributors.\n"
                    "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                    "# Indexer Key零拷贝页视图候选。\n" + ast.unparse(module) + "\n")
    audit = {"kernel": fn.name, "parity_branches_replaced": hits,
             "source_bytes_per_page": 4096, "page_tensor_shape": [32, 128],
             "dynamic_device_page_offsets": True, "new_gm_allocation": False,
             "qk_ws_math_order_unchanged": True, "ffts_unchanged": True,
             "device_validated": False}
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["body_bundle"] = str(path)
    return result, audit
