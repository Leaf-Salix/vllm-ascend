"""冻结末块发布修正版，显式抽取每半块的 softmax 统计量。"""

import difflib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
OLD_PREFIX = WORKSPACE / ".cache/csa-sparse-final-publish-4ffccb7b"
PREFIX = WORKSPACE / ".cache/csa-sparse-final-publish-fix-4ffccb7b"
PACKAGE = "dsv4_csa_sparse_final_publish_4ffccb7b"


def main():
    if (ROOT / "task.txt").exists():
        raise RuntimeError("不得编辑已入队副本")
    for side in ("baseline", "candidate"):
        shutil.copytree(Path(f"{OLD_PREFIX}-{side}"), Path(f"{PREFIX}-{side}"),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    relative = Path("vllm_ascend/ops/pypto") / PACKAGE / "decode_sparse_attn_csa.py"
    path = Path(f"{PREFIX}-candidate") / relative
    before = path.read_text()
    after = before
    for name, source in (("m_mi", "m_valid"), ("m_li", "l_valid")):
        old = f"{name} = pl.slice({source}, [FINAL_HEAD_TILE, 1], [pub_local_head, 0])"
        # A2/A3 TEXTRACT supports ND Vec tiles. Reshape the complete contiguous
        # column first (zero offset), then extract its row segment explicitly.
        new = (f"{name} = pl.reshape(pl.tile.extract(pl.reshape({source}, [1, H // 2]), "
               "0, pub_local_head, [1, FINAL_HEAD_TILE], target_memory=pl.MemorySpace.Vec), "
               "[FINAL_HEAD_TILE, 1])")
        if after.count(old) != 1:
            raise ValueError(old)
        after = after.replace(old, new)
    compile(after, str(path), "exec")
    path.write_text(after)
    (ROOT / "fix.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile=f"a/{relative}", tofile=f"b/{relative}")))
    old = ROOT.parent / "csa_sparse_final_publish_20260929"
    (ROOT / "compile.py").write_text((old / "compile.py").read_text().replace(
        ".cache/csa-sparse-final-publish-4ffccb7b", ".cache/csa-sparse-final-publish-fix-4ffccb7b"))
    driver = Path(f"{PREFIX}-candidate") / "tests/pypto_test/dsv4_csa_sparse_diagnostic.py"
    text = driver.read_text().replace("import os\n", "")
    text = text.replace(
        '    suffix = "_perf" if variant == "performance" else ""\n'
        '    mod = importlib.import_module(f"vllm_ascend.ops.pypto.'
        'deepseek_v4_flash_dspark{suffix}.decode_sparse_attn_csa")',
        '    from vllm_ascend.ops.pypto.variant import variant_package\n'
        '    mod = importlib.import_module(f"{variant_package()}.decode_sparse_attn_csa")')
    text = text.replace('    os.environ["PTO_CSA_VARIANT"] = args.variant\n', '')
    text = text.replace('    kernel, mod = build_kernel(args.variant)\n',
        '    if args.input:\n'
        '        # 固定历史输入只取前 16 个独立序列，覆盖跨 query 流水。\n'
        '        payload["batch"] = 16\n'
        '        for name in ("q", "position_ids", "cmp_sparse_indices", "expected"):\n'
        '            payload[name] = payload[name][:96].contiguous()\n'
        '        for name in ("ori_block_table", "cmp_block_table", "seqused_kv"):\n'
        '            payload[name] = payload[name][:16].contiguous()\n'
        '    kernel, mod = build_kernel(args.variant)\n')
    (ROOT / "sparse_diagnostic.py").write_text(text)
    (ROOT / "sparse_case.py").write_text('''"""使用冻结适配器和私有算子运行独立 Sparse 诊断。"""
import importlib.util
import sys
from pathlib import Path

source = Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(source / "tests/pypto_test"))
driver = Path(__file__).with_name("sparse_diagnostic.py")
spec = importlib.util.spec_from_file_location("private_sparse_diagnostic", driver)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.main()
''')


if __name__ == "__main__":
    main()
