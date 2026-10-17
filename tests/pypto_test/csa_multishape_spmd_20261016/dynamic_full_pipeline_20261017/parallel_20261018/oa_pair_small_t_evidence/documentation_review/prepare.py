#!/usr/bin/env python3
"""复制 OA 冻结候选四文件，仅整理说明与两处 import 的多行格式。

本脚本不导入 PyPTO、不编译、不执行 kernel。除 module/function/class docstring
外的完整 AST 必须相等；import 名称和顺序也单独核对。不使用 hash。
"""
from __future__ import annotations

import argparse
import ast
import copy
import difflib
import json
from pathlib import Path
import shutil

DEFAULT_SOURCE = Path("/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/oa_pair_manual_evidence/source_candidate")
FILES = (
    "cube_schedule.py",
    "vector_schedule.py",
    "output_projection_a_stages.py",
    "phase_sync.py",
)

OA_OVERVIEW = """动态 CSA 的 O_A 分组投影与分组量化。

输入输出与阅读顺序
    AIC 消费 attention 的 packed heads，按八个 O group 独立做
    K4096→N1024，写 FP32[P_MM,8192]。输出组流水开启、OA_COLS=128 且
    设备实际 0<T<=96 时，使用 o_projection_a_pair_group_stage_nz；
    其它区间保留原独立列块 helper。AIV 再对每行每组 1024 列独立取
    amax，写 INT8[P_MM,8192] 和 FP32[8,P_SCALE]，不跨八组归约。
    wo_a 保持 Native BF16[8,4096,1024] 的实际 ND 或 NZ 存储；各入口
    声明所需布局，pair 专用于 NZ，不做格式转换或导入期设备探针。

动态轴、容量与别名
    tokens 是调用方从真实业务 Tensor 动态轴得到的设备 Scalar T，不从
    packed/workspace 容量反推。packed 是 BF16[PACK_TOTAL_ROWS,4096]，
    PACK_TOTAL_ROWS=8*P_PACK，组 g 的第 t 行位于 g*P_PACK+t，P_PACK>=T。
    FP32/INT8 工作区共享 P_MM 容量合同但不共享存储；P_SCALE 独立且>=T。
    O_B 使用的 ROW_TILE 为 32/96/128 之一，P_MM>=ceil(T/ROW_TILE)*ROW_TILE，
    并须足以保存实际 T 行。packed 总行数须整除 8，各视图连续且不得非法别名。
    所有容量轴均 pl.dynamic；固定八组、模型维度和 tile 不是 B/S 白名单。
    外层捕获前拒绝不一致容量/layout；不能让生产者跳写而消费者读取旧值。

分工、数学与尾行
    原路径以 (group,ceil(T/128) 行块,N 列块) 展平并按 logical%WORKERS
    分给 AIC，每块先 K256 matmul，再按原顺序追加 15 个 K256。
    小 T pair 路径专用于 24 AIC：组 g 的三个 pair 归 3g..3g+2，每核
    共用输入和 Left，分别累加两个 N128 输出；末两个 N128 块分别归
    (3g+3+s)%24，s 为 0/1。组内五 owner 互异，组 7 的 single 回绕到 0/1。
    pair 显式使用 Mat/Left/Right 与两个 FP32 Acc，每条链按 K128 顺序
    累加；single 保留原 K256 首块和后续流水。M128 是 tile 容量，所有
    GM 读写只覆盖实际 T 行；不把 tile 容量当作可访问的 GM 行数。
    量化以 (group,32 行块) 展平轮转 AIV，块内每 8 行流水；有效行数贯穿
    load、row_max、常数、div、cast 和 store，不读取 O_A 未写的 FP32 padding。
    每组最后一个活动 32 行块是唯一尾 owner，额外将 INT8 的 [T,P_USED) 写零，
    P_USED=ceil(T/ROW_TILE)*ROW_TILE；O_B 才能安全整块读取。T=0 不产生任务。
    量化保留 127/amax、amax*(1/127) 及 INT32 rint→FP16 round→INT8 trunc；
    反量化 scale 不能替换为 recip(量化乘数)，也不增加 BF16 往返舍入。

同步与生命周期
    原 groups helper 保留 logical%WORKERS；pair helper 使用与 cube/vector
    同步的五 owner 映射。各 helper 均不创建 spmd、task 或 notify/wait。
    O_A 先获取所消费 packed 行的全部 attention 生产者；每个 AIC 完成
    本核全部 O_A 组后才进入 O_B 等待。AIV 获取本组全部 O_A owner 后量化，
    真实 INT8、scale 与所需尾零完成后才发布。O_B 获取该组全部量化 owner
    后可读整 ROW_TILE。组间就绪次序可以变化，组内归约与尾零合同不变。
    外层负责业务 fence、缓存维护和阶段信号。函数返回 GM 别名不代表
    异步流水已完成，也不代表其它核完成；消费者读完前不得复用 GM。
    每个输出行列块唯一写；InOut 保留未写区，P_USED 以后的 INT8 和 T 以后的
    scale 不承诺初始化。BYPASS 策略如需使用，须在外层 spmd 直属设置。
    实际编译、精度与完整 L1 验证状态见对应测试档案，不由源码说明推定。
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{label}: 预期唯一文本锚点，实际 {count}")
    return text.replace(old, new, 1)


def replace_module_docstring(text: str, new_doc: str) -> str:
    tree = ast.parse(text, type_comments=True)
    node = tree.body[0]
    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)):
        raise ValueError("模块开头不是 docstring")
    lines = text.splitlines(keepends=True)
    return "".join(lines[:node.lineno - 1]) + '"""' + new_doc + '"""\n' + "".join(lines[node.end_lineno:])


def document(name: str, text: str) -> str:
    if name == "output_projection_a_stages.py":
        text = replace_module_docstring(text, OA_OVERVIEW)
        text = replace_once(
            text,
            '"""全八组兼容入口：与按组通知分支复用同一数学，不另维护计算版本。"""',
            '"""原独立列块路径的全八组入口：复用 groups helper，不包含专用 pair 分工。"""',
            "OA 原整阶段入口说明",
        )
        text = replace_once(
            text,
            '"""仅既有N128且1≤T≤96：每组3pair+2single，实际GM容量/有效行仍动态。\n\n'
            '    pair p归3*g+p，写2p/2p+1；single s归(3*g+3+s)%24，写6+s。\n'
            '    本组五个owner互异。两个判断独立，外层在本组所有本核写入完成后才发布。',
            '"""仅输出组流水开启、N128 且 0<T<=96：每组三个 pair、两个 single。\n\n'
            '    pair p 归 3*g+p，写列块 2p/2p+1；single s 归 (3*g+3+s)%24，写 6+s。\n'
            '    本组五个 owner 互异，每核在同组只执行一种角色；外层完成本组全部\n'
            '    本核写入后才发布。实际 GM 容量与有效行仍动态，不能按 M128 写满尾行。',
            "OA pair 合同",
        )
    elif name == "cube_schedule.py":
        text = replace_once(
            text,
            '输出尾段按策略选择整阶段或逐输出组通知；逐组路径不改变本核O_A任务序，\n'
            '已做完自己O_A的AIC可读就绪量化组做O_B，而不等待其它组完成量化。',
            '输出尾段按策略选择整阶段或逐输出组通知。逐组 N128 且 0<T<=96 时，\n'
            '使用三 pair、两 single 的五 owner 分工；其它区间沿用 logical%24。每核\n'
            '先完成自己的全部 O_A，再消费就绪量化组做 O_B，保持无环的生产消费方向。\n'
            'pair 会改变原每核任务和组间就绪次序，不保证每个组都更早完成。',
            "Cube 输出总览",
        )
        text = replace_once(
            text,
            'from .output_projection_a_stages import o_projection_a_groups_stage_nz, o_projection_a_stage_nz, o_projection_a_pair_group_stage_nz',
            'from .output_projection_a_stages import (\n'
            '    o_projection_a_groups_stage_nz,\n'
            '    o_projection_a_stage_nz,\n'
            '    o_projection_a_pair_group_stage_nz,\n'
            ')',
            "Cube 新增 import 格式；不改名称或顺序",
        )
        text = replace_once(
            text,
            '        # 固定8是模型的输出组数，不是B/S档位。每组内部任务数由本次T决定；\n'
            '        # 首项公式保留全局logical%24分核及原每核组序，不把所有核绑到同一组。\n'
            '        # 本核完成某组即可通知AIV；该AIC继续其它组的O_A，与已就绪组量化重叠。\n'
            '        # 只沿现有N128/T≤96区间共享A；同一谓词必须与AIV量化等待一致。\n'
            '        # 全部本核OA仍先于OB等待，不能把未完成OA移到可能阻塞的quant之后。',
            '        # 固定八组来自模型结构；实际行数和工作区容量仍动态。\n'
            '        # 每核先完成自身全部 O_A，再进入 O_B 等待，避免量化与未来 O_A 互等。\n'
            '        # N128 且 0<T<=96 使用与 Vector 匹配的五 owner；否则保留 logical%24。\n'
            '        # 两条路径只在本核负责的组发布一次，发布前完成该组全部本核写入。',
            "Cube OA 调度注释",
        )
        text = replace_once(
            text,
            '        # 保留整阶段路径供区间策略与同轮对照；两分支复用全部数学helper。',
            '        # 关闭组流水时保留原整阶段数学与通知，不使用 pair 分工或五 owner 等待。',
            "Cube 关闭路径注释",
        )
    elif name == "vector_schedule.py":
        text = replace_once(
            text,
            '组流水启用时，量化只等本输出组的O_A owner；末块尾零属于同组通知合同，\n'
            '每组独立槽位。HCpost仍等待所有O_B partial，保留八组反量化左折叠顺序。',
            '组流水启用时，量化只等本输出组的真实 O_A owner。N128 且 0<T<=96\n'
            '等三 pair 加两 single 的五核，其它区间按原逻辑列块 owner 等待。量化\n'
            '仍按 (group,32 行块)%48 分工，每组末块尾零在同组发布前完成。HCpost\n'
            '仍等待所有 O_B partial，保留八组反量化左折叠顺序。',
            "Vector 输出总览",
        )
        text = replace_once(
            text,
            'from .phase_sync import begin_invocation, publish_phase, wait_producer, wait_row_owners, wait_oa_pair_group',
            'from .phase_sync import (\n'
            '    begin_invocation,\n'
            '    publish_phase,\n'
            '    wait_producer,\n'
            '    wait_row_owners,\n'
            '    wait_oa_pair_group,\n'
            ')',
            "Vector 新增 import 格式；不改名称或顺序",
        )
        text = replace_once(
            text,
            '                # amax需要本行的全部1024列。首版按整组所有行的真实O_A owner\n'
            '                # 获取数据，已比等待全八组更细；不声称做到逐行块最小依赖。',
            '                # amax 需要本行全部 1024 列；等待本组所有行的真实 O_A owner。\n'
            '                # 依赖按输出组区分，不声称做到逐行块最小等待。',
            "Vector 量化等待说明",
        )
    elif name == "phase_sync.py":
        text = replace_once(
            text,
            '    同一工作区的 G、核编号映射、阶段槽数及阶段含义必须保持不变；仅改变\n'
            '    业务 T/B/metadata 不改变此合同。若新策略改变参与核集合或阶段布局，\n'
            '    必须绑定另一份清零的新工作区，不能只因容量足够就复用旧信号内容。',
            '    同一工作区调用 begin 的物理核集合、G、核编号映射、阶段槽数与阶段\n'
            '    含义必须保持不变。每个阶段的活动 owner 可随 T 和合法策略变化，\n'
            '    由当次消费者公式与当代发布匹配；这不要求清空未使用槽。改变物理\n'
            '    参与核或槽布局时，必须绑定另一份清零的新工作区，不能只因容量\n'
            '    足够就复用旧信号内容。',
            "物理参与核与阶段活动 owner 区分",
        )
        text = replace_once(
            text,
            '五、当前状态\n'
            '    这是可复用的动态控制组件，尚不等于完整 CSA。wait_row_owners 只适用\n'
            '    “按连续行 tile 轮转分给 worker”的布局；请求分组、split-K 和多维\n'
            '    投影应显式枚举自己的真实 owner，不能误用本函数减少依赖。',
            '五、通用与专用 owner 等待\n'
            '    wait_row_owners 只适用“按连续行 tile 轮转分给 worker”的布局；\n'
            '    请求分组、split-K 和多维投影须显式枚举真实 owner，不能误用该公式。\n'
            '    wait_oa_pair_group 专用于固定 24 AIC、输出组流水开启的 N128 小 T\n'
            '    分工，等待 3g..3g+2 与 (3g+3+s)%24。两种 helper 均在获取全部依赖\n'
            '    后统一维护业务缓存，不能交换 owner 公式。',
            "Phase 通用/专用 helper 总览",
        )
        text = replace_once(
            text,
            '"""N128小T专用五owner并集；合同0≤group<8，24AIC，三pair和两single。',
            '"""N128 小 T 的五 owner 并集：组流水开启、0<T<=96、0<=group<8、24 AIC。',
            "Phase pair 完整调用合同",
        )
    else:
        raise ValueError(name)
    return text


def is_docstring(node: ast.AST) -> bool:
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str))


class RemoveDocstrings(ast.NodeTransformer):
    def strip(self, node: ast.AST) -> ast.AST:
        self.generic_visit(node)
        if node.body and is_docstring(node.body[0]):
            node.body = node.body[1:]
        return node

    visit_Module = strip
    visit_ClassDef = strip
    visit_FunctionDef = strip
    visit_AsyncFunctionDef = strip


def executable_ast(tree: ast.AST) -> str:
    # 只移除 Python 定义的 docstring；其它字符串常量/注释类型字段/导入顺序均保留。
    stripped = RemoveDocstrings().visit(copy.deepcopy(tree))
    return ast.dump(stripped, annotate_fields=True, include_attributes=False)


def collect_docstrings(tree: ast.AST) -> dict:
    docs = {}

    class Collect(ast.NodeVisitor):
        def __init__(self):
            self.scope = []

        def record(self, node, name):
            self.scope.append(name)
            if node.body and is_docstring(node.body[0]):
                docs[".".join(self.scope)] = {
                    "line": node.body[0].lineno,
                    "text": node.body[0].value.value,
                }
            self.generic_visit(node)
            self.scope.pop()

        def visit_Module(self, node):
            self.record(node, "<module>")

        def visit_ClassDef(self, node):
            self.record(node, node.name)

        def visit_FunctionDef(self, node):
            self.record(node, node.name)

        def visit_AsyncFunctionDef(self, node):
            self.record(node, node.name)

    Collect().visit(tree)
    return docs


def imports(tree: ast.AST) -> list[str]:
    return [
        ast.dump(node, annotate_fields=True, include_attributes=False)
        for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    destination = output / "source_documented"
    if destination == source or source in destination.parents:
        raise ValueError("输出不得写入冻结源目录")
    output.mkdir(parents=True, exist_ok=True)

    original = {name: (source / name).read_text(encoding="utf-8") for name in FILES}
    prepared = {name: document(name, text) for name, text in original.items()}
    per_file = {}
    doc_changes = []
    patch_lines = []
    for name in FILES:
        old, new = original[name], prepared[name]
        old_tree = ast.parse(old, filename=str(source / name), type_comments=True)
        new_tree = ast.parse(new, filename=str(destination / name), type_comments=True)
        equal = executable_ast(old_tree) == executable_ast(new_tree)
        import_equal = imports(old_tree) == imports(new_tree)
        old_docs, new_docs = collect_docstrings(old_tree), collect_docstrings(new_tree)
        changed_keys = []
        for key in sorted(set(old_docs) | set(new_docs)):
            before, after = old_docs.get(key), new_docs.get(key)
            if (before or {}).get("text") != (after or {}).get("text"):
                changed_keys.append(key)
                doc_changes.append({
                    "file": name,
                    "scope": key,
                    "before": before,
                    "after": after,
                })
        per_file[name] = {
            "source_path": str(source / name),
            "documented_path": str(destination / name),
            "source_size_bytes": len(old.encode("utf-8")),
            "documented_size_bytes": len(new.encode("utf-8")),
            "full_ast_equal": ast.dump(old_tree, include_attributes=False) == ast.dump(new_tree, include_attributes=False),
            "all_docstrings_removed_ast_equal": equal,
            "ordered_import_ast_equal": import_equal,
            "docstring_count_before": len(old_docs),
            "docstring_count_after": len(new_docs),
            "docstrings_changed": changed_keys,
        }
        if not equal or not import_equal:
            raise AssertionError(f"{name}: 执行 AST 或 import 改变，拒绝输出")
        patch_lines.extend(difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile="a/" + name, tofile="b/" + name,
        ))

    destination.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(source / name, destination / name)
        (destination / name).write_text(prepared[name], encoding="utf-8")
        if (destination / name).read_text(encoding="utf-8") != prepared[name]:
            raise AssertionError("写入文本与已审计文本不一致")
    frozen_unchanged = all(
        (source / name).read_text(encoding="utf-8") == original[name] for name in FILES
    )
    if not frozen_unchanged:
        raise AssertionError("读取期间冻结源发生变化，不能报告同一输入")
    audit = {
        "status": "PASS",
        "adoption_status": "未采用；仅文档整理副本，原区间谓词完全保留",
        "source": str(source),
        "destination": str(destination),
        "files_count": len(FILES),
        "method": "ast.parse(type_comments=True)，剥除 Module/ClassDef/FunctionDef/AsyncFunctionDef 首项 docstring 后完整 ast.dump 比较；仅排除源码位置属性，不忽略导入、常量、函数体、类型注释或任何执行节点",
        "full_ast_equal": all(v["full_ast_equal"] for v in per_file.values()),
        "all_docstrings_removed_ast_equal": all(v["all_docstrings_removed_ast_equal"] for v in per_file.values()),
        "ordered_import_ast_equal": all(v["ordered_import_ast_equal"] for v in per_file.values()),
        "frozen_source_text_unchanged": frozen_unchanged,
        "docstring_changes_count": len(doc_changes),
        "files": per_file,
    }
    (output / "documentation.patch").write_text("".join(patch_lines), encoding="utf-8")
    (output / "ast_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "docstring_changes.json").write_text(
        json.dumps({"count": len(doc_changes), "changes": doc_changes}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": audit["status"],
        "files_count": audit["files_count"],
        "all_docstrings_removed_ast_equal": audit["all_docstrings_removed_ast_equal"],
        "ordered_import_ast_equal": audit["ordered_import_ast_equal"],
        "docstring_changes_count": audit["docstring_changes_count"],
        "full_ast_equal": audit["full_ast_equal"],
        "frozen_source_text_unchanged": frozen_unchanged,
        "destination": str(destination),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
