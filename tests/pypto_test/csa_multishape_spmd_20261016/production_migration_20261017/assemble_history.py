# SPDX-License-Identifier: Apache-2.0
"""首次汇编 CSA 单 SPMD 历史全文；之后只手工追加主日志，禁止重建覆盖。

只读两份实验树中的 Markdown。外部只读参考资料列链接，其余全文保留。
只重定位代码区域外的 Markdown 相对链接；不改历史数字、结论或代码块。
不计算 hash、不删原文、不嵌入二进制或原始日志。
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


STATIC_NAME = "csa_static_spmd_feasibility_20261015"
DYNAMIC_NAME = "csa_multishape_spmd_20261016"
GROUPS = (
    ("static-ledger", "静态原型总账"),
    ("static-attempts", "静态逐次尝试"),
    ("static-special", "静态专项记录、组件问题与 issue 正文"),
    ("dynamic-ledger", "动态区间入口总账与七档结果"),
    ("dynamic-special", "动态专项实现、诊断与失败尝试"),
    ("handoff-history", "交接、旧目标与工作区历史"),
)
STATIC_LEDGER = ("README.md", "PROCESS_LOG.md", "validation_log_excerpt.md")
DYNAMIC_LEDGER = (
    "GOAL_AND_CONSTRAINTS.md", "README.md", "ATTEMPTS.md", "SELECTED_SEVEN_RESULTS.md",
    "BALANCED_SEVEN_RESULTS.md", "CACHED_SEVEN_RESULTS.md", "QK_SEVEN_RESULTS.md",
)


def classify(tree_name, relative):
    if relative.parts[0] in {"references", "upstream_records"}:
        return "external-reference", "外部只读参考资料；保留原文链接和本地快照，不重复全文"
    if tree_name == STATIC_NAME:
        if relative.as_posix() in STATIC_LEDGER:
            return "static-ledger", "静态原型总览、过程总账或当时验证摘录，全文保留"
        if relative.as_posix() == "ALL_ATTEMPTS.md":
            return "static-attempts", "静态逐次尝试总表，包含失败及未完成尝试，全文保留"
        if relative.as_posix() == "COMPILE_AND_RESUME.md":
            return "handoff-history", "静态编译交接快照，全文保留，不作为当前状态"
        if relative.parts[0] == "upstream_issues":
            return "static-special", "本会话提交的 issue 正文或初稿；全部全文保留"
        return "static-special", "本会话静态专项记录或随证据包保存的文档快照，全文保留"
    if relative.as_posix() in DYNAMIC_LEDGER:
        return "dynamic-ledger", "动态目标、过程总账或七档结果，全文保留"
    if (relative.parts[0] == "handoff_snapshots"
            or relative.name.startswith(("NEXT_STEP", "GOAL_BEFORE_"))
            or relative.name in {"LOCAL_WORKTREE.md", "WORKSPACE_MIGRATION_20261017.md"}):
        return "handoff-history", "交接、旧目标或工作区快照，全文保留，不作为当前状态"
    return "dynamic-special", "本会话动态专项实现、失败尝试或静态审计记录，全文保留"


def upstream_url(path):
    if "simpler_atomic_guides_20261016" in path.parts:
        return "https://github.com/nalinaly/simpler/blob/fdwic-swimlane-deps/tests/atomic_probe/" + path.name
    fixed = "https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/examples/_kernel_extension_aclgraph/"
    if path.name == "csa-torchair-library-readme.md":
        return fixed + "torch_library/README.md"
    if path.name == "csa-torchair-pybind-readme.md":
        return fixed + "pybind/README.md"
    raise ValueError(f"外部资料需要明确原文地址：{path}")


def code_regions(text):
    """标记 fenced/缩进代码和 inline code，供链接改写时排除。

    保守地保护所有四空格/制表符开头的行；本次源集另核对这些行中的
    Markdown 链接，不把代码示例误当为可点击相对链接。
    """
    mask = bytearray(len(text))
    regions = []
    active = None
    offset = 0
    for line in text.splitlines(keepends=True):
        match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*?)(?:\r?\n)?$", line)
        protected = active is not None
        if active is not None:
            if match and match[1][0] == active[0] and len(match[1]) >= active[1] and not match[2].strip():
                active = None
        elif match and not (match[1][0] == "`" and "`" in match[2]):
            active = (match[1][0], len(match[1]))
            protected = True
        elif line.startswith(("    ", "\t")):
            protected = True
        if protected:
            mask[offset:offset + len(line)] = b"\x01" * len(line)
            regions.append((offset, offset + len(line)))
        offset += len(line)
    # HTML code/pre blocks also remain verbatim.
    for match in re.finditer(r"<(pre|code)\b[^>]*>.*?</\1\s*>", text, re.I | re.S):
        mask[match.start():match.end()] = b"\x01" * (match.end() - match.start())
        regions.append((match.start(), match.end()))
    position = 0
    while position < len(text):
        if mask[position] or text[position] != "`":
            position += 1
            continue
        end = position
        while end < len(text) and text[end] == "`":
            end += 1
        marker = text[position:end]
        close = end
        while True:
            close = text.find(marker, close)
            if close < 0:
                position = end
                break
            marker_end = close + len(marker)
            if (not mask[close]
                    and (close == 0 or text[close - 1] != "`")
                    and (marker_end == len(text) or text[marker_end] != "`")):
                mask[position:marker_end] = b"\x01" * (marker_end - position)
                regions.append((position, marker_end))
                position = marker_end
                break
            close = marker_end
    return mask, regions


def destination_span(text, start):
    """定位 Markdown destination 本体，不包含尖括号、标题或右括号。"""
    while start < len(text) and text[start].isspace():
        start += 1
    if start == len(text):
        return None
    if text[start] == "<":
        end = start + 1
        while end < len(text):
            if text[end] == "\\":
                end += 2
            elif text[end] == ">":
                return start + 1, end
            else:
                end += 1
        return None
    depth = 0
    end = start
    while end < len(text):
        char = text[end]
        if char == "\\":
            end += 2
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                break
            depth -= 1
        elif char.isspace():
            break
        end += 1
    return (start, end) if end > start else None


def relocate_target(target, source, output):
    if (not target or target.startswith(("/", "~"))
            or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target)):
        return target
    # Fragment-only links refer to the original document's own headings.
    split_at = min([index for index in (target.find("#"), target.find("?")) if index >= 0]
                   or [len(target)])
    filename, suffix = target[:split_at], target[split_at:]
    destination = source.parent / filename if filename else source
    return Path(os.path.relpath(destination, output.parent)).as_posix() + suffix


def relocate_links(text, source, output):
    mask, regions = code_regions(text)
    spans = set()
    for match in re.finditer(r"\]\(", text):
        if mask[match.start()]:
            continue
        span = destination_span(text, match.end())
        if span and not any(mask[span[0]:span[1]]):
            spans.add(span)
    for match in re.finditer(r"(?m)^ {0,3}\[[^\]\n]+\]:[ \t]*", text):
        if mask[match.start()]:
            continue
        span = destination_span(text, match.end())
        if span and not any(mask[span[0]:span[1]]):
            spans.add(span)
    edits = []
    for start, end in sorted(spans):
        before = text[start:end]
        after = relocate_target(before, source, output)
        if before != after:
            edits.append({"source_char_start": start, "source_char_end": end,
                          "original": before, "relocated": after})
    rewritten = text
    for edit in reversed(edits):
        rewritten = rewritten[:edit["source_char_start"]] + edit["relocated"] + rewritten[edit["source_char_end"]:]
    # 用逆向文本还原检查“只改链接，没有截断或改写正文”；不计算 hash。
    shift = 0
    for edit in edits:
        edit["embedded_char_start"] = edit["source_char_start"] + shift
        shift += len(edit["relocated"]) - len(edit["original"])
    restored = rewritten
    for edit in reversed(edits):
        start = edit["embedded_char_start"]
        assert restored[start:start + len(edit["relocated"])] == edit["relocated"]
        restored = restored[:start] + edit["original"] + restored[start + len(edit["relocated"]):]
    assert restored == text
    assert all(not any(start <= edit["source_char_start"] < end for start, end in regions)
               for edit in edits)
    return rewritten, edits, len(regions)


def document_order(item):
    group_index = [value[0] for value in GROUPS].index(item["group"])
    name = item["relative_in_tree"]
    ordered = STATIC_LEDGER if item["group"] == "static-ledger" else DYNAMIC_LEDGER
    rank = ordered.index(name) if name in ordered else len(ordered)
    return group_index, rank, item["tree"], name


def assemble():
    directory = Path(__file__).resolve().parent
    repository = directory.parents[3]
    test_root = repository / "tests" / "pypto_test"
    output = repository / "vllm_ascend" / "ops" / "pypto" / "DSV4_CSA_SPMD_OPTIMIZATION_LOG.md"
    inventory_path = directory / "history_inventory.json"
    if output.exists() or inventory_path.exists():
        raise SystemExit("拒绝覆盖：主日志或首次归档清单已存在。后续仅在主日志的“持续追加”区手工追加。")
    inputs = []
    originals = {}
    for tree_name in (STATIC_NAME, DYNAMIC_NAME):
        tree_root = test_root / tree_name
        for path in sorted(tree_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() != ".md":
                continue
            raw = path.read_bytes()
            text = raw.decode("utf-8")
            originals[path] = raw
            relative = path.relative_to(tree_root)
            group, reason = classify(tree_name, relative)
            inputs.append({"source": path.relative_to(repository).as_posix(), "tree": tree_name,
                           "relative_in_tree": relative.as_posix(), "source_bytes": len(raw),
                           "group": group, "reason": reason, "mode": "link_only" if group == "external-reference" else "full_text",
                           "path": path, "text": text})
    embedded = sorted((item for item in inputs if item["mode"] == "full_text"), key=document_order)
    references = sorted((item for item in inputs if item["mode"] == "link_only"), key=lambda item: item["source"])
    for index, item in enumerate(embedded, 1):
        item["anchor"] = f"history-document-{index:03d}"
        item["ordinal"] = index
        item["body"], item["link_edits"], item["protected_code_regions"] = relocate_links(item["text"], item["path"], output)
        item["embedded_bytes"] = len(item["body"].encode("utf-8"))
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    source_bytes = sum(item["source_bytes"] for item in inputs)
    full_source_bytes = sum(item["source_bytes"] for item in embedded)
    chunks = []
    byte_count = 0

    def emit(text):
        nonlocal byte_count
        chunks.append(text)
        byte_count += len(text.encode("utf-8"))

    emit(f"""# CSA 完整单 SPMD 优化总记录

首次历史归档：{timestamp}。本文件位于正式算子目录，集中保留本会话从静态原型到动态区间策略的完整过程。
原文、脚本、原始日志、泳道图和二进制继续保留在 `tests/pypto_test/`；此文不嵌入原始日志或二进制。
**主日志只首次生成一次。后续必须在末尾“持续追加”区手工记录，禁止运行脚本覆盖新记录。**

<a id="current-objective"></a>
## 当前目标与状态

目标：把 CSA 全部 incore task 融合为一个完整 SPMD，支持动态输入形状和按实际工作量覆盖区间的分档，
并让每个验收档位的完整 L1 incore 首尾耗时相对同轮原动态分 task PTO **降低至少 30%**：
`SPMD ≤ dynamic PTO × 0.70`。旧的 `128K/B16 < 700 μs` 单点目标已被替换；下方原文中的旧目标只是当时快照。

- 起始七档均为 S=6：历史 128K 的 B4/8/16/24，历史 8K 的 B16/24/32。七档当前均未达到 30%，不能用平均收益代替逐档验收。
- 分档根据 CSA 自身 B、历史长度、本轮 query 数与总 T，覆盖区间并验证代表值和边界；允许不同形状编译不同二进制。
  不把七个回归点做成支持白名单，不照搬 HC_pre 的阈值，不把目前 S=6 的实现边界缩减成最终目标。
- 档内由实际 device metadata 决定有效范围、地址、循环和尾块，不新增 Host 取值。
- 使用正式权重 `/data/model/DeepSeek-V4-Flash-0731-w8a8`，同输入、同设备配置、同轮多次采样中位数比较完整 CSA。
  计时包含内部等待和阶段空隙，排除外围 AICPU/orchestrator 首尾；Native、局部阶段、CPU 模型和插桩数据不能替代验收。
- 先功能、再精度、最后性能。输出、Top-K、cache/state、动态 metadata、图重放、保护区与核间同步的既有门槛保持不变。

**需要覆盖历史表述的最新核实：**

1. 当前 A3 的 `aic_gather` 底层仍是 `AIV UB → GM FIFO → AIC L1`，不是绕过 GM 的直接 UB→L1。
   历史文档里“直传/消除 GM 往返”的设想必须按后续源码审计纠正，不能当作已获得的性能收益。
2. 现有 B16 插桩中 HC 首个数学开始到最后 mix/norm 发布约 60.78 μs；该档历史完整结果距 30% 目标仍差约 173.60 μs。
   只优化 HC 前段不足以单独证明完整目标可达；这只是量级比较，不是同轮消融或性能下限。
3. `hc_sinkhorn_scalar_v1` 只完成独立 AST 候选和静态审计，未进行 PyPTO parse/lowering、编译或设备验证。
   Scalar `/` 与 A3 `vdiv` 的舍入、`vcadd` 内部归约树均未证明逐 bit 一致；8 行串行还有性能风险。
4. 正式包首批组件已经开始抽取，但完整生成器与后端退避仍待独立化；**尚非完整生产接入，更不是整模型服务已接通。**

## 阅读说明与目录

下列每篇历史文档都标明原始路径并保留全文，包含失败、初稿、过期目标和当时未解决的问题。
除代码区域外的 Markdown 相对链接重定位外，不改写原内容。历史正文的“当前”“已通过”“下一步”仅指写作当时，
不能覆盖上面的现目标或末尾新记录。源码块、原 issue 正文及重复保存的证据快照均保留。

本次盘点 {len(inputs)} 篇 Markdown，共 {source_bytes:,} 字节；其中 {len(embedded)} 篇本会话文档全文纳入，
原文 {full_source_bytes:,} 字节；{len(references)} 篇外部只读参考仅列原文链接和本地快照。
清单与逐链接重定位记录：
[history_inventory.json](../../../tests/pypto_test/{DYNAMIC_NAME}/production_migration_20261017/history_inventory.json)。
归档脚本：
[assemble_history.py](../../../tests/pypto_test/{DYNAMIC_NAME}/production_migration_20261017/assemble_history.py)。

- [当前目标与状态](#current-objective)
""")
    for group_id, title in GROUPS:
        emit(f"- [{title}](#{group_id})\n")
    emit("- [外部只读参考原文](#external-references)\n- [历史归档结束与持续追加](#ongoing-log)\n\n")
    emit("<!-- CSA_SPMD_HISTORY_ARCHIVE_BEGIN -->\n\n")
    for group_id, title in GROUPS:
        selected = [item for item in embedded if item["group"] == group_id]
        emit(f'<a id="{group_id}"></a>\n## {title}\n\n')
        for item in selected:
            emit(f'- [{item["ordinal"]:03d} · {item["tree"]}/{item["relative_in_tree"]}](#{item["anchor"]})\n')
        emit("\n")
        for item in selected:
            local_link = Path(os.path.relpath(item["path"], output.parent)).as_posix()
            emit(f'<a id="{item["anchor"]}"></a>\n### 历史原文 {item["ordinal"]:03d}：{item["relative_in_tree"]}\n\n')
            emit(f'> **当时快照，全文保留。** 来源：[{item["source"]}]({local_link})；'
                 f'原文 {item["source_bytes"]:,} 字节。{item["reason"]}。\n\n')
            marker = item["anchor"]
            emit(f'<!-- SOURCE_BEGIN {marker} -->\n')
            item["body_byte_start"] = byte_count
            emit(item["body"])
            item["body_byte_end"] = byte_count
            emit(f'\n<!-- SOURCE_END {marker} -->\n\n---\n\n')
    emit('<a id="external-references"></a>\n## 外部只读参考原文\n\n')
    emit("这些文件是外部资料快照，不属于本会话编写的尝试记录，因此只列链接；原快照不删除。\n"
         "引用链接保留当时来源，不表示本次重新核验了上游最新状态。所有 `upstream_issues/**` 的本会话 issue 正文已在前文全文纳入。\n\n")
    for item in references:
        local_link = Path(os.path.relpath(item["path"], output.parent)).as_posix()
        item["upstream_url"] = upstream_url(item["path"])
        emit(f'- [{item["source"]}]({local_link})（{item["source_bytes"]:,} 字节）；'
             f'[上游原文]({item["upstream_url"]})。{item["reason"]}。\n')
    emit("\n<!-- CSA_SPMD_HISTORY_ARCHIVE_END -->\n\n"
         "**历史全文归档到此结束。下面的内容为后续持续记录，禁止再次生成本文件覆盖本区。**\n\n"
         '<a id="ongoing-log"></a>\n## 持续追加\n\n'
         "### 2026-10-17：统一工作区、建立总记录及首批生产组件抽取\n\n"
         "按用户要求，vllm-ascend 工作统一到 `vllm-ascend-dsv4-pto-0251rc1`，分支为 "
         "`dsv4-flash-pto-v0.25.1rc1`。动态实验及历史产物已迁入，原静态基线与失败尝试保留。"
         "远端 `a919f119` 已以本地合并提交 `643d8039` 合入；这条迁移记录本身不表示已 push。"
         "原日志中的旧绝对路径仍表示当时实际执行位置，不伪造历史运行位置。\n\n"
         "用户随后要求把正式动态 SPMD 实现放在 `vllm_ascend/ops/pypto/`。首批已抽取到 "
         "`deepseek_v4_flash_csa_spmd/` 的组件为 `__init__.py`、`kernels.py`、`policy.py`、"
         "`signals.py`、`signal_codegen.py`；测试侧策略和信号 lowering 改为薄包装，生成器改读正式数学核体。"
         "首批源码等价审计记录了 73 个数学函数 AST、信号 AST、576 个策略输入及 5 个拒绝用例，"
         "并检查生产文件没有测试目录导入。证据见 "
         f"[source_equivalence.json](../../../tests/pypto_test/{DYNAMIC_NAME}/production_migration_20261017/source_equivalence.json)。\n\n"
         "**首批组件迁入，完整生成器和后端退避仍待独立化；尚非完整生产接入。** "
         "B17/H65536 新入口的完整生成源码 AST 与迁移前一致，仅 signals 模块导入路径变化；证据见 "
         f"[entry_equivalence.json](../../../tests/pypto_test/{DYNAMIC_NAME}/production_migration_20261017/entry_equivalence.json)。"
         "新入口 CPU lowering 的 `full_codegen_passed=true`，正式 signals 包导入通过；证据见 "
         f"[cpu_lowering/result.json](../../../tests/pypto_test/{DYNAMIC_NAME}/production_migration_20261017/cpu_lowering/result.json)。"
         "没有运行 NPU；源码/CPU 证据不代替真实 runtime ABI、设备精度、性能或整网验证。"
         "七档至少降低 30% 的完整目标仍未达到。\n")
    result = "".join(chunks).encode("utf-8")
    assert len(result) == byte_count
    for item in embedded:
        assert result[item["body_byte_start"]:item["body_byte_end"]] == item["body"].encode("utf-8")
    for path, raw in originals.items():
        if path.read_bytes() != raw:
            raise RuntimeError(f"归档期间源文档改变，请重新执行：{path}")
    enumerated = {path.relative_to(repository).as_posix() for tree_name in (STATIC_NAME, DYNAMIC_NAME)
                  for path in (test_root / tree_name).rglob("*") if path.is_file() and path.suffix.lower() == ".md"}
    assert enumerated == {item["source"] for item in inputs}, "输入目录在归档过程中新增或丢失文档"
    excluded_fields = {"path", "text", "body"}
    inventory = {
        "created_at": timestamp, "repository": str(repository), "output": str(output.relative_to(repository)),
        "generation_policy": "首次创建；若主日志或清单存在则拒绝覆盖；后续仅手工追加主日志",
        "source_trees": [STATIC_NAME, DYNAMIC_NAME],
        "source_document_count": len(inputs), "source_bytes": source_bytes,
        "full_text_document_count": len(embedded), "full_text_source_bytes": full_source_bytes,
        "link_only_document_count": len(references),
        "full_text_embedded_bytes": sum(item["embedded_bytes"] for item in embedded),
        "initial_log_bytes": len(result),
        "relative_link_edits": sum(len(item["link_edits"]) for item in embedded),
        "issue_body_full_text_count": sum(item["relative_in_tree"].startswith("upstream_issues/") for item in embedded),
        "verification": {
            "all_markdown_in_inputs_accounted_for": True,
            "each_embedded_body_exactly_matches_full_relocated_source": True,
            "reverse_link_edits_restore_full_original_text": True,
            "code_regions_not_rewritten": True,
            "source_contents_rechecked_before_write": True,
            "source_tree_reenumerated_before_write": True,
            "hashes_computed": False,
            "original_documents_deleted_or_modified": False,
            "binary_or_raw_log_contents_embedded": False,
            "body_byte_offsets_are_initial_archive_offsets": True,
        },
        "documents": [{key: value for key, value in item.items() if key not in excluded_fields}
                      for item in embedded + references],
    }
    # 创建模式而非覆盖模式；不会重建已有日志，避免丢掉持续追加内容。
    with output.open("xb") as file:
        file.write(result)
    with inventory_path.open("x", encoding="utf-8") as file:
        json.dump(inventory, file, ensure_ascii=False, indent=2)
        file.write("\n")
    written = output.read_bytes()
    assert written == result
    print(json.dumps({key: inventory[key] for key in (
        "source_document_count", "source_bytes", "full_text_document_count", "full_text_source_bytes",
        "link_only_document_count", "full_text_embedded_bytes", "initial_log_bytes", "relative_link_edits",
        "issue_body_full_text_count")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    assemble()
