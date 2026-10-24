# SPDX-License-Identifier: Apache-2.0
"""导出已有真实decode问答，不重新生成，不截断已记录的回答。"""

import argparse
import html
import json
from pathlib import Path

from tokenizers import Tokenizer


def read(path):
    return json.loads(path.read_text())


def fenced(text):
    fence = "````````"
    while fence in text:
        fence += "`"
    return f"{fence}text\n{text}\n{fence}\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--comparisons", type=Path, nargs="+", help="导出指定完整对照报告中的Native、MPMD、SPMD三侧")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / "完整模型输入"
    inputs.mkdir(exist_ok=True)
    quality = {}
    if args.comparisons:
        labels = {"native": "Native", "mpmd_tnd": "TND-MPMD", "spmd_tnd": "TND-SPMD"}
        scenes = []
        for path in args.comparisons:
            comparison = read(path)
            quality[comparison["batch"]] = {
                label: {
                    row["key"]: row
                    for row in comparison["modes"][mode].get("answer_quality", {}).get("answers", [])
                    if 0 in row["ranks"]
                }
                for mode, label in labels.items()
            }
            roots = {
                label: Path(comparison["configuration_provenance"][mode]["results"]) for mode, label in labels.items()
            }
            note = "本页独立展示Native、TND-MPMD和TND-SPMD的实际回答，不以某一侧回答代替另一侧。"
            if comparison.get("native_reference_reused"):
                note += "Native沿用历史正确性参考，版本信息保存在对照报告中；这里不作性能比较。"
            if any(
                not all(
                    row.get("all_csa_graph_replay_proven", False)
                    for row in read(root / "functional_check.json")["ranks"]
                )
                for label, root in roots.items()
                if label != "Native"
            ):
                note += "本次TND整模型结果未通过模型级图重放验收；算子单卡图重放通过不能代替这一项。"
            elif all(
                read(root / "configuration.json").get("require_tnd_graph_replay", False)
                for label, root in roots.items()
                if label != "Native"
            ):
                note += (
                    "本次两种TND的真实长历史FULL decode已通过满批、尾批及padding的21层图重放检查，"
                    "且这些步骤无Python重新提交。首个S1步及短历史空闲补齐调用单独记录，不计入此结论。"
                )
            scenes.append((comparison["batch"], roots, note))
    else:
        scenes = [
            (
                batch,
                {"Native": here / "runs" / native, "TND-MPMD": here / "runs" / mpmd},
                "本档TND-SPMD与TND-MPMD全部输出token及接受事件一致，PTO列展示MPMD原文。"
                if batch == 16
                else "此历史实验的SPMD在收尾时OOM，尚无完整结果；只展示已完成的Native与MPMD。",
            )
            for batch, native, mpmd in ((16, "v02_native_b16", "v09_mpmd_b16"), (24, "v12_native_b24", "v12_mpmd_b24"))
        ]
    bank = Path(read(scenes[0][1]["Native"] / "configuration.json")["bank"])
    plan = read(bank / "plan.json")
    cases = {row["key"]: row for row in plan["cases"]}
    questions = {row["key"]: row for row in read(bank.parent / "mixed128k_questions.json")["cases"]}
    tokenizer = Tokenizer.from_file(str(Path(plan["model"]) / "tokenizer.json"))
    manifest = {
        "comparison_reports": [str(path.resolve()) for path in args.comparisons or []],
        "cases": {},
        "batches": {},
    }
    css = """
body{font-family:system-ui,sans-serif;margin:24px;color:#18212d;background:#f7f8fb}
main{max-width:1600px;margin:auto}
section{background:white;padding:22px;margin:20px 0;border:1px solid #dce1eb;border-radius:8px}
.answers{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:24px}.answers>div{min-width:0}
pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.65;font-family:inherit;background:#f5f7fa;padding:16px}
nav{line-height:2.2}nav a{margin-right:16px}.note{padding:14px;background:#fff4d6}small{color:#536078}
@media(max-width:850px){.answers{grid-template-columns:1fr}}
"""
    for number, (batch, roots, spmd_note) in enumerate(scenes, 1):
        config = read(roots["Native"] / "configuration.json")
        limit = config["decode_tokens"]
        respect_eos = not config.get("ignore_eos", True)
        data = {}
        for mode, root in roots.items():
            assert read(root / "functional_check.json")["status"] == "PASS"
            data[mode] = read(root / "rank0.mixed.json")["requests"]
            # 只合并已经证明内容完全相同的DP副本，不省略不同问题或不同回答。
            for rank in range(1, 16):
                other = read(root / f"rank{rank}.mixed.json")["requests"]
                assert [(r["key"], r["output_token_ids"], r.get("finish_reason")) for r in other] == [
                    (r["key"], r["output_token_ids"], r.get("finish_reason")) for r in data[mode]
                ]
        title = f"128K / B{batch}：{' / '.join(roots)}完整问题及原始回答"
        note = (
            f"正常EOS停止，最多{limit}token；逐条标出结束原因，达到上限不作为完整答案通过。"
            if respect_eos
            else f"固定生成{limit}token，可能尚未答完；此模式只证明输出前缀。"
        )
        note += "所有实际输出全文保留，没有摘录或补写；正常结束也不等于答案内容正确。"
        sections, md, rows, nav = [], [f"# {title}\n", note, spmd_note], [], []
        for index, request in enumerate(data["Native"]):
            key = request["key"]
            question = questions[key]["question"]
            kind = questions[key]["kind"]
            assert data["TND-MPMD"][index]["key"] == key
            answers = {}
            for mode in roots:
                assert data[mode][index]["key"] == key
                token_ids = data[mode][index]["output_token_ids"]
                row = data[mode][index]
                assert (0 < len(token_ids) <= limit) if respect_eos else len(token_ids) == limit
                answers[mode] = {
                    "text": row["output_text"]
                    if respect_eos
                    else tokenizer.decode(token_ids, skip_special_tokens=False),
                    "token_ids": token_ids,
                    "finish_reason": row.get("finish_reason", "fixed_length"),
                    "stop_reason": row.get("stop_reason"),
                    "quality": quality.get(batch, {}).get(mode, {}).get(key),
                }
            if key not in manifest["cases"]:
                prompt_tokens = read(bank / cases[key]["tokens"])
                name = f"Q{key[-2:]}_{kind}_{len(prompt_tokens)}tokens_完整输入.txt"
                (inputs / name).write_text(tokenizer.decode(prompt_tokens, skip_special_tokens=False))
                manifest["cases"][key] = {
                    "question": question,
                    "kind": kind,
                    "input_file": str(Path(inputs.name) / name),
                    "input_tokens": len(prompt_tokens),
                }
            input_file = manifest["cases"][key]["input_file"]
            by_mode = {
                mode: sum(a != b for a, b in zip(answers["Native"]["token_ids"], answer["token_ids"]))
                + abs(len(answers["Native"]["token_ids"]) - len(answer["token_ids"]))
                for mode, answer in answers.items()
                if mode != "Native"
            }
            differences = by_mode["TND-MPMD"]
            difference_text = "，".join(f"{mode} {count}" for mode, count in by_mode.items())
            heading = f"Q{key[-2:]} · {kind} · 与Native不同位置token数：{difference_text}"
            nav.append(f'<a href="#{key}">Q{key[-2:]}</a>')
            body = ""
            for mode, value in answers.items():
                verdict = value["quality"]["verdict"]["content"] if value["quality"] else "未评分"
                completion = f"结束：{value['finish_reason']}；{len(value['token_ids'])} token；题意内容：{verdict}"
                body += (
                    f"<div><h3>{mode}</h3><p>{html.escape(completion)}</p><pre>{html.escape(value['text'])}</pre></div>"
                )
            native_quality = answers["Native"]["quality"]
            gold = json.dumps(native_quality["expected"], ensure_ascii=False, indent=2) if native_quality else None
            sections.append(
                f'<section id="{key}"><h2>{heading}</h2>'
                f'<a href="{html.escape(input_file, quote=True)}">查看含128K背景的完整模型输入</a>'
                f"<h3>完整问题后缀</h3><pre>{html.escape(question)}</pre>"
                + (
                    f"<details><summary>按题意计算的标准结果</summary><pre>{html.escape(gold)}</pre></details>"
                    if gold
                    else ""
                )
                + f'<div class="answers">{body}</div></section>'
            )
            md += [f"## {heading}\n", f"[含128K背景的完整输入]({input_file})\n", "### 完整问题后缀\n", fenced(question)]
            if gold:
                md += ["### 按题意计算的标准结果\n", fenced(gold)]
            for mode, value in answers.items():
                md += [
                    f"### {mode}：全部已记录回答\n",
                    f"结束：{value['finish_reason']}；长度：{len(value['token_ids'])} token。\n",
                ]
                if value["quality"]:
                    md += [fenced(json.dumps(value["quality"]["verdict"], ensure_ascii=False))]
                md += [fenced(value["text"])]
            rows.append(
                {
                    "key": key,
                    "question": question,
                    "kind": kind,
                    "answers": answers,
                    "positional_token_mismatches": differences,
                    "positional_token_mismatches_vs_native": by_mode,
                }
            )
        implementations = "Native_vs_TND_MPMD_SPMD" if "TND-SPMD" in roots else "Native_vs_TND_MPMD"
        policy = f"正常EOS{limit}" if respect_eos else f"固定{limit}token"
        stem = f"{number:02d}_128K_B{batch}_{implementations}_{policy}_完整问答"
        (output / (stem + ".md")).write_text("\n".join(md) + "\n")
        page = f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{title}</title><style>{css}</style>'
        page += f'<main><h1>{title}</h1><p class="note">{note}</p><p>{spmd_note}</p>'
        page += "<p>16个DP副本逐token相同，本页展示每个不同问题一份。位置差异数可能包含格式变化造成的后续错位。</p>"
        page += f"<nav>{''.join(nav)}</nav>{''.join(sections)}</main></html>"
        (output / (stem + ".html")).write_text(page)
        manifest["batches"][str(batch)] = {
            "sources": {m: str(r) for m, r in roots.items()},
            "sampling": {"ignore_eos": not respect_eos, "max_tokens": limit},
            "questions": rows,
            "spmd_status": spmd_note,
        }
    (output / "90_问题与原始输出token.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (output / "00_阅读说明.md").write_text(
        "# CSA TND真实问答对照\n\n优先打开01、02的HTML，左右对照Native和PTO。Markdown提供同样的全文。\n\n"
        "- 各档问题与实现见文件名。16个DP副本逐token相同时才合并重复副本；不同问题和实现仍独立展示。\n"
        "- 问题后缀与实际回答全文保留，结束策略和上限见文件名及页面。length表示达到上限，不能当作完整答案通过。\n"
        "- 正常结束、题意内容正确、格式合规分别记录；解析不完整需人工核对，不把Native输出当标准答案。\n"
        "- `完整模型输入/`另存全部实际prompt，包含128K背景和问题，保留特殊token文本表示。\n"
        "- 各实现是否完成、是否复用历史Native参考，均在对应页面说明；不伪造或补写回答。\n"
        "- 功能完成与模型图重放分开判断；有图重放限制时在页面注明，本目录不提供性能结论。\n"
        "- JSON保留输出token、原文及源结果路径；不包含权重或KV缓存。\n"
    )
    print(output)
    print(f"已导出{len(scenes)}档对照、{len(manifest['cases'])}个完整输入；保留全部实际输出和结束原因。")


if __name__ == "__main__":
    main()
