"""发布用户指定的四个 bug issue，逐个回读；失败时保留记录且不自动重试。"""

import argparse
import datetime as dt
import json
import re
import subprocess
from pathlib import Path


def read_issue(url):
    match = re.fullmatch(r"https://github.com/hw-native-sys/pypto/issues/([1-9][0-9]*)", url)
    assert match, url
    read = subprocess.run(
        ["gh", "api", "--hostname", "github.com",
         "repos/hw-native-sys/pypto/issues/" + match.group(1)],
        text=True, stdout=subprocess.PIPE, check=True)
    raw = json.loads(read.stdout)
    return {"number": raw["number"], "title": raw["title"], "body": raw["body"],
            "labels": raw["labels"], "url": raw["html_url"], "state": raw["state"].upper()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "issue_manifest.json").read_text())
    identity = json.loads((root / "repository_identity.json").read_text())
    assert identity["github_host"] == "github.com"
    assert identity["issue_repo"] == "hw-native-sys/pypto"
    assert {r["id"] for r in manifest} == {
        "p1_group_inout", "p4_ptoas_branch_view", "p5_ptoas_gather_phi", "p7_ptoas_column_carry"
    }
    record_file = root / "submitted.json"
    if record_file.exists() and not args.resume:
        raise RuntimeError("已有发布记录，不重复创建；先只读核对 GitHub")
    if not args.publish:
        print(json.dumps([{"id": r["id"], "title": r["title"]} for r in manifest], ensure_ascii=False))
        return
    results = json.loads(record_file.read_text()) if record_file.exists() else []
    for entry in manifest:
        assert entry["repo"] == identity["issue_repo"] and entry["labels"] == ["bug"]
        body = root / entry["body_file"]
        text = body.read_text()
        assert "/data/" not in text and len(text.encode()) == entry["body_bytes"]
        prior = next((r for r in results if r["id"] == entry["id"]), None)
        if prior is not None:
            assert prior.get("url"), "未知创建结果必须先人工核对，不能自动重试"
            verified = read_issue(prior["url"])
            assert verified["title"] == entry["title"] and verified["body"] == text
            assert verified["url"] == prior["url"] and verified["state"] == "OPEN"
            (root / (entry["id"] + "_verified.json")).write_text(json.dumps(verified, ensure_ascii=False, indent=2) + "\n")
            prior.update({"state": "created_verified", "number": verified["number"],
                          "labels_requested": entry["labels"], "labels_actual": verified["labels"]})
            record_file.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps(prior, ensure_ascii=False), flush=True)
            continue
        published = {
            "id": entry["id"], "repo": entry["repo"], "title": entry["title"],
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "state": "mutation_pending", "body_file": entry["body_file"],
        }
        results.append(published)
        record_file.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
        with (root / (entry["id"] + "_create.log")).open("w") as log:
            result = subprocess.run(
                ["bash", "/tmp/csa-issue-create.sh", "create", "github.com", entry["repo"],
                 entry["title"], str(body), *entry["labels"]],
                text=True, stdout=subprocess.PIPE, stderr=log, check=False)
        (root / (entry["id"] + "_create_response.txt")).write_text(result.stdout)
        published["exit_code"] = result.returncode
        if result.returncode != 0:
            published["state"] = "unknown_check_before_retry"
            record_file.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
            raise RuntimeError("创建未明确成功，先回读，不自动重试：" + entry["id"])
        url = result.stdout.strip()
        assert re.fullmatch(r"https://github.com/hw-native-sys/pypto/issues/[1-9][0-9]*", url), url
        published.update({"url": url, "state": "created_readback_pending"})
        record_file.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
        verified = read_issue(url)
        (root / (entry["id"] + "_verified.json")).write_text(json.dumps(verified, ensure_ascii=False, indent=2) + "\n")
        assert verified["title"] == entry["title"] and verified["body"] == text
        assert verified["url"] == url and verified["state"] == "OPEN"
        published.update({"state": "created_verified", "number": verified["number"],
                          "labels_requested": entry["labels"], "labels_actual": verified["labels"]})
        record_file.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(published, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
