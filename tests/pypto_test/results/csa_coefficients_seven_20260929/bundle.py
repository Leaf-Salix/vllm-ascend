"""Gather original JSONs using the prior matrix's validated download mapping."""

import json

from collect import ROOT, load_previous

if __name__ == "__main__":
    source = json.loads((ROOT / "source.json").read_text())
    load_previous("bundle").main()
    path = ROOT / "download/README.md"
    path.write_text(path.read_text().replace("源码c93ec723", "源码" + source["operator_commit"]))
