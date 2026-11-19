"""levin-phase-floor prompt 07c Part A, section 2.5: each reference integrand's captured blob as a file of its own, for
tools.filon_reference (one captured call per pickle), after 07b Part A's slice_refs_07b.py (a record). New in 07c.

One file per integrand, ``ref-<draw>-<shard>-<serial>-<g|a><index>``:
- R4 (state/selection_r4.json) and R5 (state/selection_r5.json): the capture of the integrand's call at 1e-7 if the draw
  holds one, else at 1e-8 (the manifests' entries ``c-...`` / ``m-...``), sliced only once its replay at the tree
  reproduces 07b's census at its own rtol and, where replayed, at the other (``x-...``, or the other drawn call);
- R31's record: its rehearsal capture (ref-G+q+r-h-3-20363-1.pkl), sliced once its replay reproduces R31's record.
Each blob is copied at its recorded offset and length and checked against its SHA-256; nothing is unpickled. Writes
state/refs/inputs/<key>.pkl (refusing to overwrite) and out/refs/inputs.json. Usage: python -B slice_refs_07c.py
"""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import WIDER, header, sha256  # noqa: E402

REP = WIDER / "out" / "replays" / "tree"


def main():
    out_dir = WIDER / "state" / "refs" / "inputs"
    out_json = WIDER / "out" / "refs" / "inputs.json"
    if out_json.exists():
        raise SystemExit(f"{out_json} exists")
    out_dir.mkdir(parents=True, exist_ok=True)
    man = json.loads((WIDER / "state" / "calls_07c.json").read_text())["calls"]
    man.update(json.loads((WIDER / "state" / "calls_07c_r5.json").read_text())["calls"])
    ins = [WIDER / "state" / "calls_07c.json", WIDER / "state" / "calls_07c_r5.json"]
    todo = []
    for draw in ("r4", "r5"):
        sp = WIDER / "state" / f"selection_{draw}.json"
        ins.append(sp)
        sel = json.loads(sp.read_text())
        for k in sel["taken"]:
            sh, se, gi = k.split("-")
            sids = []
            for c in sel["calls"][k]:
                rt = c["rtol"]
                sids.append(f"c-{rt}-{sh}-{se}-{c['index']}" if draw == "r4" else f"m-{rt}-{sh}-{se}-{gi}")
            sids += [s for s in man if s.startswith("x-") and s.endswith(f"-{sh}-{se}-{gi[1:]}") and man[s].get("captured_as") in sids]
            src = sorted(s for s in sids if not s.startswith("x-"))[0]  # 1e-7 sorts before 1e-8
            todo.append((f"ref-{draw.upper()}-{k}", draw.upper(), src, sids))
    todo.append(("ref-R31-3-20363-g1", "R31", "r31-3-20363-1", ["r31-3-20363-1"]))
    doc = {"header": header(Path(__file__).resolve(), "levin-phase-floor 07c Part A: the reference integrands, one file each", ins),
           "inputs": {}}
    for key, draw, src, sids in todo:
        checked = {}
        for s in sids:
            p = REP / f"{s}.json"
            checked[s] = json.loads(p.read_text())["reproduces_census"] if p.exists() else None
        if not all(checked.values()):
            raise SystemExit(f"{key}: a replay at the tree does not reproduce its record: {checked}")
        c = man[src]
        with open(c["path"], "rb") as f:
            f.seek(c["offset"])
            blob = f.read(c["length"])
        if hashlib.sha256(blob).hexdigest() != c["sha256"]:
            raise SystemExit(f"{key}: sha256 differs")
        p = out_dir / f"{key}.pkl"
        if p.exists():
            raise SystemExit(f"{p} exists")
        p.write_bytes(blob)
        doc["inputs"][key] = {"path": str(p), "sha256": sha256(p), "bytes": len(blob), "draw": draw, "source_sid": src,
                              "scored_sids": sids, "replays_reproduce": checked}
        print(f"{key}: from {src}; scores {sids}")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=1))
    print(f"{len(doc['inputs'])} inputs; wrote {out_json} sha256 {sha256(out_json)}")


if __name__ == "__main__":
    main()
