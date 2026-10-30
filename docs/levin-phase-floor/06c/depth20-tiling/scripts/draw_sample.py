"""depth20-tiling step 0: draw the sample by sampling_rule.md (whose SHA-256 is checked against the
value recorded in log.md). Reads state/inventory.json only. Writes state/sample.json.

Usage: PYTHONPATH=<tree>:<scripts> python -B draw_sample.py
"""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dprov import BLOCKS, STATE, TASK, header, sha256  # noqa: E402

RULE_SHA = "d47a35db9b11330a1b4e7000a9af06ab77fe25442f5a5bc5d367b4e1b2772edc"
ELIGIBLE_WALL_S = 120.0
N = 30


def key(e):
    return hashlib.sha256(f"{e['unit']}:{e['index']}".encode()).hexdigest()


def main():
    rule = TASK / "sampling_rule.md"
    inv_p = STATE / "inventory.json"
    for l in header(Path(__file__).resolve(), "depth20-tiling: the sample, drawn by sampling_rule.md", [rule, inv_p]):
        print(l)
    if sha256(rule) != RULE_SHA:
        raise SystemExit("sampling_rule.md differs from the recorded rule")
    inv = json.loads(inv_p.read_text())
    cells = {}
    for e in inv:
        e = dict(e, key=key(e), eligible=e["unit_wall_s"] <= ELIGIBLE_WALL_S)
        cells.setdefault((e["rtol"], e["block"]), []).append(e)
    order_cells = [(rt, b) for rt in ("1e-7", "1e-8") for b in BLOCKS]
    assert set(cells) == set(order_cells), set(order_cells) - set(cells)
    taken, seen = [], set()
    for c in order_cells:
        el = sorted((e for e in cells[c] if e["eligible"]), key=lambda e: e["key"])
        if el:
            pick, why = el[0], "rule 1: eligible, smallest key"
        else:
            pick = sorted(cells[c], key=lambda e: (e["unit_wall_s"], e["key"]))[0]
            why = "rule 1: no eligible call; smallest unit wall_s"
        taken.append(dict(pick, cell=list(c), drawn_by=why))
        seen.add((pick["unit"], pick["index"]))
    fill_order = sorted(order_cells, key=lambda c: (-len(cells[c]), ("1e-7", "1e-8").index(c[0]), BLOCKS.index(c[1])))
    while len(taken) < N:
        added = False
        for c in fill_order:
            if len(taken) >= N:
                break
            el = sorted((e for e in cells[c] if e["eligible"] and (e["unit"], e["index"]) not in seen), key=lambda e: e["key"])
            if el:
                taken.append(dict(el[0], cell=list(c), drawn_by="rule 2: stratified fill"))
                seen.add((el[0]["unit"], el[0]["index"]))
                added = True
        if not added:
            break
    for i, e in enumerate(taken):
        e["sample_no"] = i
    out = STATE / "sample.json"
    out.write_text(json.dumps(taken, indent=1))
    print(f"{'#':>2} {'cell':<14} {'unit':<18} {'idx':>3} {'wall_s':>8} {'elig':>5}  drawn_by")
    for e in taken:
        print(f"{e['sample_no']:>2} {e['rtol']+' '+e['block']:<14} {e['unit']:<18} {e['index']:>3} {e['unit_wall_s']:>8.1f} {str(e['eligible']):>5}  {e['drawn_by']}")
    print(f"cells with no eligible call: {[c for c in order_cells if not any(e['eligible'] for e in cells[c])]}")
    print(f"sample {len(taken)} calls on {len({e['unit'] for e in taken})} units; wrote {out} sha256 {sha256(out)}")


if __name__ == "__main__":
    main()
