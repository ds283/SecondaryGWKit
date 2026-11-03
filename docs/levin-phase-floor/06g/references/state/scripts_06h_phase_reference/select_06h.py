"""levin-phase-floor prompt 06h section 5, the phase re-measurement: the two selections the captures
run on. New in 06h.

The calls are 06f's 20 M2 calls (06f's out/sample/selection.json, its entries with ``"sample":
"M2"``, read only; its SHA-256 is checked against the committed copy in the code worktree,
docs/levin-phase-floor/06f/out/sample/selection.json). Each is captured twice, at ``p`` (06h's
code commit's parent, whose code is 06e's) and at ``h`` (06h's code), and each capture must
reproduce its own tree's census record bit for bit (x1_capture.py compares value, abserr, regions,
maximum depth, converged and the label). So each selection carries that census's record:
- ``selection_p.json``: from 06e's census checkpoint (prov.RUN06E, its SHA-256 checked against
  prov.CHECKPOINT06E_SHA256). These must equal the census fields of 06f's own selection, which was
  drawn from the same checkpoint (checked);
- ``selection_h.json``: from 06h's census checkpoint (prov.census_run_06h(), the census agent's run,
  read only; its SHA-256 printed).
A call is matched by the unit's ``<rtol>-<shard>-<serial>`` (each census names its units by its own
tree) and the call index, and its label must equal 06f's. Every record of a unit is read; if more
than one holds the call (an over-budget pass), they must agree. The units are renamed
``<tree>-<rtol>-<shard>-<serial>`` (x1_capture.py runs them in that tree); 06f's name is kept as
``unit_06f`` and the census's own as ``census_unit``. A call missing from a census, or whose
records disagree, stops this script with nothing written.

Usage: python -B select_06h.py   (from var/levin-phase-floor-06h/phase/; writes out/sample/)
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from prov import (  # noqa: E402
    CHECKPOINT06E_SHA256,
    REC06F,
    RUN06E,
    TREES,
    census_run_06h,
    header,
    sha256,
)

SEL06F = REC06F / "out" / "sample" / "selection.json"
SEL06F_DOCS = (
    TREES["h"][0]
    / "docs"
    / "levin-phase-floor"
    / "06f"
    / "out"
    / "sample"
    / "selection.json"
)


def rest_of(unit):
    return unit.split("-", 1)[1]


def scan(checkpoint, rests):
    """{(rest, index): [(census unit, unit status, call record), ...]} for the wanted calls."""
    want = {}
    for rest, index in rests:
        want.setdefault(rest, set()).add(index)
    found = {}
    with open(checkpoint) as f:
        for line in f:
            head = line[:200]
            if not any(f'-{r}"' in head for r in want):
                continue
            rec = json.loads(line)
            unit = rec.get("unit", "")
            rest = unit.split("-", 1)[-1]
            if rest not in want:
                continue
            data = rec.get("data") or {}
            for c in (data.get("calls") or []) + (data.get("other_calls") or []):
                if c["index"] in want[rest]:
                    found.setdefault((rest, c["index"]), []).append(
                        (unit, data.get("status"), c)
                    )
    return found


def key(c):
    return (
        c["value"]["hex"],
        c["abserr"]["hex"],
        c["num_regions"],
        c["max_depth"],
        bool(c["converged"]),
        c["notify_label"],
    )


def main():
    run06h = census_run_06h()
    ck = {"p": RUN06E / "checkpoint.jsonl", "h": run06h / "checkpoint.jsonl"}
    for l in header(
        Path(__file__).resolve(),
        "levin-phase-floor 06h: the phase re-measurement's selections (06f's 20 M2 calls, at p and h)",
        [SEL06F, SEL06F_DOCS, ck["p"], ck["h"]],
    ):
        print(l)
    if sha256(SEL06F) != sha256(SEL06F_DOCS):
        raise SystemExit("06f's selection differs from its committed copy")
    if sha256(ck["p"]) != CHECKPOINT06E_SHA256:
        raise SystemExit("06e's census checkpoint is not a0dc4d7f...")
    sel06f = [e for e in json.loads(SEL06F.read_text()) if e.get("sample") == "M2"]
    print(f"06f's M2 calls: {len(sel06f)}")
    if len(sel06f) != 20:
        raise SystemExit("expected 20 M2 calls")
    rests = [(rest_of(e["unit"]), e["index"]) for e in sel06f]
    out = {}
    problems = []
    for tree in ("p", "h"):
        found = scan(ck[tree], rests)
        sel = []
        for e in sel06f:
            k = (rest_of(e["unit"]), e["index"])
            recs = found.get(k) or []
            if not recs:
                problems.append(f"{tree}: {e['unit']} #{e['index']} not in the census")
                continue
            if len({key(c) for _u, _s, c in recs}) != 1:
                problems.append(
                    f"{tree}: {e['unit']} #{e['index']}: {len(recs)} census records disagree"
                )
                continue
            unit, status, c = recs[0]
            if c["notify_label"] != e["notify_label"]:
                problems.append(
                    f"{tree}: {e['unit']} #{e['index']}: label {c['notify_label']!r} != 06f's"
                )
                continue
            r = dict(e)
            r.update(
                unit=f"{tree}-{k[0]}",
                unit_06f=e["unit"],
                census_unit=unit,
                census_unit_status=status,
                census_n_records=len(recs),
                census_value=c["value"]["hex"],
                census_abserr=c["abserr"]["hex"],
                census_num_regions=c["num_regions"],
                census_max_depth=c["max_depth"],
                census_converged=bool(c["converged"]),
                census=("06e " if tree == "p" else "06h ") + str(ck[tree]),
            )
            if tree == "p":
                same = all(
                    r[f] == e[f]
                    for f in (
                        "census_value",
                        "census_abserr",
                        "census_num_regions",
                        "census_max_depth",
                        "census_converged",
                    )
                )
                if not same:
                    problems.append(
                        f"p: {e['unit']} #{e['index']}: 06e's record differs from 06f's selection"
                    )
            sel.append(r)
        out[tree] = sel
    for p_ in problems:
        print("PROBLEM:", p_)
    if problems:
        print("nothing written")
        return 1
    dest = ROOT / "out" / "sample"
    dest.mkdir(parents=True, exist_ok=True)
    for tree, sel in out.items():
        path = dest / f"selection_{tree}.json"
        if path.exists():
            raise SystemExit(f"{path} exists; not overwritten")
        path.write_text(json.dumps(sel, indent=1))
        print(f"wrote {path} sha256 {sha256(path)}")
    print(
        "\nunit (06f) | index | label | p: regions, depth, conv, abserr | h: regions, depth, conv, abserr | h census unit, status"
    )
    for ep, eh in zip(out["p"], out["h"]):
        print(
            f"{ep['unit_06f']} | {ep['index']} | {ep['notify_label']} | {ep['census_num_regions']}, {ep['census_max_depth']}, "
            f"{ep['census_converged']}, {float.fromhex(ep['census_abserr']):.4e} | {eh['census_num_regions']}, "
            f"{eh['census_max_depth']}, {eh['census_converged']}, {float.fromhex(eh['census_abserr']):.4e} | "
            f"{eh['census_unit']}, {eh['census_unit_status']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
