"""levin-phase-floor prompt 07c Part A, section 2.6 table 4: under each variant, whether an unconverged chase call ends with
every depth-20 leaf below a refused leaf, from its instrumented copy ``<v>_i``. New in 07c.

d1_descent_07c.py's ``analyse()`` is run on out/replays/<v>_i/ (its source read and its one directory name, ``"instr"``,
replaced by ``"<v>_i"``; nothing else changed). Each ``<v>_i`` replay must equal the plain ``<v>`` replay of the same call
bit for bit (value, abserr, converged, num_regions, max_depth, the refusal counts, the region arrays' SHA-256 and every
leaf record); a call where they differ is reported and left out of the column.

Writes out/d1/variants_descent.json ({v: {sid: {all_depth20_below_refused, n_depth20, ...}}}, and the identity check).
Usage: python -B variants_descent_07c.py V [V ...]
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    out = ROOT / "out" / "d1" / "variants_descent.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    src = (HERE / "d1_descent_07c.py").read_text()
    doc, ident = {}, {}
    for v in sys.argv[1:]:
        s = src.replace('"replays" / "instr" /', f'"replays" / "{v}_i" /')
        if s.count(f'"{v}_i"') != 2:
            raise SystemExit("the directory name was not replaced exactly twice")
        ns = {"__name__": f"d1_{v}", "__file__": str(HERE / "d1_descent_07c.py")}
        exec(compile(s, f"d1_descent_07c[{v}_i]", "exec"), ns)
        doc[v], ident[v] = {}, {"equal": 0, "differ": []}
        for p in sorted((ROOT / "out" / "replays" / f"{v}_i").glob("*.instrument.json")):
            sid = p.name[: -len(".instrument.json")]
            a = json.loads((ROOT / "out" / "replays" / v / f"{sid}.json").read_text())
            b = json.loads((ROOT / "out" / "replays" / f"{v}_i" / f"{sid}.json").read_text())
            same = a["replay"] == b["replay"] and a["regions_sha256"] == b["regions_sha256"] and a["leaves"] == b["leaves"]
            if not same:
                ident[v]["differ"].append(sid)
                continue
            ident[v]["equal"] += 1
            nd = ns["analyse"](sid)[0]
            doc[v][sid] = {k: nd[k] for k in ("n_depth20", "all_depth20_below_refused", "descends_from_refused", "routes",
                                              "converged", "num_regions", "n_refused")}
    out.write_text(json.dumps(dict(doc, _identity=ident), indent=1))
    for v in doc:
        unc = [s for s, x in doc[v].items() if not x["converged"]]
        print(f"{v}: _i equals plain on {ident[v]['equal']} (differ {len(ident[v]['differ'])}); unconverged {len(unc)}, "
              f"of which every depth-20 leaf below a refused leaf {sum(1 for s in unc if doc[v][s]['all_depth20_below_refused'])}")
    print("wrote", out)


if __name__ == "__main__":
    main()
