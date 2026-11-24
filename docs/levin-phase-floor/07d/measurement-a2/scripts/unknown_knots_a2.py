"""levin-phase-floor prompt 07d Part A' (the orchestrator's go-ahead, item 3): why neither knot reader reads the knots of
the "unknown" leaves. For every leaf record: the call kind, whether its knots are known, and for the calls whose are not,
the shape of the integrand the readers walk (type of f[0], whether it has a closure, the type of theta's ``theta``, and
whether that is a bound method), with both readers' errors re-raised on one call per notify-label class. Also: every
analytic leaf's knot status, so the class is placed against all analytic leaves. New in Part A'.
Usage: PYTHONPATH=<tree>:<scripts> python -B unknown_knots_a2.py OUT
"""
import json
import sys
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prov import ROOT, header  # noqa: E402


def main():
    out = Path(sys.argv[1])
    if out.exists():
        raise SystemExit(f"{out} exists")
    import leafref_v4 as R4

    files = []
    for d in sorted((ROOT / "out" / "leaves_a2").iterdir()):
        if d.is_dir() and (d.name.startswith("partA_") or d.name.startswith("nb")):
            files += sorted(d.glob("*.json"))
    an = Counter()
    shapes = Counter()
    labels = Counter()
    examples = {}
    for p in files:
        doc = json.loads(p.read_text())
        Ls = [L for L in doc["leaves"] if "error" not in L]
        if not Ls or "|analytic|" not in Ls[0]["id"]:
            continue
        unk = [L for L in Ls if L["n_knots"] is None]
        an["analytic leaves"] += len(Ls)
        an["unknown"] += len(unk)
        if not unk:
            continue
        spec = json.loads(Path(doc["header"]["spec"]).read_text())
        rec = R4.load_capture(spec["capture"])
        f0 = rec["f"][0]
        th = rec["theta"]["theta"]
        sh = (type(f0).__name__, getattr(f0, "__qualname__", None), f0.__closure__ is not None if hasattr(f0, "__closure__") else "no attr",
              type(th).__name__, getattr(th, "__qualname__", None), hasattr(th, "__self__"))
        shapes[json.dumps(sh)] += len(unk)
        lab = " ".join(rec["notify_label"].split()[:3])
        labels[lab] += len(unk)
        if sh not in [tuple(json.loads(k)) for k in examples]:
            errs = []
            try:
                from tools.filon_reference import production_edges
                production_edges(rec)
            except Exception:  # noqa: BLE001
                errs.append(traceback.format_exc(limit=-2))
            try:
                list(zip(f0.__code__.co_freevars, [c_.cell_contents for c_ in f0.__closure__]))
            except Exception:  # noqa: BLE001
                errs.append(traceback.format_exc(limit=-1))
            examples[json.dumps(sh)] = {"leaf": unk[0]["id"], "notify_label": rec["notify_label"],
                                        "f0_freevars": list(getattr(getattr(f0, "__code__", None), "co_freevars", []) or []),
                                        "errors": errs}
    doc = {"header": header("levin-phase-floor 07d Part A': why the unknown-knot leaves' knots are not read", __file__, files[:1]),
           "counts": dict(an), "shapes": dict(shapes), "labels": dict(labels), "examples": examples}
    out.write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc["counts"]), json.dumps(doc["shapes"], indent=1), json.dumps(doc["labels"], indent=1))
    for k, v in examples.items():
        print(k, v["leaf"], v["notify_label"], v["f0_freevars"])
        for e in v["errors"]:
            print(e)


if __name__ == "__main__":
    main()
