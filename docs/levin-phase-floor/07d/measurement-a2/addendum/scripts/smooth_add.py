"""levin-phase-floor 07d Part A' addendum, item A.3: the smooth-phase reference on the leaves whose ref48 verdict the noise
term decides (state/second_pass_add.json; the rule: state/rule_ref48_add.md, A.3).

tools.filon_reference.reference on the captured call with x_span replaced by the leaf's [start, end]; route by route_of;
edges for route B the leaf's pieces (leafref_v4.call_knots restricted to the leaf, its ends included); workers 1.
Value: the FCC resolution-2 reading (the tool's ``value_hex``); stated error eps_s = scatter of the four readings +
2 x the FCC resolution-2 reading's fit_noise_1sigma (route B; 0 for route A). Writes OUT_DIR/<leaf>.json; refuses to
overwrite. ``--first K`` takes only the first K leaves in the file's order (the cost probe); else leaves are dealt
round-robin to N workers, worker I taking its share.
Usage: PYTHONPATH=<tree>:<a2 scripts>:<addendum scripts> python -B smooth_add.py OUT_DIR I N [--first K | --list FILE]
"""

import datetime
import hashlib
import json
import socket
import sys
import time
from pathlib import Path

ADD = Path(__file__).resolve().parent.parent
fx = float.fromhex


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out_dir, part, nparts = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    first = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[4] == "--first" else None
    sel = json.loads((ADD / "state" / "selection_add.json").read_text())
    lst = sys.argv[5] if len(sys.argv) > 5 and sys.argv[4] == "--list" else ADD / "state" / "second_pass_add.json"
    sp = json.loads(Path(lst).read_text())
    ids = sp["leaves"][:first] if first else sp["leaves"][part::nparts]
    import AdaptiveLevin.levin_quadrature as LQ
    import leafref_v4 as R4
    from prov import TREE_LQ_SHA
    from tools import filon_reference as FR

    if sha(LQ.__file__) != TREE_LQ_SHA:
        raise SystemExit("the module is not the tree's")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in ids:
        p = out_dir / (i.replace("|", "_").replace("#", "_") + ".json")
        if p.exists():
            raise SystemExit(f"{p} exists")
        lv = sel["leaves"][i]
        doc = json.loads(Path(lv["file"]).read_text())
        spec = json.loads(Path(doc["header"]["spec"]).read_text())
        rec = R4.load_capture(spec["capture"])
        xs0, xs1 = (fx(v) for v in spec["x_span"])
        a, b = fx(lv["start"]), fx(lv["end"])
        edges_all, _, _ = R4.call_knots(rec, xs0, xs1)
        kn = R4.interior(edges_all, a, b)
        call = dict(rec)
        call["x_span"] = (a, b)
        route = FR.route_of(call)
        edges = [a] + (kn or []) + [b] if route == "B" else None
        t0 = time.perf_counter()
        err = None
        try:
            ref = FR.reference(call, 0.0, route=route, edges=edges, workers=1)
        except Exception as e:  # noqa: BLE001
            ref, err = None, repr(e)[:400]
        wall = time.perf_counter() - t0
        r = {"header": {"title": f"levin-phase-floor 07d Part A' addendum: smooth-phase reference on {i}",
                        "written": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "host": socket.gethostname(), "script": str(Path(__file__).resolve()), "script_sha256": sha(__file__),
                        "filon_reference_sha256": sha(FR.__file__), "leaf_record": lv["file"]},
             "id": i, "route": route, "n_edges": None if edges is None else len(edges), "error": err, "wall_s": wall}
        if ref is not None:
            fcc2 = next(x for x in ref["readings"] if x["impl"] == "fcc" and x["res"] == 2)
            fn1 = ((fcc2.get("tmap") or {}).get("fit_noise_1sigma") or 0.0) if route == "B" else 0.0
            r.update(value_hex=ref["value_hex"], value_30=ref["value_30"], scatter=ref["scatter"], fit_noise_1sigma=fn1,
                     eps_s=ref["scatter"] + 2 * fn1, counts=None,
                     readings=[{k: x.get(k) for k in ("impl", "res", "value_hex", "value_30", "imag_30", "wall_s")}
                               for x in ref["readings"]],
                     tmap=fcc2.get("tmap"), jitter=fcc2.get("jitter"))
        p.write_text(json.dumps(r, indent=1, default=str))
        print(i, route, "error" if err else f"scatter {r['scatter']:.3g}", f"{wall:.1f} s", flush=True)


if __name__ == "__main__":
    main()
