"""levin-phase-floor prompt 06g section 5, the references: each capture's route, and Route B's default
edges, before any reading. New in 06g.

For every sliced capture (out/inputs.json): ``tools.filon_reference.route_of``; for Route A, the
tool's ``RouteA`` (K exactly, C, the representation switches ``u_breaks``); for Route B,
``production_edges`` (06i's closure walk), which raises on a capture whose closures differ from the
layout 06i read. The brief: such a capture's edges "must be passed", each case a deviation; where
they would come from is not settled, so a raise here is a stop (the driver runs no reading). No
integral is formed.

Writes out/routes.json; exits 1 if any capture raises. Reads only.

Usage: PYTHONPATH=<tree g>:<this directory> python -B routes_refs.py INPUTS_JSON
"""

import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402


def main():
    inputs = json.loads(Path(sys.argv[1]).read_text())["captures"]
    import mpmath as mp
    import tools.filon_reference as F

    ins = [Path(F.__file__)] + [Path(v["pickle"]) for v in inputs.values()]
    for l in header(
        Path(__file__).resolve(),
        "levin-phase-floor 06g: the references' routes and Route B's default edges",
        ins,
    ):
        print(l)
    out = {}
    n_bad = 0
    for sid in sorted(inputs):
        v = inputs[sid]
        row = {"unit": v["unit"], "index": v["index"], "kind": v["kind"]}
        try:
            call = F._load_pickle(v["pickle"], "ray.cloudpickle")
            row["route"] = F.route_of(call)
            row["x_span"] = [float(x).hex() for x in call["x_span"]]
            if row["route"] == "A":
                A = F.RouteA(call)
                with mp.workdps(F.DPS):
                    row["K_exact"] = str(A.K_exact)
                    row["K_float"] = float(A.K_exact)
                    row["C_30"] = mp.nstr(A.C_mp(), 30)
                row["u_breaks"] = A.u_breaks()
                row["orders"] = [str(p.nu) for p in A.group.phases]
            else:
                edges, n_tables = F.production_edges(call)
                row["n_edges"] = len(edges)
                row["n_segments"] = len(edges) - 1
                row["n_tables"] = n_tables
            row["error"] = None
        except Exception:
            row["error"] = traceback.format_exc()
            n_bad += 1
        expect = "A" if v["kind"] == "analytic" else "B"
        row["route_as_expected"] = row.get("route") == expect
        out[sid] = row
        print(
            f"{sid} {v['kind']:8s} route {row.get('route')}"
            + (f" (expected {expect})" if not row["route_as_expected"] else "")
            + (
                f"; segments {row.get('n_segments')} tables {row.get('n_tables')}"
                if row.get("route") == "B"
                else ""
            )
            + (
                f"; K {row.get('K_float') or float('nan'):.6g} breaks {len(row.get('u_breaks') or [])} orders {row.get('orders')}"
                if row.get("route") == "A"
                else ""
            )
            + (f"; ERROR {row['error'].splitlines()[-1]}" if row["error"] else "")
        )
    dst = RC.ROOT / "out" / "routes.json"
    if dst.exists():
        raise SystemExit(f"{dst} exists; not overwritten")
    dst.write_text(json.dumps(out, indent=1))
    n_unexp = sum(1 for r in out.values() if not r["route_as_expected"])
    print(
        f"{len(out)} captures; {n_bad} raised; {n_unexp} not on the expected route; wrote {dst} sha256 {RC.sha256(dst)}"
    )
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
