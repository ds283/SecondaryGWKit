"""levin-phase-floor prompt 07d Part A, section 2.6: Route A's cost per panel, for the projection, before any reading.
New in 07d.

07c's projection model has no Route A term (07c read no analytic block). This script times the tool's own Route A chunk
functions (``_route_a_fcc_chunk``, ``_route_a_qawo_chunk``, at the tool's DPS, in one process) on the first NP panels of
the tool's own panel lists (``RouteA.panels``) of a few drawn analytic integrands spanning |K|, at both resolutions. **No
value is kept or printed:** the chunk's result is discarded, so nothing here is a reading; only the wall per panel is
recorded. Writes out/refs/costA.json. Usage: PYTHONPATH=<tree>:<scripts> python -B refs_costA_07d.py KEY [KEY ...]
(keys of out/refs/inputs_ra.json or inputs_rt.json)
"""

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from prov import header, sha256  # noqa: E402

NP = {"fcc": 6, "qawo": 40}


def main():
    import mpmath as mp
    import ray.cloudpickle as cp

    import tools.filon_reference as F

    out_p = ROOT / "out" / "refs" / "costA.json"
    if out_p.exists():
        raise SystemExit(f"{out_p} exists")
    ins = {}
    srcs = []
    for s in ("ra", "rt"):
        p = ROOT / "out" / "refs" / f"inputs_{s}.json"
        ins |= json.loads(p.read_text())["inputs"]
        srcs.append(p)
    rows = {}
    for key in sys.argv[1:]:
        e = ins[key]
        call = cp.loads(Path(e["pickle"]).read_bytes())
        st = F._make_state(call, "A", None)
        A = st["route"]
        r = {"K_abs": e["features"]["K_abs"], "per_panel_s": {}}
        with mp.workdps(F.DPS):
            for res in (1, 2):
                pf = A.panels("fcc", res)[: NP["fcc"]]
                t0 = time.perf_counter()
                F._route_a_fcc_chunk(st, pf, F.ROUTE_A_FCC[res]["N"])
                r["per_panel_s"][f"fcc_r{res}"] = (time.perf_counter() - t0) / len(pf)
                # the first NP QAWO panels of the first u-segment (the tool's dyadic tiling), not the whole list
                cfg = F.ROUTE_A_QAWO[res]
                ue = A.u_segments(cfg["du"])
                l, rr = mp.exp(ue[0]), mp.exp(ue[1])
                h_max = min(cfg["Pmax"] / (2 * r["K_abs"]), float(rr - l) / 2)
                h = 2.0 ** int(mp.floor(mp.log(h_max, 2)))
                n = min(NP["qawo"], int(mp.floor((rr - l) / (2 * h))))
                pq = [(l + h + 2 * h * j, h) for j in range(n)] if n > 0 else F.dyadic_tiling(l, rr, h_max)[: NP["qawo"]]
                t0 = time.perf_counter()
                F._route_a_qawo_chunk(st, pq, cfg["epsrel"])
                r["per_panel_s"][f"qawo_r{res}"] = (time.perf_counter() - t0) / len(pq)
        rows[key] = r
        print(key, json.dumps(r), flush=True)
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07d Part A: Route A cost per panel (no values kept)",
                 srcs + [Path(F.__file__)])
    out_p.write_text(json.dumps({"header": hdr, "np": NP, "rows": rows}, indent=1))
    print("wrote", out_p, sha256(out_p))


if __name__ == "__main__":
    main()
