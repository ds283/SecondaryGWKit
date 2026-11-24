"""levin-phase-floor 07d Part A' addendum, item B.2: the trust check of the region V93T12 returns trusted on the call
R:a:W2-0-12977-5 (span [0x1.3744465540140p+4, 0x1.45b4e8df10b9ep+4], depth 4), and of its two halves (each check's values estimate its own span: the halves' are scored against nothing here;
the region's V_P, halves' sum and Q against the region's ref48), reproduced by
checkrepro_a2.CallContext.check (bit for bit with I6 at 6 pi, measurement-a2/out/repro/verify_repro_a2.json) at
T = 6, 8, 10, 12 pi under the CC_93 fallback, with the V93 family's declaration (pair93, kappa 4) for the CC sub-solves as
item 1 took it; and the region's ref48 from out/dissect_b_meas (the registered run add-dissect-b). Usage: PYTHONPATH=<tree>:<a2 scripts>:<addendum scripts> python -B check_b_region.py"""
import json, math, pickle, sys
from pathlib import Path
ADD = Path(__file__).resolve().parent.parent
A2 = ADD.parent
PARTA = A2.parent / "measurement"
SID = "R:a:W2-0-12977-5"
fx = float.fromhex
import AdaptiveLevin.levin_quadrature as LQ
import checkrepro_a2 as CR
import leafref_v4 as R4
from prov import TREE_LQ_SHA, header, sha256
from ref48_add import gl
from replay_07d import load_capture
assert sha256(LQ.__file__) == TREE_LQ_SHA
man = {}
for p in sorted((PARTA / "state").glob("calls_07d*.json")):
    man.update(json.loads(p.read_text())["calls"])
c = man[SID]
rec = load_capture(c)
kw = dict(rec["kw"])
if c["override"]:
    census = {}
    for p in sorted((PARTA / "state").glob("census_refs_07d*.pkl")):
        census.update(pickle.loads(p.read_bytes())["calls"])
    cen = None if c["census"].startswith("r32:") else census[c["census"]]
    kw["rtol"] = c["override"]["rtol"]
    kw["atol"] = c["override"]["atol"] if c["override"].get("atol") is not None else cen["call"]["atol"]
C = CR.CallContext(LQ, rec, kw)
G = R4.Integrand(LQ, rec)
edges, _, _ = R4.call_knots(rec, min(rec["x_span"]), max(rec["x_span"]))
a, b = fx("0x1.3744465540140p+4"), fx("0x1.45b4e8df10b9ep+4")
m = 0.5 * (a + b)
out = {"sid": SID, "regions": {}}
meas = {}
for q in sorted((ADD / "out" / "dissect_b_meas").glob("*.json")):
    mq = json.loads(q.read_text())
    meas[(mq["start"], mq["end"])] = mq
for name, (x0, x1) in {"region": (a, b), "left half": (a, m), "right half": (m, b)}.items():
    grid, D = LQ.chebyshev_matrices((x0, x1), C.N)
    V = float(C.A.build_Levin_data(grid, D, need_AmatT=False, vectorize_cache=C.cache)[4])
    mq = meas[(a.hex(), b.hex())]
    ref = fx(mq["ref48"])
    ch, noise, fin = mq["change"], mq["noise"], mq["finest"]
    rows = {}
    for Tp in (6, 8, 10, 12):
        ch_ = C.check(x0, x1, Tp * math.pi, "cc93")
        keep = {k: ch_.get(k) for k in ("trusted", "refused_by_rate_rule", "rule", "r", "delta1", "delta2", "nu_B", "V_P",
                                         "halves_sum", "Q", "sub_routes", "residual", "own_route", "total_err_trusted",
                                         "total_err_depth_cap", "value_returned")}
        for k in ("V_P", "halves_sum", "Q"):
            if keep.get(k) and name == "region":
                keep[f"err_{k}"] = fx(keep[k]) - ref
        rows[str(Tp)] = keep
    out["regions"][name] = {"start": x0.hex(), "end": x1.hex(), "V_over_pi": V / math.pi, "ref48": ref.hex(), "change": ch,
                            "noise": noise, "finest": fin, "checks": rows}
    print(name, f"V/pi {V / math.pi:.3g} ref48 {ref:.6e} eps {ch + noise:.3g}")
    for Tp, k in rows.items():
        print(f"  T {Tp}pi trusted {k['trusted']} r {k['r']} sub_routes {k['sub_routes']} D {k['total_err_trusted']} "
              f"errQ {k.get('err_Q')} errVP {k.get('err_V_P')} errH {k.get('err_halves_sum')}")
out["header"] = header("levin-phase-floor 07d Part A' addendum: item B.2, the trust check of V93T12's returned region",
                       __file__, [A2 / "out" / "replays" / "V93T12" / "R_a_W2-0-12977-5.json"])
(ADD / "out" / "check_b_region.json").write_text(json.dumps(out, indent=1, default=str))
