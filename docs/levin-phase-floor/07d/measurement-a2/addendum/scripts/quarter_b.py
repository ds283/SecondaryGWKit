"""levin-phase-floor 07d Part A' addendum, item B.2: the sub-solve that changes route at 12 pi in the trust check of
V93T12's returned region on R:a:W2-0-12977-5: the right half's right quarter (cR, b) of [0x1.3744465540140p+4,
0x1.45b4e8df10b9ep+4] (checkrepro_a2's sixth sub-solve), with its sibling (c, cR) beside it.
Mode "tree": each quarter's V, its Levin value (the tree's solve with the gate off), its CC_93 (checkrepro_a2's cc93, as
the check takes it) and its ref48 (state/rule_ref48_add.md A.2, sigma_hat from leafref_v4.v3_sigma).
Mode "variant": V93T12's own force_direct solve of each quarter: its value, abserr_direct and phase_err (the declaration
the variant's check would carry for a CC sub-solve). Writes out/quarter_b_<mode>.json.
Usage: PYTHONPATH=<tree>:<a2 scripts>:<addendum scripts> python -B quarter_b.py tree|variant"""
import json, math, pickle, sys
from pathlib import Path
ADD = Path(__file__).resolve().parent.parent
A2 = ADD.parent
PARTA = A2.parent / "measurement"
SID = "R:a:W2-0-12977-5"
fx = float.fromhex
mode = sys.argv[1]
if mode == "variant":
    import variant_loader as VL
    vj = json.loads((A2 / "out" / "variants" / "variants.json").read_text())["variants"]["V93T12"]
    LQ = VL.install(Path(vj["path"]), vj["sha256"], "a2_V93T12")
else:
    import AdaptiveLevin.levin_quadrature as LQ
import checkrepro_a2 as CR
import leafref_v4 as R4
from prov import header
from ref48_add import gl
from replay_07d import load_capture
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
a, b = fx("0x1.3744465540140p+4"), fx("0x1.45b4e8df10b9ep+4")
cc = (b + a) / 2.0
cR = (b + cc) / 2.0
out = {"sid": SID, "mode": mode, "quarters": {}}
for name, (x0, x1) in {"RR (cR, b), sixth": (cR, b), "RL (c, cR), fifth": (cc, cR)}.items():
    grid, D = LQ.chebyshev_matrices((x0, x1), C.N)
    V = float(C.A.build_Levin_data(grid, D, need_AmatT=False, vectorize_cache=C.cache)[4])
    q = {"start": x0.hex(), "end": x1.hex(), "V_over_pi": V / math.pi}
    if mode == "variant":
        d = LQ._adaptive_levin_subregion((x0, x1), rec["f"], C.A, id_label=C.id_label, chebyshev_order=C.N, rtol=C.rtol,
                                         vectorize_cache=C.cache, force_direct=True)
        q.update(value=float(d["value"]).hex(), abserr_direct=float(d["abserr_direct"]),
                 phase_err=float(d.get("phase_err") or 0.0),
                 declared=max(float(d["abserr_direct"]), float(d.get("phase_err") or 0.0)))
    else:
        G = R4.Integrand(LQ, rec)
        edges, _, _ = R4.call_knots(rec, min(rec["x_span"]), max(rec["x_span"]))
        kn = R4.interior(edges, x0, x1)
        brk = [x0] + (kn or []) + [x1]
        sg = R4.v3_sigma(G, x0, x1, kn, V)["sigma_hat"]
        lv = {n: gl(G, brk, n) for n in (512, 1024)}
        ch = abs(lv[1024][0] - lv[512][0]); noise = 2 * sg * math.sqrt(lv[1024][1]); fin = 1024
        if ch > noise:
            lv[2048] = gl(G, brk, 2048); ch = abs(lv[2048][0] - lv[1024][0]); noise = 2 * sg * math.sqrt(lv[2048][1]); fin = 2048
        ref = lv[fin][0]
        lev = C.solve((x0, x1), -math.inf)
        c93 = C.cc93((x0, x1))
        q.update(ref48=ref.hex(), eps48=ch + noise, change=ch, noise=noise, finest=fin,
                 levin_value=float(lev["value"]).hex(), levin_err=float(lev["value"]) - ref,
                 levin_decl=max(float(lev.get("abserr") or 0.0), float(lev.get("phase_err") or 0.0)),
                 cc93=float(c93).hex(), cc93_err=float(c93) - ref)
    out["quarters"][name] = q
    print(name, json.dumps(q))
out["header"] = header(f"levin-phase-floor 07d Part A' addendum: item B.2, the quarters ({mode})", __file__,
                       [A2 / "out" / "replays" / "V93T12" / "R_a_W2-0-12977-5.json"])
(ADD / "out" / f"quarter_b_{mode}.json").write_text(json.dumps(out, indent=1))
