"""levin-phase-floor prompt 07a section 6, the user's references (new in 07a; read only): are the captured integrands the
same? For each selected call, its capture at h (out/refs/captures_h.json) and its capture with the parent's module
(out/refs/captures_parent.json) are unpickled (each blob's SHA-256 checked) and compared: x_span (hex), the keyword
arguments, and the integrand's two components f[0], f[1], the raw phase theta["theta"], theta["theta_deriv"] and
theta["theta_mod_2pi"] sampled at 1001 points evenly spaced over x_span, by bit pattern. Then, among the h captures, the
same comparison across the two rtols of one call (kw's rtol and atol aside) and between index 2 (G+q-r) and index 3
(G-q+r) of one row and rtol. Writes OUT_JSON. Usage: PYTHONPATH=<tree h>:<scripts> python -B refs_integrand_07a.py OUT_JSON
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import WIDER, header, sha256  # noqa: E402

CH = WIDER / "out" / "refs" / "captures_h.json"
CP = WIDER / "out" / "refs" / "captures_parent.json"


def load(e):
    import ray.cloudpickle as cp

    with open(Path(e["dir"]) / e["pickle"], "rb") as f:
        f.seek(e["offset"])
        blob = f.read(e["length"])
    if hashlib.sha256(blob).hexdigest() != e["sha256"]:
        raise SystemExit(f"{e['sid']}: sha256 differs")
    return cp.loads(blob)


def samples(rec):
    a, b = rec["x_span"]
    xs = np.linspace(min(a, b), max(a, b), 1001)
    th = rec["theta"]
    out = {"f0": [float(rec["f"][0](x)) for x in xs], "f1": [float(rec["f"][1](x)) for x in xs],
           "theta": [float(th["theta"](x)) for x in xs], "theta_deriv": [float(th["theta_deriv"](x)) for x in xs]}
    if "theta_mod_2pi" in th:
        out["theta_mod_2pi"] = [float(th["theta_mod_2pi"](x)) for x in xs]
    return {k: [v_.hex() for v_ in v] for k, v in out.items()}


def cmp(r1, s1, r2, s2, kw_aside=()):
    res = {"x_span": [float(x).hex() for x in r1["x_span"]] == [float(x).hex() for x in r2["x_span"]]}
    k1 = {k: repr(v) for k, v in r1["kw"].items() if k not in kw_aside}
    k2 = {k: repr(v) for k, v in r2["kw"].items() if k not in kw_aside}
    res["kw"] = k1 == k2
    if k1 != k2:
        res["kw_differs"] = sorted(k for k in set(k1) | set(k2) if k1.get(k) != k2.get(k))
    for k in s1:
        res[k] = s1[k] == s2.get(k)
    res["all_equal"] = all(v for k, v in res.items() if k != "kw_differs")
    return res


def main():
    out_p = Path(sys.argv[1])
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07a: the reference calls' integrands compared", [CH, CP])
    H = {e["sid"]: e for e in json.loads(CH.read_text())}
    Pa = {e["sid"]: e for e in json.loads(CP.read_text())}
    recs, samp = {}, {}
    for sid in sorted(H):
        recs[("h", sid)] = load(H[sid])
        recs[("p", sid)] = load(Pa[sid])
        samp[("h", sid)] = samples(recs[("h", sid)])
        samp[("p", sid)] = samples(recs[("p", sid)])
    doc = {"header": hdr, "h_vs_parent": {}, "across_rtol": {}, "index2_vs_index3": {}}
    for sid in sorted(H):
        doc["h_vs_parent"][sid] = cmp(recs[("h", sid)], samp[("h", sid)], recs[("p", sid)], samp[("p", sid)])
    for sid in sorted(H):
        if "-1e-7-" in sid:
            s8 = sid.replace("-1e-7-", "-1e-8-")
            doc["across_rtol"][f"{sid}~{s8}"] = cmp(recs[("h", sid)], samp[("h", sid)], recs[("h", s8)], samp[("h", s8)], ("rtol", "atol"))
        if sid.endswith("-2"):
            s3 = sid[:-2] + "-3"
            doc["index2_vs_index3"][f"{sid}~{s3}"] = cmp(recs[("h", sid)], samp[("h", sid)], recs[("h", s3)], samp[("h", s3)])
    doc["kw_example"] = {k: repr(v) for k, v in recs[("h", sorted(H)[0])]["kw"].items()}
    out_p.write_text(json.dumps(doc, indent=1))
    for sec in ("h_vs_parent", "across_rtol", "index2_vs_index3"):
        for k, v in doc[sec].items():
            print(sec, k, "ALL EQUAL" if v["all_equal"] else {kk: vv for kk, vv in v.items() if vv is not True})
    print("kw", doc["kw_example"])
    print(f"wrote {out_p} sha256 {sha256(out_p)}")


if __name__ == "__main__":
    main()
