"""Orchestrator's own re-score of the rehearsal agent's 'which store understates' table.
Values and abserrs read directly from R30's and R31's stores (my stop-1 reader); the reference value
and scatter from each readings file (FCC r2 = top-level value_30); exact arithmetic in Fractions.
The reference is in the captured call's own units, so v and e are the replays' call-level outputs
(refcap_replay_a07.json for R30, refcap_replay_h.json for R31); each replay's stored_value/stored_abserr is
checked here against the store record read by my own reader.
usage: python -I -B understates_rescore.py <understates.json> <readings dir> <R30 primary> <R31 primary> <replay_a07.json> <replay_h.json>"""
import json, re, sys, hashlib
from fractions import Fraction as F
from decimal import Decimal
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stop1" / "scripts"))
from qsi_compare_07b import read

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
u = json.load(open(sys.argv[1])); rd = Path(sys.argv[2])
print("script", sha(__file__)); print("input", sha(sys.argv[1]), sys.argv[1])
A, B = read(sys.argv[3]), read(sys.argv[4])
REP = {"R30": {c["sid"] if "sid" in c else Path(c["pickle"]).stem: c for c in json.load(open(sys.argv[5]))["calls"]},
       "R31": {c["sid"] if "sid" in c else Path(c["pickle"]).stem: c for c in json.load(open(sys.argv[6]))["calls"]}}
for a in sys.argv[5:7]: print("input", sha(a), a)
def find(S, key, rtol, path):
    hits = [v for kk, v in S.items() if (float.fromhex(kk[0]), float.fromhex(kk[1]), float.fromhex(kk[2]), float.fromhex(kk[3])) == tuple(key) and kk[5] == rtol]
    assert len(hits) == 1, (key, len(hits))
    md = hits[0][1]
    m = re.fullmatch(r"WKB_Levin\.subintervals\[(\d+)\]\.groups\[(\d+)\]", path)
    return md["WKB_Levin"]["subintervals"][int(m[1])]["groups"][int(m[2])]
for row in u["rows"]:
    f = rd / (row["sid"] + ".json"); r = json.load(open(f))
    ref = F(Decimal(r["value_30"])); s = F(r["scatter"])
    assert r["value_hex"] == r["readings"][3]["value_hex"] and r["readings"][3]["impl"] == "fcc" and r["readings"][3]["res"] == 2
    out = [row["sid"], row["label"], tuple(row["key"])]
    for name, S in (("R30", A), ("R31", B)):
        g = find(S, row["key"], -7.0, row["record"]); assert g["label"] == row["label"]
        c = REP[name][row["sid"]]["got"]
        assert c["stored_value"] == g["value"].hex() and c["stored_abserr"] == g["abserr"].hex(), (name, row["sid"])
        assert c["regions"] == g["regions"] and c["converged"] == g["converged"]
        v, e = F(float.fromhex(c["value"])), F(float.fromhex(c["abserr"]))
        d = abs(v - ref)
        verdict = "understates" if d - e > s else ("honest" if e - d > s else "undecided")
        out.append("%s %.4g (v %s e %s regions %d conv %s) %s" % (name, float(d / e), g["value"].hex(), g["abserr"].hex(), g["regions"], g["converged"], verdict))
    out.append("s/min(e) %.3g" % float(s / min(F(float.fromhex(REP[n][row["sid"]]["got"]["abserr"])) for n in ("R30", "R31"))))
    print(" | ".join(str(x) for x in out))
