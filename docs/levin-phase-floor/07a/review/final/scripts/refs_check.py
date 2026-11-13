"""Orchestrator's own scoring of the two references the user asked for at the W5 stop (3-678 and 1-34599, group
indices 2 and 3, both rtols). Read only; prints only. Independent of the census agent's scoring.
Each reading (census/out/refs/readings/r<row>-1e-7-<i>-<impl>-r<res>.json) is accepted only if: tool SHA-256 d37e633b...;
its input pickle's SHA-256 as recorded; route B; every non-monotone segment direct; value_30 rounds to value_hex.
The reference is FCC r2 (07's V4 rule). Spread = max - min over accepted readings; also |FCC r2 - FCC r1|. h's and 07's
value and abserr are read from the two census checkpoints at my own indices (line SHA-256 checked). Exact Fractions."""
import glob, hashlib, json
from decimal import Decimal
from fractions import Fraction as F
from pathlib import Path
M = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
RD = M / "var/levin-phase-floor-07a/census/out/refs/readings"
IX = M / "var/levin-phase-floor-07a/review/stop3"
RUNS = {"h": (M / "var/runs/levin-phase-floor-07a-census-20261112T132744", json.load(open(IX / "index07a.json"))),
        "07": (M / "var/runs/levin-phase-floor-07-census-20261108T171137", json.load(open(IX / "index07.json")))}
TOOL = "d37e633bf54a179114b2462529eaaa3836cccff8fe22e0250e6d36d61992c7be"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
print("script", sha(__file__))
def call(run, key, i):
    r, ix = RUNS[run]; e = ix["units"][key]
    with open(r / "checkpoint.jsonl", "rb") as f:
        f.seek(e[0]); raw = f.read(e[1])
    assert hashlib.sha256(raw).hexdigest() == e[2]
    d = json.loads(raw)["data"]; m = [c for c in d["calls"] + d["other_calls"] if c["index"] == i]
    assert len(m) == 1; return m[0]
fx = lambda v: F(float.fromhex(v["hex"]))
for row in ("3-678", "1-34599"):
    for i in (2, 3):
        rd = {}
        for p in sorted(glob.glob(str(RD / f"r{row}-1e-7-{i}-*.json"))):
            r = json.load(open(p))
            ok = (r["tool_sha256"] == TOOL and sha(r["pickle"]) == r["pickle_sha256"] and r["route"] == "B"
                  and float(r["value_30"]).hex() == r["value_hex"] and r["index"] == i)
            segs = r["segments"] if isinstance(r["segments"], list) else None
            print(f"  reading {Path(p).name} sha {sha(p)[:12]} checks {ok} segments_listed {segs is not None}")
            if ok:
                rd[f"{r['impl']}_r{r['res']}"] = F(Decimal(r["value_30"]))
        ref = rd["fcc_r2"]; spread = max(rd.values()) - min(rd.values()); d12 = abs(rd["fcc_r2"] - rd["fcc_r1"])
        print(f"{row} index {i}: readings {sorted(rd)}; ref (FCC r2) {float(ref)!r}; spread {float(spread):.4g}; |FCC r2 - r1| {float(d12):.4g}")
        for rt in ("1e-7", "1e-8"):
            key = f"{row}@{rt}"
            ch, c7 = call("h", key, i), call("07", key, i)
            assert ch["notify_label"] == c7["notify_label"]
            vh, eh, v7, e7 = fx(ch["value"]), fx(ch["abserr"]), fx(c7["value"]), fx(c7["abserr"])
            print(f"   {rt} {ch['group']}: abserr h {float(eh):.4g} 07 {float(e7):.4g}; |v-ref|/abserr h {float(abs(vh-ref)/eh):.4g}  07 {float(abs(v7-ref)/e7):.4g}; "
                  f"spread x10 <= min(abserr) {spread*10 <= min(eh, e7)}  (spread/min abserr {float(spread/min(eh,e7)):.3g}); |vh-v7|/abserr_h {float(abs(vh-v7)/eh):.4g}")
