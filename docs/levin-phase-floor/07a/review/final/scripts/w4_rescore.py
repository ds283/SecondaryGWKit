"""Orchestrator's own re-score of W4 (levin-phase-floor 07a Part C, §C.2). Read only but for OUT_JSON.

usage: python -I -B w4_rescore.py OUT_JSON

Adapted from 07's review/scripts/v4_rescore.py; independent of the census agent's scripts. Reads:
- the 24 calls' identities from 07's census/state/v4_selection.json (fixed before 07's census);
- h's and 07's value, abserr and converged from the two census checkpoints, at my own indices (review/stop3/index07a.json,
  index07.json), each line accepted only if its SHA-256 matches;
- each reading from 07's census/out/v4/readings/<sid>-<impl>-r<res>.json, accepted only if: its tool SHA-256 is d37e633b...;
  its input pickle has the SHA-256 the reading records; route B; every non-monotone segment `direct`; value_30 rounds to
  value_hex;
- z00, z12 and Z3's seven from 06g's scores_stage3/scores.json rows (committed; unchanged since e182bc7a), 06g's
  `unusable` left out, the reference 06g's FCC r2.
Rule (07's, as ruled at 07's stop 4): readings agree iff max - min <= tol; reference FCC r2; honest iff |v - ref| <= abserr.
tol is computed two ways and both reported: tol_h7 = min(abserr_h, abserr_07)/100 (07's rule with 07 as the before),
tol_07 = 07's own tol, min(abserr_07, abserr_06m)/100, taken from 07's agent's v4_scores.json. Exact Fractions throughout.
Each call's class (window_both / window_h_only / own / aux_only / untouched) from h's route_07a and the two guard_07s.
"""
import hashlib, json, sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
C07 = MAIN / "var/levin-phase-floor-07/census"
R7A = MAIN / "var/runs/levin-phase-floor-07a-census-20261112T132744"
R07 = MAIN / "var/runs/levin-phase-floor-07-census-20261108T171137"
IX = MAIN / "var/levin-phase-floor-07a/review/stop3"
S06G = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-4bf50d3/docs/levin-phase-floor/06g/references/out/scores_stage3/scores.json")
TOOL = "d37e633bf54a179114b2462529eaaa3836cccff8fe22e0250e6d36d61992c7be"
ZS = ["z00", "z12", "z07", "z16", "z18", "z21", "z24", "z28", "z29"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fx(v):
    return Fraction(float.fromhex(v["hex"] if isinstance(v, dict) else float(v).hex()))


def line(run, e):
    with open(run / "checkpoint.jsonl", "rb") as f:
        f.seek(e[0]); raw = f.read(e[1])
    assert hashlib.sha256(raw).hexdigest() == e[2]
    return json.loads(raw)["data"]


def call_of(d, i):
    m = [c for c in d["calls"] + d["other_calls"] if c["index"] == i]
    assert len(m) == 1
    return m[0]


def klass(ch, c7):
    gh, g7 = ch["guard_07"], c7["guard_07"]
    lh = gh["n_points"] > 0 or gh["n_windows"] > 0
    l7 = g7["n_points"] > 0 or g7["n_windows"] > 0
    if lh and l7: return "window_both"
    if lh: return "window_h_only"
    if l7: return "window_07_only"
    ra = ch["route_07a"]
    return "own" if ra["own_differs"] else ("aux_only" if ra["aux_differs"] else "untouched")


def main():
    out = Path(sys.argv[1])
    i7a = json.load(open(IX / "index07a.json")); i07 = json.load(open(IX / "index07.json"))
    assert i7a["sha256"] == "461df2ec66969a113e9f6d590f05a4d5b300d7c1e0cb62d05c7385d23e8de56b"
    assert i07["sha256"] == "03dfe786f69343a0e86f436936161ace728630f26d687061e186d0883dd5cb10"
    sel = json.loads((C07 / "state/v4_selection.json").read_text())["calls"]
    s06g = {r["sid"]: r for r in json.loads(S06G.read_text())["rows"]}
    ag07 = {x["sid"]: x for x in (lambda a: a["calls"] + a["z"])(json.loads((C07 / "out/v4/v4_scores.json").read_text()))}
    todo = [(c["sid"], c["unit"], c["index"], None) for c in sel] + [(z, s06g[z]["unit"], s06g[z]["index"], z) for z in ZS]
    rows, problems = [], []
    for sid, u07, i, z in todo:
        assert u07.startswith("g-1e-7-") or u07.startswith("g-1e-8-"), u07
        rt, row = u07[2:6], u07[7:]
        key = row + "@" + rt
        dh, d7 = line(R7A, i7a["units"][key]), line(R07, i07["units"][key])
        assert d7["unit"] == u07
        ch, c7 = call_of(dh, i), call_of(d7, i)
        assert ch["notify_label"] == c7["notify_label"] and json.dumps(ch["x_span"]) == json.dumps(c7["x_span"])
        if z is None:
            rd = {}
            for impl in ("qawo", "fcc"):
                for res in (1, 2):
                    p = C07 / f"out/v4/readings/{sid}-{impl}-r{res}.json"
                    if not p.exists():
                        continue
                    r = json.loads(p.read_text())
                    ok = (r["tool_sha256"] == TOOL and sha(r["pickle"]) == r["pickle_sha256"] and r["unit"] == u07
                          and r["index"] == i and r["route"] == "B"
                          and all(s["direct"] for s in r["segments"] if not s["monotone"])
                          and float(r["value_30"]).hex() == r["value_hex"])
                    if ok:
                        rd[f"{impl}_r{res}"] = Fraction(Decimal(r["value_30"]))
                    else:
                        problems.append((sid, impl, res))
        else:
            r6 = s06g[z]
            rd = {k: Fraction(Decimal(v["value_30"])) for k, v in r6["readings"].items() if k not in r6["unusable"]}
            assert Fraction(Decimal(r6["ref_30"])) == rd["fcc_r2"]
        vh, eh, v7, e7 = fx(ch["value"]), fx(ch["abserr"]), fx(c7["value"]), fx(c7["abserr"])
        ref = rd["fcc_r2"]; spread = max(rd.values()) - min(rd.values())
        tol_h7 = min(eh, e7) / 100
        tol07 = Fraction(ag07[sid]["tol"]) if "tol" in ag07[sid] else None
        rh, r7 = abs(vh - ref) / eh, abs(v7 - ref) / e7
        agree_h7 = spread <= tol_h7
        agree_07 = (spread <= tol07) if tol07 is not None else None
        rows.append(dict(sid=sid, key=key, index=i, cls=klass(ch, c7), basis=sorted(rd),
                         conv_h=ch["converged"], conv_07=c7["converged"],
                         bitequal_h_07=ch["value"] == c7["value"] and ch["abserr"] == c7["abserr"],
                         ratio_h=float(rh), ratio_07=float(r7), abserr_h_over_07=float(eh / e7),
                         spread_over_tol_h7=float(spread / tol_h7), agree_h7=agree_h7, agree_07tol=agree_07,
                         verdict=("honest" if rh <= 1 else "NOT honest") if agree_h7 else "reported, not scored",
                         ratio_07_agent=ag07[sid]["ratio_g"]))
    res = dict(script_sha256=sha(__file__), s06g_sha256=sha(S06G), rows=rows, reading_problems=problems,
               not_honest=[r["sid"] for r in rows if r["verdict"] == "NOT honest"],
               not_scored=[r["sid"] for r in rows if r["verdict"] == "reported, not scored"],
               agree_differs_by_tol=[r["sid"] for r in rows if r["agree_07tol"] is not None and r["agree_07tol"] != r["agree_h7"]])
    out.write_text(json.dumps(res, indent=1, default=str))
    print("# w4_rescore.py sha256", res["script_sha256"])
    print(f"{'sid':4} {'class':13} {'conv h/07':9} {'bit=':5} {'sprd/tol':>9} {'ratio_h':>9} {'ratio_07':>9} {'e_h/e_07':>9}  verdict")
    for r in rows:
        print(f"{r['sid']:4} {r['cls']:13} {str(r['conv_h'])[0]}/{str(r['conv_07'])[0]:7} {str(r['bitequal_h_07'])[0]:5} "
              f"{r['spread_over_tol_h7']:9.3g} {r['ratio_h']:9.3g} {r['ratio_07']:9.3g} {r['abserr_h_over_07']:9.3g}  {r['verdict']}")
    print("reading problems:", problems)
    print("not honest:", res["not_honest"], " not scored:", res["not_scored"], " agree differs by tol:", res["agree_differs_by_tol"])
    print("ratio_07 equals 07 agent's ratio_g in all:", all(abs(r["ratio_07"] - r["ratio_07_agent"]) <= 1e-15 * max(1, r["ratio_07"]) for r in rows))


if __name__ == "__main__":
    main()
