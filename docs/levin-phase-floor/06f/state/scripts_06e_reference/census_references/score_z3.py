"""levin-phase-floor prompt 06e section 5: Z3 scored by state/predictions.md Part B. New in 06e.
Reads only: references/out/ref-sNN-P128.txt and -P64.txt (samples 2, 3, 8, 10, 13, 14, 15, 22, 23,
27), the orchestrator's 02-ref-sNN.txt and 04-ref-sNN-P64.txt (samples 1, 9, 25; depth20-tiling's
orchestrator_check, read only), inputs/inputs.json, the census run's checkpoint (g's value and
abserr) and 06c's census checkpoint (06c's, checked against depth20-tiling's census_abserr).
Resolutions: S96(P128), S96(P64), S128(P64); reference S128(P64); counts iff max - min of the three
<= min(abserr_06c, abserr_g)/100, exact (Fraction); rule |v_g - ref| <= abserr_g, exact.
Usage: python -B score_z3.py CENSUS_RUN_DIR RUN06C_DIR"""
import hashlib
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORCH = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06c/depth20-tiling/orchestrator_check")
SAMPLES = (1, 2, 3, 8, 9, 10, 13, 14, 15, 22, 23, 25, 27)
OLD = (1, 9, 25)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse(path):
    """{name: hex} for the S lines, the replay line's value/abserr, the replay's tree."""
    t = Path(path).read_text()
    out = {"S": {m.group(1): m.group(2) for m in re.finditer(r"^S(\d+) = (\S+) ", t, re.M)}}
    m = re.search(r"^driver replay value (\S+) abserr (\S+)", t, re.M)
    out["replay"] = (m.group(1), m.group(2)) if m else None
    m = re.search(r"^tree (\S+) (\S+)", t, re.M)
    out["tree"] = (m.group(1), m.group(2)) if m else None
    m = re.search(r"^pickle (\S+) sha256 (\S+)", t, re.M)
    out["pickle_sha256"] = m.group(2) if m else None
    return out


def call_from(run_dir, uid, index):
    with open(Path(run_dir) / "checkpoint.jsonl") as f:
        for line in f:
            if f'"unit": "{uid}"' not in line[:200]:
                continue
            r = json.loads(line)
            if r["unit"] != uid:
                continue
            for c in r["data"]["calls"] + r["data"]["other_calls"]:
                if c["index"] == index:
                    return c
    return None


def main():
    run_g, run_c = Path(sys.argv[1]), Path(sys.argv[2])
    print("script", Path(__file__).name, "sha256", sha(__file__))
    print("census checkpoint sha256", sha(run_g / "checkpoint.jsonl"), "; 06c checkpoint sha256", sha(run_c / "checkpoint.jsonl"))
    inputs = json.loads((HERE / "inputs" / "inputs.json").read_text())
    rows = []
    for n in SAMPLES:
        e = inputs[str(n)]
        if n in OLD:
            f1, f2 = ORCH / f"02-ref-s{n:02d}.txt", ORCH / f"04-ref-s{n:02d}-P64.txt"
        else:
            f1, f2 = HERE / "out" / f"ref-s{n:02d}-P128.txt", HERE / "out" / f"ref-s{n:02d}-P64.txt"
        a, b = parse(f1), parse(f2)
        _t, rest = e["unit"].split("-", 1)
        rtol, shard, serial = rest.rsplit("-", 2)
        uid = f"g-{rtol}-{shard}-{serial}"
        cg = call_from(run_g, uid, e["index"])
        cc = call_from(run_c, e["unit"], e["index"])
        vg, eg = Fraction(float.fromhex(cg["value"]["hex"])), Fraction(float.fromhex(cg["abserr"]["hex"]))
        vc, ec = Fraction(float.fromhex(cc["value"]["hex"])), Fraction(float.fromhex(cc["abserr"]["hex"]))
        checks = {
            "label_and_span_g": cg["notify_label"] == e["notify_label"] and [x["hex"] for x in cg["x_span"]] == [float.fromhex(h).hex() for h in e["x_span"]],
            "06c_checkpoint_equals_capture_record": (cc["value"]["hex"], cc["abserr"]["hex"]) == (float.fromhex(e["census06c_value"]).hex(), float.fromhex(e["census06c_abserr"]).hex()),
            "pickle_sha256_both_runs": a["pickle_sha256"] == b["pickle_sha256"] == e["sha256"],
        }
        if n not in OLD:  # the replay at g in each new run's own output must be g's census call
            checks["replay_equals_g_call"] = a["replay"] == b["replay"] == (cg["value"]["hex"], cg["abserr"]["hex"])
            checks["tree_g"] = a["tree"][1] == b["tree"][1] == "6d81fad43cbf1f17cff8d0f48fc7b65966e343ab"
        r1, r2, r3 = (Fraction(float.fromhex(x)) for x in (a["S"]["96"], b["S"]["96"], b["S"]["128"]))
        s64 = a["S"].get("64")
        ref = r3
        scatter = max(r1, r2, r3) - min(r1, r2, r3)
        m = min(ec, eg)
        counts = all(checks.values()) and scatter <= m / 100
        orch_reading = max(abs(r3 - r2), abs(r3 - r1)) / m
        viol = abs(vg - ref) > eg
        rows.append(dict(n=n, unit=uid, index=e["index"], block=e["block"], counts=counts, checks=checks,
                         scatter_over_min=float(scatter / m), orch_reading=float(orch_reading),
                         g_ratio=float(abs(vg - ref) / eg), c_ratio=float(abs(vc - ref) / ec), viol=viol,
                         ref=float(ref).hex(), S96_P128=a["S"]["96"], S96_P64=b["S"]["96"], S128_P64=b["S"]["128"], S64_P128=s64,
                         abserr_g=cg["abserr"]["hex"], abserr_06c=cc["abserr"]["hex"], value_g=cg["value"]["hex"], value_06c=cc["value"]["hex"],
                         files=[str(f1), str(f2)], file_sha256=[sha(f1), sha(f2)]))
    print("\nsample unit #index block | counts | range/min(abserr) (orch reading) | |v_g-ref|/e_g | |v_06c-ref|/e_06c | checks")
    for r in rows:
        print(f"s{r['n']:02d} {r['unit']} #{r['index']} {r['block']} | {'COUNTED' if r['counts'] else 'dropped'} | "
              f"{r['scatter_over_min']:.4g} ({r['orch_reading']:.4g}) | {r['g_ratio']:.4g}{' VIOLATES' if r['viol'] else ''} | {r['c_ratio']:.4g} | "
              f"{'all hold' if all(r['checks'].values()) else r['checks']}")
    counted = [r for r in rows if r["counts"]]
    fail = [r for r in counted if r["viol"]]
    v = "fails" if fail else ("undecided" if len(counted) < 5 else "holds")
    print(f"\ncounted {len(counted)} {[r['n'] for r in counted]}; dropped {len(rows) - len(counted)} {[r['n'] for r in rows if not r['counts']]}")
    print(f"violations among counted: {[(r['n'], round(r['g_ratio'], 4)) for r in fail]}")
    print(f"Z3: {v}")
    (HERE / "out" / "z3_scores.json").write_text(json.dumps({"verdict": v, "rows": rows}, indent=1))
    print("wrote references/out/z3_scores.json")


if __name__ == "__main__":
    main()
