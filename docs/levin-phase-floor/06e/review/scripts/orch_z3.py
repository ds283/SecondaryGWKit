"""Orchestrator's own re-scoring of 06e's Z3 from the reference outputs and the two census
checkpoints. Reads files only. Exact arithmetic on the recorded doubles (Fraction of float.fromhex).

Resolutions: S96(P128), S96(P64), S128(P64); the reference is S128(P64). A reference counts iff
range(max - min) <= min(abserr_06c, abserr_g)/100. Rule: |v_g - ref| <= abserr_g."""

import json
import re
import sys
from fractions import Fraction

C06 = "var/runs/levin-phase-floor-06c-census-20261029T130214/checkpoint.jsonl"
C06E = "var/runs/levin-phase-floor-06e-census-20261030T081318/checkpoint.jsonl"
REF = "var/levin-phase-floor-06e/census/references"
ORCH = "../SecondaryGWKit-lpf06e-6d81fad/docs/levin-phase-floor/06c/depth20-tiling/orchestrator_check"
EXTRA = sys.argv[1:]  # optional extra P64-style files "NN=path" to report beside


def units(p, want):
    out = {}
    for line in open(p):
        r = json.loads(line)
        if r["unit"] in want:
            out[r["unit"]] = r["data"]
    return out


def find_call(d, index, label):
    hits = [
        c
        for c in (d.get("calls") or []) + (d.get("other_calls") or [])
        if c["index"] == index and c["notify_label"] == label
    ]
    assert len(hits) == 1, (d["unit"], index, label, len(hits))
    return hits[0]


def parse(path):
    t = open(path).read()
    s = {m.group(1): m.group(2) for m in re.finditer(r"^S(\d+) = (\S+)", t, re.M)}
    sha = re.search(r"script sha256 (\S+)", t).group(1)
    tree = re.search(r"^tree \S+ (\S+)", t, re.M).group(1)
    rep = re.search(r"driver replay value (\S+) abserr (\S+)", t)
    return s, sha, tree, (rep.group(1), rep.group(2))


F = lambda h: Fraction(float.fromhex(h))
inputs = json.load(open(f"{REF}/inputs/inputs.json"))
samples = sorted(int(k) for k in inputs)
want = {inputs[str(n)]["unit"] for n in samples}
g = units(C06E, want)
c = units(C06, want)

counted = violations = 0
print("sample unit idx | P128 file tree | range/min(abserr) counted | |vg-ref|/ag  |v06c-ref|/a06c | replay at g == census g")
for n in samples:
    inp = inputs[str(n)]
    if n in (1, 9, 25):
        f128, f64 = f"{ORCH}/02-ref-s{n:02d}.txt", f"{ORCH}/04-ref-s{n:02d}-P64.txt"
    else:
        f128, f64 = f"{REF}/out/ref-s{n:02d}-P128.txt", f"{REF}/out/ref-s{n:02d}-P64.txt"
    s128, sha1, tree1, rep1 = parse(f128)
    s64, sha2, tree2, rep2 = parse(f64)
    assert sha1 == sha2 == "8a0d291d09d74d444b47ff1f34b0f9000e4a045c4b94c36e5dec3158f2ea863b"
    res = [F(s128["96"]), F(s64["96"]), F(s64["128"])]
    ref = res[2]
    cg = find_call(g[inp["unit"]], inp["index"], inp["notify_label"])
    cc = find_call(c[inp["unit"]], inp["index"], inp["notify_label"])
    assert cc["value"]["hex"] == inp["census06c_value"] and cc["abserr"]["hex"] == inp["census06c_abserr"]
    vg, ag = F(cg["value"]["hex"]), F(cg["abserr"]["hex"])
    vc, ac = F(cc["value"]["hex"]), F(cc["abserr"]["hex"])
    scatter = (max(res) - min(res)) / min(ac, ag)
    ok = scatter <= Fraction(1, 100)
    rg = abs(vg - ref) / ag
    rc = abs(vc - ref) / ac
    # the replay at g is only in the files run at g (not 1, 9, 25, which ran at 06c's tree)
    rep_ok = (
        "n/a (06c tree)"
        if n in (1, 9, 25)
        else str(rep1 == rep2 == (cg["value"]["hex"], cg["abserr"]["hex"]))
    )
    if ok:
        counted += 1
        violations += rg > 1
    print(
        f"{n:2d} {inp['unit']} {inp['index']} | {tree1[:7]} | {float(scatter):.4g} {'COUNTED' if ok else 'dropped'} | "
        f"{float(rg):.6g}{' VIOLATES' if ok and rg > 1 else ''}  {float(rc):.6g} | {rep_ok}"
        f"  conv_g={cg['converged']} depth_g={cg['max_depth']}"
    )
    if n == 3:
        print(f"     s03: ref S128(P64) {s64['128']}; S96(P64) {s64['96']}; S96(P128) {s128['96']}")
        print(f"          v_g {cg['value']['hex']} abserr_g {cg['abserr']['hex']}; v_06c {cc['value']['hex']} abserr_06c {cc['abserr']['hex']}")
print(f"counted {counted}, violations {violations} ->", "FAILS" if violations else ("UNDECIDED" if counted < 5 else "HOLDS"))
for e in EXTRA:
    n, path = e.split("=")
    inp = inputs[n]
    s, sha, tree, rep = parse(path)
    cg = find_call(g[inp["unit"]], inp["index"], inp["notify_label"])
    ag = F(cg["abserr"]["hex"])
    print(f"extra s{n} {path}: tree {tree[:7]} replay==census g {rep == (cg['value']['hex'], cg['abserr']['hex'])}; " + "; ".join(
        f"S{k}={v} |vg-S{k}|/ag={float(abs(F(cg['value']['hex']) - F(v)) / ag):.6g}" for k, v in s.items()))
