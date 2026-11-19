"""levin-phase-floor prompt 07c Part A extension (R2-28), question 1: the leaf-reference spec for R2-28. New in the
extension, after 07b's extension's x1_spec.py (a record), whose break-point expressions it keeps.

R2-28 has no window and no stationary point (its replays: n_windows 0, stationary_points []), so one reading of
leafref_07c.py over its whole span serves every leaf of every partition. The spec's points:
- every leaf end of a:R2-28 and a:R2-28~o under the tree and the seven variants (out/replays/<v>/), and of 07a's one
  region (07b Part A's replay at 07a's code, var/levin-phase-floor-07b/measurement/out/replays/tree/R2-28.json, read
  only);
- on every leaf that is trusted (``trusted`` true, Levin or direct) in the tree's, S's and 07a's partitions of a:R2-28:
  its halves', quarters' and eighths' break points, by ``_levin_leaf.break_point``'s expressions (c = (b + a)/2; quarters
  (c + a)/2 and (b + c)/2; each quarter's midpoint the same way).
One span: the call's x_span. The capture is the 1e-7 call's (the ~o calls are the same blob with an rtol override:
first run, record 48, refused them). Writes state/r2-28/spec.json (refuses to overwrite). Usage: python -B r2_28_spec_07c.py
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R07B = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07b/measurement")
VARIANTS = ["tree", "S", "S2", "N", "Eg", "Ea", "SEg", "SEa"]
SUBDIVIDE = {("tree", "a:R2-28"), ("S", "a:R2-28"), ("07a", "a:R2-28")}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pieces(lo, hi):
    cm = (hi + lo) / 2.0
    cL, cR = (cm + lo) / 2.0, (hi + cm) / 2.0
    return [cm, cL, cR] + [(q1 + q0) / 2.0 for q0, q1 in ((lo, cL), (cL, cm), (cm, cR), (cR, hi))]


def main():
    out = ROOT / "state" / "r2-28" / "spec.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    srcs = {}
    for v in VARIANTS:
        for sid in ("a:R2-28", "a:R2-28~o"):
            srcs[(v, sid)] = ROOT / "out" / "replays" / v / f"{sid}.json"
    srcs[("07a", "a:R2-28")] = R07B / "out" / "replays" / "tree" / "R2-28.json"
    points, cap, span, named = set(), None, None, []
    for (v, sid), p in srcs.items():
        d = json.loads(p.read_text())
        assert d["n_windows"] == 0 and not d["stationary_points"], (v, sid)
        c = d["header"]["capture"]
        c = {k: c[k] for k in ("path", "offset", "length", "sha256", "override")}
        # The ~o calls carry an rtol/atol override of the same captured blob: the integrand is the same, so the
        # reference is read from the 1e-7 capture (override None); the blob itself must be the same.
        if sid == "a:R2-28" and v == "tree":
            cap = c
        assert {k: c[k] for k in ("path", "offset", "length", "sha256")} == {k: cap[k] for k in ("path", "offset", "length", "sha256")}, (v, sid, c)
        sp = tuple(d["x_span"])
        span = span or sp
        assert sp == span
        for i, L in enumerate(d["leaves"]):
            lo, hi = float.fromhex(L["start"]), float.fromhex(L["end"])
            points |= {lo, hi}
            if (v, sid) in SUBDIVIDE and L["trusted"]:
                points |= set(pieces(lo, hi))
                named.append({"partition": v, "sid": sid, "i": i, "lo": L["start"], "hi": L["end"], "type": L["type"]})
    spec = {"sid": "r2-28", "capture": cap, "spans": [list(span)], "points": [p.hex() for p in sorted(points)],
            "named": named, "sources": {f"{v}|{sid}": {"path": str(p), "sha256": sha256(p)} for (v, sid), p in srcs.items()}}
    out.write_text(json.dumps(spec, indent=1))
    print(f"{len(points)} points, {len(named)} subdivided leaves; wrote {out} {sha256(out)}")


if __name__ == "__main__":
    main()
