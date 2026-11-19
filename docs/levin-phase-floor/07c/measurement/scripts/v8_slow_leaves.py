"""levin-phase-floor 06g, descriptive (the census brief's addendum, var/levin-phase-floor-06g/review/
dispatch_census_addendum.md, sha256 36010d19...): the slow leaves behind V8's lost calls. Read only; reruns nothing.

Scope: every untrusted (b) leaf a lost call returns, outside the 720, at each rtol. A lost call is one of
out/v8_calls.json's "lost" entries with stationary_point_row False (that file is read, never written). A leaf is
taken from the call's pickle record's ``checked_leaves_06g`` (every leaf the trust check was taken on, returned or
bisected): returned, 1/4 <= r < 1, delta1 > phase_err (test 3's class "1/4<=r<1|d1>phi"), and trusted_final 0
(untrusted as returned). Returned (b)-class leaves found floor-limited after their check (trusted_final -1) are
outside the scope and counted separately.

Items (per rtol, and each split by group):
1. r in [1/4, 1/2), [1/2, 3/4), [3/4, 1): counts and quantiles; a histogram of r in bins of 1/32.
2. delta1/phase_err: quantiles from the median up, and per r bin.
3. r against depth: the chains are partly recorded. A bisected leaf is in checked_leaves_06g only if its trust
   check was taken before it was split; a returned leaf's recorded ancestors are the checked leaves of the same
   call whose span contains its span at a smaller depth. Reported: how many depth levels of each chain are
   recorded; for chains with at least three recorded ancestors at depth >= 10, max |log2(r_ancestor / r_leaf)| over
   those ancestors (and the same over ancestors at depth >= 16, added after a first look at the output); r along the chain for a fixed sample (the leftmost scoped leaf of each of the first eight lost
   calls in (row, index) order); and r by depth over every checked leaf of the lost calls.
4. Position, among the scoped leaves at depth 20. **The clustering rule, fixed before the result was looked at:**
   in each call, the depth-20 scoped leaves are sorted by start, and two consecutive ones belong to the same cluster
   if the gap between the first's end and the next's start is at most 64 depth-20 widths of that call
   (64 (b - a) 2^-20, with (a, b) the call's x_span). Each cluster is given by its centre (mean of its leaves'
   midpoints), its leaf count and its centre's position u = (centre - a) / (b - a) in the call's interval; its
   distance to the nearest end of the call's x_span (the call's sub-interval ends), in units of b - a, is reported,
   and a cluster "is at an end" if that distance is at most 2^-10. Sub-interval ends other than the call's own, and
   the QuadSourceIntegral partition's other points, are not in the call's records, so no other endpoints are tested.
5. By group: every item above per group label.
Usage: python -B v8_slow_leaves.py RUN_DIR"""
import collections
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import WIDER, header  # noqa: E402

GAP_WIDTHS = 64
END_TOL = 2.0 ** -10
BINS = ((0.25, 0.5), (0.5, 0.75), (0.75, 1.0))


def q(vals, ps=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)):
    v = sorted(vals)
    if not v:
        return "none"
    return ", ".join(f"p{p*100:g} {v[min(len(v)-1, int(p*(len(v)-1)))]:.4g}" for p in ps)


def binname(r):
    for lo, hi in BINS:
        if lo <= r < hi:
            return f"[{lo:g}, {hi:g})"
    return "outside"


def main():
    run = Path(sys.argv[1])
    v8 = WIDER / "out" / "v8_calls.json"
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06g: the slow leaves behind V8's lost calls (descriptive)",
                    [v8, run / "checkpoint.jsonl"]):
        print(l)
    d = json.loads(v8.read_text())
    lost = [x for x in d["lost"] if not x["stationary_point_row"]]
    want = collections.defaultdict(list)
    for x in lost:
        want[x["unit"]].append(x)
    offs = {}
    with open(run / "checkpoint.jsonl") as f:
        for line in f:
            u = line[10:60].split('"')[0]
            if u in want:
                dd = json.loads(line)["data"]
                offs[u] = (dd["pickle"], dd["pickle_offset"], dd["pickle_length"], dd["pickle_record_sha256"])
    print(f"lost calls outside the 720: {len(lost)} in {len(want)} units; pickle records located {len(offs)}")
    leaves = []  # scoped leaves
    floor_lim = collections.Counter()
    all_checked = []  # (rtol, group, depth, r) over every checked leaf of the lost calls
    chains = []
    handles = {}
    for u in sorted(want):
        p, off, ln, sha = offs[u]
        h = handles.get(p) or handles.setdefault(p, open(run / p, "rb"))
        h.seek(off)
        blob = h.read(ln)
        assert hashlib.sha256(blob).hexdigest() == sha, u
        rec = pickle.loads(blob)
        byidx = {c["index"]: c for c in rec["checked_leaves_06g"] if c is not None}
        for x in want[u]:
            g = x["g"]
            rtol, grp = x["rtol"], (g["group"] if g["kind"] == "group" else g["notify_label"])
            c = byidx[g["index"]]
            a0, b0 = (float.fromhex(s) for s in g["x_span"])
            n = len(c["depth"])
            recs = [{"start": float(c["start"][i]), "end": float(c["end"][i]), "depth": int(c["depth"][i]),
                     "r": float(c["convergence_ratio"][i]), "d1": float(c["delta1"][i]), "phi": float(c["phase_err"][i]),
                     "ret": bool(c["returned"][i]), "tf": int(c["trusted_final"][i])} for i in range(n)]
            for e in recs:
                all_checked.append((rtol, grp, e["depth"], e["r"], e["ret"]))
            for e in recs:
                if not (e["ret"] and 0.25 <= e["r"] < 1.0 and e["d1"] > e["phi"]):
                    continue
                if e["tf"] != 0:
                    floor_lim[(rtol, e["tf"])] += 1
                    continue
                anc = sorted((o for o in recs if o["depth"] < e["depth"] and o["start"] <= e["start"] and o["end"] >= e["end"]),
                             key=lambda o: o["depth"])
                e2 = dict(e, rtol=rtol, group=grp, unit=u, row=x["row"], index=g["index"], span=(a0, b0), anc=anc)
                leaves.append(e2)
    for h in handles.values():
        h.close()
    out = []
    P = out.append
    P("# The slow leaves behind V8's lost calls (descriptive; the census brief's addendum)")
    P("")
    P(f"Computed by `scripts/v8_slow_leaves.py` from `out/v8_calls.json` (read only, sha256 "
      f"`{hashlib.sha256(v8.read_bytes()).hexdigest()[:8]}…`) and the census's region pickles (`checked_leaves_06g`), "
      "every record SHA-checked. Nothing was rerun. The scope, the items and the clustering rule are the script's "
      "docstring, fixed before its result was looked at.")
    P("")
    P(f"- Lost calls outside the 720: {len(lost)} ({len(want)} units).")
    for rtol in ("1e-7", "1e-8"):
        L = [e for e in leaves if e["rtol"] == rtol]
        P(f"- {rtol}: scoped leaves (returned, untrusted, 1/4 <= r < 1, delta1 > phi_P) {len(L)}; returned (b)-class leaves "
          f"exempted floor-limited after their check (outside the scope) {floor_lim[(rtol, -1)]}; "
          f"(b)-class returned leaves flagged trusted {floor_lim[(rtol, 1)]} (expected 0).")
    P("")
    groups_all = sorted({e["group"] for e in leaves})
    for rtol in ("1e-7", "1e-8"):
        P(f"## rtol {rtol}")
        P("")
        for grp in [None] + groups_all:
            L = [e for e in leaves if e["rtol"] == rtol and (grp is None or e["group"] == grp)]
            if not L:
                continue
            P(f"### {'all groups' if grp is None else 'group ' + grp} ({len(L)} leaves, "
              f"{len({(e['unit'], e['index']) for e in L})} calls)")
            P("")
            # 1. r
            rs = [e["r"] for e in L]
            cnt = collections.Counter(binname(r) for r in rs)
            P("**1. r.** " + "; ".join(f"{b}: {cnt.get(b, 0)}" for b in (binname(0.3), binname(0.6), binname(0.8))) + f". Quantiles: {q(rs)}.")
            hist = collections.Counter(min(31, int(r * 32)) for r in rs)
            P("  Histogram of r in bins of 1/32 (bin lower edge: count): " +
              ", ".join(f"{k/32:.4g}: {hist[k]}" for k in range(8, 32) if hist.get(k)))
            near = {t: sum(1 for r in rs if abs(r - t) <= 0.01) for t in (0.25, 0.5, 0.75)}
            P(f"  Within 0.01 of 1/4, 1/2, 3/4: {near[0.25]}, {near[0.5]}, {near[0.75]}.")
            P("")
            # 2. delta1/phi
            ratio = [e["d1"] / e["phi"] if e["phi"] > 0 else math.inf for e in L]
            P(f"**2. delta1/phi_P.** {q(ratio, (0.5, 0.75, 0.9, 0.99, 1.0))}; min {min(ratio):.4g}.")
            for lo, hi in BINS:
                rr = [rt for e, rt in zip(L, ratio) if lo <= e["r"] < hi]
                P(f"  - r in [{lo:g}, {hi:g}): {len(rr)} leaves; delta1/phi_P {q(rr, (0.0, 0.5, 0.9, 1.0))}; at most 2: {sum(1 for x in rr if x <= 2)}, "
                  f"at most 10: {sum(1 for x in rr if x <= 10)}")
            P("")
            # 3. r against depth
            P(f"**3. r against depth.** Depth of the scoped leaves: {dict(sorted(collections.Counter(e['depth'] for e in L).items()))}.")
            nrec = [len(e["anc"]) for e in L]
            P(f"  Chains are partly recorded (a bisected leaf is recorded only if it was checked before its split): recorded ancestors per leaf {q(nrec, (0.0, 0.5, 0.9, 1.0))} "
              f"of {L[0]['depth']} levels for a depth-{L[0]['depth']} leaf; recorded at depth >= 10: {q([sum(1 for o in e['anc'] if o['depth'] >= 10) for e in L], (0.0, 0.5, 0.9, 1.0))}.")
            dev = []
            for e in L:
                deep = [o for o in e["anc"] if o["depth"] >= 10 and o["r"] > 0 and math.isfinite(o["r"])]
                if len(deep) >= 3:
                    dev.append(max(abs(math.log2(o["r"] / e["r"])) for o in deep))
            P(f"  Chains with >= 3 recorded ancestors at depth >= 10: {len(dev)}; max |log2(r_ancestor / r_leaf)| over them: {q(dev, (0.0, 0.25, 0.5, 0.75, 0.9, 1.0))}; "
              f"within a factor 2 throughout (<= 1): {sum(1 for x in dev if x <= 1)}.")
            dev16 = []
            for e in L:
                deep = [o for o in e["anc"] if o["depth"] >= 16 and o["r"] > 0 and math.isfinite(o["r"])]
                if len(deep) >= 3:
                    dev16.append(max(abs(math.log2(o["r"] / e["r"])) for o in deep))
            P(f"  The same over the last levels only (recorded ancestors at depth >= 16, chains with >= 3): {len(dev16)}; max |log2(r_ancestor / r_leaf)|: "
              f"{q(dev16, (0.0, 0.25, 0.5, 0.75, 0.9, 1.0))}; within a factor 2 throughout: {sum(1 for x in dev16 if x <= 1)}; within 10 %: "
              f"{sum(1 for x in dev16 if x <= math.log2(1.1))}.")
            if grp is None:
                ac = [t for t in all_checked if t[0] == rtol]
                P("  r by depth over every checked leaf of these calls (returned or bisected), median [p10, p90] (n):")
                for dep in sorted({t[2] for t in ac}):
                    v = sorted(t[3] for t in ac if t[2] == dep)
                    fin = [x for x in v if math.isfinite(x)]
                    if fin:
                        P(f"    - depth {dep}: {fin[len(fin)//2]:.3g} [{fin[int(0.1*(len(fin)-1))]:.3g}, {fin[int(0.9*(len(fin)-1))]:.3g}] ({len(v)}; not finite {len(v)-len(fin)})")
                calls_sorted = sorted({(e["row"], e["index"]) for e in L})[:8]
                P("  Sample chains (the leftmost scoped leaf of each of the first eight lost calls in (row, index) order): depth: r of each recorded ancestor, then the leaf.")
                for row, idx in calls_sorted:
                    e = min((e for e in L if (e["row"], e["index"]) == (row, idx)), key=lambda e: e["start"])
                    P(f"    - {row} call {idx} {e['group']}, leaf depth {e['depth']} r {e['r']:.4g} d1/phi {e['d1']/e['phi']:.3g}: " +
                      ", ".join(f"{o['depth']}: {o['r']:.3g}" for o in e["anc"]))
            P("")
            # 4. position
            per_call = collections.defaultdict(list)
            for e in L:
                if e["depth"] == 20:
                    per_call[(e["row"], e["index"], e["group"])].append(e)
            ncl, at_end, cl_sizes, examples = [], 0, [], []
            us = []
            n_clusters = 0
            for key, es in sorted(per_call.items()):
                a0, b0 = es[0]["span"]
                w = (b0 - a0) * 2.0 ** -20
                es.sort(key=lambda e: e["start"])
                cls = [[es[0]]]
                for e in es[1:]:
                    if e["start"] - cls[-1][-1]["end"] <= GAP_WIDTHS * w:
                        cls[-1].append(e)
                    else:
                        cls.append([e])
                ncl.append(len(cls))
                desc_ = []
                for c in cls:
                    cen = sum((e["start"] + e["end"]) / 2 for e in c) / len(c)
                    u_ = (cen - a0) / (b0 - a0)
                    dist = min(u_, 1 - u_)
                    at_end += dist <= END_TOL
                    n_clusters += 1
                    cl_sizes.append(len(c))
                    us.append(u_)
                    desc_.append(f"u={u_:.6f} ({len(c)} leaves, distance to an end {dist:.2e})")
                examples.append((key, a0, b0, desc_))
            P(f"**4. Position (depth-20 scoped leaves, {sum(len(v) for v in per_call.values())} in {len(per_call)} calls).** Clusters per call: "
              f"median {sorted(ncl)[len(ncl)//2] if ncl else None}, max {max(ncl) if ncl else None}, distribution {dict(sorted(collections.Counter(ncl).items()))}; "
              f"leaves per cluster {q(cl_sizes, (0.0, 0.5, 0.9, 1.0))}; clusters within 2^-10 of an end of the call's x_span: {at_end} of {n_clusters}.")
            hu = collections.Counter(min(9, int(u_ * 10)) for u_ in us)
            P("  Cluster centres' position u in tenths of the interval: " + ", ".join(f"[{k/10:.1f}, {(k+1)/10:.1f}): {hu.get(k, 0)}" for k in range(10)) + ".")
            if grp is None:
                P("  Example calls (the first eight in (row, index) order): x_span (a, b) and clusters:")
                for key, a0, b0, desc_ in examples[:8]:
                    P(f"    - {key[0]} call {key[1]} {key[2]}, x_span ({a0:.9g}, {b0:.9g}): " + "; ".join(desc_[:8]) + (" ..." if len(desc_) > 8 else ""))
            P("")
    text = "\n".join(out) + "\n"
    (WIDER / "out" / "v8_slow_leaves.md").write_text(text)
    data = [{k: v for k, v in e.items() if k != "anc"} | {"anc": [(o["depth"], o["r"], o["d1"], o["phi"]) for o in e["anc"]]} for e in leaves]
    (WIDER / "out" / "v8_slow_leaves.json").write_text(json.dumps(data))
    print(text)


if __name__ == "__main__":
    main()
