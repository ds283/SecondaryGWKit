"""levin-phase-floor prompt 07c Part A, section 2.4: for each variant, which calls and which leaves it changes against the
tree's replay, one call it does not touch shown bit-identical to the tree, and (every variant, as (E-all) must) every
window whose stop changes against 07b's (the tree's replay, which reproduces 07b's census). New in 07c.

Per call of section 2.1 (out/replays/tree/ and out/replays/<v>/, 454 each):
- touched: the replay's value, abserr, converged, num_regions, max_depth or refusal counts differ, or the region arrays'
  SHA-256 differs;
- the leaves, keyed by (start, end) in hex: only in the variant (made), only in the tree (not made), and on the same span
  with a differing record, by the fields that differ; each counted by route (the leaf record's ``type``) and depth;
- the windows: n_windows, n_windows_stopped, and per window (by lo, hi) stopped, stopped_after_split, iroff1_count.
Untouched example: the first call (sorted sid) whose replay and region SHA-256 equal the tree's, of each set it holds.

Writes out/variants/reach.json and out/variants/reach.md. Usage: python -B variant_reach_07c.py V [V ...]
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RP = ROOT / "out" / "replays"


def load(v, sid):
    return json.loads((RP / v / f"{sid}.json").read_text())


def lkey(l):
    return (l["start"], l["end"])


def main():
    vs = sys.argv[1:]
    out_j, out_m = ROOT / "out" / "variants" / "reach.json", ROOT / "out" / "variants" / "reach.md"
    for p in (out_j, out_m):
        if p.exists():
            raise SystemExit(f"{p} exists")
    sids = sorted(p.name[:-5] for p in (RP / "tree").glob("*.json"))
    tree = {s: load("tree", s) for s in sids}
    doc = {}
    md = ["# Section 2.4: what each variant changes against the tree (07b's code), on the 454 calls of section 2.1", ""]
    for v in vs:
        touched, untouched = [], {}
        made, unmade, changed, fields = Counter(), Counter(), Counter(), Counter()
        made_depth = Counter()
        win = []
        calls = {}
        for s in sids:
            a, b = tree[s], load(v, s)
            same = a["replay"] == b["replay"] and a["regions_sha256"] == b["regions_sha256"]
            set_ = a["set"]
            if same:
                if set_ not in untouched:
                    untouched[set_] = {"sid": s, "regions_sha256_tree": a["regions_sha256"], "regions_sha256_variant": b["regions_sha256"],
                                       "replay_tree": a["replay"], "replay_variant": b["replay"]}
            else:
                touched.append(s)
            la = {lkey(l): l for l in (a.get("leaves") or [])}
            lb = {lkey(l): l for l in (b.get("leaves") or [])}
            c_made = Counter(lb[k]["type"] for k in lb.keys() - la.keys())
            c_unmade = Counter(la[k]["type"] for k in la.keys() - lb.keys())
            for k in lb.keys() - la.keys():
                made_depth[(lb[k]["type"], lb[k]["depth"])] += 1
            c_changed = Counter()
            for k in la.keys() & lb.keys():
                if la[k] != lb[k]:
                    c_changed[la[k]["type"]] += 1
                    for f in la[k]:
                        if la[k][f] != lb[k].get(f):
                            fields[f] += 1
            made.update(c_made)
            unmade.update(c_unmade)
            changed.update(c_changed)
            wa = {(w["lo"], w["hi"]): w for w in a.get("windows") or []}
            wb = {(w["lo"], w["hi"]): w for w in b.get("windows") or []}
            for k in sorted(wa.keys() | wb.keys()):
                x, y = wa.get(k), wb.get(k)
                fx = None if x is None else {f: x.get(f) for f in ("stopped", "stopped_after_split", "iroff1_count", "leaves_at_stop")}
                fy = None if y is None else {f: y.get(f) for f in ("stopped", "stopped_after_split", "iroff1_count", "leaves_at_stop")}
                if fx != fy:
                    win.append({"sid": s, "lo": k[0], "hi": k[1], "tree": fx, "variant": fy})
            if a.get("n_windows") != b.get("n_windows") or a.get("n_windows_stopped") != b.get("n_windows_stopped"):
                win.append({"sid": s, "n_windows": [a.get("n_windows"), b.get("n_windows")],
                            "n_windows_stopped": [a.get("n_windows_stopped"), b.get("n_windows_stopped")]})
            if not same:
                calls[s] = {"set": set_, "made": dict(c_made), "unmade": dict(c_unmade), "changed": dict(c_changed),
                            "converged": [a["replay"]["converged"], b["replay"]["converged"]],
                            "num_regions": [a["replay"]["num_regions"], b["replay"]["num_regions"]]}
        by_set = Counter(tree[s]["set"] for s in touched)
        doc[v] = {"touched": len(touched), "touched_by_set": dict(by_set), "untouched_example": untouched,
                  "leaves_made": dict(made), "leaves_not_made": dict(unmade), "leaves_changed_same_span": dict(changed),
                  "fields_changed_same_span": dict(fields), "made_by_route_depth": {f"{r}|{d}": n for (r, d), n in sorted(made_depth.items())},
                  "windows_changed": win, "calls": calls}
        md += [f"## {v}", "",
               f"- calls touched: **{len(touched)}** of {len(sids)}, by set {dict(sorted(by_set.items()))}",
               f"- leaves made that the tree does not make: {dict(made)}; tree leaves not made: {dict(unmade)}; same span, record differs: "
               f"{dict(changed)} (fields: {dict(fields.most_common())})",
               f"- windows whose stop changes against 07b's: {len(win)}" + ("" if not win else " — " + "; ".join(
                   f"{w['sid']} " + (f"[{w['lo']}, {w['hi']}] tree {w['tree']} -> {w['variant']}" if "lo" in w else
                                     f"n_windows {w['n_windows']}, stopped {w['n_windows_stopped']}") for w in win)),
               "- untouched, bit-identical to the tree (replay and region arrays' SHA-256): " + "; ".join(
                   f"{st}: {u['sid']} regions {u['regions_sha256_tree'][:16]}… = {u['regions_sha256_variant'][:16]}…, value "
                   f"{u['replay_tree']['value']} = {u['replay_variant']['value']}" for st, u in sorted(untouched.items())), ""]
    out_j.write_text(json.dumps(doc, indent=1))
    out_m.write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
