"""Final review (levin-phase-floor 07c Part C, §C.2): my own re-score of Y2, Y3 and Y5 on 07c's census. Read only but
for OUT_JSON. Adapted from 07b's review/final/scripts/u_rescore.py (U2, U3, U5); independent of the census agent's
scripts.

usage: python -I -B y235.py OUT_JSON

07c's checkpoint read through my index (final/index07c.json, file SHA-256 asserted, each line's SHA-256 checked); region
records read only where needed, each accepted only against the line's pickle_record_sha256.
- Y2 (07a's W2): a returned Levin leaf (type 0) holding a scan sign change, by the scan's grid test recomputed from the
  pickle's full scan (sign_change_x +- halfwidth inside the leaf), exact; my grid set compared with the scan's own; the
  scan's 'ends' listings of Levin leaves counted apart. Holds iff 0.
- Y3 (07a's W3): on every scanned call, each sign change has a located root within h and each located root lies within h
  of a sign change (exact). Every location_skipped_phases entry listed by reason; a reason other than 'zero_rate' is a
  violation; scanned calls with a skipped phase and a sign change listed. Holds iff 0 violations.
- Y5 (07a's W5, gated solves): sum of route_07a.n_cc_with_levin_half; sum of route_07a.n_partition_nonforced; sum of
  guard_07.num_guarded_regions. Holds iff all three are 0. Forced partitions (route_07a.partition.forced) descriptive.
"""
import hashlib, json, pickle, sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

M = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
FIN = M / "var/levin-phase-floor-07c/review/final"
RC = M / "var/runs/levin-phase-floor-07c-census-20261119T201419"
SHA_C = "33d94ce21077cd23f29da65ef967b64676db6722c081b999946bbc91c822a38e"


def fh(v):
    return Fraction(float.fromhex(v["hex"] if isinstance(v, dict) else float(v).hex()))


def main():
    out = Path(sys.argv[1])
    ix = json.load(open(FIN / "index07c.json"))
    assert ix["sha256"] == SHA_C
    C = Counter()
    u2_viol, u2_ends_levin, u2_grid_mismatch, u2_nonlevin = [], [], [], Counter()
    u3_viol, u3_skipped, u3_bad_reason, u3_skip_sc, u3_end_pts = [], Counter(), [], [], []
    u5 = Counter(); u5_calls = defaultdict(list); forced_calls = 0
    listed = Counter()  # the scan's listings of returned leaves, by (how, branch): descriptive
    f = open(RC / "checkpoint.jsonl", "rb")
    for key in sorted(ix["units"]):
        e = ix["units"][key]
        f.seek(e[0]); raw = f.read(e[1])
        assert hashlib.sha256(raw).hexdigest() == e[2]
        dh = json.loads(raw)["data"]
        rt = dh["rtol"]
        C["units"] += 1; C["units_done"] += dh["status"] == "done"
        need_pickle = False
        for kind in ("calls", "other_calls"):
            for ch in dh[kind]:
                C["calls"] += 1
                gh, ra = ch["guard_07"], ch["route_07a"]
                a = ra["n_cc_with_levin_half"]; b = ra["n_partition_nonforced"]; fo = ra["partition"]["forced"]
                g = gh["num_guarded_regions"]
                u5["cc_with_levin_half"] += a; u5["partition_nonforced"] += b; u5["partition_forced"] += fo
                u5["num_guarded_regions"] += g
                forced_calls += fo > 0
                for nm, v in (("cc_with_levin_half", a), ("partition_nonforced", b), ("num_guarded_regions", g)):
                    if v:
                        u5_calls[nm].append((key, kind, ch["index"], v))
                for x in gh["location_skipped_phases"]:
                    u3_skipped[x.get("reason", "<no reason>")] += 1
                    if x.get("reason") != "zero_rate":
                        u3_bad_reason.append((key, kind, ch["index"], x))
        callsh = {c["index"]: c for c in dh["calls"] + dh["other_calls"]}
        for s in dh["scan_07"] or []:
            C[f"scanned_{s['kind']}_{rt}"] += 1
            x = callsh[s["index"]]; g = x["guard_07"]
            sc = [fh(v) for v in s["sign_change_x"]]
            assert s["n_sign_changes"] == len(sc)
            if sc:
                need_pickle = True
            hh = fh(s["h"])
            if g["location_skipped_phases"] and sc:
                u3_skip_sc.append((key, s["index"], s["kind"], len(sc)))
            roots = [fh(p["x"]) for p in g["points"] if p["kind"] == "root"]
            if any(p["kind"] == "end" for p in g["points"]):
                u3_end_pts.append((key, s["index"]))
            bad = [float(z) for z in sc if not any(abs(z - r) <= hh for r in roots)]
            bad += [float(r) for r in roots if not any(abs(z - r) <= hh for z in sc)]
            C["y3_scored"] += 1
            if sc or roots:
                C["y3_with_change_or_root"] += 1
            if bad:
                u3_viol.append((key, s["index"], s["kind"], bad[:4]))
        if not need_pickle:
            continue
        with open(RC / dh["pickle"], "rb") as g_:
            g_.seek(dh["pickle_offset"]); rawp = g_.read(dh["pickle_length"])
        assert hashlib.sha256(rawp).hexdigest() == dh["pickle_record_sha256"], key
        p = pickle.loads(rawp); C["pickles_read"] += 1
        # the scan's index addresses calls of its kind; group and analytic regions are kept apart
        regs = {("group", r["index"]): r for r in p["regions"]} | {("analytic", r["index"]): r for r in p["other_regions"]}
        # the pickle's full scans, keyed by (kind, index) where they carry a kind, else by index (asserted unique)
        if all("kind" in e_ for e_ in p["scan_07"]):
            sfull = {(e_["kind"], e_["index"]): e_["scan"] for e_ in p["scan_07"]}
            assert len(sfull) == len(p["scan_07"])
            C["pickle_scan_keyed_by_kind"] += 1
        else:
            sfull = {e_["index"]: e_["scan"] for e_ in p["scan_07"]}
            assert len(sfull) == len(p["scan_07"]), key
            C["pickle_scan_keyed_by_index"] += 1
        for s in dh["scan_07"]:
            if not s["n_sign_changes"]:
                continue
            i = s["index"]
            kd = "group" if s["kind"] in ("group", "calls") else ("analytic" if s["kind"] in ("analytic", "other_calls") else None)
            if kd is None:  # unknown kind label: fall back to the index alone, asserted unique
                cands = [v for (k_, j), v in regs.items() if j == i]
                assert len(cands) == 1, (key, i, s["kind"])
                r = cands[0]
            else:
                r = regs[(kd, i)]
            sf = sfull[(s["kind"], i)] if (s["kind"], i) in sfull else sfull[i]
            lo, hi = r["start"].astype(float), r["end"].astype(float)
            mine = set()
            for xc, hw in zip(sf["sign_change_x"], sf["sign_change_halfwidth"]):
                a, b = Fraction(xc) - Fraction(hw), Fraction(xc) + Fraction(hw)
                for kk in range(len(lo)):
                    l, e_ = sorted((Fraction(float(lo[kk])), Fraction(float(hi[kk]))))
                    if l <= a and b <= e_:
                        mine.add(kk)
            sg = {x["region_index"] for x in sf["listed"] if x["how"] == "grid"}
            C["y2_calls_with_sign_change"] += 1
            C["y2_calls_with_nonempty_grid_set"] += bool(mine)
            C["y2_sign_changes"] += len(sf["sign_change_x"])
            for x in sf["listed"]:
                listed[(x["how"], x.get("branch"))] += 1
            if mine != sg:
                u2_grid_mismatch.append((key, i, sorted(mine), sorted(sg)))
            for kk in mine:
                if int(r["type"][kk]) == 0:
                    u2_viol.append((key, i, kk))
                else:
                    u2_nonlevin[("window" if int(r["window_07"][kk]) else "", "guarded" if int(r["guarded_07"][kk]) else "")] += 1
            for x in sf["listed"]:
                if x["how"] == "ends" and x["branch"] == "Levin":
                    u2_ends_levin.append((key, i, x["region_index"]))
    res = dict(
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        index_sha256=hashlib.sha256((FIN / "index07c.json").read_bytes()).hexdigest(), checkpoint_sha256=SHA_C,
        counts=dict(C),
        Y2=dict(n=len(u2_viol), holds=not u2_viol, violations=u2_viol[:50], ends_levin=u2_ends_levin,
                n_grid_mismatch=len(u2_grid_mismatch), grid_mismatch=u2_grid_mismatch[:20],
                nonlevin_by_tag={str(k): v for k, v in u2_nonlevin.items()},
                scan_listings_by_how_branch={str(k): v for k, v in listed.items()}),
        Y3=dict(n=len(u3_viol), holds=not u3_viol and not u3_bad_reason, violations=u3_viol[:50],
                skipped_by_reason=dict(u3_skipped), bad_reason=u3_bad_reason[:50], skipped_with_sign_change=u3_skip_sc,
                calls_with_end_points=len(u3_end_pts)),
        Y5=dict(sums=dict(u5), holds=u5["cc_with_levin_half"] == 0 and u5["partition_nonforced"] == 0
                and u5["num_guarded_regions"] == 0, calls_with_forced=forced_calls,
                calls={k: v[:50] for k, v in u5_calls.items()}, n_calls={k: len(v) for k, v in u5_calls.items()}),
    )
    out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str)[:20000])


if __name__ == "__main__":
    main()
