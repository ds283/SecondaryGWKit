"""levin-phase-floor prompt 06b section 5: the census's scoring and descriptive items. New in 06b.

Scores W1, W3 and W4 and draws W2's sample by the rules of state/predictions.md (Part A, the
prompt's text; Part B, the selection rule and the readings), reading:
- 06a's run (prov.RUN06A): its checkpoint, and its per-worker pickles for the units that need
  their region arrays;
- this census run (RUN_DIR): its checkpoint and per-worker pickles (W3's per-region floors are in
  the pickle records' ``w3``).

``select RUN_DIR``: W2's eligible calls and the draw; writes state/w2_selection.json (the 24 calls,
each with its captured inputs' pickle, offset, length and sha256, and 06a's value and abserr) and
prints the eligible counts by stratum.

``report RUN_DIR [W2_REFERENCE_JSON]``: W1, W3, W4 (and W2 from the reference output), and the
descriptive items; writes out/w3_regions.csv (every region b accepted as phase-limited at 1e-7,
with its recomputed floors), out/moved_calls.csv (every group call that moved) and
out/census_rows_06b.csv.gz (one line per unit).

Read-only outside state/w2_selection.json and out/. Imports nothing from either tree.
Usage: python -B analyse06b.py select RUN_DIR ; python -B analyse06b.py report RUN_DIR [REF_JSON]
"""

import collections
import csv
import gzip
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import RUN06A, WIDER, header, sha256  # noqa: E402

ROWS_JSON = WIDER / "state" / "rows.json"
CAPTURE_JSON = WIDER / "state" / "w2_capture.json"
SELECTION_JSON = WIDER / "state" / "w2_selection.json"
DEPTH_MAX = 20
RTOLS = ("1e-7", "1e-8")


def read_checkpoint(run_dir):
    recs = {}
    with open(Path(run_dir) / "checkpoint.jsonl") as f:
        for line in f:
            r = json.loads(line)
            recs[r["unit"]] = r["data"]
    return recs


def load_rec(run_dir, d):
    with open(Path(run_dir) / d["pickle"], "rb") as f:
        f.seek(d["pickle_offset"])
        blob = f.read(d["pickle_length"])
    if hashlib.sha256(blob).hexdigest() != d["pickle_record_sha256"]:
        raise RuntimeError(f"{d['unit']}: pickle record sha256 differs")
    return pickle.loads(blob)


def rowkey(uid):
    """"<shard>-<serial>" of a unit id or capture key ("[tree-]<rtol>-<shard>-<serial>")."""
    return "-".join(uid.rsplit("-", 2)[-2:])


def hx(d):
    return None if d is None else d["hex"]


def fl(d):
    return float.fromhex(d["hex"])


def load_all(run_dir):
    rows = json.loads(ROWS_JSON.read_text())["rows"]
    by_key = {f"{r['shard']}-{r['serial']}": r for r in rows}
    A = read_checkpoint(RUN06A)
    Bc = read_checkpoint(run_dir)
    return rows, by_key, A, Bc


def call_moved(cb, ca):
    return hx(cb["value"]) != hx(ca["value"]) or hx(cb["abserr"]) != hx(ca["abserr"])


def sha_key(uid, idx):
    return hashlib.sha256(f"{uid}:{idx}".encode()).hexdigest()


# ------------------------------------------------------------------------------------------ select


def cmd_select(run_dir):
    run_dir = Path(run_dir)
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06b: W2's sample", [ROWS_JSON, CAPTURE_JSON, RUN06A / "checkpoint.jsonl", run_dir / "checkpoint.jsonl"]):
        print(l)
    rows, by_key, A, Bc = load_all(run_dir)
    cap = json.loads(CAPTURE_JSON.read_text())["capture"]
    strata = collections.defaultdict(list)
    n_cap = n_band = n_done = n_moved = n_inputs = 0
    not_moved = []
    for key, lst in cap.items():
        rk = rowkey(key)
        R = by_key[rk]
        ua, ub = f"h-{key}", f"b-{key}"
        da, db = A[ua], Bc[ub]
        ca = {c["index"]: c for c in da["calls"]}
        cb = {c["index"]: c for c in db["calls"]}
        inp = {e["index"]: e for e in (db.get("w2_inputs") or [])}
        for idx, label in lst:
            n_cap += 1
            if R["band"] not in ("low", "mid"):
                continue
            n_band += 1
            if db["status"] != "done":
                continue
            n_done += 1
            if not call_moved(cb[idx], ca[idx]):
                not_moved.append((ub, idx))
                continue
            n_moved += 1
            e = inp.get(idx)
            if e is None or e.get("error"):
                continue
            n_inputs += 1
            strata[(R["band"], R["k_key"])].append(
                {
                    "unit": ub,
                    "index": idx,
                    "notify_label": label,
                    "band": R["band"],
                    "k_key": R["k_key"],
                    "k": R["k"],
                    "row": R["name"],
                    "dir": str(run_dir),
                    "pickle": e["pickle"],
                    "offset": e["offset"],
                    "length": e["length"],
                    "sha256": e["sha256"],
                    "value_b": hx(cb[idx]["value"]),
                    "abserr_b": hx(cb[idx]["abserr"]),
                    "value_06a": hx(ca[idx]["value"]),
                    "abserr_06a": hx(ca[idx]["abserr"]),
                    "order_key": sha_key(ub, idx),
                }
            )
    print(f"captured calls {n_cap}; band low/mid {n_band}; b unit done {n_done}; moved {n_moved}; inputs captured {n_inputs}")
    print(f"captured low/mid calls that did not move: {len(not_moved)} {not_moved[:10]}")
    for s in strata.values():
        s.sort(key=lambda e: e["order_key"])
    bands = {}
    for band in ("low", "mid"):
        keys = sorted([k for k in strata if k[0] == band], key=lambda k: float(strata[k][0]["k"]))
        bands[band] = keys
        print(f"band {band}: strata " + ", ".join(f"{k[1]} (k {float(strata[k][0]['k']):.4g}): {len(strata[k])}" for k in keys))

    def round_robin(keys, used, want):
        out = []
        pos = {k: 0 for k in keys}
        while len(out) < want:
            took = False
            for k in keys:
                if len(out) >= want:
                    break
                while pos[k] < len(strata[k]) and (strata[k][pos[k]]["unit"], strata[k][pos[k]]["index"]) in used:
                    pos[k] += 1
                if pos[k] < len(strata[k]):
                    e = strata[k][pos[k]]
                    pos[k] += 1
                    out.append(e)
                    used.add((e["unit"], e["index"]))
                    took = True
            if not took:
                break
        return out

    used = set()
    picked = {b: round_robin(bands[b], used, 12) for b in ("low", "mid")}
    for b, other in (("low", "mid"), ("mid", "low")):
        short = 12 - len(picked[b])
        if short > 0:
            picked[other] += round_robin(bands[other], used, short)
    sel = picked["low"] + picked["mid"]
    print(f"\nselected {len(sel)} (low {len(picked['low'])}, mid {len(picked['mid'])}); shortfall {24 - len(sel)}")
    for i, e in enumerate(sel):
        print(f"  {i + 1:2d} {e['band']:<4} {e['k_key']:<6} {e['unit']:<18} call {e['index']}  {e['notify_label'].split('regime')[-1]}")
    if SELECTION_JSON.exists():
        raise SystemExit(f"{SELECTION_JSON} exists")
    SELECTION_JSON.write_text(json.dumps(sel, indent=1))
    print(f"wrote {SELECTION_JSON} sha256 {sha256(SELECTION_JSON)}")


# ------------------------------------------------------------------------------------------ report


def quant(xs, qs=(0.5, 0.9, 0.99, 1.0)):
    if not xs:
        return "n=0"
    s = sorted(xs)
    return "n=%d " % len(s) + " ".join(f"q{q:g}={s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]:.3g}" for q in qs)


def cmd_report(run_dir, ref_path=None):
    run_dir = Path(run_dir)
    inputs = [ROWS_JSON, CAPTURE_JSON, RUN06A / "checkpoint.jsonl", run_dir / "checkpoint.jsonl"]
    if ref_path:
        inputs.append(Path(ref_path))
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06b: the census's scores and descriptive items", inputs):
        print(l)
    rows, by_key, A, Bc = load_all(run_dir)
    status = collections.Counter((d["rtol"], d["status"]) for d in Bc.values())
    print(f"\nunits: {dict(status)}; 06a units {len(A)}")
    over = [u for u, d in Bc.items() if d["status"] == "over budget"]
    errs = [u for u, d in Bc.items() if d["status"] == "error"]
    print(f"over budget {len(over)}, in error {len(errs)}")

    # ------------------------------------------------------------ the reading's cross-check
    print("\n== the reading (Part B): 'phase-limited in 06a' == 'b asked the premise' on group calls")
    xc = collections.Counter()
    xc_bad = []
    for ub, db in Bc.items():
        da = A["h-" + ub.split("-", 1)[1]]
        for cb, ca in zip(db["calls"], da["calls"]):
            assert cb["index"] == ca["index"] and cb["notify_label"] == ca["notify_label"]
            k = (db["rtol"], bool(ca["phase_limited"]), cb["premise_checks"] > 0)
            xc[k] += 1
            if k[1] != k[2]:
                xc_bad.append((ub, cb["index"]))
    print(f"   (rtol, 06a phase-limited, b asked premise): {dict(sorted(xc.items()))}; disagreements {len(xc_bad)} {xc_bad[:10]}")

    # ------------------------------------------------------------ W1
    print("\n== W1 -- nothing else moves")
    w1 = {}
    w1_fail_calls = collections.defaultdict(list)
    w1_fail_rows = collections.defaultdict(list)
    for rtol in RTOLS:
        n_calls = n_calls_ok = n_rows = n_rows_ok = 0
        n_rows_analytic_pl = 0
        for ub, db in Bc.items():
            if db["rtol"] != rtol:
                continue
            da = A["h-" + ub.split("-", 1)[1]]
            row_clean = True
            for cb, ca in zip(db["calls"], da["calls"]):
                if ca["phase_limited"]:
                    row_clean = False
                    continue
                n_calls += 1
                same = (
                    hx(cb["value"]) == hx(ca["value"])
                    and hx(cb["abserr"]) == hx(ca["abserr"])
                    and cb["num_regions"] == ca["num_regions"]
                    and cb["max_depth"] == ca["max_depth"]
                )
                n_calls_ok += same
                if not same:
                    w1_fail_calls[rtol].append((ub, cb["index"]))
            if any(c["premise_checks"] > 0 for c in db["other_calls"]):
                if row_clean:
                    n_rows_analytic_pl += 1
                row_clean = False
            if row_clean:
                n_rows += 1
                same = hx(db["result"]["total"]) == hx(da["result"]["total"]) and hx(db["result"]["total_abserr"]) == hx(da["result"]["total_abserr"])
                n_rows_ok += same
                if not same:
                    w1_fail_rows[rtol].append(ub)
        w1[rtol] = (n_calls, n_calls_ok, n_rows, n_rows_ok)
        print(
            f"   rtol {rtol}: group calls with no phase-limited region in 06a {n_calls}, equal to 06a's (value, abserr, regions, max depth) {n_calls_ok}; "
            f"rows all of whose calls are such {n_rows}, equal in total and total_abserr {n_rows_ok} "
            f"(rows excluded only by an analytic block's call phase-limited in 06a: {n_rows_analytic_pl})"
        )
        if w1_fail_calls[rtol] or w1_fail_rows[rtol]:
            print(f"      differing calls {w1_fail_calls[rtol][:20]}; differing rows {w1_fail_rows[rtol][:20]}")
    w1_false = any(w1_fail_calls[r] or w1_fail_rows[r] for r in RTOLS)
    w1_verdict = "FAILS" if w1_false else ("UNDECIDED" if (over or errs) else "HOLDS")
    print(f"   W1 {w1_verdict}")

    # ------------------------------------------------------------ W3
    print("\n== W3 -- the exit now fires only where bisection would not help (1e-7)")
    w3_rows = []
    n_pl_regions_arrays = 0
    n_w3_err = 0
    units_with_w3 = 0
    for ub, db in Bc.items():
        if db["rtol"] != "1e-7":
            continue
        n_here = sum(c["n_phase_limited_regions"] for c in db["calls"]) + sum(c["n_phase_limited_regions"] for c in db["other_calls"])
        n_pl_regions_arrays += n_here
        if db["w3"] is None:
            raise RuntimeError(f"{ub}: no W3 record")
        if db["w3"]["n_errors"]:
            n_w3_err += db["w3"]["n_errors"]
        if db["w3"]["n_regions"] != n_here:
            print(f"   !! {ub}: W3 recomputed {db['w3']['n_regions']} regions, the calls hold {n_here}")
        if db["w3"]["n_regions"] == 0:
            continue
        units_with_w3 += 1
        rec = load_rec(run_dir, db)
        for w in rec["w3"]:
            if "error" in w:
                continue
            for g in w["regions"]:
                w3_rows.append({"unit": ub, "row": db["row"], "call_index": w["index"], "notify_label": w["notify_label"], **g})
    n_viol = sum(not g["premise_holds"] for g in w3_rows)
    n_mis = sum(not g["own_equals_stored"] for g in w3_rows)
    ratios = [g["ratio"] for g in w3_rows if g["ratio"] is not None]
    # re-score from the hex numbers themselves, not the worker's bool
    n_viol_hex = sum(
        not (float.fromhex(g["phase_err_L"]) + float.fromhex(g["phase_err_R"]) >= 0.5 * float.fromhex(g["phase_err_own"]))
        for g in w3_rows
    )
    by_kind = collections.Counter(("group" if "regime" in (g["notify_label"] or "") else "analytic", g["type"], g["depth"]) for g in w3_rows)
    print(
        f"   regions b accepted as phase-limited at 1e-7: {len(w3_rows)} (in {units_with_w3} units; the region arrays hold {n_pl_regions_arrays}); "
        f"recomputation errors {n_w3_err}"
    )
    print(f"   own floor recomputed != stored phase_err: {n_mis}")
    print(f"   premise violated (worker's bool) {n_viol}; re-scored from the hex floors {n_viol_hex}")
    print(f"   (halves' sum)/(own floor): min {min(ratios) if ratios else None}; {quant(ratios, (0, 0.01, 0.5, 0.99, 1.0))}")
    print(f"   by (call kind, region type, depth): {dict(sorted(by_kind.items()))}")
    print(f"   halves on the fallback branch (L, R is_direct): {collections.Counter((g['own_is_direct'], g['L_is_direct'], g['R_is_direct']) for g in w3_rows)}")
    w3_verdict = "FAILS" if n_viol_hex else ("UNDECIDED" if (n_w3_err or any(Bc[u]['rtol'] == '1e-7' for u in over + errs)) else "HOLDS")
    print(f"   W3 {w3_verdict}")
    out_csv = WIDER / "out" / "w3_regions.csv"
    keys = ["unit", "row", "call_index", "notify_label", "start", "end", "depth", "type", "phase_err_stored", "phase_err_own",
            "own_equals_stored", "own_is_direct", "phase_err_L", "phase_err_R", "L_is_direct", "R_is_direct", "halves_sum", "ratio", "premise_holds"]
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for g in w3_rows:
            wr.writerow({k: g[k] for k in keys})
    print(f"   wrote {out_csv} ({len(w3_rows)} regions) sha256 {sha256(out_csv)}")

    # ------------------------------------------------------------ W4
    print("\n== W4 -- the cap")
    w4_hits = []
    n_moved = collections.Counter()
    maxd_moved = collections.Counter()
    maxd_all = {}
    for ub, db in Bc.items():
        da = A["h-" + ub.split("-", 1)[1]]
        rt = db["rtol"]
        for cb, ca in zip(db["calls"], da["calls"]):
            maxd_all[rt] = max(maxd_all.get(rt, 0), cb["max_depth"])
            if call_moved(cb, ca):
                n_moved[(rt, "group")] += 1
                maxd_moved[(rt, "group")] = max(maxd_moved[(rt, "group")], cb["max_depth"])
                if cb["max_depth"] >= DEPTH_MAX:
                    w4_hits.append((ub, cb["index"]))
        for cb in db["other_calls"]:
            maxd_all[rt] = max(maxd_all.get(rt, 0), cb["max_depth"])
            if cb["premise_failed"] > 0:
                n_moved[(rt, "analytic")] += 1
                maxd_moved[(rt, "analytic")] = max(maxd_moved[(rt, "analytic")], cb["max_depth"])
                if cb["max_depth"] >= DEPTH_MAX:
                    w4_hits.append((ub, cb["index"]))
    print(f"   moved calls: {dict(n_moved)}; their largest depth: {dict(maxd_moved)}; largest depth of any b call: {maxd_all}")
    print(f"   moved calls at depth >= {DEPTH_MAX}: {len(w4_hits)} {w4_hits[:20]}; units over budget {len(over)}")
    for ub, idx in w4_hits:
        db = Bc[ub]
        c = next(x for x in db["calls"] + db["other_calls"] if x["index"] == idx)
        print(f"      {ub} ({db['row']}) call {idx} {c['notify_label']}: value {c['value']['repr']} abserr {c['abserr']['repr']} converged {c['converged']} "
              f"regions {c['num_regions']} evaluations {c['evaluations']} depth hist {c['depth_hist_levin_cc']} premise asked {c['premise_checks']} failed {c['premise_failed']}; "
              f"unit wall {db['wall_s']:.1f} s; row total_converged {db['result']['total_converged']}")
    w4_verdict = "FAILS" if (w4_hits or over) else ("UNDECIDED" if errs else "HOLDS")
    print(f"   W4 {w4_verdict}")

    # ------------------------------------------------------------ W2
    w2_verdict = None
    if ref_path:
        print("\n== W2 -- honest where it moved")
        ref = json.loads(Path(ref_path).read_text())
        res = ref["results"]
        n_err = sum(1 for r in res if r.get("error"))
        n_rerun_bad = sum(1 for r in res if not r.get("error") and not r["rerun_equals_census"])
        dropped = [r for r in res if not r.get("error") and not r["agree"]]
        left = [r for r in res if not r.get("error") and r["agree"] and r["rerun_equals_census"]]
        fails = [r for r in left if not r["holds"]]
        print(f"   calls {len(res)}; reference errors {n_err}; re-run differs from b {n_rerun_bad}; dropped (|S9-S10| > abserr_b/100) {len(dropped)}; left {len(left)}; failing {len(fails)}")
        for r in res:
            if r.get("error"):
                print(f"   {r['unit']} call {r['index']}: ERROR")
                continue
            eb = float.fromhex(r["abserr_b"])
            print(
                f"   {r['band']:<4} {r['k_key']:<6} {r['unit']:<18} {r['index']} rerun {r['rerun_equals_census']} agree {r['agree']} "
                f"|S9-S10|/abserr_b {float.fromhex(r['S_diff']) / eb:.3g} err_b/abserr_b {r['err_b_over_abserr_b']:.3g} holds {r['holds']}; "
                f"before: err_06a/abserr_06a {r['err_06a_over_abserr_06a']:.3g}; abserr_06a/abserr_b {float.fromhex(r['abserr_06a']) / eb:.3g}; "
                f"floor sum (2^10 pieces)/abserr_b {float.fromhex(r['floor_sum_10']) / eb:.3g}; direct pieces {r['n_direct_9']}/{r['n_direct_10']}"
            )
        if fails:
            w2_verdict = "FAILS"
        elif n_err or n_rerun_bad or not left:
            w2_verdict = "UNDECIDED"
        else:
            w2_verdict = "HOLDS"
        e06 = [r["err_06a_over_abserr_06a"] for r in res if not r.get("error")]
        eb_ = [r["err_b_over_abserr_b"] for r in res if not r.get("error")]
        print(f"   err_b/abserr_b {quant(eb_, (0, 0.5, 1.0))}; before, err_06a/abserr_06a {quant(e06, (0, 0.5, 1.0))}; "
              f"06a calls outside their own abserr: {sum(1 for r in res if not r.get('error') and not r['within_06a'])}")
        print(f"   W2 {w2_verdict}")

    # ------------------------------------------------------------ descriptive
    print("\n== descriptive: how often the exit fired")
    for rtol in RTOLS:
        a_calls = a_rows = b_calls = b_rows = b_an_calls = b_an_rows = 0
        a_tpl = b_tpl = 0
        for ub, db in Bc.items():
            if db["rtol"] != rtol:
                continue
            da = A["h-" + ub.split("-", 1)[1]]
            na = sum(c["phase_limited"] for c in da["calls"])
            nb = sum(c["phase_limited"] for c in db["calls"])
            nban = sum(c["phase_limited"] for c in db["other_calls"])
            a_calls += na
            a_rows += na > 0
            b_calls += nb
            b_rows += nb > 0
            b_an_calls += nban
            b_an_rows += (nb + nban) > 0
            a_tpl += da["result"]["total_phase_limited"]
            b_tpl += db["result"]["total_phase_limited"]
        print(
            f"   rtol {rtol}: phase-limited group calls 06a {a_calls} on {a_rows} rows -> b {b_calls} on {b_rows} rows; "
            f"total_phase_limited rows 06a {a_tpl} -> b {b_tpl}; b's analytic-block calls phase-limited {b_an_calls} (rows with any phase-limited call, group or analytic, {b_an_rows})"
        )
        # 06a's phase-limited calls and regions that fail the premise: b's premise counter on the same calls
        g_checks = g_failed = g_calls_failing = g_calls_holding = an_checks = an_failed = an_calls = an_calls_failing = 0
        for ub, db in Bc.items():
            if db["rtol"] != rtol:
                continue
            da = A["h-" + ub.split("-", 1)[1]]
            for cb, ca in zip(db["calls"], da["calls"]):
                if ca["phase_limited"]:
                    g_checks += cb["premise_checks"]
                    g_failed += cb["premise_failed"]
                    if cb["premise_failed"] > 0:
                        g_calls_failing += 1
                    if cb["premise_failed"] < cb["premise_checks"]:
                        g_calls_holding += 1
            for cb in db["other_calls"]:
                if cb["premise_checks"] > 0:
                    an_calls += 1
                    an_checks += cb["premise_checks"]
                    an_failed += cb["premise_failed"]
                    an_calls_failing += cb["premise_failed"] > 0
        print(
            f"      06a's phase-limited group calls: premise asked {g_checks} times in b, failed {g_failed}; calls with a failed premise {g_calls_failing}, "
            f"calls with a premise that held {g_calls_holding}"
        )
        print(f"      analytic-block calls phase-limited in 06a (by the reading) {an_calls}: premise asked {an_checks}, failed {an_failed}; calls with a failed premise {an_calls_failing}")

    # region-level: 06a's phase-limited regions (group calls) found again in b as phase-limited or not
    print("\n   06a's phase-limited regions (group calls; region arrays matched by start and end, as hex):")
    for rtol in RTOLS:
        held = failed = 0
        for ub, db in Bc.items():
            if db["rtol"] != rtol:
                continue
            da = A["h-" + ub.split("-", 1)[1]]
            if not any(c["phase_limited"] for c in da["calls"]):
                continue
            ra = load_rec(RUN06A, da)
            rb = load_rec(run_dir, db)
            mb = {}
            for arr in rb["regions"]:
                for s, e, pl in zip(arr["start"], arr["end"], arr["phase_limited"]):
                    mb[(arr["index"], float(s).hex(), float(e).hex())] = bool(pl)
            for arr in ra["regions"]:
                for s, e, pl in zip(arr["start"], arr["end"], arr["phase_limited"]):
                    if pl:
                        if mb.get((arr["index"], float(s).hex(), float(e).hex())):
                            held += 1
                        else:
                            failed += 1
        print(f"   rtol {rtol}: 06a phase-limited regions {held + failed}: still phase-limited in b {held}; bisected in b (premise failed) {failed}")

    print("\n== descriptive: depth (row depth = largest max_depth over the row's group calls, as 06a; b also with the analytic blocks)")
    for rtol in RTOLS:
        ha, hb, hb2 = collections.Counter(), collections.Counter(), collections.Counter()
        for ub, db in Bc.items():
            if db["rtol"] != rtol:
                continue
            da = A["h-" + ub.split("-", 1)[1]]
            ha[max([c["max_depth"] for c in da["calls"]], default=None)] += 1
            hb[max([c["max_depth"] for c in db["calls"]], default=None)] += 1
            hb2[max([c["max_depth"] for c in db["calls"] + db["other_calls"]], default=None)] += 1
        keyf = lambda kv: (-1 if kv[0] is None else kv[0])
        print(f"   rtol {rtol}: 06a {dict(sorted(ha.items(), key=keyf))}")
        print(f"              b   {dict(sorted(hb.items(), key=keyf))}")
        print(f"              b with analytic blocks {dict(sorted(hb2.items(), key=keyf))}")

    print("\n== descriptive: totals against 06a, by band and k (rows whose total moved)")
    moved_rows = []
    for rtol in RTOLS:
        grp = collections.defaultdict(list)
        n_moved_rows = 0
        for ub, db in Bc.items():
            if db["rtol"] != rtol:
                continue
            da = A["h-" + ub.split("-", 1)[1]]
            R = by_key[rowkey(ub)]
            ta, tb = fl(da["result"]["total"]), fl(db["result"]["total"])
            ea = fl(da["result"]["total_abserr"])
            if hx(da["result"]["total"]) == hx(db["result"]["total"]):
                continue
            n_moved_rows += 1
            rel = abs(tb - ta) / abs(ta) if ta != 0 else math.inf
            vs = abs(tb - ta) / ea if ea > 0 else math.inf
            grp[(R["band"], R["k_key"])].append((rel, vs))
            moved_rows.append((rtol, ub, R["band"], R["k_key"], rel, vs, fl(db["result"]["total_abserr"]) / ea if ea > 0 else math.inf))
        print(f"   rtol {rtol}: rows whose total moved {n_moved_rows}")
        for key in sorted(grp, key=lambda k: (k[0], float(by_key_k(rows, k[1])))):
            v = grp[key]
            print(f"      {key[0]:<4} {key[1]:<6} n={len(v):4d}  |dtotal|/|total_06a| max {max(x[0] for x in v):.3g} median {sorted(x[0] for x in v)[len(v) // 2]:.3g};  "
                  f"|dtotal|/total_abserr_06a max {max(x[1] for x in v):.3g} median {sorted(x[1] for x in v)[len(v) // 2]:.3g}; >1: {sum(x[1] > 1 for x in v)}")
        rat = [m[6] for m in moved_rows if m[0] == rtol]
        print(f"      total_abserr_b/total_abserr_06a on the moved rows: {quant(rat, (0, 0.5, 1.0))}")

    print("\n== descriptive: rows whose total moved by more than total_abserr_06a -- which component moved")
    for rtol in RTOLS:
        comp = collections.Counter()
        lines = []
        for ub, db in Bc.items():
            if db["rtol"] != rtol:
                continue
            da = A["h-" + ub.split("-", 1)[1]]
            ta, tb = fl(da["result"]["total"]), fl(db["result"]["total"])
            ea = fl(da["result"]["total_abserr"])
            if abs(tb - ta) <= ea:
                continue
            moved = tuple(k for k in ("WKB_Levin", "analytic_rad", "numeric_quad") if hx(da["result"][k]) != hx(db["result"][k]))
            comp[moved] += 1
            an = da["result"]["analytic_rad"], db["result"]["analytic_rad"]
            lines.append((abs(tb - ta) / ea, ub, moved, hx(da["result"]["total"]), hx(db["result"]["total"]), hx(an[0]), hx(an[1]),
                          hx(da["result"]["total_abserr"]), hx(db["result"]["total_abserr"])))
        print(f"   rtol {rtol}: {len(lines)} rows; components that moved: {dict(comp)}")
        for l in sorted(lines, reverse=True)[:8]:
            print(f"      |dtotal|/abserr_06a {l[0]:.3g} {l[1]} moved {l[2]}: total 06a {float.fromhex(l[3]):.6g} b {float.fromhex(l[4]):.6g}; "
                  f"analytic_rad 06a {float.fromhex(l[5]) if l[5] else None:.6g} b {float.fromhex(l[6]) if l[6] else None:.6g}; abserr 06a {float.fromhex(l[7]):.3g} b {float.fromhex(l[8]):.3g}")

    print("\n== descriptive: the 720 stationary-point rows of 06a's scan (07's; not scored)")
    sp = set()
    for ua, da in A.items():
        if da["rtol"] == "1e-7" and any(s and s["n_sign_changes"] > 0 for s in (da.get("scan") or [])):
            sp.add(rowkey(ua))
    for rtol in RTOLS:
        moved = deeper = shallower = pla = plb = 0
        dd = collections.Counter()
        for rk in sp:
            da, db = A[f"h-{rtol}-{rk}"], Bc[f"b-{rtol}-{rk}"]
            moved += hx(da["result"]["total"]) != hx(db["result"]["total"])
            a = max(c["max_depth"] for c in da["calls"])
            b = max(c["max_depth"] for c in db["calls"])
            dd[b - a] += 1
            pla += any(c["phase_limited"] for c in da["calls"])
            plb += any(c["phase_limited"] for c in db["calls"])
        print(f"   rtol {rtol}: rows {len(sp)}; total moved {moved}; row depth change b-06a {dict(sorted(dd.items()))}; rows with a phase-limited group call 06a {pla} -> b {plb}")

    print("\n== descriptive: convergence (total_converged)")
    for rtol in RTOLS:
        ca = sum(1 for u, d in A.items() if d["rtol"] == rtol and not d["result"]["total_converged"])
        cb = sum(1 for u, d in Bc.items() if d["rtol"] == rtol and not d["result"]["total_converged"])
        ga = sum(1 for u, d in A.items() if d["rtol"] == rtol for c in d["calls"] if not c["converged"])
        gb = sum(1 for u, d in Bc.items() if d["rtol"] == rtol for c in d["calls"] if not c["converged"])
        print(f"   rtol {rtol}: total_converged False on 06a {ca} -> b {cb} rows; unconverged group calls 06a {ga} -> b {gb}")

    print("\n== descriptive: cost")
    for name, recs in (("06a", A), ("b", Bc)):
        for rtol in RTOLS:
            w = [d["wall_s"] for d in recs.values() if d["rtol"] == rtol]
            rb = [d["rebuild_s"] for d in recs.values() if d["rtol"] == rtol]
            print(f"   {name} rtol {rtol}: Levin wall (timed section) sum {sum(w):.0f} s, {quant(w, (0.5, 0.99, 1.0))}; rebuild sum {sum(rb):.0f} s")
    w3s = sum(d["w3"]["seconds"] for d in Bc.values() if d["w3"])
    print(f"   b, W3's recomputation (outside the timed section) sum {w3s:.1f} s")

    # the per-row file and the moved calls
    with gzip.open(WIDER / "out" / "census_rows_06b.csv.gz", "wt", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["unit", "row", "rtol", "band", "k_key", "total_06a", "total_b", "abserr_06a", "abserr_b", "converged_06a", "converged_b",
                     "rowdepth_06a", "rowdepth_b", "pl_calls_06a", "pl_calls_b", "premise_checks_b", "premise_failed_b", "wall_b"])
        for ub, db in sorted(Bc.items()):
            da = A["h-" + ub.split("-", 1)[1]]
            R = by_key[rowkey(ub)]
            allc = db["calls"] + db["other_calls"]
            wr.writerow([ub, db["row"], db["rtol"], R["band"], R["k_key"], hx(da["result"]["total"]), hx(db["result"]["total"]),
                         hx(da["result"]["total_abserr"]), hx(db["result"]["total_abserr"]), da["result"]["total_converged"], db["result"]["total_converged"],
                         max([c["max_depth"] for c in da["calls"]], default=None), max([c["max_depth"] for c in db["calls"]], default=None),
                         sum(c["phase_limited"] for c in da["calls"]), sum(c["phase_limited"] for c in db["calls"]),
                         sum(c["premise_checks"] for c in allc), sum(c["premise_failed"] for c in allc), round(db["wall_s"], 4)])
    with open(WIDER / "out" / "moved_calls.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["unit", "index", "group", "value_06a", "value_b", "abserr_06a", "abserr_b", "regions_06a", "regions_b", "depth_06a", "depth_b",
                     "phase_limited_06a", "phase_limited_b", "premise_checks_b", "premise_failed_b"])
        for ub, db in sorted(Bc.items()):
            da = A["h-" + ub.split("-", 1)[1]]
            for cb, ca in zip(db["calls"], da["calls"]):
                if call_moved(cb, ca):
                    wr.writerow([ub, cb["index"], cb["group"], hx(ca["value"]), hx(cb["value"]), hx(ca["abserr"]), hx(cb["abserr"]), ca["num_regions"],
                                 cb["num_regions"], ca["max_depth"], cb["max_depth"], ca["phase_limited"], cb["phase_limited"], cb["premise_checks"], cb["premise_failed"]])
    print(f"\nwrote out/census_rows_06b.csv.gz and out/moved_calls.csv")
    print(f"\n== verdicts: W1 {w1_verdict}; W2 {w2_verdict}; W3 {w3_verdict}; W4 {w4_verdict}")


def by_key_k(rows, k_key):
    for r in rows:
        if r["k_key"] == k_key:
            return r["k"]


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "select":
        cmd_select(sys.argv[2])
    elif cmd == "report":
        cmd_report(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
