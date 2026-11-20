"""Section 9.5's S3 prediction, scored on R32 (prompt 07c section 9.5, the S3 item). New in 07c's rehearsal.

The prediction: R31's record (G+q+r at (k, q, r) = (1e5, 1e6, 1e6), z = 144627.7, R31's record
WKB_Levin.subintervals[1].groups[0], 143 regions at R31, unconverged) converges at 86 regions, honest at 0.053 of its
abserr; its row moves, with total_converged possibly false -> true.

Read, all read only:
- R32's store, through replay07c_r32.read_store: the record's stored value, abserr, converged and regions; and R31's
  store for the same, for the record; the row's total_converged in each from out/qsi_rows_r31_r32.csv.
- The replay at h (state/capture/replay; out/replay_calls.checks.json): the one call whose stored form ((1 + z) times the
  call's value and abserr) equals R32's record; its own (unscaled) value, abserr, converged, num_regions, max_depth,
  refusals_07b and chase_07c.
- The capture of that call from R32 (out/refcap.json, refcapcheck's record; its blob in state/inputs/), compared byte for
  byte with the capture from R31 on which 07c Part A took the reference (07b's rehearsal
  state/inputs/ref-G+q+r-h-3-20363-1.pkl, and Part A's copy state/refs/inputs/ref-R31-3-20363-g1.pkl, SHA-256 df2d478f...).
  The blobs differ by the unit name, the tree path and str hashes (as every rehearsal's block captures do), so the
  integrand is compared instead: block_reference_r30.py (no --reference) run on each capture at the tree
  (out/s3_record/), whose 1001 samples (x, integrand, theta, theta mod 2pi, theta') must be bit-identical, with equal
  x_span and kw. Then Part A's reference is a reference for R32's call.
- 07c Part A's (S) replay record of that call, scores_07c.json table "r31-3-20363-1" "S" (value, abserr, converged,
  regions, max_depth), compared bit for bit with the replay's call.
- 07c Part A's readings of ref-R31-3-20363-g1 (out/refs/readings/ of var/levin-phase-floor-07c/measurement): the reference
  is split FCC at resolution 2 (the fsum of its pieces' value_30), its stated error the spread of every reading taken
  (largest minus smallest), as Part A's score_07c.py my_refs() reads them; evaluated here in exact Fractions of the
  decimal strings.

The verdict (07's rule, as Part A scored it): scored if the stated error is at most abserr/100; honest iff
|v - ref| <= abserr, evaluated exactly (Fractions of the double and the decimal readings). The prediction holds iff R32's
record is converged with 86 regions and its call is honest; the ratio is reported against 0.053.

Usage: PYTHONPATH=<tree> python -B s3_record_r32.py OUT_JSON
"""

import hashlib
import json
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
Z = MAIN / "var" / "levin-phase-floor-07c" / "rehearsal"
PA = MAIN / "var" / "levin-phase-floor-07c" / "measurement"
sys.path.insert(0, str(Z / "scripts" / "capture"))
import replay07c_r32 as RP  # noqa: E402

KEY = (100000.0, 1000000.0, 1000000.0, 144627.70674226896)
PATH = "WKB_Levin.subintervals[1].groups[0]"
LABEL = "G+q+r"
R31_PKLS = [MAIN / "var/levin-phase-floor-07b/rehearsal/state/inputs/ref-G+q+r-h-3-20363-1.pkl",
            PA / "state/refs/inputs/ref-R31-3-20363-g1.pkl"]
R31_PKL_SHA = "df2d478f91034ad60f1ee125e7d760c7a8b7a8f2c659644b60eeb65c9ac3a249"
SCORES = PA / "out/tables/scores_07c.json"
READ = PA / "out/refs/readings"
REF = "ref-R31-3-20363-g1"
PREDICTED_REGIONS = 86
SAMP_R32 = Z / "out/s3_record/R32-ref-G+q+r-h-3-122-1.json"
SAMP_R31 = Z / "out/s3_record/R31-ref-G+q+r-h-3-20363-1.json"
PREDICTED_RATIO = 0.053


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def F(dec):
    return Fraction(Decimal(dec))


def main():
    out_p = Path(sys.argv[1])
    ins = [SAMP_R32, SAMP_R31, Z / "out/qsi_rows_r31_r32.csv", Z / "state/replay_rows.json", Z / "out/replay_calls.checks.json", Z / "out/refcap.json", SCORES, *R31_PKLS,
           *sorted(READ.glob(f"{REF}-*.json"))]
    print(f"# script {Path(__file__).resolve()} sha256 {sha256(__file__)}")
    for p in ins:
        print(f"# input {p} sha256 {sha256(p)}")
    res = {"key": list(KEY), "record": PATH, "label": LABEL}
    # the stores
    s32 = RP.read_store(RP.STORES["h"])[KEY]
    s31 = RP.read_store(RP.R31_STORE)[KEY]
    rec = {}
    for tag, s in (("R31", s31), ("R32", s32)):
        g = [r for kind, name, r, _ in RP.levin_records(s["metadata"]) if name == PATH][0]
        assert g["label"] == LABEL, g["label"]
        rec[tag] = {"value": float(g["value"]).hex(), "abserr": float(g["abserr"]).hex(), "converged": g["converged"],
                    "regions": g["regions"]}
    import csv
    rr = [x for x in csv.DictReader(open(Z / "out/qsi_rows_r31_r32.csv")) if (float(x["k"]), float(x["q"]), float(x["r"]), float(x["z_response"])) == KEY]
    assert len(rr) == 1
    rec["R31"]["row_total_converged"], rec["R32"]["row_total_converged"] = rr[0]["total_converged_31"], rr[0]["total_converged_32"]
    res["store"] = rec
    # the replay's call
    rows = json.loads((Z / "state/replay_rows.json").read_text())["rows"]
    e = [e for e in rows if tuple(e["key"]) == KEY][0]
    uid = f"h-1e-7-{e['R32']['shard']}-{e['R32']['serial']}"
    units = RP.load_units(Z / "state/capture/replay")
    u, bye, _ = units[uid]
    hits = [c for c in u["calls"] if c["notify_label"].endswith(f"group {LABEL}")
            and RP.stored_form(c, KEY[3]) == (rec["R32"]["value"], rec["R32"]["abserr"])]
    assert len(hits) == 1, len(hits)
    c = hits[0]
    chk = [x for x in json.loads((Z / "out/replay_calls.checks.json").read_text())["checks"]
           if tuple(x["key"]) == KEY and x["path"] == PATH][0]
    call = {"uid": uid, "index": c["index"], "notify_label": c["notify_label"], "value": c["value"]["hex"],
            "abserr": c["abserr"]["hex"], "converged": c["converged"], "num_regions": c["num_regions"],
            "max_depth": c["max_depth"], "refusals_07b": c.get("refusals_07b"), "chase_07c": c.get("chase_07c"),
            "replay_reproduces_record": chk["reproduces"], "ray_initialized": None if bye is None else bye.get("ray_initialized")}
    res["call"] = call
    # the capture against R31's
    caps = [x for x in json.loads((Z / "out/refcap.json").read_text())["captures"] if tuple(x["key"]) == KEY and x["record"] == PATH]
    assert len(caps) == 1, len(caps)
    cap = caps[0]
    blob = Path(cap["pickle"]).read_bytes()
    r31_blobs = [p.read_bytes() for p in R31_PKLS]
    res["capture"] = {"pickle": cap["pickle"], "sha256": hashlib.sha256(blob).hexdigest(), "reproduces_store": cap["reproduces_store"],
                      "index": cap.get("index"), "equals_R31_capture": [b == blob for b in r31_blobs],
                      "R31_capture_sha256": [hashlib.sha256(b).hexdigest() for b in r31_blobs],
                      "R31_capture_sha256_expected": R31_PKL_SHA}
    # the integrand: block_reference_r30.py (no --reference) on each capture at the tree, out/s3_record/ (the blobs differ
    # by the unit name, the tree path and str hashes, as every rehearsal's block captures do)
    sj = {t: json.loads(p_.read_text()) for t, p_ in (("R32", SAMP_R32), ("R31", SAMP_R31))}
    res["capture"]["samples"] = {
        "R32_json": str(SAMP_R32), "R31_json": str(SAMP_R31),
        "R32_pickle_is_refcap": sj["R32"]["header"]["pickle"] == cap["pickle"] and sj["R32"]["header"]["pickle_sha256"] == res["capture"]["sha256"],
        "R31_pickle_sha256": sj["R31"]["header"]["pickle_sha256"],
        "samples_bit_identical_1001": sj["R32"]["samples"] == sj["R31"]["samples"],
        "x_span_equal": sj["R32"]["x_span"] == sj["R31"]["x_span"], "kw_equal": sj["R32"]["kw"] == sj["R31"]["kw"],
        "R32_replay_reproduces_capture": sj["R32"]["replay_reproduces_capture"],
        "R31_capture_replayed_at_tree": sj["R31"]["replay"], "R31_capture_record": sj["R31"]["capture_record"],
        "R31_capture_replay_equals_call": (sj["R31"]["replay"]["value"], sj["R31"]["replay"]["abserr"], sj["R31"]["replay"]["num_regions"],
                                           sj["R31"]["replay"]["converged"]) == (call["value"], call["abserr"], call["num_regions"], call["converged"]),
        "ray_initialized": [sj[t]["ray_initialized"] for t in ("R32", "R31")]}
    same_integrand = (res["capture"]["samples"]["samples_bit_identical_1001"] and res["capture"]["samples"]["x_span_equal"]
                      and res["capture"]["samples"]["kw_equal"] and res["capture"]["samples"]["R31_pickle_sha256"] == R31_PKL_SHA)
    res["capture"]["same_integrand_as_reference_capture"] = same_integrand
    # Part A's (S) record
    sa = json.loads(SCORES.read_text())["table"]["r31-3-20363-1"]["S"]
    eq = {"value": sa["value"] == call["value"], "abserr": sa["abserr"] == call["abserr"], "converged": sa["converged"] == call["converged"],
          "regions": sa["regions"] == call["num_regions"], "max_depth": sa["max_depth"] == call["max_depth"]}
    res["partA_S"] = {"record": {k: sa[k] for k in ("value", "abserr", "converged", "regions", "max_depth")}, "equal": eq,
                      "equal_all": all(eq.values()),
                      "partA_ratio_S": json.loads(SCORES.read_text())["scored"]["r31-3-20363-1"]["by"]["S"]["ratio"]}
    # the reference, exactly
    rd = {}
    fs = json.loads((READ / f"{REF}-fccsplit.json").read_text())
    for r_, x in fs["readings"].items():
        rd[f"fccsplit-r{r_}"] = sum((F(p["value_30"]) for p in x["pieces"]), Fraction(0))
    for impl in ("fcc", "qawo"):
        for r_ in (1, 2):
            q = READ / f"{REF}-{impl}-r{r_}.json"
            if q.exists():
                d = json.loads(q.read_text())
                if d.get("value_30") is not None:
                    rd[f"{impl}-r{r_}"] = F(d["value_30"])
    ref = rd["fccsplit-r2"]
    spread = max(rd.values()) - min(rd.values())
    v, ab = Fraction(float.fromhex(call["value"])), Fraction(float.fromhex(call["abserr"]))
    d = abs(v - ref)
    scored = spread <= ab / 100
    honest = d <= ab
    res["reference"] = {"key": REF, "basis": "+".join(sorted(rd)), "ref_decimal_30": f"{float(ref):.17g}",
                        "readings": {k: f"{float(x):.17g}" for k, x in rd.items()}, "spread": float(spread),
                        "spread_over_abserr": float(spread / ab), "scored": scored}
    res["ratio"] = float(d / ab)
    res["honest"] = honest
    pred = (rec["R32"]["converged"] is True and rec["R32"]["regions"] == PREDICTED_REGIONS and call["converged"] is True
            and call["num_regions"] == PREDICTED_REGIONS and scored and honest and same_integrand)
    res["prediction"] = {"converged_at_86_regions": rec["R32"]["converged"] is True and rec["R32"]["regions"] == PREDICTED_REGIONS,
                         "honest": honest, "ratio_predicted": PREDICTED_RATIO, "holds": pred}
    print(json.dumps(res, indent=1))
    print(f"\nR31: value {rec['R31']['value']} abserr {rec['R31']['abserr']} converged {rec['R31']['converged']} regions {rec['R31']['regions']}; "
          f"R32: value {rec['R32']['value']} abserr {rec['R32']['abserr']} converged {rec['R32']['converged']} regions {rec['R32']['regions']} "
          f"(row total_converged {rec['R31']['row_total_converged']} -> {rec['R32']['row_total_converged']})")
    print(f"the call: value {call['value']} abserr {call['abserr']} converged {call['converged']} regions {call['num_regions']} "
          f"max depth {call['max_depth']}; refusals {call['refusals_07b']}; chase {call['chase_07c']}; replay reproduces R32 {call['replay_reproduces_record']}")
    print(f"capture from R32 byte-equal to R31's (07b rehearsal, Part A copy): {res['capture']['equals_R31_capture']} (not expected: unit name, tree path, str hashes)")
    print(f"integrand: {dict((k, v) for k, v in res['capture']['samples'].items() if k not in ('R31_capture_replayed_at_tree', 'R31_capture_record'))}")
    print(f"same integrand as the reference's capture: {same_integrand}")
    print(f"equal to Part A's (S) record bit for bit: {eq}")
    print(f"reference {REF}: {res['reference']['ref_decimal_30']} (fccsplit r2), spread {res['reference']['spread']:.3e} "
          f"= {res['reference']['spread_over_abserr']:.3e} of abserr: scored {scored}")
    print(f"|v - ref| / abserr = {res['ratio']!r} (Part A's (S): {res['partA_S']['partA_ratio_S']!r}): {'honest' if honest else 'NOT honest'}")
    print(f"section 9.5's S3 prediction (converged at 86 regions, honest at 0.053): {'HOLDS' if pred else 'FAILS'}")
    out_p.write_text(json.dumps(res, indent=1))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0 if pred else 1


if __name__ == "__main__":
    sys.exit(main())
