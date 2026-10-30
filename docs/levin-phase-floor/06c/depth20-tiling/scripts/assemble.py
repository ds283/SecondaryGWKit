"""depth20-tiling step 5: assemble the per-call numbers (results.json, every number in hex too) and
the tables (state/tables.md, pasted into results.md). Reads state/sample.json, state/captured.json,
state/cheap_pass.json, state/classify/, state/runs/{replay,pinned}/, state/ref/. Writes only those
two files.

The descriptive reading ("reading", post hoc, after the rule's class was computed): each flagged
deep level not attributed by the rule is labelled
  - "Levin P vs CC halves": P is a Levin solve and a half is a Clenshaw-Curtis solve;
  - "offset-dominated, first-order prediction off by X": tests (i) and (ii) hold, (iii) fails, and
    pinning removes >= 99 % of the signed residual in mp;
  - else "other: <the rule's largest component>".

Usage: PYTHONPATH=<tree>:<scripts> python -B assemble.py
"""

import collections
import glob
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dprov import BLOCKS, STATE, TASK, header, hx  # noqa: E402


def f(h):
    return float.fromhex(h)


def label(L):
    if L.get("attributed_to_tiling"):
        return "attributed"
    if "direct" in L["types"][1:]:
        return "Levin P vs CC halves"
    t = L["tests"]
    if t["i_no_tile"] and t["ii_pinned_le_10floor"] and not t["iii_change_matches_offsets"] and abs(L["s_mp_pinned"]) <= 0.01 * abs(L["s_mp_computed"]):
        return f"offset-dominated, prediction off {L['dS_mismatch'] / abs(L['dS_predicted']):.2f}"
    return "other: " + L["largest_component"]


def main():
    ins = [STATE / "sample.json", STATE / "captured.json", STATE / "cheap_pass.json"] + sorted(Path(p) for p in glob.glob(str(STATE / "classify" / "*.json"))) \
        + sorted(Path(p) for p in glob.glob(str(STATE / "runs" / "*" / "*.json"))) + sorted(Path(p) for p in glob.glob(str(STATE / "ref" / "*.json")))
    hdr = header(Path(__file__).resolve(), "depth20-tiling: assembly", ins)
    for l in hdr:
        print(l)
    sample = json.loads((STATE / "sample.json").read_text())
    calls = []
    for e in sample:
        n = e["sample_no"]
        cl = json.loads((STATE / "classify" / f"{n:02d}.json").read_text())
        rp = json.loads((STATE / "runs" / "replay" / f"{n:02d}_replay.json").read_text())
        pn = json.loads((STATE / "runs" / "pinned" / f"{n:02d}_pinned.json").read_text())
        refp = STATE / "ref" / f"{n:02d}.ref.json"
        ref = json.loads(refp.read_text()) if refp.exists() else None
        deep = [L for L in cl["levels"] if L["depth"] >= 16]
        flagged = [L for L in deep if L.get("flagged")]
        labels = [label(L) for L in flagged]
        reading_a = bool(flagged) and all(l_ == "attributed" or l_.startswith("offset-dominated") or (l_ == "Levin P vs CC halves" and L["depth"] == max(x["depth"] for x in flagged)) for l_, L in zip(labels, flagged))
        c = {"sample_no": n, "rtol": e["rtol"], "block": e["block"], "unit": e["unit"], "index": e["index"], "notify_label": e["notify_label"],
             "class": cl["summary"]["class"], "is_a": cl["summary"]["is_a"], "is_b": cl["summary"]["is_b"], "class_c_component": cl["summary"]["class_c_component"],
             "deep_flagged": len(flagged), "deep_attributed": sum(1 for L in flagged if L.get("attributed_to_tiling")),
             "deep_labels": {str(L["depth"]): l_ for L, l_ in zip(flagged, labels)},
             "reading_tiling_holds_the_deep_levels": reading_a,
             "line_reproduces_census": cl["summary"]["all_levels_reproduce_census"],
             "deep_levels": [{k: L.get(k) for k in ("depth", "types", "tile", "parent_exact", "census_resid", "census_floor", "resid_pinned_double", "floor_pinned",
                                                     "resid_over_floor", "resid_pinned_over_floor_pinned", "s_mp_computed", "s_mp_pinned", "dS_measured",
                                                     "dS_predicted", "dS_mismatch", "dbl_minus_mp_computed", "flagged", "tests", "attributed_to_tiling")} for L in deep],
             "stationary": {k: cl["stationary"][k] for k in ("max_abs_theta_prime_span", "region16", "region16_min_abs_theta_prime", "region16_min_at_x", "region16_min_at_z",
                                                              "region16_theta_prime_at_min", "region16_theta_second_at_min", "region16_min_over_span_max", "region16_sign_changes",
                                                              "span_sign_changes", "holds")},
             "census": {"value": rp["census"]["value"], "abserr": rp["census"]["abserr"], "num_regions": rp["census"]["num_regions"], "max_depth": rp["census"]["max_depth"],
                        "converged": rp["census"]["converged"], "call_elapsed_s": rp["census"]["call_elapsed_s"]},
             "replay": {"equals_census": rp["equals_census"], "wall_s": rp["wall_s"], **{k: rp["result"][k] for k in ("num_at_depth_20", "num_subregion_solves", "depth_hist")}},
             "pinned": {"wall_s": pn["wall_s"], "patch_selfcheck": {k: v for k, v in pn["patch_selfcheck"].items() if k != "rows"},
                        **{k: pn["result"][k] for k in ("value", "abserr", "num_regions", "max_depth", "converged", "num_at_depth_20", "num_subregion_solves",
                                                         "num_phase_limited_regions", "num_untrusted_regions", "depth_hist")}},
             }
        vc, ec = f(c["census"]["value"]), f(c["census"]["abserr"])
        vp, ep = f(c["pinned"]["value"]), f(c["pinned"]["abserr"])
        c["pinned_minus_census"] = hx(vp - vc)
        c["pinned_minus_census_over_census_abserr"] = abs(vp - vc) / ec
        c["pinned_minus_census_over_sum_abserr"] = abs(vp - vc) / (ec + ep)
        if ref is not None:
            r = f(ref["ref"])
            cd = f(ref["conv_diff"])
            c["reference"] = {"ref": ref["ref"], "S_low": ref["S_low"], "conv_diff": ref["conv_diff"], "npanels": ref["npanels"],
                              "conv_over_min_abserr": cd / min(ec, ep), "converged_1_100": cd <= min(ec, ep) / 100,
                              "census_err_over_abserr": abs(vc - r) / ec, "pinned_err_over_abserr": abs(vp - r) / ep,
                              "census_err": hx(vc - r), "pinned_err": hx(vp - r), "wall_s": ref["wall_s"]}
        else:
            c["reference"] = None
        calls.append(c)
    (TASK / "results.json").write_text(json.dumps({"header": hdr, "calls": calls}, indent=1))

    T = []
    T.append("### Classification, per sampled call\n")
    T.append("Rule class: (a) tiling, (b) stationary point, (c) other, by `scripts/classify.py`'s fixed rule. "
             "Deep levels = depths 16–20; flagged = Levin parent with residual > 10 floor; attributed = no tile, pinned residual ≤ 10 floor, "
             "and the residual's change within 10 % (+ floor) of the offset terms. Labels of non-attributed levels are the post-hoc reading.\n")
    T.append("| # | rtol | block | unit, call | rule class | flagged / attributed (d16–20) | non-attributed deep levels | (b) test: min\\|θ′\\| / max_span\\|θ′\\| on the d16 region (θ′ there) | θ′ sign change in span | reading: tiling holds the deep levels |")
    T.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in calls:
        s = c["stationary"]
        na = "; ".join(f"d{d}: {l_}" for d, l_ in c["deep_labels"].items() if l_ != "attributed") or "—"
        sc = "; ".join(f"x={z['x']:.6g}" + (f" (z={z['z']:.6g})" if z["z"] is not None else "") + f", θ″={z['theta_second']:.3g}" for z in s["span_sign_changes"]) or "none"
        T.append(f"| {c['sample_no']} | {c['rtol']} | {c['block']} | {c['unit']} #{c['index']} | **{c['class']}**{' (a and b)' if c['is_a'] and c['is_b'] else ''} | "
                 f"{c['deep_flagged']} / {c['deep_attributed']} | {na} | {s['region16_min_over_span_max']:.2g} ({s['region16_theta_prime_at_min']:.3g} at x={s['region16_min_at_x']:.6g}) | {sc} | "
                 f"{'yes' if c['reading_tiling_holds_the_deep_levels'] else 'no'} |")
    T.append("")
    T.append("### Classification totals\n")
    T.append("| rtol | block | calls | (a) | (b) | (c) | reading: tiling holds |")
    T.append("|---|---|---|---|---|---|---|")
    for rt in ("1e-7", "1e-8", "all"):
        for blk in BLOCKS + ["all"]:
            cs = [c for c in calls if (rt == "all" or c["rtol"] == rt) and (blk == "all" or c["block"] == blk)]
            if not cs or (blk != "all" and rt == "all"):
                continue
            k = collections.Counter(c["class"] for c in cs)
            T.append(f"| {rt} | {blk} | {len(cs)} | {k['a']} | {k['b']} | {k['c']} | {sum(c['reading_tiling_holds_the_deep_levels'] for c in cs)} |")
    T.append("")
    rf = [L["resid_over_floor"] for c in calls for L in c["deep_levels"] if L["resid_over_floor"] is not None and L["types"][0] == "Levin"]
    rfp = [L["resid_pinned_over_floor_pinned"] for c in calls for L in c["deep_levels"] if L.get("flagged") and L["resid_pinned_over_floor_pinned"] is not None]
    T.append(f"Sample, Levin parents at depths 16–20 ({len(rf)} levels): residual/floor median {statistics.median(rf):.3g}, max {max(rf):.3g}. "
             f"Flagged levels with the ends pinned ({len(rfp)}): residual/floor median {statistics.median(rfp):.3g}, max {max(rfp):.3g}; "
             f"≤ 10 on {sum(x <= 10 for x in rfp)}.\n")
    T.append("### Counterfactual: the call re-run with the end nodes pinned, against R18's census call\n")
    T.append("| # | rtol | block | census: depth / regions / solves / wall s / conv | pinned: depth / regions / solves / wall s / conv | census value; abserr | pinned value; abserr | \\|Δv\\| / abserr_census | ref? | census \\|v−ref\\|/abserr | pinned \\|v−ref\\|/abserr | ref conv / min abserr |")
    T.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in calls:
        C, P, R = c["census"], c["pinned"], c["reference"]
        T.append(f"| {c['sample_no']} | {c['rtol']} | {c['block']} | {C['max_depth']} / {C['num_regions']} / {c['replay']['num_subregion_solves']} / {C['call_elapsed_s']:.2f} / {C['converged']} | "
                 f"{P['max_depth']} / {P['num_regions']} / {P['num_subregion_solves']} / {P['wall_s']:.2f} / {P['converged']} | `{C['value']}`; `{C['abserr']}` | `{P['value']}`; `{P['abserr']}` | "
                 f"{c['pinned_minus_census_over_census_abserr']:.3g} | {('yes' if R['converged_1_100'] else 'NOT CONVERGED') if R else 'no'} | {R['census_err_over_abserr']:.3g} | {R['pinned_err_over_abserr']:.3g} | {R['conv_over_min_abserr']:.2g} |"
                 if R else
                 f"{c['sample_no']} | {c['rtol']} | {c['block']} | {C['max_depth']} / {C['num_regions']} / {c['replay']['num_subregion_solves']} / {C['call_elapsed_s']:.2f} / {C['converged']} | "
                 f"{P['max_depth']} / {P['num_regions']} / {P['num_subregion_solves']} / {P['wall_s']:.2f} / {P['converged']} | `{C['value']}`; `{C['abserr']}` | `{P['value']}`; `{P['abserr']}` | "
                 f"{c['pinned_minus_census_over_census_abserr']:.3g} | no | — | — | — |")
        if not R:
            T[-1] = "| " + T[-1]
    T.append("")
    T.append("### Depth before and after pinning (the 30 sampled calls)\n")
    hb = collections.Counter(c["census"]["max_depth"] for c in calls)
    ha = collections.Counter(c["pinned"]["max_depth"] for c in calls)
    T.append("| max depth | census calls | pinned calls |")
    T.append("|---|---|---|")
    for d in sorted(set(hb) | set(ha)):
        T.append(f"| {d} | {hb.get(d, 0)} | {ha.get(d, 0)} |")
    T.append("")
    T.append(f"Converged: census {sum(c['census']['converged'] for c in calls)} of 30, pinned {sum(c['pinned']['converged'] for c in calls)} of 30. "
             f"Regions summed: census {sum(c['census']['num_regions'] for c in calls)}, pinned {sum(c['pinned']['num_regions'] for c in calls)}. "
             f"Solves summed: census {sum(c['replay']['num_subregion_solves'] for c in calls)}, pinned {sum(c['pinned']['num_subregion_solves'] for c in calls)}. "
             f"Wall summed: census calls {sum(c['census']['call_elapsed_s'] for c in calls):.1f} s, pinned {sum(c['pinned']['wall_s'] for c in calls):.1f} s.\n")
    refd = [c for c in calls if c["reference"] and c["reference"]["converged_1_100"]]
    T.append(f"References attempted: {[c['sample_no'] for c in calls if c['reference']]}; converged to 1/100 of the smaller abserr: {[c['sample_no'] for c in refd]}.\n")
    if refd:
        T.append(f"Honesty where a reference exists ({len(refd)} calls: {', '.join(str(c['sample_no']) for c in refd)}): pinned |v−ref|/abserr max "
                 f"{max(c['reference']['pinned_err_over_abserr'] for c in refd):.3g}, median {statistics.median(c['reference']['pinned_err_over_abserr'] for c in refd):.3g}; "
                 f"census max {max(c['reference']['census_err_over_abserr'] for c in refd):.3g}, median {statistics.median(c['reference']['census_err_over_abserr'] for c in refd):.3g}. "
                 f"Pinned within its abserr on {sum(c['reference']['pinned_err_over_abserr'] <= 1 for c in refd)}; census within on {sum(c['reference']['census_err_over_abserr'] <= 1 for c in refd)}.\n")
    (STATE / "tables.md").write_text("\n".join(T) + "\n")
    print("\n".join(T))


if __name__ == "__main__":
    main()
