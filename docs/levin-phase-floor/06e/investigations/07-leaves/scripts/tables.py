"""Tables for the 07-leaves measurement, from the JSON records in out/. Writes out/tables.md."""

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lc  # noqa: E402

OUT = lc.OUT


def g(x, fmt=".2e"):
    if x is None:
        return "--"
    return format(x, fmt)


def load_q12(case_key):
    runs = {}
    hdrs = []
    for p in sorted(glob.glob(str(OUT / f"q12_{case_key}*.json"))):
        if "thetamod" in p:
            continue
        d = json.load(open(p))
        hdrs.append((Path(p).name, d["header"]))
        runs.update(d["runs"])
    return runs, hdrs


def rung_key(k):
    phi, d = k.split("pi_d")
    return (float(phi), int(d))


def summary_table(runs):
    lines = [
        "| Phi_w | d_max | abserr | true error | request | window declared | outside declared | outside true (sum) | outside declared by component |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k in sorted(runs, key=rung_key):
        r = runs[k]
        comps = ", ".join(
            f"{c} {v:.1e}" for c, v in sorted(r["outside_by_component"].items(), key=lambda t: -t[1])
        )
        phi, d = rung_key(k)
        lines.append(
            f"| {phi:g}pi | {d} | {g(r['abserr'])} | {g(r['true_error'], '+.1e')} | {g(r['request'], '.2e')} | "
            f"{g(r['window_sum_total_err'])} | {g(r['outside_sum_total_err'])} | "
            f"{g(r['outside_sum_true_error'], '+.1e')} | {comps} |"
        )
    return lines


def carriers_table(runs, min_share=0.01):
    """For each run, the outside leaves grouped by (component, type), groups with >= min_share of
    the outside declared error."""
    lines = [
        "| Phi_w | d_max | component | type | leaves | depth | dtheta_near (rad) | abs theta' near | declared (sum) | abs true (sum) | max abs true | true (signed sum) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k in sorted(runs, key=rung_key):
        r = runs[k]
        tot = r["outside_sum_total_err"]
        groups = defaultdict(list)
        for leaf in r["outside"]:
            groups[(leaf["component"], leaf["type"])].append(leaf)
        phi, d = rung_key(k)
        for (comp, typ), L in sorted(groups.items(), key=lambda t: -sum(x["total_err"] for x in t[1])):
            s = sum(x["total_err"] for x in L)
            if s < min_share * tot:
                continue
            dep = sorted({x["depth"] for x in L})
            dth = [x["dtheta_near"] for x in L]
            thp = [abs(x["theta_prime_near"]) for x in L]
            lines.append(
                f"| {phi:g}pi | {d} | {comp} | {typ} | {len(L)} | {dep[0]}-{dep[-1]} | "
                f"{min(dth):.3g}-{max(dth):.3g} | {min(thp):.2g}-{max(thp):.2g} | {s:.2e} | "
                f"{sum(abs(x['true_error']) for x in L):.2e} | {max(abs(x['true_error']) for x in L):.2e} | "
                f"{sum(x['true_error'] for x in L):+.1e} |"
            )
    return lines


def adjacent_table(runs):
    """The leaf adjacent to the window on each side."""
    lines = [
        "| Phi_w | d_max | side | span | depth | type | component | declared | true |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k in sorted(runs, key=rung_key):
        r = runs[k]
        lo, hi = r["stationary_points"][0]["window"]
        phi, d = rung_key(k)
        for leaf in r["outside"]:
            if leaf["b"] == lo or leaf["a"] == hi:
                lines.append(
                    f"| {phi:g}pi | {d} | {leaf['side']} | [{leaf['a']:.6f}, {leaf['b']:.6f}] | {leaf['depth']} | "
                    f"{leaf['type']} | {leaf['component']} | {leaf['total_err']:.2e} | {leaf['true_error']:+.1e} |"
                )
    return lines


def q3_tables(case_key):
    p = OUT / f"q3_{case_key}.json"
    if not p.exists():
        return []
    d = json.load(open(p))
    lines = [
        f"### Decomposition, {case_key} (`{p.name}`)",
        "",
        "Levin leaves: the step-(4) residual V_P - V_L - V_R split into resolution / node / endpoint-theta / other "
        "(each the parent's component minus the halves'); 'ends' lists which of P, L, R has a computed end node "
        "off its span end (offset in units of 1e-16). CC (direct) leaves: the nested pair split into CC_31's and "
        "CC_16's errors, each resolution + node + argument + other; and the summed nested pair, floor and true error "
        "after 1, 2, 3 further bisections.",
        "",
        "| config | span | dtheta_near | theta' | kind | declared basis | residual / nested | floor | split | theta_mod_2pi: residual | true error of P (V_P - exact) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cfg, c in d["configs"].items():
        for r in c["chosen"]:
            span = f"[{r['span'][0]:.7f}, {r['span'][1]:.7f}]"
            ret = r["returned"]
            if "parts" in r:
                sp = r["residual_split"]
                ends = []
                for n in ("P", "L", "R"):
                    part = r["parts"][n]
                    if part["kind"] == "Levin" and not part["grid_ends_exact"]:
                        offs = "/".join(f"{o * 1e16:+.2f}" for o in part["end_node_offsets"])
                        ends.append(f"{n}({offs})")
                    elif part["kind"] == "CC":
                        ends.append(f"{n}=CC")
                split = (
                    f"res {sp['resolution']:+.1e}, node {sp['node']:+.1e}, end-theta {sp['endpoint_theta']:+.1e}, "
                    f"other {sp['other']:+.1e}; ends {' '.join(ends) or 'all exact'}"
                )
                lines.append(
                    f"| {cfg} | {span} | {r['dtheta_near']:.3g} | {r['theta_prime_near']:.2g} | Levin d{ret['depth']} | "
                    f"{ret['type']} | {r['residual']:.2e} | {r['floor']:.2e} | {split} | "
                    f"{r['theta_mod_2pi']['residual']:.2e} | {r['parts']['P']['error']:+.1e} |"
                )
            else:
                p_ = r["parent"]
                fi, co = p_["fine"], p_["coarse"]
                refine = "; ".join(
                    f"+{q['extra_depth']}: nested {q['sum_nested_pair']:.1e}, floor {q['sum_floor']:.1e}, true {q['sum_true_error']:+.1e}"
                    for q in r["refine"]
                )
                split = (
                    f"CC31 err {fi['error']:+.1e} (res {fi['resolution']:+.0e}, node {fi['node']:+.1e}, arg {fi['argument']:+.1e}); "
                    f"CC16 err {co['error']:+.1e} (res {co['resolution']:+.1e}); refine {refine}"
                )
                lines.append(
                    f"| {cfg} | {span} | {r['dtheta_near']:.3g} | {r['theta_prime_near']:.2g} | CC d{ret['depth']} | "
                    f"{ret['type']} | {p_['nested_pair']:.2e} | {p_['floor']:.2e} | {split} | -- | {fi['error']:+.1e} |"
                )
    lines.append("")
    # history: residual vs floor by depth
    lines += [
        "History of every Levin leaf made outside the window (all configs pooled): median and max of the "
        "step-(4) residual, the eq. (151) floor, and the true errors of V_P and V_L + V_R, by depth; "
        "and the share of leaves whose residual exceeds 10x the floor.",
        "",
        "| config | depth | leaves | median residual | max residual | median floor | median abs(V_P err) | median abs(V_L+V_R err) | residual > 10 floor |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    import statistics as st

    for cfg, c in d["configs"].items():
        by = defaultdict(list)
        for h in c["history"]:
            by[h["depth"]].append(h)
        for dep in sorted(by):
            H = by[dep]
            lines.append(
                f"| {cfg} | {dep} | {len(H)} | {st.median(h['residual'] for h in H):.1e} | "
                f"{max(h['residual'] for h in H):.1e} | {st.median(h['floor'] for h in H):.1e} | "
                f"{st.median(abs(h['P_error']) for h in H):.1e} | {st.median(abs(h['halves_error']) for h in H):.1e} | "
                f"{sum(1 for h in H if h['residual'] > 10 * h['floor'])}/{len(H)} |"
            )
    lines.append("")
    return lines


def q4_tables(case_key):
    lines = []
    files = sorted(glob.glob(str(OUT / f"q4_{case_key}_*pi.json")), key=lambda p: float(Path(p).name.split("_")[-1][:-7]))
    if not files:
        return lines
    lines += [
        f"### Inside the window, {case_key}",
        "",
        "Whole window: the sums over its leaves. Then sampled depth-0 pieces, summed over the leaves inside each at the two depths; "
        "then three single leaves with the fine rule's true error split in mpmath.",
        "",
        "| Phi_w | d_max | leaves | nested pair | C1 | C2 | C3 | declared | true (signed sum) | abs true (sum) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    piece_lines = [
        "",
        "| Phi_w | d_max | piece | dtheta at ends | leaves | nested pair | C1 | C2 | C3 | declared | true (signed sum) | abs true (sum) | max abs true per leaf |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    split_lines = [
        "",
        "| Phi_w | d_max | leaf | dtheta mid | nested pair | C3 | true | resolution | node | argument | other |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in files:
        d = json.load(open(p))
        k = d["phi_w_over_pi"]
        for dk, r in d["runs"].items():
            a = r["aggregate"]
            lines.append(
                f"| {k:g}pi | {dk[1:]} | {r['window_leaves']} | {a['nested']:.2e} | {a['C1']:.2e} | {a['C2']:.2e} | "
                f"{a['C3']:.2e} | {a['declared']:.2e} | {a['true_signed']:+.1e} | {a['true_abs']:.2e} |"
            )
            for q in r["pieces"]:
                piece_lines.append(
                    f"| {k:g}pi | {dk[1:]} | {q['piece']} | {q['dtheta'][0]:.3g} / {q['dtheta'][1]:.3g} | {q['leaves']} | "
                    f"{q['nested_sum']:.1e} | {q['C1_sum']:.1e} | {q['C2_sum']:.1e} | {q['C3_sum']:.2e} | {q['declared_sum']:.2e} | "
                    f"{q['true_sum']:+.1e} | {q['true_abs_sum']:.1e} | {q['max_abs_true']:.1e} |"
                )
            for s in r["split"]:
                split_lines.append(
                    f"| {k:g}pi | {dk[1:]} | [{s['span'][0]:.8f}, {s['span'][1]:.8f}] | {s['dtheta_mid']:.3g} | "
                    f"{s['nested']:.1e} | {s['C3']:.1e} | {s['true']:+.1e} | {s['resolution']:+.0e} | {s['node']:+.1e} | "
                    f"{s['argument']:+.1e} | {s['other']:+.1e} |"
                )
    return lines + piece_lines + split_lines + [""]


def lg_tables():
    p = OUT / "lg_seed0_call15.json"
    if not p.exists():
        return []
    d = json.load(open(p))
    rec = d["stuck_call"]
    lines = [
        "## LiouvilleGreen seed 0, test_JJJ, Levin call 15",
        "",
        f"x_span {rec['x_span']}, kwargs {rec['kwargs']}.",
        "",
        "| d_max | wall s | abserr | request | window leaves | window declared | window by component | outside leaves | outside declared | outside by component |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for dk, r in rec["runs"].items():
        wc = ", ".join(f"{c} {v:.1e}" for c, v in r["window_by_component"].items())
        oc = ", ".join(f"{c} {v:.1e}" for c, v in r["outside_by_component"].items())
        lines.append(
            f"| {dk[1:]} | {r['wall_s']:.0f} | {r['abserr']:.2e} | {r['request']:.2e} | {r['window_leaves']} | "
            f"{r['window_sum_total_err']:.2e} | {wc} | {r['outside_leaves']} | {r['outside_sum_total_err']:.2e} | {oc} |"
        )
    lines += ["", "Window floor term by term (sums over the window leaves):", "",
              "| d_max | C1 | C2 | C3 | D (declared theta_abserr) | max rel. diff. of C1+C2+C3+D from phase_err |",
              "|---|---|---|---|---|---|"]
    for dk, r in rec["runs"].items():
        if "window_floor_terms" in r:
            t = r["window_floor_terms"]
            lines.append(f"| {dk[1:]} | {t['C1']:.2e} | {t['C2']:.2e} | {t['C3']:.2e} | {t['D']:.2e} | {t['recon_max_rel_diff']:.1e} |")
    lines += ["", "Outside leaves carrying the declared error (top 5 per depth):", "",
              "| d_max | span | depth | type | component | dtheta_near | theta' near | abserr | floor | declared |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for dk, r in rec["runs"].items():
        for o in sorted(r["outside"], key=lambda o: -o["total_err"])[:5]:
            lines.append(
                f"| {dk[1:]} | [{o['a']:.5f}, {o['b']:.5f}] | {o['depth']} | {o['type']} | {o['component']} | "
                f"{g(o['dtheta_near'], '.3g')} | {o['theta_prime_near']:.3g} | {o['abserr']:.1e} | {o['floor']:.1e} | {o['total_err']:.1e} |"
            )
    lines += ["", "Gauss-Legendre references on sampled leaves (fsum; the integrand from the basis's own eval_basis):", "",
              "| d_max | span | type | depth | declared | nested / residual | floor | GL 64-32 | GL 128-64 | GL 256-128 | value - GL256 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for dk, r in rec["runs"].items():
        for q in r.get("gl_refs", []):
            gd = q["gl_diffs"]
            lines.append(
                f"| {dk[1:]} | [{q['span'][0]:.6f}, {q['span'][1]:.6f}] | {q['type']} | {q['depth']} | {q['declared']:.1e} | "
                f"{q['nested_or_residual']:.1e} | {q['floor']:.1e} | {gd['64-32']:+.1e} | {gd['128-64']:+.1e} | {gd['256-128']:+.1e} | "
                f"{q['value_minus_gl256']:+.1e} |"
            )
    return lines + [""]


def main():
    lines = ["# 07's driver: where the declared error sits, leaf by leaf", ""]
    hdr = lc.header(__file__)
    lines.append(
        f"tables.py run at {hdr['utc']} on {hdr['host']}; base {hdr['base_sha']}, patch sha256 {hdr['patch_sha256']}, "
        f"levin_quadrature.py sha256 {hdr['levin_quadrature_sha256']}; numpy {hdr['numpy']}, scipy {hdr['scipy']}, mpmath {hdr['mpmath']}."
    )
    lines.append("")
    lines.append("Script sha256: " + ", ".join(f"{k} `{v}`" for k, v in hdr["scripts_sha256"].items()))
    lines.append("")
    lines.append("Every JSON in out/ carries its own header (script, its sha256, every script's, base, patch, levin_quadrature.py, host, UTC, versions).")
    lines.append("")
    for case_key in ("c1e6w0", "c1e8w0", "c1e8w1"):
        runs, hdrs = load_q12(case_key)
        if not runs:
            continue
        lines += [f"## {case_key}: {lc.CASES[case_key]}", ""]
        lines += ["Inputs: " + ", ".join(f"`{n}` ({h['utc']})" for n, h in hdrs), ""]
        lines += ["### Per run", ""] + summary_table(runs) + [""]
        lines += ["### Outside leaves carrying >= 1% of the outside declared error, grouped by component and type", ""]
        lines += carriers_table(runs) + [""]
        lines += ["### The leaves adjacent to the window", ""] + adjacent_table(runs) + [""]
        tm = OUT / f"q12_{case_key}_thetamod.json"
        if tm.exists():
            T = json.load(open(tm))["runs"]
            lines += [
                "### The same runs with an exact theta_mod_2pi supplied (`" + tm.name + "`)",
                "",
                "| Phi_w | d_max | abserr raw / mod | true raw / mod | window declared raw / mod | outside declared raw / mod |",
                "|---|---|---|---|---|---|",
            ]
            for k in sorted(T, key=rung_key):
                r, t = runs[k], T[k]
                phi, d = rung_key(k)
                lines.append(
                    f"| {phi:g}pi | {d} | {r['abserr']:.3e} / {t['abserr']:.3e} | {r['true_error']:+.2e} / {t['true_error']:+.2e} | "
                    f"{r['window_sum_total_err']:.3e} / {t['window_sum_total_err']:.3e} | {r['outside_sum_total_err']:.3e} / {t['outside_sum_total_err']:.3e} |"
                )
            lines.append("")
        lines += q3_tables(case_key)
        lines += q4_tables(case_key)
    en = OUT / "endnodes.json"
    if en.exists():
        E = json.load(open(en))["files"]
        lines += ["## Levin residual above the floor against end-node tiling (`endnodes.json`)", "",
                  "Every Levin leaf made outside the window whose own value is resolved (abs(V_P err) < 1e-15). 'Tile': the computed end nodes of the parent and its halves coincide (left half's bottom = parent's, right half's top = parent's, left's top = right's bottom).", "",
                  "| file | class | leaves | max residual | parent's end nodes off a, b |", "|---|---|---|---|---|"]
        for f, T in E.items():
            for k, v in sorted(T.items()):
                lines.append(f"| {f} | {k} | {v['n']} | {v['max_residual']:.1e} | {v['parent_ends_off']} |")
        lines.append("")
    lines += lg_tables()
    (OUT / "tables.md").write_text("\n".join(lines))
    print("wrote", OUT / "tables.md")


if __name__ == "__main__":
    main()
