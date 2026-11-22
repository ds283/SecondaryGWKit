"""levin-phase-floor prompt 07d Part A, section 2.2 (D-N): the integrand's noise on the leaf sample, by class, with its
source and structure. New in 07d. Reads the fill (out/leaves/fill_state.json, the selection, leafcc_07d.py's outputs);
the sample is every leaf of the Clenshaw-Curtis strata in the fill's prefix (up to and including each stratum's 40th
scored leaf), measured before any candidate is formed.

Per leaf (leafcc_07d.py ``noise``): sigma/mean|g| from the least-squares residual at d_lo and d_hi; the plateau of the
fine (2N-1) and CC_{8N-7} interpolants' coefficients (standardChop); per component sigma (each f_i against its rms, the
raw phase against u mean|theta_raw|, theta_mod_2pi unwrapped); the phase-predicted scatter of g, sigma_theta_mod x
rms|f| (|f| = sqrt(sum f_i^2)), and the amplitude-predicted one, sqrt(sum_i sigma_fi^2), each against sigma_g; the lag-1
autocorrelation of g's residual; the spread (sample standard deviation) of CC_{2N-1} over the shifted spans against
s_CC = sigma_hat sqrt(sum w^2); the knots inside the leaf; the declared theta_abserr x rms|f| where supplied.
**Classes:** group label (``group <signs>``) or analytic block (``analytic <block>``), k band, route (the stratum's
family: gate, chase, window), knots inside the leaf (0, 1, 2-4, 5+, unknown).
Writes out/tables/d_n.json and out/tables/d_n.md. Usage: PYTHONPATH=<tree>:<scripts> python -B d_n_table_07d.py
"""

import json
import math
import pickle
import re
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from prov import header, sha256  # noqa: E402

fx = float.fromhex
FILL = 40
U = 2.0 ** -53


def prefix_leaves(sel, state, families):
    """(stratum, rank, leaf record, verdict) for every leaf of each stratum's fill prefix."""
    cache = {}
    out = []
    for st, S in state["strata"].items():
        if st.split("|")[0].rstrip("1234") not in families and st.split("|")[0] not in families:
            continue
        n_sc = 0
        for res in S["results"]:
            if n_sc >= FILL:
                break
            if res["file"] is None:
                out.append((st, res["rank"], None, res["verdict"]))
                continue
            if res["file"] not in cache:
                cache.clear()
                cache[res["file"]] = {l["id"]: l for l in json.loads((ROOT / "out" / "leaves" / res["file"]).read_text())["leaves"]}
            l = cache[res["file"]].get(f"{st}#{res['rank']}")
            out.append((st, res["rank"], l, res["verdict"]))
            n_sc += res["verdict"] in ("bounded", "understates")
    return out


def q(xs):
    xs = [x for x in xs if x is not None and x == x and math.isfinite(x)]
    if not xs:
        return None
    a = np.array(xs)
    return {"n": len(xs), "p10": float(np.percentile(a, 10)), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "min": float(a.min()), "max": float(a.max())}


def knots_bin(n):
    if n is None:
        return "unknown"
    return "0" if n == 0 else ("1" if n == 1 else ("2-4" if n <= 4 else "5+"))


def main():
    sel = json.loads((ROOT / "state" / "selection_leaves_07d.json").read_text())["selection"]
    state = json.loads((ROOT / "out" / "leaves" / "fill_state.json").read_text())
    idx = {r["sid"]: r for r in pickle.loads((ROOT / "state" / "census_index_07d.pkl").read_bytes())["rows"]}
    man = {}
    for p in sorted((ROOT / "state").glob("calls_07d*.json")):
        man.update(json.loads(p.read_text())["calls"])
    rows = []
    for st, rk, l, v in prefix_leaves(sel, state, {"gate_q", "chase", "window", "guarded"}):
        if l is None or "error" in l:
            continue
        leaf_sel = sel[st]["order"][rk]
        r = idx[man[leaf_sel["sid"]]["census"]]
        lab = r["notify_label"]
        m = re.search(r"group (\S+)", lab)
        cls_label = f"group {m.group(1)}" if m else re.sub(r".*(analytic \S+).*", r"\1", lab)
        nz = l["noise"]
        comp = nz["components"]
        sg = comp["g"]["sigma"]
        fmag = comp["g"]["rms_fmag"]
        s_th = comp.get("theta_mod_2pi", {}).get("sigma")
        sig_f = math.sqrt(sum(comp[k]["sigma"] ** 2 for k in comp if re.fullmatch(r"f\d+", k)))
        she = [fx(x) for x in nz["shift_errors"] if x is not None]
        dec = nz.get("declared")
        rows.append({
            "stratum": st, "family": "gate" if st.startswith("gate") else st.split("|")[0],
            "kind": r["kind"], "rtol": r["rtol"], "label": cls_label, "k_band": r["k_band"], "knots": l["n_knots"],
            "knots_bin": knots_bin(l["n_knots"]), "V_over_6pi": l["V_over_6pi"],
            "sigma_rel_hi": nz["sigma_over_mean_abs_g"], "sigma_rel_lo": nz["sigma_lo"] / max(comp["g"]["mean_abs"], 1e-300),
            "lo_over_hi": nz["sigma_lo"] / sg if sg > 0 else None,
            "plateau_fine_rel": None if nz["plateau_fine"]["sigma"] is None else nz["plateau_fine"]["sigma"] / max(comp["g"]["mean_abs"], 1e-300),
            "plateau_fine_over_ls": None if nz["plateau_fine"]["sigma"] is None or sg == 0 else nz["plateau_fine"]["sigma"] / sg,
            "plateau_185_over_ls": None if nz["plateau_185"]["sigma"] is None or sg == 0 else nz["plateau_185"]["sigma"] / sg,
            "chopped_fine": nz["plateau_fine"]["sigma"] is not None,
            "theta_raw_sigma_over_u_theta": comp["theta_raw"]["sigma"] / comp["theta_raw"]["u_mean_abs"] if comp["theta_raw"]["u_mean_abs"] > 0 else None,
            "theta_mod_sigma": s_th, "theta_raw_mean_abs": comp["theta_raw"]["mean_abs"],
            "phase_pred_over_sigma_g": None if s_th is None or sg == 0 else s_th * fmag / sg,
            "amp_pred_over_sigma_g": sig_f / sg if sg > 0 else None,
            "f_sigma_rel": [comp[k]["sigma"] / comp[k]["rms"] if comp[k]["rms"] > 0 else None for k in sorted(comp) if re.fullmatch(r"f\d+", k)],
            "acf1_g": comp["g"]["acf"][0], "acf1_theta_mod": (comp.get("theta_mod_2pi") or {}).get("acf", [None])[0],
            "shift_spread_over_s_cc": (statistics.stdev(she) / nz["s_cc"]) if nz["s_cc"] > 0 and len(she) > 1 else None,
            "declared_pred_over_sigma_g": None if not dec or sg == 0 else max(dec["theta_abserr_lo"], dec["theta_abserr_hi"]) * fmag / sg,
            "declared": dec,
        })
    def group_by(keyf):
        out = {}
        for x in rows:
            out.setdefault(keyf(x), []).append(x)
        return out

    metrics = ["sigma_rel_hi", "lo_over_hi", "plateau_fine_over_ls", "plateau_185_over_ls", "theta_raw_sigma_over_u_theta",
               "phase_pred_over_sigma_g", "amp_pred_over_sigma_g", "acf1_g", "acf1_theta_mod", "shift_spread_over_s_cc",
               "declared_pred_over_sigma_g"]
    tabs = {}
    for name, kf in (("all", lambda x: "all"), ("kind", lambda x: x["kind"]), ("label", lambda x: x["label"]),
                     ("k_band", lambda x: f"{x['kind']}|k{x['k_band']}"), ("route", lambda x: f"{x['kind']}|{x['family']}"),
                     ("knots", lambda x: f"{x['kind']}|{x['knots_bin']}")):
        tabs[name] = {k: {"n": len(v), "chopped_fine": sum(1 for x in v if x["chopped_fine"]),
                          **{m: q([x[m] for x in v]) for m in metrics}} for k, v in sorted(group_by(kf).items())}
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07d Part A: D-N, the integrand's noise",
                 [ROOT / "out" / "leaves" / "fill_state.json", ROOT / "state" / "selection_leaves_07d.json"])
    od = ROOT / "out" / "tables"
    od.mkdir(parents=True, exist_ok=True)
    (od / "d_n.json").write_text(json.dumps({"header": hdr, "n_leaves": len(rows), "tables": tabs, "leaves": rows}, indent=1))

    def f(qq, k="median"):
        return "-" if qq is None else f"{qq[k]:.3g}"

    lines = [l.replace("# ", "", 1) for l in hdr] + ["", f"Leaves: {len(rows)} (every Clenshaw-Curtis stratum's fill prefix).", ""]
    for name, t in tabs.items():
        lines += [f"## by {name}", "",
                  "| class | n | chopped (fine) | sigma/mean|g| median (p10-p90) | sigma(d_lo)/sigma(d_hi) | plateau47/LS | plateau185/LS | "
                  "theta_raw sigma / u|theta| | phase-pred/sigma_g | amp-pred/sigma_g | acf1 g | acf1 theta | shift spread / s_CC | declared/sigma_g |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, v in t.items():
            s = v["sigma_rel_hi"]
            lines.append(f"| {k} | {v['n']} | {v['chopped_fine']} | " + ("-" if s is None else f"{s['median']:.3g} ({s['p10']:.2g}-{s['p90']:.2g})")
                         + " | " + " | ".join(f(v[m]) for m in metrics[1:]) + " |")
        lines.append("")
    (od / "d_n.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))
    print("wrote", od / "d_n.json", sha256(od / "d_n.json"))


if __name__ == "__main__":
    main()
