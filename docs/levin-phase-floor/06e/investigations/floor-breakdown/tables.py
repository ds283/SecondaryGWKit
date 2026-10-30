"""Tables from driver_run.json, node_chain.json and scaling.json -> tables.md."""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

e = lambda x: "--" if x is None else f"{x:.2e}"  # noqa: E731


def main():
    dr = json.loads((HERE / "driver_run.json").read_text())
    nc = json.loads((HERE / "node_chain.json").read_text())
    sc = json.loads((HERE / "scaling.json").read_text())
    import numpy as np

    for k, srun in sc["runs"].items():
        name, dpart = k.rsplit(" d=", 1)
        fn = HERE / f"scaling_leaves_{name.replace(' ', '_').replace('=', '')}_d{dpart}.json"
        rows = json.loads(fn.read_text())
        lv = [r["floor_code"] for r in rows if r["kind"] == "Levin"]
        cc = [r["floor_code"] for r in rows if r["kind"] == "CC"]
        srun["median_floor_levin"] = float(np.median(lv)) if lv else None
        srun["median_floor_cc"] = float(np.median(cc)) if cc else None
    hdr = C.header(__file__)
    L = [
        "# 06d's round-off floor on 07's (R) cases: term by term",
        "",
        f"tables.py run: worktree HEAD `{hdr['git_head']}`, dirty {hdr['tree_dirty']}, "
        f"levin_quadrature.py sha256 `{hdr['levin_quadrature_sha256']}`, host {hdr['host']}, "
        f"{hdr['utc']}, numpy {hdr['numpy']}, scipy {hdr['scipy']}, mpmath {hdr['mpmath']}.",
        "",
        "Inputs: driver_run.json (header: HEAD "
        f"`{dr['header']['git_head']}`, dirty {dr['header']['tree_dirty']}, "
        f"{dr['header']['utc']}), node_chain.json ({nc['header']['utc']}), scaling.json.",
        "",
        "Script sha256: "
        + ", ".join(f"{k} `{v}`" for k, v in hdr["scripts_sha256"].items()),
        "",
        "07's scripts (read, not modified) sha256: "
        + ", ".join(f"{k} `{v}`" for k, v in hdr["scripts_07_sha256"].items()),
        "",
        "## 1-2. The driver's own run at 06d: the floor's terms summed over the leaves, against the round-off",
        "",
        "| | c=1e6 w=0 | c=1e8 w=0 | c=1e8 w=1 |",
        "|---|---|---|---|",
    ]
    names = [c["name"] for c in C.CASES]
    S = [dr["cases"][n]["summary"] for n in names]
    N = [nc["cases"][n]["totals"] for n in names]

    def row(label, vals):
        L.append(f"| {label} | " + " | ".join(vals) + " |")

    row("leaves (Levin / CC), all floor-limited", [f"{s['num_regions'] - s['num_cc']} / {s['num_cc']}, {s['num_floor_limited']}" for s in S])
    row("request max(atol, rtol abs(V))", [e(s["request"]) for s in S])
    row("**declared abserr**", [e(s["abserr"]) for s in S])
    row("**true error** (V - closed form)", [e(abs(s["true_error"])) for s in S])
    row("declared / true", [f"{s['abserr'] / abs(s['true_error']):.0f}" for s in S])
    for t in ("L1_eps_f_h2", "L2_G1_over_G0", "L3_maxG1k2_over_G0", "C1_n4_uS", "C2_argument", "C3_node"):
        row(f"floor term {t}, summed", [e(s["sum_terms"][t]) for s in S])
    row("C3_node share of the floor", [f"{s['sum_terms']['C3_node'] / s['sum_floor']:.4f}" for s in S])
    row("actual round-off, sum abs per leaf", [e(s["sum_abs_roundoff"]) for s in S])
    row("actual round-off, abs of sum", [e(s["abs_sum_roundoff"]) for s in S])
    row("  CC node rounding: sum abs / abs sum", [f"{e(n['sum_abs_R'])} / {e(n['abs_sum_R'])}" for n in N])
    row("  Levin endpoint-phase rounding: sum abs / abs sum", [f"{e(s['levin_endpoint_part_sum_abs'])} / {e(s['levin_endpoint_part_abs_sum'])}" for s in S])
    row("  the rest (solve, sums, f, weights): sum abs / abs sum", [f"{e(s['rest_part_sum_abs'])} / {e(s['rest_part_abs_sum'])}" for s in S])
    row("truncation (mp scheme - exact): sum abs / abs sum", [f"{e(s['sum_abs_truncation'])} / {e(s['abs_sum_truncation'])}" for s in S])
    row("C3 / sum abs(node part)", [f"{n['A_C3'] / n['sum_abs_R']:.0f}" for n in N])
    row("C3 / abs(sum node part)", [f"{n['A_C3'] / n['abs_sum_R']:.0f}" for n in N])
    row("leaves with abs(round-off) > own floor", [str(s["leaves_roundoff_above_floor"]) for s in S])
    L += [
        "",
        "### The CC node term's pessimism, factor by factor (sums over the driver's CC leaves)",
        "",
        "A = C3 as coded; B = A with the bound delta = 5u(max(|a|,|b|) + h) replaced by each node's actual "
        "|x^_k - x_k|; C = B with max theta' replaced by theta'(x_k); D = sum |w||dx||g'| with the true g'; "
        "E = sum w dx g' (signed, first order); R = measured node part.",
        "",
        "| | c=1e6 w=0 | c=1e8 w=0 | c=1e8 w=1 |",
        "|---|---|---|---|",
    ]
    row("A / B (bound on dx vs actual dx)", [f"{n['A_C3'] / n['B']:.1f}" for n in N])
    row("B / C (max theta' vs theta' at the node)", [f"{n['B'] / n['C']:.2f}" for n in N])
    row("C / D (theta' sum abs f + sum abs f' vs abs g')", [f"{n['C'] / n['D']:.2f}" for n in N])
    row("D / sum abs E (cancellation inside a leaf)", [f"{n['D'] / n['sum_abs_E']:.1f}" for n in N])
    row("sum abs E / abs sum E (cancellation across leaves)", [f"{n['sum_abs_E'] / n['abs_sum_E']:.1f}" for n in N])
    row("E / R (first-order model vs measured; summed)", [f"{n['abs_sum_E'] / n['abs_sum_R']:.6f}" for n in N])
    meds = [nc["cases"][n]["median_factors"] for n in names]
    row("max abs(dx) over all CC nodes, in u", [f"{m['max_dx_u']:.3f}" for m in meds])
    row("the bound delta, in u", [f"{m['delta_u']:.3f}" for m in meds])
    L += ["", "## 3. Scaling with h: [0, 1] cut into 2^d equal leaves, each solved by 06d's subregion solve", ""]
    for n in names:
        L += [
            f"### {n}",
            "",
            "| d | h | Levin / CC | summed floor | L1 sum | L2 sum | L3 sum | C1 sum | C2 sum | C3 sum | med floor Levin | med floor CC | med L3 | med C3 | med abs(ro) Levin (rest) | med abs(ro) CC (node) | sum abs(ro) | abs(sum ro) | leaves ro>floor (Levin rest>floor) | worst Levin rest/floor | node pattern(s) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for k, s in sc["runs"].items():
            if not k.startswith(n + " "):
                continue
            t = s["terms"]
            r = s["roundoff"]
            lev_meds = None
            L.append(
                f"| {s['depth']} | {s['h']:.1e} | {s['levin']} / {s['cc']} | {e(s['sum_floor'])} | "
                + " | ".join(e(t[x]["sum"]) for x in ("L1_eps_f_h2", "L2_G1_over_G0", "L3_maxG1k2_over_G0", "C1_n4_uS", "C2_argument", "C3_node"))
                + f" | {e(s.get('median_floor_levin'))} | {e(s.get('median_floor_cc'))} | {e(t['L3_maxG1k2_over_G0']['median'])} | {e(t['C3_node']['median'])}"
                + f" | {e(r['median_abs_levin'])} ({e(r['median_abs_levin_rest'])}) | {e(r['median_abs_cc'])} ({e(r['median_abs_cc_node'])})"
                + f" | {e(r['sum_abs_all'])} | {e(r['abs_sum_all'])} | {s['leaves_roundoff_above_floor']} ({s['levin_leaves_rest_above_floor']})"
                + f" | {s['worst_rest_over_floor_levin']:.2g} | {s['cc_node_disp_patterns']} |"
            )
        L.append("")
    L += ["### Local power of h of each summed term, sum ~ h^p, between consecutive depths", ""]
    keys = ("L1_eps_f_h2", "L2_G1_over_G0", "L3_maxG1k2_over_G0", "C3_node")
    L += ["| case | d -> d' | " + " | ".join(keys) + " | summed floor | abs(sum ro) |", "|---|---|" + "---|" * (len(keys) + 2)]
    for n in names:
        runs = sorted([s_ for k, s_ in sc["runs"].items() if k.startswith(n + " ")], key=lambda s_: s_["depth"])
        for r0, r1 in zip(runs[:-1], runs[1:]):
            dl = math.log(r1["h"] / r0["h"])

            def p(v0, v1):
                return f"{math.log(v1 / v0) / dl:+.2f}" if v0 and v1 else "--"

            L.append(
                f"| {n} | {r0['depth']} -> {r1['depth']} | "
                + " | ".join(p(r0["terms"][k_]["sum"], r1["terms"][k_]["sum"]) for k_ in keys)
                + f" | {p(r0['sum_floor'], r1['sum_floor'])} | {p(r0['roundoff']['abs_sum_all'], r1['roundoff']['abs_sum_all'])} |"
            )
    L.append("")
    (HERE / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
