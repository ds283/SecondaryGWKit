"""levin-phase-floor prompt 06i: the white jitter of a cosmological group's theta_mod_2pi, by Route B
sample. A diagnostic for Route B's t(x) construction; no integral of a sample is formed. New in 06i.

For each Route B sample: on 40 segments spread over the span (samples.route_b_edges), the
least-squares smooth phase's residual rms sigma (route_b.FIT[1]), against |theta'| there; and the
noise floor that jitter of that size puts under a quadrature that samples the jittered integrand
with node spacing Delta x = P / (|theta'| n) (06e's Gauss-Legendre: P = 128 rad, n = 96):
    floor ~ sqrt( int |f|^2 sigma(x)^2 Delta x dx ),
with sigma(x) = kappa |theta'(x)|, kappa the median measured sigma/|theta'|. Set against
min(abserr_06c, abserr_06e)/100 and 06e's Gauss-Legendre scatter.
Usage (census tree as cwd): python -B r0_jitter.py
"""

import json
import math
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import filon  # noqa: E402
import prov  # noqa: E402
import route_b as RB  # noqa: E402
import samples as S  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"


def one(n):
    rec = S.load(n)
    th = rec["theta"]
    f_sin, f_cos = rec["f"]
    edges, _ = S.route_b_edges(rec)
    nseg = len(edges) - 1
    fit = RB.FIT[1]
    rows = []
    for j in range(40):
        i = min(nseg - 1, round(j * (nseg - 1) / 39))
        sg = S.Segment(rec, edges[i], edges[i + 1], M=fit["M"], K=fit["K"])
        td = abs(th["theta_deriv"](float(sg.mid)))
        rows.append(
            dict(
                x=float(sg.mid),
                theta_deriv=td,
                sigma=sg.sigma,
                kappa=sg.sigma / td,
                branch=sg.branch_margin,
            )
        )
    kappa = float(np.median([r["kappa"] for r in rows]))
    a, b = (float(v) for v in rec["x_span"])
    xs = np.linspace(a, b, 4001)
    f2 = np.array([f_sin(x) ** 2 + f_cos(x) ** 2 for x in xs])
    tdv = np.abs(np.array([th["theta_deriv"](x) for x in xs]))
    P, nn = 128.0, 96
    integrand = f2 * (kappa * tdv) ** 2 * (P / (tdv * nn))
    floor_gl = math.sqrt(np.trapezoid(integrand, xs))
    z = S.z3_row(n)
    m = min(float.fromhex(z["abserr_g"]), float.fromhex(z["abserr_06c"]))
    gl_scatter = z["scatter_over_min"] * m
    return dict(
        sample=n,
        segments=nseg,
        kappa_median=kappa,
        kappa_range=[min(r["kappa"] for r in rows), max(r["kappa"] for r in rows)],
        max_branch_margin=max(r["branch"] for r in rows),
        jitter_floor_GL_P128_n96=floor_gl,
        min_abserr_over_100=m / 100,
        floor_over_threshold=floor_gl / (m / 100),
        GL_scatter_06e=gl_scatter,
        GL_scatter_over_threshold=z["scatter_over_min"] * 100,
        rows=rows,
    )


def main():
    filon.set_precision()
    res = [one(n) for n in S.ROUTE_B]
    out = {
        "header": prov.header([S.pickle_path(n) for n in S.ROUTE_B] + [S.Z3]),
        "samples": res,
    }
    (OUT / "r0_jitter.json").write_text(json.dumps(out, indent=1))
    L = [
        "sample | segments | kappa = sigma/|theta'| (median, range) | jitter floor (GL P128 n96) | min abserr/100 | floor/thr | 06e GL scatter/thr"
    ]
    for r in res:
        L.append(
            f"{r['sample']} | {r['segments']} | {r['kappa_median']:.3g} ({r['kappa_range'][0]:.2g}-{r['kappa_range'][1]:.2g}) | "
            f"{r['jitter_floor_GL_P128_n96']:.3e} | {r['min_abserr_over_100']:.3e} | {r['floor_over_threshold']:.3g} | {r['GL_scatter_over_threshold']:.3g}"
        )
    (OUT / "r0_jitter.txt").write_text(
        json.dumps(out["header"], indent=1) + "\n\n" + "\n".join(L) + "\n"
    )
    print("\n".join(L))


if __name__ == "__main__":
    main()
