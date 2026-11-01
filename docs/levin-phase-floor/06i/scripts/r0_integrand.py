"""levin-phase-floor prompt 06i Sec 1, Gate R0, second part: each route's integrand at 20 points
against the captured integrand f_sin sin(theta_mod_2pi) + f_cos cos(theta_mod_2pi), at the census
tree. New in 06i.

Route A (sample 9, whose two order-5/2 factors exercise the near-region residual): the 20 points
are uniform in u over the span. Route A's integrand at u is Re[F(u) exp(i(K e^u + C + R(fl(e^u))))]
with K and C exact and e^u exact (mpmath). The measured phase difference is decomposed into the
three roundings the census tree's group does not declare (prompt 06h Sec 0.1):
  (a) the product: fl(K^ x^) - K^ x^ = -fma(K^, x^, -fl(K^ x^)),
  (b) K's rounding: -K_lo x^,
  (c) the abscissa: K (x^ - e^u),
and the residual after removing them is set against the group's theta_abserr(u).

Route B (sample 2): 20 points, one in each twentieth of the x span, at 37 % of its segment's t
span; x^ = fl(x(t)) from the segment's map; Route B's integrand in x, Re[F(x^) exp(i p(x^))] (p the
segment's least-squares smooth phase, route_b.py), against the captured integrand at x^. They differ
by the phase delta = rem(theta_mod_2pi(x^) - p(x^)), the captured phase's jitter about the smooth one.
Usage (census tree as cwd): python -B r0_integrand.py
"""

import bisect
import json
import math
import sys
from pathlib import Path

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import filon  # noqa: E402
import prov  # noqa: E402
import samples as S  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"


def route_a(n):
    rec = S.load(n)
    A = S.RouteA(rec)
    th = rec["theta"]
    K = A.K_mp()
    C = A.C_mp()
    Kh = A.group.K
    Klo = mp.mpf(A.K_lo.numerator) / A.K_lo.denominator
    rows = []
    for j in range(20):
        u = A.u_a + (A.u_b - A.u_a) * j / 19
        if j == 19:
            u = A.u_b
        xh = math.exp(u)
        e = mp.exp(mp.mpf(u))
        R = A.group.residual(xh)
        theta_A = K * e + C + R
        Fv = A.F(u)
        F = mp.mpc(Fv.real, Fv.imag)
        I_A = (F * mp.expj(theta_A)).real
        m = th["theta_mod_2pi"](u)
        I_cap = A.f_sin(u) * math.sin(m) + A.f_cos(u) * math.cos(m)
        dphase = S.rem2pi(mp.mpf(m) - theta_A)
        a_ = -math.fma(Kh, xh, -(Kh * xh))
        b_ = -Klo * xh
        c_ = K * (mp.mpf(xh) - e)
        resid = dphase - (a_ + b_ + c_)
        decl = th["theta_abserr"](u)
        absF = abs(Fv)
        diff = abs(mp.mpf(I_cap) - I_A)
        rows.append(
            dict(
                u=u,
                K_e_u=float(K * e),
                I_A=mp.nstr(I_A, 20),
                I_captured=repr(I_cap),
                abs_diff=float(diff),
                declaration_times_absF=decl * absF,
                diff_over_declaration=(
                    float(diff) / (decl * absF) if decl * absF > 0 else math.inf
                ),
                dphase=float(dphase),
                a_product=a_,
                b_K_lo=float(b_),
                c_abscissa=float(c_),
                residual_after_abc=float(resid),
                theta_abserr=decl,
                residual_over_declaration=float(abs(resid)) / decl,
            )
        )
    return dict(
        sample=n,
        K_hat=Kh,
        K_lo=float(Klo),
        K_lo_fsum=A.K_lo_fsum,
        C_exact=mp.nstr(C, 20),
        C_attr=A.group.C,
        nus=A.nus,
        holds_strict=all(r["diff_over_declaration"] <= 1 for r in rows),
        holds_after_abc=all(r["residual_over_declaration"] <= 1 for r in rows),
        max_diff_over_declaration=max(r["diff_over_declaration"] for r in rows),
        max_residual_after_abc_over_declaration=max(
            r["residual_over_declaration"] for r in rows
        ),
        rows=rows,
    )


def route_b(n):
    import route_b as RB

    rec = S.load(n)
    th = rec["theta"]
    f_sin, f_cos = rec["f"]
    edges, ntab = S.route_b_edges(rec)
    fit = RB.FIT[1]
    rows = []
    a, b = edges[0], edges[-1]
    for j in range(20):
        # points uniform in x, moved into the segment containing them, at t = 37 % of its span
        x0 = a + (b - a) * (j + 0.5) / 20
        i = bisect.bisect_right(edges, x0) - 1
        sg = RB.fit_segments(rec, edges, i, i + 1, fit["M"], fit["K"])[0]
        t = sg.T_l + (sg.T_r - sg.T_l) * mp.mpf("0.37")
        y = sg.solve_y(t)
        xh = float(sg.mid + sg.half * y)
        yh = sg.y_of(mp.mpf(xh))
        th_ = sg.p(yh)
        m = th["theta_mod_2pi"](xh)
        delta = S.rem2pi(mp.mpf(m) - th_)
        Fv = complex(f_cos(xh), -f_sin(xh))
        F = mp.mpc(Fv.real, Fv.imag)
        I_B = (F * mp.expj(th_)).real
        I_cap = f_sin(xh) * math.sin(m) + f_cos(xh) * math.cos(m)
        diff = abs(mp.mpf(I_cap) - I_B)
        rows.append(
            dict(
                x=xh,
                theta_deriv=th["theta_deriv"](xh),
                eps_t=float(th_ - t),
                delta=float(delta),
                segment_sigma=sg.sigma,
                I_B=mp.nstr(I_B, 20),
                I_captured=repr(I_cap),
                abs_diff=float(diff),
                abs_diff_over_absF=float(diff) / abs(Fv),
                monotone=sg.monotone,
            )
        )
    return dict(
        sample=n,
        segments=len(edges) - 1,
        tables=ntab,
        declaration="none: cosmological groups declare no phase error at the census tree",
        max_abs_diff_over_absF=max(r["abs_diff_over_absF"] for r in rows),
        rows=rows,
    )


def main():
    filon.set_precision()
    A = route_a(9)
    B = route_b(2)
    res = {
        "header": prov.header([S.pickle_path(9), S.pickle_path(2)]),
        "route_A": A,
        "route_B": B,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "r0_integrand.json").write_text(json.dumps(res, indent=1))
    L = [
        f"Route A, sample 9: K^ = {A['K_hat']!r}, K_lo = {A['K_lo']:.6e} (fsum {A['K_lo_fsum']:.6e}), "
        f"C exact = {A['C_exact']} (attribute {A['C_attr']!r}), orders {A['nus']}",
        "u | K e^u | |I_cap - I_A| | theta_abserr |F| | ratio | dphase | (a) | (b) | (c) | resid | resid/theta_abserr",
    ]
    for r in A["rows"]:
        L.append(
            f"{r['u']:+.4f} | {r['K_e_u']:.3e} | {r['abs_diff']:.3e} | {r['declaration_times_absF']:.3e} | "
            f"{r['diff_over_declaration']:.3g} | {r['dphase']:+.3e} | {r['a_product']:+.2e} | {r['b_K_lo']:+.2e} | "
            f"{r['c_abscissa']:+.2e} | {r['residual_after_abc']:+.2e} | {r['residual_over_declaration']:.3g}"
        )
    L.append(
        f"strict (|diff| <= theta_abserr |F| at all 20): {A['holds_strict']} (max ratio {A['max_diff_over_declaration']:.4g}); "
        f"after removing (a)+(b)+(c): {A['holds_after_abc']} (max {A['max_residual_after_abc_over_declaration']:.4g})"
    )
    L.append("")
    L.append(
        f"Route B, sample 2: {B['segments']} segments, {B['tables']} tables; declaration: {B['declaration']}"
    )
    L.append(
        "x | theta' | eps_t = t^ - t | delta = rem(theta_mod - p) | segment sigma | |I_cap - I_B| / |F|"
    )
    for r in B["rows"]:
        L.append(
            f"{r['x']:.6f} | {r['theta_deriv']:.3e} | {r['eps_t']:+.3e} | {r['delta']:+.3e} | {r['segment_sigma']:.2e} | {r['abs_diff_over_absF']:.3e}"
        )
    L.append(f"max |I_cap - I_B|/|F| = {B['max_abs_diff_over_absF']:.3e}")
    (OUT / "r0_integrand.txt").write_text(
        json.dumps(res["header"], indent=1) + "\n\n" + "\n".join(L) + "\n"
    )
    print("\n".join(L))


if __name__ == "__main__":
    main()
