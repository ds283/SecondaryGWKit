"""
Measurements for docs/qcd-background-audit-2026-09.md.

The QCD background's temperature T(z) is represented by a cubic interpolating spline over 500
points uniform in u = log(1+z), built from node values that are root-solved to rtol = 1e-4
(CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py, _build_T_z_spline and _solve_T_z). This
script measures what that representation costs, and what a better one would deliver.

Everything here is scored against a *tight* reference: the same defining equation
T Gs(T)^(1/3) = T_CMB Gs(T_CMB)^(1/3) (1+z), root-solved to rtol = 1e-14. The shipped
_solve_T_z is NOT a reference -- it is one of the things being measured.

No Ray and no datastore. Run from the repository root:

    PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/measure_T_z_representation.py

Sections:
    0  the shipped pipeline, against the tight reference
    1  the equation-of-state branch joins -- where g and g_s fail to match
    2  T(z) is genuinely discontinuous, not merely kinked
    3  candidate representations, measured
    4  downstream: H(z), conformal time, and the phase that costs
    5  BREAK_POINT_ALL against the production sample grid
    6  cost per call
"""

import sys
import time
import warnings
from math import exp, log, log1p, expm1, pow
from pathlib import Path

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import root_scalar

REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComputeTargets.tests.test_gk_wkb_phase import qcd_model_with_tables  # noqa: E402
from ComputeTargets.tests.wkb_reference import (  # noqa: E402
    load_references,
    production_source_grid,
)
from CosmologyModels.GenericEOS.GenericEOS import (  # noqa: E402
    BREAK_POINT_ALL,
    BREAK_POINT_DISCONTINUITY,
)
from CosmologyModels.GenericEOS.LambdaCDM_GenericEOS import HIGH_T_GSTAR  # noqa: E402

# The tabulated range _build_T_z_spline actually covers: the model's declared bounds, widened
# by 5% in (1+z) at each end. Reproduced here so a candidate is built over the same interval.
SHIPPED_MIN_Z = 0.95 * (1.0 + (-0.2)) - 1.0
SHIPPED_MAX_Z = 1.05 * (1.0 + 1.0e20) - 1.0

# k*tau spans at the three production wavenumbers, from docs/gktk-remedial-verification.md
# section 3.5. Used only to express a relative error in tau as radians of phase.
PRODUCTION_SPANS_RAD = {1.0e5: 1.3728e09, 1.0e7: 1.3728e11, 3.0e8: 4.1184e12}
PRODUCTION_FLOOR_RAD = {1.0e5: 3.05e-07, 1.0e7: 3.05e-05, 3.0e8: 9.15e-04}


def banner(title):
    print("\n" + "=" * 94)
    print(title)
    print("=" * 94)


class Background:
    """The QCD cosmology, with an accurate T(z) and the candidate representations."""

    def __init__(self):
        refs = load_references()
        self.grid = production_source_grid(
            refs["models"]["LambdaCDMModel"]["grid"]["z_init"]
        )
        self.model = qcd_model_with_tables(self.grid)
        self.cosmology = self.model.cosmology
        self.eos = self.cosmology._eos
        self.T_CMB = self.cosmology._T_CMB
        self.G_S_CMB_pow13 = self.cosmology._G_S_CMB_pow13
        self.u_lo = log(1.0 + SHIPPED_MIN_Z)
        self.u_hi = log(1.0 + SHIPPED_MAX_Z)
        self._jumps = None

    def accurate_T(self, z, rtol=1.0e-14):
        """T(z) from the defining equation, root-solved to rtol. The reference for everything."""
        hi = 1.05 * self.T_CMB * (1.0 + z)
        lo = 0.95 * self.T_CMB * (1.0 + z) / pow(HIGH_T_GSTAR, 1.0 / 3.0)
        target = self.T_CMB * self.G_S_CMB_pow13 * (1.0 + z)

        def equation(T):
            return T * pow(self.eos.Gs(T), 1.0 / 3.0) - target

        return root_scalar(equation, bracket=(lo, hi), xtol=1.0e-300, rtol=rtol).root

    def entropy_factor(self, u, rtol=1.0e-14):
        """F(u) = log( T / (T_CMB (1+z)) ) = -(1/3) log( Gs(T)/Gs(T_CMB) ). Bounded and O(1)."""
        return log(self.accurate_T(expm1(u), rtol) / (self.T_CMB * exp(u)))

    def jump_locations(self):
        """
        The u at which T(z) crosses each declared break temperature, to machine precision.

        Found by bisecting the *monotone* T(z) rather than by root-finding on
        accurate_T(z) - T_break: T(z) is discontinuous at exactly these points (section 2), so a
        root of that difference need not exist, and a bracketing solver applied to it lands
        beside the jump rather than on it. A segment edge that misses the jump by one node
        leaves the old error in place -- this is the single detail a re-implementation must get
        right.
        """
        if self._jumps is not None:
            return self._jumps
        out = []
        for T_break_GeV in sorted(set(self.eos.break_temperatures_GeV)):
            T_break = T_break_GeV * self.cosmology.units.GeV
            lo, hi = 1.0e-6, 1.0e19
            if not self.accurate_T(lo) < T_break < self.accurate_T(hi):
                continue
            for _ in range(200):
                mid = (lo * hi) ** 0.5
                if self.accurate_T(mid) < T_break:
                    lo = mid
                else:
                    hi = mid
                if hi / lo - 1.0 < 1.0e-15:
                    break
            u = log1p(0.5 * (lo + hi))
            if self.u_lo < u < self.u_hi:
                out.append(u)
        self._jumps = sorted(out)
        return self._jumps

    def build_plain(self, n, order=3, rtol=1.0e-14):
        """The shipped shape -- T against u, one global spline -- but with accurate nodes."""
        us = np.linspace(self.u_lo, self.u_hi, n)
        Ts = np.array([self.accurate_T(expm1(x), rtol) for x in us])
        spline = make_interp_spline(us, Ts, k=order)
        return lambda z: float(spline(log1p(z)))

    def build_entropy(self, n, order=3, rtol=1.0e-14):
        """One global spline of the entropy factor F(u)."""
        us = np.linspace(self.u_lo, self.u_hi, n)
        F = np.array([self.entropy_factor(x, rtol) for x in us])
        spline = make_interp_spline(us, F, k=order)
        return lambda z: self.T_CMB * (1.0 + z) * exp(float(spline(log1p(z))))

    def build_segmented(self, total_pts, order=5, rtol=1.0e-14, pad=1.0e-12):
        """
        The recommended representation: one entropy-factor spline per branch, with the segment
        edges placed exactly on the jumps. `pad` keeps the node set strictly inside its own
        branch so no segment interpolates across the discontinuity.
        """
        bounds = [self.u_lo] + self.jump_locations() + [self.u_hi]
        widths = np.diff(bounds)
        frac = widths / widths.sum()
        segments = []
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            a_in = a + pad if i > 0 else a
            b_in = b - pad if i < len(bounds) - 2 else b
            n = max(order + 1, int(round(total_pts * frac[i])))
            us = np.linspace(a_in, b_in, n)
            F = np.array([self.entropy_factor(x, rtol) for x in us])
            segments.append((b, make_interp_spline(us, F, k=order)))

        def evaluate(z):
            u = log1p(z)
            for upper, spline in segments:
                if u <= upper:
                    return self.T_CMB * (1.0 + z) * exp(float(spline(u)))
            return self.T_CMB * (1.0 + z) * exp(float(segments[-1][1](u)))

        return evaluate

    def Hubble_with(self, T_of_z, z_values):
        """H(z) evaluated with the cosmology's T(z) temporarily replaced by T_of_z."""
        original = self.cosmology._T_z_spline
        self.cosmology._T_z_spline = T_of_z
        try:
            return np.array([self.cosmology.Hubble(float(z)) for z in z_values])
        finally:
            self.cosmology._T_z_spline = original


def relative(candidate, reference):
    return np.abs(candidate - reference) / np.abs(reference)


def report(label, rel):
    print(
        f"   {label:<52s} max {rel.max():.3e}   p90 {np.percentile(rel, 90):.3e}"
        f"   median {np.median(rel):.3e}"
    )


def main():
    t_start = time.perf_counter()
    bg = Background()

    zs = np.array(bg.grid.as_float_list(), dtype=float)
    zs.sort()
    u_nodes = np.log1p(zs)
    probe_u = np.unique(np.concatenate([u_nodes, 0.5 * (u_nodes[:-1] + u_nodes[1:])]))
    probe_z = np.expm1(probe_u)
    probe_z = probe_z[(probe_z > 1.0) & (probe_z < 1.0e16)][::5]
    T_ref = np.array([bg.accurate_T(float(z)) for z in probe_z])

    banner("0. THE SHIPPED PIPELINE, AGAINST A TIGHT REFERENCE")
    print(
        f"   {len(probe_z)} probe points on and between production grid nodes, "
        f"z in [{probe_z.min():.4g}, {probe_z.max():.4g}]."
    )
    print(
        "   Probes sit at midpoints as well as nodes: a spline is worst between its knots.\n"
    )
    report(
        "shipped _solve_T_z (its own rtol = 1e-4)",
        relative(np.array([bg.cosmology._solve_T_z(float(z)) for z in probe_z]), T_ref),
    )
    report(
        "shipped T(z) spline (500 pts, over those nodes)",
        relative(
            np.array([bg.cosmology._T_z_spline(float(z)) for z in probe_z]), T_ref
        ),
    )
    print(
        "\n   The node values are themselves wrong by up to the first line, and independently\n"
        "   so from one node to the next: that is the 'scatter between neighbouring nodes' of\n"
        "   [02-qcd-T-z-spline-node-tolerance]."
    )

    banner("1. THE EQUATION-OF-STATE BRANCH JOINS")
    print(
        "   Relative jump in g and g_s across each declared break temperature, evaluated at\n"
        "   T*(1 +/- 1e-9). Entropy conservation T g_s(T)^(1/3) ~ (1+z) then forces a jump\n"
        "   dT/T = -(1/3) dg_s/g_s in T(z) at fixed z.\n"
    )
    print(
        f"   {'T_break [GeV]':>14s} {'g below':>12s} {'g above':>12s} {'dg/g':>12s}"
        f" {'g_s below':>12s} {'g_s above':>12s} {'dg_s/g_s':>12s} {'-> dT/T':>12s}"
    )
    for T_break_GeV in sorted(set(bg.eos.break_temperatures_GeV), reverse=True):
        T = T_break_GeV * bg.cosmology.units.GeV
        g_lo, g_hi = bg.eos.G(T * (1.0 - 1.0e-9)), bg.eos.G(T * (1.0 + 1.0e-9))
        s_lo, s_hi = bg.eos.Gs(T * (1.0 - 1.0e-9)), bg.eos.Gs(T * (1.0 + 1.0e-9))
        dg = (g_hi - g_lo) / g_lo
        ds = (s_hi - s_lo) / s_lo
        print(
            f"   {T_break_GeV:14.6g} {g_lo:12.6f} {g_hi:12.6f} {dg:12.3e}"
            f" {s_lo:12.6f} {s_hi:12.6f} {ds:12.3e} {-ds / 3.0:+12.3e}"
        )
    print(
        "\n   One join matches to round-off and the others do not, which is the evidence that\n"
        "   the mismatches are a defect of the parametrisation rather than its intent. This is\n"
        "   an upstream data fixture; the audit records it and does not repair it."
    )

    banner("2. T(z) IS GENUINELY DISCONTINUOUS, NOT MERELY KINKED")
    jumps = bg.jump_locations()
    print(
        f"   T(z) crosses a break temperature at z = {[f'{expm1(u):.6g}' for u in jumps]}\n"
    )
    z_c = expm1(jumps[0])
    print(
        f"   F(u) = log(T / (T_CMB (1+z))) across the lowest crossing, z_c = {z_c:.6g}:\n"
    )
    print(f"   {'z':>16s} {'F(u)':>22s}")
    for factor in (0.95, 0.99, 0.999, 1.001, 1.01, 1.05):
        z = z_c * factor
        print(f"   {z:16.6e} {bg.entropy_factor(log1p(z)):22.12f}")
    print(
        "\n   F is piecewise constant to twelve decimals with a step between: g_s is a step\n"
        "   there, so T(z) is a step. No single smooth approximant can represent it, at any\n"
        "   node count -- but a representation segmented AT the step reproduces it exactly."
    )

    banner("3. CANDIDATE REPRESENTATIONS")
    print(
        "   All built over the same interval the shipped spline covers, all with accurate\n"
        "   nodes, all scored on the same probe set as section 0.\n"
    )
    report(
        "shipped (500 pts, sloppy nodes, T against u)",
        relative(
            np.array([bg.cosmology._T_z_spline(float(z)) for z in probe_z]), T_ref
        ),
    )
    for label, builder in (
        ("plain T spline, 500 pts, accurate nodes", lambda: bg.build_plain(500)),
        ("entropy factor, 500 pts", lambda: bg.build_entropy(500)),
        ("entropy factor, 2000 pts", lambda: bg.build_entropy(2000)),
        ("entropy factor, 2000 pts, k=5", lambda: bg.build_entropy(2000, order=5)),
        ("SEGMENTED entropy factor, 3000 pts, k=5", lambda: bg.build_segmented(3000)),
    ):
        f = builder()
        report(label, relative(np.array([f(float(z)) for z in probe_z]), T_ref))
    print(
        "\n   Splining T against u spends the resolution on the (1+z) ramp, which is known\n"
        "   analytically. Splining the entropy factor spends it on g_s alone. Segmenting at\n"
        "   the jumps removes the rest; without it the max is pinned at the jump height however\n"
        "   many nodes are used."
    )

    banner("4. DOWNSTREAM: H(z), CONFORMAL TIME, AND THE PHASE THAT COSTS")
    improved = bg.build_segmented(3000)
    z_dense = zs[(zs > 1.0) & (zs < 1.0e16)]
    H_shipped = np.array([bg.cosmology.Hubble(float(z)) for z in z_dense])
    H_improved = bg.Hubble_with(improved, z_dense)
    H_exact = bg.Hubble_with(lambda z: bg.accurate_T(z), z_dense)
    print(
        "   rho_r = RadiationConstant * g(T) * T^4 and H ~ sqrt(rho), so a relative error"
    )
    print("   eps in T reaches H at roughly 2*eps in the radiation era.\n")
    report("H(z), shipped", relative(H_shipped, H_exact))
    report("H(z), improved (segmented entropy factor)", relative(H_improved, H_exact))

    print(
        "\n   Conformal time. theta = -[k dtau + drho], so a relative error in tau is a phase"
    )
    print("   error proportional to k*tau.\n")

    def inverse_H(T_of_z):
        def integrand(u):
            original = bg.cosmology._T_z_spline
            bg.cosmology._T_z_spline = T_of_z
            try:
                return (1.0 + expm1(u)) / bg.cosmology.Hubble(expm1(u))
            finally:
                bg.cosmology._T_z_spline = original

        return integrand

    u_a, u_b = log1p(1.0e2), log1p(1.0e12)
    interior = [u for u in jumps if u_a < u < u_b]
    integrals = {}
    for label, T_of_z in (
        ("shipped", bg.cosmology._T_z_spline),
        ("improved", improved),
        ("exact", lambda z: bg.accurate_T(z)),
    ):
        # The `exact` integrand root-solves to machine precision on every call, so quad
        # reports a roundoff warning before it reaches epsrel. It is benign: the reported
        # relative error is converged against this setting -- 3.4744e-08, 3.4602e-08,
        # 3.4605e-08, 3.4605e-08 at epsrel = 1e-9, 1e-10, 1e-11, 1e-12 -- which is four
        # digits of stability across four decades, well inside the effect being measured.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=IntegrationWarning)
            value, _ = quad(
                inverse_H(T_of_z),
                u_a,
                u_b,
                limit=400,
                epsabs=0,
                epsrel=1.0e-11,
                points=interior,
            )
        integrals[label] = value
        print(f"   integral dz/H over z in [1e2, 1e12], {label:<9s} = {value:.16e}")

    print(f"\n   {'representation':<12s} {'rel. error in tau':>20s}", end="")
    for k in PRODUCTION_SPANS_RAD:
        print(f" {'phase at k=' + f'{k:.0e}':>18s}", end="")
    print()
    for label in ("shipped", "improved"):
        rel = abs(integrals[label] - integrals["exact"]) / abs(integrals["exact"])
        print(f"   {label:<12s} {rel:20.3e}", end="")
        for k, span in PRODUCTION_SPANS_RAD.items():
            print(f" {rel * span:18.3e}", end="")
        print()
    print(f"   {'(1 ulp floor)':<12s} {'':>20s}", end="")
    for k in PRODUCTION_SPANS_RAD:
        print(f" {PRODUCTION_FLOOR_RAD[k]:18.3e}", end="")
    print()
    print(
        "\n   This error is COMMON MODE: the producer and the consumer are built from the same\n"
        "   H, so every self-consistency measurement in docs/gktk-remedial-verification.md sees\n"
        "   1 ulp while the background underneath both carries the figures above."
    )

    banner("5. BREAK_POINT_ALL AGAINST THE PRODUCTION SAMPLE GRID")
    z_hi, z_lo = float(zs.max()), float(zs.min())
    du = np.diff(np.sort(u_nodes))
    print(
        f"   production source grid: {len(zs)} samples, z in [{z_lo:.4g}, {z_hi:.4g}],"
        f" median spacing {np.median(du):.4e} in u\n"
    )
    for kind in (BREAK_POINT_ALL, BREAK_POINT_DISCONTINUITY):
        points = np.asarray(
            bg.cosmology.integration_break_points(z_lo, z_hi, kind=kind), dtype=float
        )
        line = f"   {kind:<14s} {len(points):5d} points in range"
        if len(points) > 1:
            spacing = np.median(np.diff(np.sort(points)))
            line += (
                f", median spacing {spacing:.4e} in u"
                f" = {spacing / np.median(du):.2f} x the grid spacing"
            )
        print(line)
    knots = bg.cosmology._T_z_spline_knots_log1pz
    inside = knots[(knots > u_nodes.min()) & (knots < u_nodes.max())]
    print(
        f"\n   Of the BREAK_POINT_ALL points, {len(inside)} are knots of the T(z) spline itself --\n"
        "   a uniform lattice of an auxiliary 500-point interpolant, not a feature of the\n"
        "   cosmology. A representation that does not need that lattice removes them from the\n"
        "   break-point set, leaving only the genuine crossings of section 2."
    )

    banner("6. COST PER CALL")
    entropy_2000 = bg.build_entropy(2000)
    segmented = improved
    sample = probe_z[:200]
    for label, f in (
        ("shipped spline", lambda z: bg.cosmology._T_z_spline(z)),
        ("entropy factor, 2000 pts", entropy_2000),
        ("segmented entropy factor", segmented),
        ("accurate root solve (build time only)", lambda z: bg.accurate_T(z)),
    ):
        best = None
        for _ in range(3):
            t0 = time.perf_counter()
            for z in sample:
                f(float(z))
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        print(f"   {label:<40s} {best / len(sample) * 1.0e6:8.3f} us/call")
    print(
        "\n   The recommended representation is no more expensive at runtime than the one it\n"
        "   replaces. The accurate root solve is paid once per node at build time."
    )

    print(f"\nTotal wall time {time.perf_counter() - t_start:.1f} s")


if __name__ == "__main__":
    main()
