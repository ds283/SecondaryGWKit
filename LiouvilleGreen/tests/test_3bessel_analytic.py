"""
Three-Bessel integrals against their closed forms.

The tolerances here were set when `bessel_phase` built the phase by integrating Q = theta/x as an
ODE, whose phase and modulus fits carried a ~2e-8 relative floor that these integrals inherited.
That construction was replaced by the two-region amplitude-residual one
(prompts/transfer-remedial, prompts 03-05) and the phase groups were rebuilt as K t + C + R(t)
(prompt 07), so the tolerances were re-measured from scratch by prompt 08 rather than scaled.

Measured at the fixed triple k, q, s = 1.3, 1.7, 2.1, max_x = 1e12, atol = 1e-14, rtol = 1e-10,
on the tree immediately before the campaign (f17f2d4) and on the tree after prompt 07 -- relative
error against the closed form:

    oracle   before     after      gain
    J000     1.625e-08  1.397e-10    116x
    J110     4.615e-08  5.857e-12   7880x
    J220     2.561e-08  2.521e-14    1.0e6x
    J222     9.552e-09  3.504e-14    2.7e5x
    J231     3.349e-08  1.491e-13    2.2e5x
    Y000     2.156e-08  4.771e-11    452x
    Y022     2.014e-08  1.543e-14    1.3e6x

**What limits the two constants below is not the Bessel phase.** The J000 and Y000 residuals are
set by DEFAULT_3BESSEL_CHEBYSHEV_ORDER = 12: raising it to 20 at the same triple moves J000 from
1.397e-10 to 2.071e-13 and Y000 from 4.771e-11 to 3.508e-13, three orders in each case. That is a
genuine finding and it is not prompt 08's to fix -- the constant is deliberately left alone; see
prompts/transfer-remedial/IMPLEMENTATION_STATE.md
[08-3bessel-chebyshev-order-is-now-the-limit], which also records that raising the order makes
five of the seven oracles *worse*, so 12 is not simply too low. The five that are not (0,0,0) sit
at 1e-12 to 1e-14 and would support tolerances four orders tighter than the ones set here.

The tolerances are therefore sized by the worst oracle over random draws, not by the best. Over
42 seeded draws from uniform(0.1, 5) -- the distribution `test_JJJ`, `test_YJJ` and
`test_YJJ_log_singularity` sample from, unseeded -- the worst triangle relative error was
1.740e-09 (Y000) and the worst non-triangle absolute error 4.056e-10 (J000). REL_TOLERANCE and
ABS_TOLERANCE below are 100x tighter than before and keep a factor of 25 to 57 over those, which
is the right margin for a test that draws its own wavenumbers on every run.

The convergence figures this module used to draw on every run are now behind
THREE_BESSEL_DIAGNOSTIC_PLOTS -- see the note on DIAGNOSTIC_PLOTS below. Set it to anything other
than "0" or "false" to get them back, along with `test_YJJ_log_scaling`, which draws the eps
scaling and asserts nothing.
"""

import os
import time
import unittest
from datetime import datetime
from pathlib import Path
from random import uniform

import numpy as np

from LiouvilleGreen.bessel_phase import (
    DEFAULT_AMPLITUDE_RTOL,
    DEFAULT_PHASE_ATOL,
    bessel_phase,
)
from LiouvilleGreen.three_bessel_integrals import quad_JJJ, quad_YJJ
from utilities import format_time

ABS_TOLERANCE = 1e-8
REL_TOLERANCE = 1e-7

# The convergence figures this module can draw are a diagnostic, not part of any assertion. Each
# one evaluates the integral on a 250-point grid in x purely to show it settling onto the closed
# form; the assertion itself reads only the single evaluation at max_x that follows. Measured at
# (k, q, s) = (1.3, 1.7, 2.1), max_x = 1e12, atol = 1e-14, rtol = 1e-10, that grid costs 42.5 s
# against 0.14 s for the asserted evaluation -- a factor of ~300 -- and the module ran the helper
# 47 times, which made it 1121 s of a 1321 s suite (85 % of the whole thing) and wrote 110 files.
# So the grid and the figures are off unless THREE_BESSEL_DIAGNOSTIC_PLOTS is set to something
# other than "0" or "false". Nothing any assertion reads depends on them, and the numbers the
# assertions do read are bit-for-bit what they were: the final evaluator call is unchanged, and
# the grid never fed back into it.
DIAGNOSTIC_PLOTS = os.environ.get("THREE_BESSEL_DIAGNOSTIC_PLOTS", "").lower() not in (
    "",
    "0",
    "false",
)

# Neither singularity band is tightened, and what limits them is genuine near-singular behaviour
# rather than anything this campaign touched. `test_YJJ_log_singularity` walks s to within eps of
# |k - q| and of k + q for eps down to 1e-10, where the closed forms carry
# log|(k-q+s)(k+q-s) / ((k+q+s)(k-q-s))| and both the integral and its reference lose conditioning.
# Measured over all 40 cases of one full run under the new oracle (21.2 min), |relerr| and |abserr|
# grow monotonically as eps falls -- Y022 lo, for instance, runs 1.7e-10, 5.0e-11, 2.7e-10,
# 2.0e-09, 6.1e-09, 3.8e-08, 3.1e-08, 2.2e-06, 5.1e-05, 2.8e-04 from eps = 0.1 to 1e-10. The worst
# case of the run is |relerr| = 2.794e-04 and |abserr| = 4.888e-03, both at Y022, eps = 1e-10, s
# near |k - q|.
#
# Two consequences. (1) A factor 36 of headroom on the relative band, over an error that grows by
# an order for each order eps falls and over wavenumbers the test draws afresh on every run, is
# not margin to spend: 1e-2 stays. (2) The `or` in the assertions is load-bearing, not belt and
# braces -- at eps = 1e-10 the absolute error *exceeds* SINGULARITY_ABS_TOLERANCE (4.888e-03
# against 1e-3) and the case passes on the relative band alone. So the absolute band cannot be
# tightened either without the near-singular cases failing on it.
SINGULARITY_ABS_TOLERANCE = 1e-3
SINGULARITY_REL_TOLERANCE = 1e-2

# with max eta around 10500 and a largest k of order 10^7/Mpc or so, the largest argument we would
# appear to need is about 1E4 * 1E7 ~ 1E11
MAX_X = 1e12


def is_triangle(k: float, q: float, s: float):
    return np.fabs(k - q) < s < k + q


def _figure_path(label, mu, nu, sigma, k, q, s, max_x, timestamp):
    path = Path(
        f"test_3bessel_analytic/{timestamp.isoformat()}/{label}_mu={mu:.3f}_nu={nu:.3f}_sigma={sigma:.3f}_k={k:.3f}_q={q:.3f}_s={s:.3f}_maxx={max_x:.5g}.pdf"
    ).resolve()
    path.parents[0].mkdir(parents=True, exist_ok=True)
    return path


def _plot_convergence(
    evaluator,
    mu_phase,
    nu_phase,
    sigma_phase,
    mu: float,
    nu: float,
    sigma: float,
    k: float,
    q: float,
    s: float,
    max_x: float,
    analytic_result: float,
    label: str,
    timestamp,
    quad_atol: float,
    quad_rtol: float,
):
    """
    Draw the integral against its closed form over a 250-point grid in x, to show it converging.

    Diagnostic only -- see the DIAGNOSTIC_PLOTS note above for what this costs and why it is off
    by default. `seaborn` and `matplotlib` are imported here rather than at module scope so that
    a default run does not pay for them either.
    """
    import seaborn as sns
    from matplotlib import pyplot as plt

    x_grid = np.logspace(np.log10(100.0), np.log10(max_x), 250)
    y_grid = [
        evaluator(
            mu_phase,
            nu_phase,
            sigma_phase,
            mu,
            nu,
            sigma,
            k,
            q,
            s,
            x,
            atol=quad_atol,
            rtol=quad_rtol,
        ).value
        for x in x_grid
    ]

    sns.set_theme()

    fig = plt.figure()
    ax = plt.gca()

    ax.plot(x_grid, y_grid, color="b", label="Numeric + Levin")

    if analytic_result is not None:
        ax.axhline(analytic_result, color="r", linestyle=(0, (1, 1)), label="Analytic")

    ax.text(0.0, 1.03, f"k={k:.5g}", transform=ax.transAxes, fontsize="x-small")
    ax.text(0.3, 1.03, f"q={q:.5g}", transform=ax.transAxes, fontsize="x-small")
    ax.text(0.6, 1.03, f"s={s:.5g}", transform=ax.transAxes, fontsize="x-small")
    ax.text(0.0, 1.08, f"$\\mu$={mu:.5g}", transform=ax.transAxes, fontsize="x-small")
    ax.text(0.3, 1.08, f"$\\nu$={nu:.5g}", transform=ax.transAxes, fontsize="x-small")
    ax.text(
        0.6, 1.08, f"$\\sigma$={sigma:.5g}", transform=ax.transAxes, fontsize="x-small"
    )

    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.legend(loc="best")
    ax.grid(True)

    fig_path = _figure_path(label, mu, nu, sigma, k, q, s, max_x, timestamp)
    fig.savefig(fig_path)
    fig.savefig(fig_path.with_suffix(".png"))

    plt.close()


def compute_3Bessel(
    evaluator,
    mu: float,
    nu: float,
    sigma: float,
    k: float,
    q: float,
    s: float,
    max_x: float,
    analytic_result: float,
    label: str,
    timestamp,
    phase_atol=DEFAULT_PHASE_ATOL,
    amplitude_rtol=DEFAULT_AMPLITUDE_RTOL,
    quad_atol=1e-14,
    quad_rtol=1e-10,
    mu_phase=None,
    nu_phase=None,
):
    """
    Evaluate the three-Bessel integral at `max_x`, which is the number the callers assert on.

    `mu_phase` and `nu_phase` may be supplied by a caller that already holds them: they depend
    only on `mu`/`k` and `nu`/`q`, so a caller sweeping `s` at fixed `k`, `q` would otherwise
    rebuild the same two objects on every step. `sigma_phase` depends on `s` and is always built
    here.

    When DIAGNOSTIC_PLOTS is set, the convergence figure is drawn first, from the same three
    phase objects. It does not affect the returned result.
    """
    # These used to be atol=1e-25, rtol=5e-14 -- tolerances of the phase ODE solve, which the
    # two-region construction (prompts/transfer-remedial prompt 05) no longer performs. Those
    # arguments are now accepted, ignored and warned about, so the builds here were *already*
    # running at the defaults; passing the defaults explicitly changes no number and removes
    # three DeprecationWarnings per case. See IMPLEMENTATION_STATE.md standing note 20.
    if mu_phase is None:
        mu_phase = bessel_phase(
            mu + 0.5,
            1.075 * k * max_x,
            phase_atol=phase_atol,
            amplitude_rtol=amplitude_rtol,
        )
    if nu_phase is None:
        nu_phase = bessel_phase(
            nu + 0.5,
            1.075 * q * max_x,
            phase_atol=phase_atol,
            amplitude_rtol=amplitude_rtol,
        )
    sigma_phase = bessel_phase(
        sigma + 0.5,
        1.075 * s * max_x,
        phase_atol=phase_atol,
        amplitude_rtol=amplitude_rtol,
    )

    if DIAGNOSTIC_PLOTS:
        _plot_convergence(
            evaluator,
            mu_phase,
            nu_phase,
            sigma_phase,
            mu,
            nu,
            sigma,
            k,
            q,
            s,
            max_x,
            analytic_result,
            label,
            timestamp,
            quad_atol=quad_atol,
            quad_rtol=quad_rtol,
        )

    result = evaluator(
        mu_phase,
        nu_phase,
        sigma_phase,
        mu,
        nu,
        sigma,
        k,
        q,
        s,
        max_x,
        atol=quad_atol,
        rtol=quad_rtol,
    )
    return result


class J000:
    mu = 0.0
    nu = 0.0
    sigma = 0.0

    @staticmethod
    def analytic(k, q, s):
        if not is_triangle(k, q, s):
            return 0.0

        return (np.pi / 4.0) / (k * q * s)


class J110:
    mu = 1.0
    nu = 1.0
    sigma = 0.0

    @staticmethod
    def analytic(k, q, s):
        if not is_triangle(k, q, s):
            return 0.0

        k_sq = k * k
        q_sq = q * q
        s_sq = s * s
        return (np.pi / 8.0) * (k_sq + q_sq - s_sq) / (k_sq * q_sq * s)


class J220:
    mu = 2.0
    nu = 2.0
    sigma = 0.0

    @staticmethod
    def analytic(k, q, s):
        if not is_triangle(k, q, s):
            return 0.0

        k_sq = k * k
        k3 = k_sq * k
        k4 = k_sq * k_sq
        q_sq = q * q
        q3 = q_sq * q
        s_sq = s * s

        pre_factor = np.pi / 32.0
        numerator = (
            3.0 * k4
            + 2.0 * k_sq * (q_sq - 3.0 * s_sq)
            + 3.0 * (q_sq - s_sq) * (q_sq - s_sq)
        )
        denominator = k3 * q3 * s

        return pre_factor * numerator / denominator


class J222:
    mu = 2.0
    nu = 2.0
    sigma = 2.0

    @staticmethod
    def analytic(k, q, s):
        if not is_triangle(k, q, s):
            return 0.0

        k_sq = k * k
        q_sq = q * q
        s_sq = s * s

        k3 = k_sq * k
        q3 = q_sq * q
        s3 = s_sq * s

        k4 = k_sq * k_sq
        q4 = q_sq * q_sq
        s4 = s_sq * s_sq

        s6 = s4 * s_sq

        pre_factor = np.pi / 64.0

        numerator = (
            (3.0 * k4 + 2.0 * k_sq * q_sq + 3.0 * q4) * s_sq
            + 3.0 * (k_sq + q_sq) * s4
            - 3.0 * (k_sq - q_sq) * (k_sq - q_sq) * (k_sq + q_sq)
            - 3.0 * s6
        )
        denominator = k3 * q3 * s3

        return pre_factor * numerator / denominator


class J231:
    mu = 2.0
    nu = 3.0
    sigma = 1.0

    @staticmethod
    def analytic(k, q, s):
        if not is_triangle(k, q, s):
            return 0.0

        k_sq = k * k
        q_sq = q * q
        s_sq = s * s

        k3 = k_sq * k

        k4 = k_sq * k_sq
        q4 = q_sq * q_sq
        s4 = s_sq * s_sq

        k6 = k4 * k_sq

        pre_factor = np.pi / 64.0

        numerator = (
            3.0 * k4 * (q_sq + 5.0 * s_sq)
            + (q_sq - s_sq) * (q_sq - s_sq) * (q_sq + 5.0 * s_sq)
            + k_sq * (q4 + 6.0 * q_sq * s_sq - 15.0 * s4)
            - 5.0 * k6
        )
        denominator = k3 * q4 * s_sq

        return pre_factor * numerator / denominator


class Y000:
    mu = 0.0
    nu = 0.0
    sigma = 0.0

    @staticmethod
    def analytic(k, q, s):
        prefactor = (1.0 / 4.0) / (k * q * s)

        numerator = (k - q + s) * (k + q - s)
        denominator = (k + q + s) * (k - q - s)

        return prefactor * np.log(np.abs(numerator / denominator))


class Y022:
    mu = 0.0
    nu = 2.0
    sigma = 2.0

    @staticmethod
    def analytic(k, q, s):
        k_sq = k * k
        q_sq = q * q
        s_sq = s * s

        q3 = q * q_sq
        s3 = s * s_sq

        k4 = k_sq * k_sq
        q4 = q_sq * q_sq
        s4 = s_sq * s_sq

        A = 3.0 * (q_sq + s_sq - k_sq) / 8.0 / (k * q_sq * s_sq)

        B = 3.0 * (k4 + q4 + s4) - 6.0 * (k_sq * q_sq + k_sq * s_sq) + 2.0 * q_sq * s_sq
        C = 32.0 * k * q3 * s3

        numerator = (k - q + s) * (k + q - s)
        denominator = (k + q + s) * (k - q - s)

        return A + (B / C) * np.log(np.abs(numerator / denominator))


Jintegrals = [J000, J110, J220, J222, J231]
Yintegrals = [Y000, Y022]
singularity_eps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

# fixed wavenumber triple used by test_abserr_bounds_truth below: satisfies the triangle
# inequality, no two equal, not a right-angle or degenerate configuration. Matches
# docs/adaptive-levin-benchmark/levin_bench/bessel_tier.py's K0, Q0, S0.
ABSERR_BOUNDS_K, ABSERR_BOUNDS_Q, ABSERR_BOUNDS_S = 1.3, 1.7, 2.1


def intify(x: float):
    return int(round(x, 0))


class Test3BesselAnalytic(unittest.TestCase):

    def test_JJJ(self):
        timestamp = datetime.now().replace(microsecond=0)

        for J in Jintegrals:
            k = uniform(0.1, 5.0)
            q = uniform(0.1, 5.0)
            s = uniform(0.1, 5.0)

            analytic = J.analytic(k, q, s)
            result = compute_3Bessel(
                quad_JJJ,
                J.mu,
                J.nu,
                J.sigma,
                k,
                q,
                s,
                max_x=MAX_X,
                analytic_result=analytic,
                label="JJJ",
                timestamp=timestamp,
            )
            numeric = result.value
            abserr = np.fabs(numeric - analytic)

            if is_triangle(k, q, s):
                relerr = abserr / analytic

                print(f"@@ (J{intify(J.mu)},J{intify(J.nu)},J{intify(J.sigma)}):")
                print(f"   k={k}, q={q}, s={s} satisfies the triangle inequality")
                print(f"   quadrature result = {numeric}")
                print(f"   analytic result = {analytic}")
                print(f"   relerr={relerr:.5g}, abserr={abserr:.5g}")
                print(
                    f"   reported abserr={result.abserr:.5g}, converged={result.converged}, phase_limited={result.phase_limited}"
                )
                self.assertTrue(relerr < REL_TOLERANCE or abserr < ABS_TOLERANCE)

            else:
                print(f"@@ (J{intify(J.mu)},J{intify(J.nu)},J{intify(J.sigma)}):")
                print(
                    f"   k={k}, q={q}, s={s} does not satisfy the triangle inequality"
                )
                print(f"   quadrature result = {numeric}")
                print(f"   analytic result = {analytic}")
                print(f"   abserr={abserr:.5g}")
                print(
                    f"   reported abserr={result.abserr:.5g}, converged={result.converged}, phase_limited={result.phase_limited}"
                )
                self.assertTrue(abserr < ABS_TOLERANCE)

    def test_YJJ(self):
        timestamp = datetime.now().replace(microsecond=0)

        for Y in Yintegrals:
            k = uniform(0.1, 5.0)
            q = uniform(0.1, 5.0)
            s = uniform(0.1, 5.0)

            analytic = Y.analytic(k, q, s)
            result = compute_3Bessel(
                quad_YJJ,
                Y.mu,
                Y.nu,
                Y.sigma,
                k,
                q,
                s,
                max_x=MAX_X,
                analytic_result=analytic,
                label="YJJ",
                timestamp=timestamp,
            )
            numeric = result.value

            abserr = np.fabs(numeric - analytic)
            relerr = abserr / analytic

            print(f"@@ (Y{intify(Y.mu)},J{intify(Y.nu)},J{intify(Y.sigma)}):")
            if is_triangle(k, q, s):
                print(f"   k={k}, q={q}, s={s} satisfies the triangle inequality")
            else:
                print(
                    f"   k={k}, q={q}, s={s} does not satisfy the triangle inequality"
                )

            print(f"   quadrature result = {numeric}")
            print(f"   analytic result = {analytic}")
            print(f"   relerr={relerr:.5g}, abserr={abserr:.5g}")
            print(
                f"   reported abserr={result.abserr:.5g}, converged={result.converged}, phase_limited={result.phase_limited}"
            )
            self.assertTrue(relerr < REL_TOLERANCE or abserr < ABS_TOLERANCE)

    def test_YJJ_log_singularity(self):
        timestamp = datetime.now().replace(microsecond=0)

        for Y in Yintegrals:
            k = uniform(0.1, 5.0)
            q = uniform(0.1, 5.0)

            # k and q do not vary over the eps sweep below, so neither do these two phase
            # objects: build them once per oracle rather than 20 times inside compute_3Bessel.
            mu_phase = bessel_phase(
                Y.mu + 0.5,
                1.075 * k * MAX_X,
                phase_atol=DEFAULT_PHASE_ATOL,
                amplitude_rtol=DEFAULT_AMPLITUDE_RTOL,
            )
            nu_phase = bessel_phase(
                Y.nu + 0.5,
                1.075 * q * MAX_X,
                phase_atol=DEFAULT_PHASE_ATOL,
                amplitude_rtol=DEFAULT_AMPLITUDE_RTOL,
            )

            for eps in singularity_eps:
                s_lo = np.abs(k - q) + eps
                s_hi = k + q - eps

                values = [("lo", s_lo), ("hi", s_hi)]

                for label, s in values:
                    analytic = Y.analytic(k, q, s)
                    result = compute_3Bessel(
                        quad_YJJ,
                        Y.mu,
                        Y.nu,
                        Y.sigma,
                        k,
                        q,
                        s,
                        max_x=MAX_X,
                        analytic_result=analytic,
                        label=f"YJJ_log_eps={eps:.2g}_{label}",
                        timestamp=timestamp,
                        quad_atol=1e-10,
                        quad_rtol=1e-8,
                        mu_phase=mu_phase,
                        nu_phase=nu_phase,
                    )
                    numeric = result.value

                    abserr = np.fabs(numeric - analytic)
                    relerr = abserr / analytic

                    print(
                        f"@@ (Y{intify(Y.mu)},J{intify(Y.nu)},J{intify(Y.sigma)}) eps={eps:.2g} {label}:"
                    )
                    if is_triangle(k, q, s):
                        print(
                            f"   k={k}, q={q}, s={s} satisfies the triangle inequality"
                        )
                    else:
                        print(
                            f"   k={k}, q={q}, s={s} does not satisfy the triangle inequality"
                        )

                    print(f"   quadrature result = {numeric}")
                    print(f"   analytic result = {analytic}")
                    print(f"   relerr={relerr:.5g}, abserr={abserr:.5g}")
                    print(
                        f"   reported abserr={result.abserr:.5g}, converged={result.converged}, phase_limited={result.phase_limited}"
                    )
                    self.assertTrue(
                        relerr < SINGULARITY_REL_TOLERANCE
                        or abserr < SINGULARITY_ABS_TOLERANCE
                    )

    def test_abserr_bounds_truth(self):
        """
        C12 (audit) / prompt 09: does the abserr quad_JJJ/quad_YJJ report actually bound the true
        error against the analytic oracle?

        **It now does, and this test is no longer an expected failure.**

        It used to be. Measured at the fixed triple (ABSERR_BOUNDS_K, ABSERR_BOUNDS_Q,
        ABSERR_BOUNDS_S) = (1.3, 1.7, 2.1), max_x=1e12, atol=1e-14, rtol=1e-10 (the full
        seven-oracle table is in prompts/levin-refactor/logs/09-caller-propagation.md), the
        reported abserr bounded the true error on only 2 of 7 oracles (J000, Y000) and
        underbounded it -- by up to ~11.5x -- on the other 5 (J110, J220, J222, J231, Y022). The
        diagnosis in the levin-refactor campaign was right: "abserr" measures how accurately the
        Levin rule integrated the phase it was *given*, not the accuracy of that phase itself, so
        the ~2e-8 relative fit floor of the old phase and modulus splines was invisible from
        inside the quadrature (see BesselIntegralResult's docstring in three_bessel_integrals.py,
        and README Sec 6 of that campaign). Its stated cure was for the phase construction to
        report its own fit accuracy and for adaptive_levin_sincos's theta_abserr to be wired up
        to consume it -- prompts/levin-refactor/IMPLEMENTATION_STATE.md Sec 3,
        [09-abserr-does-not-bound-phase-spline-floor].

        Both halves have since happened, for this module. prompts/transfer-remedial's prompt 05
        made the construction declare theta_abserr, and its prompt 07 rebuilt these phase groups
        so that every adaptive_levin_sincos call here passes it (summed linearly over the three
        constituents at their own arguments). At the same triple the true error is now 2.5e-14 to
        1.4e-10 relative and the reported abserr bounds it on 7 of 7, with true/reported between
        1.0e-5 and 7.1e-5.

        Note which half did the work. Prompt 05 dropped the *true* error by about eight orders
        (3.0e-10 to 3.8e-14 absolute on J110) while the reported abserr barely moved, so this
        stopped failing before prompt 07 declared anything: measured as an unexpected success on
        the prompt-06 tree. Prompt 07's declaration then made the reported abserr 0-1.9 % larger,
        which keeps it bounding. So the assertion holds for a slightly different reason than the
        cure predicted -- the phase got better rather than merely more honest -- and both changes
        push in the same direction.

        The @unittest.expectedFailure was dropped rather than the assertion weakened: the
        assertion is the thing worth having, and while it was decorated the whole module reported
        FAILED (unexpected successes=1). See prompts/transfer-remedial/IMPLEMENTATION_STATE.md
        [07-abserr-bounds-truth-is-now-an-unexpected-success], which this closes. If it ever
        fails again, the honest reading is that the reported abserr has stopped covering the
        phase error -- not that the tolerance is wrong.
        """
        k, q, s = ABSERR_BOUNDS_K, ABSERR_BOUNDS_Q, ABSERR_BOUNDS_S
        max_x = MAX_X
        quad_atol = 1e-14
        quad_rtol = 1e-10

        cases = [("JJJ", quad_JJJ, J) for J in Jintegrals] + [
            ("YJJ", quad_YJJ, Y) for Y in Yintegrals
        ]

        # gather every oracle's measurement before asserting, so a failure on an early oracle
        # does not suppress the diagnostic print for the later ones, and report all of them in
        # one message. This shape was originally chosen for @unittest.expectedFailure, which
        # expects exactly one failure from the whole test; the decorator is gone but the shape is
        # still the right one, because "which oracles underbound, and by how much" is the finding
        failures = []
        for kind, evaluator, integral_class in cases:
            mu_phase = bessel_phase(
                integral_class.mu + 0.5,
                1.075 * k * max_x,
                phase_atol=DEFAULT_PHASE_ATOL,
                amplitude_rtol=DEFAULT_AMPLITUDE_RTOL,
            )
            nu_phase = bessel_phase(
                integral_class.nu + 0.5,
                1.075 * q * max_x,
                phase_atol=DEFAULT_PHASE_ATOL,
                amplitude_rtol=DEFAULT_AMPLITUDE_RTOL,
            )
            sigma_phase = bessel_phase(
                integral_class.sigma + 0.5,
                1.075 * s * max_x,
                phase_atol=DEFAULT_PHASE_ATOL,
                amplitude_rtol=DEFAULT_AMPLITUDE_RTOL,
            )

            analytic = integral_class.analytic(k, q, s)
            result = evaluator(
                mu_phase,
                nu_phase,
                sigma_phase,
                integral_class.mu,
                integral_class.nu,
                integral_class.sigma,
                k,
                q,
                s,
                max_x,
                atol=quad_atol,
                rtol=quad_rtol,
            )

            true_abserr = np.fabs(result.value - analytic)
            ratio = true_abserr / result.abserr if result.abserr > 0 else np.inf
            bounds = true_abserr <= result.abserr

            print(
                f"@@ ({kind} {intify(integral_class.mu)},{intify(integral_class.nu)},{intify(integral_class.sigma)}): "
                f"true abserr={true_abserr:.5g}, reported abserr={result.abserr:.5g}, "
                f"ratio (true/reported)={ratio:.5g}, bounds={bounds}"
            )
            if not bounds:
                failures.append(
                    f"{kind}({integral_class.mu},{integral_class.nu},{integral_class.sigma}): "
                    f"reported abserr={result.abserr:.5g} < true abserr={true_abserr:.5g} "
                    f"(ratio={ratio:.5g})"
                )

        self.assertEqual(
            failures,
            [],
            msg=(
                "reported abserr did not bound the true error against the analytic oracle on "
                f"{len(failures)}/{len(cases)} cases -- it did on 7 of 7 when this assertion was "
                "last measured, so this is a regression in the declared phase error or in the "
                "quadrature, not a tolerance to relax (see this test's docstring): "
                + "; ".join(failures)
            ),
        )

    @unittest.skipUnless(
        DIAGNOSTIC_PLOTS,
        "diagnostic only; set THREE_BESSEL_DIAGNOSTIC_PLOTS to run it",
    )
    def test_YJJ_log_scaling(
        self,
        max_x=MAX_X,
        phase_atol=DEFAULT_PHASE_ATOL,
        amplitude_rtol=DEFAULT_AMPLITUDE_RTOL,
        quad_atol=1e-10,
        quad_rtol=1e-8,
    ):
        """
        How the near-singular YJJ integrals and their closed forms scale as eps -> 0.

        **This makes no assertion of any kind** -- it evaluates 40 near-singular integrals and
        draws four figures from them, and it reports a pass whatever those numbers are. It is a
        diagnostic that was carrying a `test_` prefix, so `unittest` ran it on every discovery
        for a result that could not fail. It is kept, under the same
        THREE_BESSEL_DIAGNOSTIC_PLOTS switch as the convergence figures, because the scaling it
        shows is worth being able to look at -- but it is skipped by default.

        The assertion that *does* cover this region, with the tolerances sized for it, is
        `test_YJJ_log_singularity` above.
        """
        import seaborn as sns
        from matplotlib import pyplot as plt

        timestamp = datetime.now().replace(microsecond=0)

        sns.set_theme()

        for Y in Yintegrals:
            k = uniform(0.1, 5.0)
            q = uniform(0.1, 5.0)

            mu_phase = bessel_phase(
                Y.mu + 0.5,
                1.075 * k * max_x,
                phase_atol=phase_atol,
                amplitude_rtol=amplitude_rtol,
            )
            nu_phase = bessel_phase(
                Y.nu + 0.5,
                1.075 * q * max_x,
                phase_atol=phase_atol,
                amplitude_rtol=amplitude_rtol,
            )

            configs = [
                ("lo", lambda k, q, eps: np.abs(k - q) + eps),
                ("hi", lambda k, q, eps: k + q - eps),
            ]

            for label, s_evaluator in configs:
                print(
                    f"@@ (Y{intify(Y.mu)},J{intify(Y.nu)},J{intify(Y.sigma)}) {label}:"
                )

                total_start = time.perf_counter()
                last_notify = total_start

                analytic_grid = []
                numeric_grid = []
                elapsed_grid = []

                for eps in singularity_eps:
                    start = time.perf_counter()

                    s = s_evaluator(k, q, eps)

                    sigma_phase = bessel_phase(
                        Y.sigma + 0.5,
                        1.075 * s * max_x,
                        phase_atol=phase_atol,
                        amplitude_rtol=amplitude_rtol,
                    )

                    analytic = Y.analytic(k, q, s)
                    result = quad_YJJ(
                        mu_phase,
                        nu_phase,
                        sigma_phase,
                        Y.mu,
                        Y.nu,
                        Y.sigma,
                        k,
                        q,
                        s,
                        max_x,
                        atol=quad_atol,
                        rtol=quad_rtol,
                    )
                    numeric = result.value

                    stop = time.perf_counter()

                    analytic_grid.append(analytic)
                    numeric_grid.append(numeric)

                    elapsed = stop - start
                    elapsed_grid.append(elapsed)

                    last_notify_elapsed = stop - last_notify
                    total_elapsed = stop - total_start
                    if elapsed > 2 * 60 or last_notify_elapsed > 5 * 60:
                        print(
                            f"   -- evaluated eps={eps:.3g} (value={numeric:.5g}) in time {format_time(elapsed)} (total time for this YJJ {format_time(total_elapsed)})"
                        )
                        last_notify = stop

                fig = plt.figure()
                ax = plt.gca()

                ax.plot(
                    singularity_eps, numeric_grid, color="b", label="Numeric + Levin"
                )
                ax.plot(
                    singularity_eps,
                    analytic_grid,
                    color="r",
                    linestyle=(0, (1, 1)),
                    label="Analytic",
                )

                ax.text(
                    0.0, 1.03, f"k={k:.5g}", transform=ax.transAxes, fontsize="x-small"
                )
                ax.text(
                    0.3, 1.03, f"q={q:.5g}", transform=ax.transAxes, fontsize="x-small"
                )
                ax.text(
                    0.6, 1.03, f"s={s:.5g}", transform=ax.transAxes, fontsize="x-small"
                )
                ax.text(
                    0.0,
                    1.08,
                    f"$\\mu$={Y.mu:.5g}",
                    transform=ax.transAxes,
                    fontsize="x-small",
                )
                ax.text(
                    0.3,
                    1.08,
                    f"$\\nu$={Y.nu:.5g}",
                    transform=ax.transAxes,
                    fontsize="x-small",
                )
                ax.text(
                    0.6,
                    1.08,
                    f"$\\sigma$={Y.sigma:.5g}",
                    transform=ax.transAxes,
                    fontsize="x-small",
                )

                ax.set_xscale("log")
                ax.set_yscale("linear")
                ax.xaxis.set_inverted(True)
                ax.set_xlabel("$\epsilon$")
                ax.legend(loc="best")
                ax.grid(True)

                fig_path = Path(
                    f"test_3bessel_analytic/{timestamp.isoformat()}/eps_dependence_{label}_mu={Y.mu:.3f}_nu={Y.nu:.3f}_sigma={Y.sigma:.3f}_k={k:.3f}_q={q:.3f}_s={s:.3f}_maxx={max_x:.5g}.pdf"
                ).resolve()
                fig_path.parents[0].mkdir(parents=True, exist_ok=True)
                fig.savefig(fig_path)
                fig.savefig(fig_path.with_suffix(".png"))

                plt.close()

                fig = plt.figure()
                ax = plt.gca()

                ax.plot(
                    singularity_eps, elapsed_grid, color="b", label="Integration time"
                )

                ax.text(
                    0.0, 1.03, f"k={k:.5g}", transform=ax.transAxes, fontsize="x-small"
                )
                ax.text(
                    0.3, 1.03, f"q={q:.5g}", transform=ax.transAxes, fontsize="x-small"
                )
                ax.text(
                    0.6, 1.03, f"s={s:.5g}", transform=ax.transAxes, fontsize="x-small"
                )
                ax.text(
                    0.0,
                    1.08,
                    f"$\\mu$={Y.mu:.5g}",
                    transform=ax.transAxes,
                    fontsize="x-small",
                )
                ax.text(
                    0.3,
                    1.08,
                    f"$\\nu$={Y.nu:.5g}",
                    transform=ax.transAxes,
                    fontsize="x-small",
                )
                ax.text(
                    0.6,
                    1.08,
                    f"$\\sigma$={Y.sigma:.5g}",
                    transform=ax.transAxes,
                    fontsize="x-small",
                )

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.xaxis.set_inverted(True)
                ax.set_xlabel("$\epsilon$")
                ax.legend(loc="best")
                ax.grid(True)

                fig_path = Path(
                    f"test_3bessel_analytic/{timestamp.isoformat()}/integration_time_{label}_mu={Y.mu:.3f}_nu={Y.nu:.3f}_sigma={Y.sigma:.3f}_k={k:.3f}_q={q:.3f}_s={s:.3f}_maxx={max_x:.5g}.pdf"
                ).resolve()
                fig_path.parents[0].mkdir(parents=True, exist_ok=True)
                fig.savefig(fig_path)
                fig.savefig(fig_path.with_suffix(".png"))

                total_stop = time.perf_counter()
                total_elapsed = total_stop - total_start

                print(f"   >> completed plot in time {format_time(total_elapsed)}")
