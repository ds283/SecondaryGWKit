"""
Prototype of the shared conformal-time primitive (review §7 and §13.3), measured for accuracy
**and throughput**, outside the production tree.

Run from the repository root:

    PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/prototype_primitive.py

This implements the design prompt 03 will move into ``ComputeTargets/cumulative_table.py``:

* nodes ``u_i = log(1 + z_i)`` on the production source grid;
* per-interval Gauss-Legendre of order ``N`` in ``u`` of the integrand ``f(z) e^u``, for
  ``f = 1/H``, ``c_s/H`` and ``(3/2)(1 + c_s^2)/(1+z)``;
* cumulative sums stored as **(hi, lo)** pairs -- ``hi = fsum(terms)``,
  ``lo = fsum([*terms, -hi])``;
* a pointwise accessor ``value(z)`` = nearest node ``(hi + lo)`` plus a local Gauss partial over
  at most half a grid interval;
* an interval accessor ``delta(z_a, z_b)`` = ``partial(a -> node) + [(hi_a - hi_b) +
  (lo_a - lo_b)] + partial(node -> b)``, **never** a difference of two pointwise values;
* a control variant with a single-double table, to show the floor a pointwise accessor cannot
  escape.

The table is anchored at the **top** node, so ``value(z) = tau(z) - tau(z_top)``, which is
exactly what ``wkb_reference_data.json`` records, and
``delta(z_a, z_b) = value(z_b) - value(z_a) = int_{z_b}^{z_a} dz/H`` -- the README §2 (c)
convention.

Output goes to ``docs/gktk-remedial/PROTOTYPE-MEASUREMENTS.md`` (by hand, from this script's
stdout) and to the campaign log.
"""

import os
import sys
import time
from math import fsum, log1p, expm1

import numpy as np
from scipy.interpolate import make_interp_spline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import reference_lib as R  # noqa: E402

from ComputeTargets.tests.wkb_reference import (  # noqa: E402
    LambdaCDMModel,
    PRODUCTION_LARGEST_K_INV_MPC,
    PRODUCTION_SUPERHORIZON_EFOLDS,
    QCDModel,
    horizon_exit_z,
    load_references,
    production_source_grid,
)

# the wavenumber at which a relative error of tau is converted to a phase error
PHASE_K = 3.0e8

# review §7's "nearest node + local 8-point Gauss"
DEFAULT_PARTIAL_ORDER = 8

BUILD_ORDERS = (4, 8, 12)

THROUGHPUT_CALLS = 100_000
THROUGHPUT_REPEATS = 3
OFF_GRID_POINTS = 25


class CountingIntegrand:
    """Wrap an integrand in ``u`` and count the calls (= Hubble evaluations, one per node)."""

    def __init__(self, f):
        self._f = f
        self.calls = 0

    def __call__(self, u):
        self.calls += 1
        return self._f(u)

    def reset(self):
        self.calls = 0


class CumulativeTablePrototype:
    """
    Gauss-Legendre-on-the-grid cumulative table with a pointwise and an interval accessor.

    :param z_nodes: the production grid, descending in z
    :param integrand_u: the integrand in ``u = log(1+z)`` (already carrying the ``e^u`` Jacobian)
    :param order: Gauss-Legendre order used per production interval when building the table
    :param partial_order: Gauss-Legendre order used for the off-grid partial
    :param double_double: if ``False``, the low-order limb is discarded (the control variant)
    """

    def __init__(
        self,
        z_nodes,
        integrand_u,
        order: int,
        partial_order: int = DEFAULT_PARTIAL_ORDER,
        double_double: bool = True,
    ):
        self._f = integrand_u
        self._double_double = double_double

        # store ascending in u so that searchsorted works; index 0 is the lowest redshift
        self.u = np.log1p(np.asarray(z_nodes, dtype=float))[::-1].copy()
        self.n = len(self.u)

        self._x, self._w = np.polynomial.legendre.leggauss(order)
        self._xp, self._wp = np.polynomial.legendre.leggauss(partial_order)
        self.order = order
        self.partial_order = partial_order

        increments = [
            self._panel(self.u[i], self.u[i + 1], self._x, self._w)
            for i in range(self.n - 1)
        ]
        self.build_integrand_calls = order * (self.n - 1)

        # anchored at the TOP node (largest u): hi[i] = int_{u_i}^{u_top}
        self.hi = np.empty(self.n)
        self.lo = np.zeros(self.n)
        for i in range(self.n):
            terms = increments[i:]
            hi = fsum(terms)
            self.hi[i] = hi
            if double_double:
                self.lo[i] = fsum([*terms, -hi])

    def _panel(self, a: float, b: float, x, w) -> float:
        half = 0.5 * (b - a)
        mid = 0.5 * (a + b)
        return half * fsum(wi * self._f(mid + half * xi) for xi, wi in zip(x, w))

    def _nearest(self, u: float) -> int:
        i = int(np.searchsorted(self.u, u))
        if i <= 0:
            return 0
        if i >= self.n:
            return self.n - 1
        return i if (self.u[i] - u) < (u - self.u[i - 1]) else i - 1

    def _partial(self, u: float, node: int) -> float:
        """``int_u^{u_node}`` of the integrand; exactly zero, and free, for an on-grid ``u``."""
        u_node = self.u[node]
        if u == u_node:
            return 0.0
        return self._panel(u, u_node, self._xp, self._wp)

    def value(self, z: float) -> float:
        """``tau(z) - tau(z_top)``, as a plain float."""
        u = log1p(z)
        node = self._nearest(u)
        return (self.hi[node] + self.lo[node]) + self._partial(u, node)

    def delta(self, z_a: float, z_b: float) -> float:
        """
        ``tau(z_b) - tau(z_a) = int_{z_b}^{z_a} dz/H``, positive when ``z_b < z_a``
        (README §2 (c)). Formed as partial + double-double table difference + partial, never as
        a difference of two pointwise values.
        """
        u_a = log1p(z_a)
        u_b = log1p(z_b)
        n_a = self._nearest(u_a)
        n_b = self._nearest(u_b)

        table = (self.hi[n_b] - self.hi[n_a]) + (self.lo[n_b] - self.lo[n_a])
        return table + self._partial(u_b, n_b) - self._partial(u_a, n_a)


def build_grid():
    lam = LambdaCDMModel()
    z_init = horizon_exit_z(
        lam.cosmology, PRODUCTION_LARGEST_K_INV_MPC, -PRODUCTION_SUPERHORIZON_EFOLDS
    )
    grid = production_source_grid(z_init)
    z_nodes = np.array([z.z for z in grid], dtype=float)
    return lam, grid, z_nodes


def off_grid_reference(functions, mp_bg, z_points, z_nodes):
    """
    Reference values of ``tau(z) - tau(z_top)`` at off-grid points: mpmath for ``LambdaCDMModel``,
    the converged adaptive quadrature of the double-precision integrand for ``QCDModel``.

    The adaptive branch integrates over the *production interval edges* above the point, plus one
    partial interval, exactly as the JSON references are built -- a single ``quad`` call across
    sixteen decades of dynamic range would not converge.
    """
    if mp_bg is not None:
        z_top = float(z_nodes[0])
        return [float(R.mp_increment(mp_bg, "tau", z, z_top)) for z in z_points]

    f = R.integrand_tau_double(functions)
    u_ascending = np.log1p(np.asarray(z_nodes, dtype=float))[::-1]
    out = []
    for z in z_points:
        u = log1p(z)
        i = int(np.searchsorted(u_ascending, u))
        edges = np.concatenate([[u], u_ascending[i:]])
        val, _, _, _ = R.quad_sum_over_intervals(f, edges, 1.5e-14)
        out.append(val)
    return out


def measure(model_name, functions, z_nodes, references, mp_bg):
    print(f"\n{'=' * 92}\n{model_name}\n{'=' * 92}")

    checkpoints = [c["z"] for c in references["checkpoints"]]
    tau_reference = references["tau_minus_top"]
    z_top = float(z_nodes[0])

    results = {}

    # ---------------------------------------------------------------------------------------
    # build cost and node accuracy, at each Gauss order
    # ---------------------------------------------------------------------------------------
    print(
        f"\n{'order':>6} {'H evals':>10} {'build [s]':>10} "
        f"{'max rel err at nodes':>22} {'as rad at k=3e8':>17}"
    )
    tables = {}
    for order in BUILD_ORDERS:
        counter = CountingIntegrand(R.integrand_tau_double(functions))
        t0 = time.perf_counter()
        table = CumulativeTablePrototype(z_nodes, counter, order)
        elapsed = time.perf_counter() - t0
        tables[order] = table

        worst = 0.0
        worst_z = None
        for z, reference in zip(checkpoints, tau_reference):
            if reference == 0.0:
                continue
            err = abs(table.value(z) - reference) / abs(reference)
            if err > worst:
                worst, worst_z = err, z
        results[("node", order)] = (worst, worst_z)
        print(
            f"{order:>6} {counter.calls:>10} {elapsed:>10.3f} "
            f"{worst:>22.2e} {worst * PHASE_K * abs(tau_reference[-1]):>17.2e}"
        )
        print(f"       (worst at z = {worst_z:.5g})")

    # ---------------------------------------------------------------------------------------
    # off-grid accuracy: 25 random points inside random intervals
    # ---------------------------------------------------------------------------------------
    rng = np.random.default_rng(20260910)
    picks = rng.integers(2, len(z_nodes) - 3, OFF_GRID_POINTS)
    fracs = rng.uniform(0.05, 0.95, OFF_GRID_POINTS)
    z_off = []
    for j, frac in zip(picks, fracs):
        z_off.append(R.fractional_point(float(z_nodes[j]), float(z_nodes[j + 1]), frac))

    t0 = time.perf_counter()
    off_reference = off_grid_reference(functions, mp_bg, z_off, z_nodes)
    print(f"\noff-grid references built in {time.perf_counter() - t0:.1f} s")

    print(
        f"\n{'order':>6} {'max rel err off-grid':>22} {'as rad at k=3e8':>17}  "
        f"(cubic spline of the nodes, for comparison)"
    )
    cubic = make_interp_spline(tables[12].u, tables[12].hi, k=3)
    quintic = make_interp_spline(tables[12].u, tables[12].hi, k=5)
    for order in BUILD_ORDERS:
        table = tables[order]
        worst = 0.0
        for z, reference in zip(z_off, off_reference):
            worst = max(worst, abs(table.value(z) - reference) / abs(reference))
        results[("offgrid", order)] = worst
        print(f"{order:>6} {worst:>22.2e} {worst * PHASE_K * 1.4e4:>17.2e}")

    worst_cubic = max(
        abs(float(cubic(log1p(z))) - reference) / abs(reference)
        for z, reference in zip(z_off, off_reference)
    )
    worst_quintic = max(
        abs(float(quintic(log1p(z))) - reference) / abs(reference)
        for z, reference in zip(z_off, off_reference)
    )
    print(
        f"{'cubic':>6} {worst_cubic:>22.2e} {worst_cubic * PHASE_K * 1.4e4:>17.2e}\n"
        f"{'quint':>6} {worst_quintic:>22.2e} {worst_quintic * PHASE_K * 1.4e4:>17.2e}"
    )
    results["cubic"] = worst_cubic
    results["quintic"] = worst_quintic

    # ---------------------------------------------------------------------------------------
    # short baselines: double-double table against the single-double control
    # ---------------------------------------------------------------------------------------
    counter_dd = CountingIntegrand(R.integrand_tau_double(functions))
    dd = CumulativeTablePrototype(z_nodes, counter_dd, 4, double_double=True)
    counter_sd = CountingIntegrand(R.integrand_tau_double(functions))
    sd = CumulativeTablePrototype(z_nodes, counter_sd, 4, double_double=False)

    print(
        f"\n{'z_node_hi':>12} {'baseline':>10} {'Delta tau':>14} "
        f"{'dd rel err':>12} {'dd [rad]':>10} {'sd rel err':>12} {'sd [rad]':>10}"
    )
    short_rows = []
    for record in references["short_baseline"]:
        for label, z_lo, reference in (
            ("full", record["z_node_lo"], record["delta_tau_full"]),
            ("37%", record["z_fraction"], record["delta_tau_fraction"]),
        ):
            z_hi = record["z_node_hi"]
            got_dd = dd.delta(z_hi, z_lo)
            got_sd = sd.delta(z_hi, z_lo)
            err_dd = abs(got_dd - reference) / abs(reference)
            err_sd = abs(got_sd - reference) / abs(reference)
            print(
                f"{z_hi:>12.5g} {label:>10} {reference:>14.7g} "
                f"{err_dd:>12.2e} {err_dd * PHASE_K * abs(reference):>10.2e} "
                f"{err_sd:>12.2e} {err_sd * PHASE_K * abs(reference):>10.2e}"
            )
            short_rows.append((z_hi, label, reference, err_dd, err_sd))
    results["short"] = short_rows

    # ---------------------------------------------------------------------------------------
    # the review §13.3 demonstration: a baseline far shorter than one grid interval, where a
    # pointwise accessor cannot help however accurate it is. Compared three ways:
    #   dd  -- the design: double-double table difference plus two partials
    #   sd  -- the single-double control table
    #   pw  -- value(z_b) - value(z_a), the pointwise-difference anti-pattern
    # ---------------------------------------------------------------------------------------
    print(
        f"\nvery short baselines (fractions of one grid interval near z = 1, where "
        f"tau ~ 1e4 Mpc)\n"
        f"{'width/interval':>15} {'Delta tau [Mpc]':>16} {'dd [rad]':>11} "
        f"{'sd [rad]':>11} {'pointwise [rad]':>16}"
    )
    j = R.short_baseline_locations(z_nodes, targets=(1.0,))[0]
    z_hi = float(z_nodes[j])
    z_lo_node = float(z_nodes[j + 1])
    tiny_rows = []
    for width in (1.0e-1, 1.0e-3, 1.0e-5):
        z_lo = R.fractional_point(z_hi, z_lo_node, width)
        if mp_bg is not None:
            reference = float(R.mp_increment(mp_bg, "tau", z_lo, z_hi))
        else:
            reference, _, _, _ = R.quad_sum_over_intervals(
                R.integrand_tau_double(functions),
                np.array([log1p(z_lo), log1p(z_hi)]),
                1.5e-14,
            )
        got_dd = dd.delta(z_hi, z_lo)
        got_sd = sd.delta(z_hi, z_lo)
        got_pw = dd.value(z_lo) - dd.value(z_hi)
        row = (
            width,
            reference,
            abs(got_dd - reference) * PHASE_K,
            abs(got_sd - reference) * PHASE_K,
            abs(got_pw - reference) * PHASE_K,
        )
        print(
            f"{row[0]:>15.0e} {row[1]:>16.8g} {row[2]:>11.2e} "
            f"{row[3]:>11.2e} {row[4]:>16.2e}"
        )
        tiny_rows.append(row)
    results["tiny"] = tiny_rows

    # ---------------------------------------------------------------------------------------
    # per-interval convergence of the fixed-order Gauss rule (review §11's open test)
    # ---------------------------------------------------------------------------------------
    print(
        f"\nper-interval fixed-order Gauss convergence against the adaptive reference\n"
        f"{'order':>6} {'intervals > 1e-12':>18} {'worst interval':>15} "
        f"{'at z':>11} {'abs [Mpc]':>11} {'as rad at k=3e8':>16}"
    )
    f_double = R.integrand_tau_double(functions)
    u_ascending = np.log1p(np.asarray(z_nodes, dtype=float))[::-1]
    _, quad_parts, _, _ = R.quad_sum_over_intervals(f_double, u_ascending, 1.5e-14)
    interval_rows = []
    for order in BUILD_ORDERS + (20,):
        x, w = np.polynomial.legendre.leggauss(order)
        worst = 0.0
        worst_index = 0
        count = 0
        for i, reference in enumerate(quad_parts):
            a, b = u_ascending[i], u_ascending[i + 1]
            half = 0.5 * (b - a)
            mid = 0.5 * (a + b)
            got = half * fsum(wi * f_double(mid + half * xi) for xi, wi in zip(x, w))
            err = abs(got - reference) / abs(reference)
            if err > 1.0e-12:
                count += 1
            if err > worst:
                worst, worst_index = err, i
        z_worst = expm1(u_ascending[worst_index])
        absolute = worst * abs(quad_parts[worst_index])
        print(
            f"{order:>6} {count:>18} {worst:>15.2e} {z_worst:>11.4g} "
            f"{absolute:>11.2e} {absolute * PHASE_K:>16.2e}"
        )
        interval_rows.append((order, count, worst, z_worst, absolute))
    results["intervals"] = interval_rows

    # ---------------------------------------------------------------------------------------
    # throughput. Timed on a table built with the RAW integrand (the counting wrapper adds a
    # Python frame per Hubble evaluation, i.e. 16 per off-grid call); the call counts come from
    # a separate counted pass. Each timing is the best of THROUGHPUT_REPEATS runs, to suppress
    # machine noise.
    # ---------------------------------------------------------------------------------------
    raw = R.integrand_tau_double(functions)
    table = CumulativeTablePrototype(z_nodes, raw, 4)
    counted = tables[4]
    counter = counted._f
    n = table.n

    rng = np.random.default_rng(1)
    z_grid_a = [float(z) for z in rng.choice(z_nodes, THROUGHPUT_CALLS)]
    z_grid_b = [float(z) for z in rng.choice(z_nodes, THROUGHPUT_CALLS)]

    def _off_grid_sample():
        return [
            R.fractional_point(float(z_nodes[j]), float(z_nodes[j + 1]), f)
            for j, f in zip(
                rng.integers(1, len(z_nodes) - 2, THROUGHPUT_CALLS),
                rng.uniform(0.05, 0.95, THROUGHPUT_CALLS),
            )
        ]

    off_a = _off_grid_sample()
    off_b = _off_grid_sample()

    def _time(fn, args):
        best = float("inf")
        for _ in range(THROUGHPUT_REPEATS):
            t0 = time.perf_counter()
            for a in args:
                fn(a)
            best = min(best, time.perf_counter() - t0)
        return best / len(args) * 1e6

    t_delta_on = _time(lambda p: table.delta(*p), list(zip(z_grid_a, z_grid_b)))
    t_delta_off = _time(lambda p: table.delta(*p), list(zip(off_a, off_b)))
    t_delta_mixed = _time(lambda p: table.delta(*p), list(zip(z_grid_a, off_b)))
    t_value_on = _time(table.value, z_grid_a)
    t_value_off = _time(table.value, off_a)

    spline = make_interp_spline(table.u, table.hi, k=3)
    t_spline = _time(lambda a: float(spline(log1p(a))), off_a)

    counter.reset()
    for a, b in zip(z_grid_a, z_grid_b):
        counted.delta(a, b)
    calls_on = counter.calls
    counter.reset()
    for a, b in zip(off_a, off_b):
        counted.delta(a, b)
    calls_off = counter.calls

    print(
        f"\nthroughput ({THROUGHPUT_CALLS} calls, best of {THROUGHPUT_REPEATS}; "
        "build order 4, partial order 8)"
    )
    print(
        f"  delta, both endpoints on-grid    {t_delta_on:8.2f} us   "
        f"integrand calls: {calls_on}"
    )
    print(
        f"  delta, both endpoints off-grid   {t_delta_off:8.2f} us   "
        f"integrand calls: {calls_off} ({calls_off / THROUGHPUT_CALLS:.1f}/call)"
    )
    print(f"  delta, one on / one off-grid     {t_delta_mixed:8.2f} us")
    print(f"  value, on-grid                   {t_value_on:8.2f} us")
    print(f"  value, off-grid                  {t_value_off:8.2f} us")
    print(
        f"  cubic spline lookup (comparator) {t_spline:8.2f} us   "
        f"ratio off-grid delta / spline = {t_delta_off / t_spline:.0f}x"
    )
    results["throughput"] = {
        "delta_on_grid_us": t_delta_on,
        "delta_off_grid_us": t_delta_off,
        "delta_mixed_us": t_delta_mixed,
        "value_on_grid_us": t_value_on,
        "value_off_grid_us": t_value_off,
        "spline_us": t_spline,
        "on_grid_integrand_calls": calls_on,
        "off_grid_integrand_calls": calls_off,
    }

    if calls_on != 0:
        print("  !! on-grid calls are NOT short-circuited")
    return results


def main():
    references = load_references()

    lam, grid, z_nodes = build_grid()
    print(
        f"production source grid: {len(z_nodes)} nodes, "
        f"z = {z_nodes[0]:.6g} .. {z_nodes[-1]:.6g}"
    )

    t0 = time.perf_counter()
    qcd = QCDModel(grid)
    qcd_build = time.perf_counter() - t0
    print(f"QCDModel construction (compute_background + splines): {qcd_build:.3f} s")

    lam_results = measure(
        "LambdaCDMModel",
        lam.functions,
        z_nodes,
        references["models"]["LambdaCDMModel"],
        R.MpLambdaCDM(lam.cosmology),
    )
    qcd_results = measure(
        "QCDModel",
        qcd.functions,
        z_nodes,
        references["models"]["QCDModel"],
        None,
    )

    print(f"\n{'=' * 92}\nSTOP-CONDITION CHECK (README §4.3)\n{'=' * 92}")
    lam_order4 = lam_results[("node", 4)][0]
    qcd_order4 = qcd_results[("node", 4)][0]
    qcd_order8 = qcd_results[("node", 8)][0]
    print(
        f"LambdaCDM tau at production nodes, Gauss order 4: {lam_order4:.2e} relative "
        f"({'PASS' if lam_order4 <= 2.0e-14 else 'FAIL'} against the <= 2e-14 acceptance)"
    )
    print(
        f"QCD tau at production nodes: order 4 {qcd_order4:.2e}, order 8 {qcd_order8:.2e} "
        f"relative; they disagree by {abs(qcd_order4 - qcd_order8):.2e} "
        f"({'ABOVE' if abs(qcd_order4 - qcd_order8) > 1e-13 else 'below'} the 1e-13 "
        "report-prominently threshold of prompt 01 §4)"
    )
    cost = qcd_results["throughput"]["delta_off_grid_us"]
    print(
        f"interval accessor, both endpoints off-grid, QCDModel: {cost:.1f} us per call "
        f"({'ABOVE' if cost > 50.0 else 'below'} the 50 us stop threshold)"
    )
    print(
        "  one endpoint off-grid: "
        f"{qcd_results['throughput']['delta_mixed_us']:.1f} us per call"
    )
    print(
        "  both endpoints on-grid (the production case, review §13.3): "
        f"{qcd_results['throughput']['delta_on_grid_us']:.2f} us per call"
    )


if __name__ == "__main__":
    main()
