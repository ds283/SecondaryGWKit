"""
The Bessel group phase's rounding (prompts/levin-phase-floor/06h-group-phase-rounding.md).

``BesselPhaseGroup.sin_cos`` carries the rounding of the product ``fl(K) x`` (TwoProduct) and of
``K`` itself (``K_lo``) exactly in its small angle, and ``BesselPhaseGroup.theta_abserr`` declares
what remains: the rounding of the abscissa ``x = fl(exp(u))`` and the group's own evaluation
(``phi_eval``). These tests check each claim against ``mpmath`` at 30 digits (50 where a check of
the reference is asked), built independently of the objects under test.

**The reference for a phase is at the exact e^u** where the declaration is scored as a function of
u (test 3): ``mpmath.exp`` of the float ``u``, then each constituent's phase at ``m_i e^u`` exactly,
composed with the group's signs. A constituent's reference phase is ``atan2(J, -Y)``; the method
check (test 7) confirms it equal to ``arg H^(1) + pi/2``. Where the represented phase at the
evaluated abscissa is scored (tests 1 and 2), the reference is at the float ``x = np.exp(u)``.
Every comparison is of angles modulo 2 pi.

**The parent** is the dispatch commit's module (``git show 2c325f5:...``, imported under another
name), i.e. 06e's code, in which ``sin_cos`` rounded ``K x`` into one double and ``theta_abserr``
was the constituents' sum. Its numbers are printed beside 06h's, and quoted in the comments.

**The cells** are seeded grids in u over:
- test_three_bessel.py's ``GROUP_CASES`` (exact, near and absent cancellation), to x = 1e12;
- test_quadsource_integral.py's ``TestThreeBesselPhaseGroups`` coefficients, to its x = 1e9;
- 06f's two violating calls' groups, ``g-1e-7-0-12689`` #19 (J4) and ``g-1e-7-3-1054`` #20 (Y4),
  by their coefficients, signs, orders and spans (docs/levin-phase-floor/06f/out/replay/), with
  production's phase_atol and amplitude_rtol, and with each call's three worst nodes of 06f's M2;
- a case with ``fl(K) != K``.

No Ray, no datastore.
"""

import hashlib
import importlib.util
import math
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

from AdaptiveLevin.levin_quadrature import _detect_vectorized
from LiouvilleGreen import three_bessel_integrals as tb
from LiouvilleGreen.bessel_phase import bessel_phase
from LiouvilleGreen.three_bessel_integrals import BesselPhaseGroup

REPO = Path(__file__).resolve().parents[2]

#: 06h's dispatch commit, whose three_bessel_integrals.py is the parent (06e's code, f4d8e17).
PARENT_SHA = "2c325f56641666b45df483adb8c1202575721c42"

REFERENCE_DPS = 30
CHECK_DPS = 50
U = 2.0**-53

#: Points per cell, drawn uniformly in u from a seeded generator, plus the span's two ends.
POINTS_PER_CELL = 48
GRID_SEED = 20261031

GROUP_ORDERS = (0.5, 1.5, 2.5)
GROUP_SIGNS = (1.0, -1.0, -1.0)
GROUP_X_MAX = 1.0e12
#: test_three_bessel.GROUP_CASES, copied (that module imports Ray-free but heavy dependencies).
GROUP_CASES = {
    "K = 0 exactly": (2.0, 1.5, 0.5),
    "K/max ~ 1e-6": (2.0 + 2.0**-20, 1.5, 0.5),
    "K/max ~ 1e-10": (2.0 + 2.0**-33, 1.5, 0.5),
    "generic": (2.1, 1.5, 0.5),
}
#: fl(K) != K: K = 2 - fl(0.3) - fl(0.1) is not representable, fl(K) = 1.6 and
#: K_lo = -8.33e-17, so K_lo x reaches 8.3e-5 rad at x = 1e12.
K_INEXACT_CASE = (2.0, 0.3, 0.1)

#: Three non-zero residuals: no order 1/2, whose residual is identically zero and so leaves two
#: terms in every other cell's residual sum. test_3bessel_analytic.py's J231 oracle uses these
#: orders. With three non-zero terms, Python's sum() (compensated for exact floats since 3.12) and
#: an array sum can differ in the last bit, which test 6 must see if it happens.
THREE_RESIDUALS_ORDERS = (2.5, 3.5, 1.5)
THREE_RESIDUALS_CASE = (1.3, 1.7, 2.1)

#: test_quadsource_integral's TestThreeBesselPhaseGroups: b = 0, orders (1/2, 5/2, 5/2), signs
#: (+, -, -), and (k, q c_s, r c_s) from its _phase_group_coefficients(), copied.
QSI_ORDERS = (0.5, 2.5, 2.5)
QSI_X_MAX = 1.0e9
_CS = math.sqrt(1.0 / 3.0)
QSI_CASES = {
    "near-resonant": (1.0e4 * _CS + 1.2e4 * _CS, 1.0e4 * _CS, 1.2e4 * _CS),
    "generic": (1.0e4, 1.0e4 * _CS, 1.2e4 * _CS),
}

#: 06f's two violating calls (docs/levin-phase-floor/06f/out/replay/*/result.json, M2.group and
#: x_span; nodes.json.gz for the three worst nodes, each with the theta_mod_2pi 06f recorded there
#: at the parent's code). Orders (1/2, 5/2, 5/2), signs (+, -, -); production builds the Bessel
#: phases with phase_atol = amplitude_rtol = 1e-12 (main.py).
F06_CASES = {
    "06f g-1e-7-0-12689 #19 (J4)": {
        "coefficients": ("0x1.32803638c7bf7p+18", "0x1.61eaaae8b74d7p+17"),
        "span": ("-0x1.66c4a907e05e8p+3", "-0x1.c978f88239ba7p+0"),
        "K": "-0x1.7b53a57f7c700p+15",
        "nodes": (
            ("-0x1.f560995c7bf98p+0", "-0x1.8c2da23c8850ep-1"),
            ("-0x1.149ad02dce52cp+1", "-0x1.fc9b116d22157p+0"),
            ("-0x1.068ea694c558ep+1", "0x1.4b2f792053f50p+1"),
        ),
    },
    "06f g-1e-7-3-1054 #20 (Y4)": {
        "coefficients": ("0x1.86a0000000000p+16", "0x1.c30e0dc8527fdp+15"),
        "span": ("-0x1.422aec355382ap+3", "-0x1.584cf5267c727p+2"),
        "K": "-0x1.e3706e4293fe8p+13",
        "nodes": (
            ("-0x1.5b1507315d095p+2", "0x1.a3772c49c1d05p-1"),
            ("-0x1.58ffcee2e8f02p+2", "-0x1.703ee8cccf14dp+0"),
            ("-0x1.584cf5267c727p+2", "-0x1.1b2e750dce26bp+1"),
        ),
    },
}
F06_ORDERS = (0.5, 2.5, 2.5)
#: the scalar declared theta_abserr 06f recorded for the nu = 1/2 and nu = 5/2 objects
F06_SCALAR_THETA_ABSERR = (8.881784197001252e-16, 5.000000000000008e-13)

#: eps_exp's sample: measure_exp_relerr.py's, drawn here again.
EXP_SEED = 20261031
EXP_POINTS = 1_000_000
EXP_RANGES = {
    "production": (-30.0, 5.0),
    "LiouvilleGreen": (-2.0, math.log(1.0e12)),
}
EXP_SAMPLE_SHA256 = {
    "production": "0bf3928c4005e65d2c436d53aa8f17e1c5ecc89b488e1cc0482d036116e774ad",
    "LiouvilleGreen": "fa869d62718dc6721e7df4781031f96ff50f37f97f048d3aa1898e359dd6ab74",
}


# -- the parent ---------------------------------------------------------------------------------

_PARENT = None


def _parent_module():
    """The dispatch commit's three_bessel_integrals.py, imported under another name."""
    global _PARENT
    if _PARENT is None:
        source = subprocess.run(
            [
                "git",
                "-C",
                str(REPO),
                "show",
                f"{PARENT_SHA}:LiouvilleGreen/three_bessel_integrals.py",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        path = Path(tempfile.mkdtemp(prefix="lpf06h-parent-")) / "tb_parent_2c325f5.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location("tb_parent_2c325f5", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _PARENT = module
    return _PARENT


# -- the reference ------------------------------------------------------------------------------


def _constituent_phase(nu, y, mpmath):
    """atan2(J_nu(y), -Y_nu(y)), inside an active workdps block: theta_nu modulo 2 pi."""
    J = mpmath.besselj(mpmath.mpf(nu), y, maxterms=10**7, maxprec=10**6)
    Y = mpmath.bessely(mpmath.mpf(nu), y, maxterms=10**7, maxprec=10**6)
    return mpmath.atan2(J, -Y)


def _constituent_phase_hankel(nu, y, mpmath):
    """arg H^(1)_nu(y) + pi/2, the other formula: J = M cos(phi), Y = M sin(phi), theta = phi + pi/2."""
    return mpmath.arg(mpmath.hankel1(mpmath.mpf(nu), y)) + mpmath.pi / 2


def _group_reference(group, x, mpmath):
    """sum_i e_i theta_i(m_i x) at the mpmath abscissa x, with the products m_i x exact."""
    total = mpmath.mpf(0)
    for phase, coefficient, sign in zip(group.phases, group.coefficients, group.signs):
        y = mpmath.mpf(coefficient) * x
        total += mpmath.mpf(sign) * _constituent_phase(phase.nu, y, mpmath)
    return total


def _wrap(d, mpmath):
    return d - 2 * mpmath.pi * mpmath.nint(d / (2 * mpmath.pi))


def _angle_error(angle, reference, mpmath):
    """The double angle minus the reference, as an angle in (-pi, pi]."""
    return float(_wrap(mpmath.mpf(float(angle)) - reference, mpmath))


def _pair_angle_error(pair, reference, mpmath):
    """The angle the (sin, cos) pair represents, atan2 of the two doubles exactly, minus the
    reference."""
    s, c = pair
    return float(
        _wrap(
            mpmath.atan2(mpmath.mpf(float(s)), mpmath.mpf(float(c))) - reference, mpmath
        )
    )


# -- the cells ----------------------------------------------------------------------------------


class _Cell:
    """One cell: the 06h group and the parent's on the same phase objects, and its points in u,
    with both references at each point."""

    def __init__(self, name, phases, coefficients, signs, u_lo, u_hi, extra_u=()):
        import mpmath

        self.name = name
        self.group = BesselPhaseGroup(phases, coefficients, signs)
        self.parent = _parent_module().BesselPhaseGroup(phases, coefficients, signs)
        index = _CELL_NAMES.index(name)
        rng = np.random.default_rng([GRID_SEED, index])
        drawn = rng.uniform(u_lo, u_hi, POINTS_PER_CELL)
        self.u = np.concatenate([[u_lo, u_hi], drawn, np.asarray(extra_u, dtype=float)])
        self.reference_at_float = []
        self.reference_at_exact = []
        with mpmath.workdps(REFERENCE_DPS):
            for u in self.u.tolist():
                x_float = mpmath.mpf(float(np.exp(u)))
                x_exact = mpmath.exp(mpmath.mpf(u))
                self.reference_at_float.append(
                    _group_reference(self.group, x_float, mpmath)
                )
                self.reference_at_exact.append(
                    _group_reference(self.group, x_exact, mpmath)
                )


_CELL_NAMES = (
    [f"GROUP_CASES {label}" for label in GROUP_CASES]
    + ["fl(K) != K", "three non-zero residuals"]
    + [f"QSI {label}" for label in QSI_CASES]
    + list(F06_CASES)
)

_CELLS = None


def _build_phases(orders, coefficients, x_max, **kwargs):
    """One BesselPhaseFunction per distinct order, built to 1.1 max(m_i) x_max and shared between
    factors of the same order, as _three_bessel_Levin shares its transfer-function object.
    """
    by_order = {}
    for order, coefficient in zip(orders, coefficients):
        by_order[order] = max(by_order.get(order, 0.0), coefficient)
    built = {
        order: bessel_phase(order, 1.1 * top * x_max, **kwargs)["phase"]
        for order, top in by_order.items()
    }
    return tuple(built[order] for order in orders)


def _u_range(phases, coefficients, x_max):
    """From the bottom of every factor's domain, x >= min_x_i / m_i, to x_max, in u."""
    x_lo = max(phase.min_x / m for phase, m in zip(phases, coefficients))
    return math.log(1.0001 * x_lo), math.log(x_max)


def _cells():
    global _CELLS
    if _CELLS is not None:
        return _CELLS
    cells = []
    for label, coefficients in GROUP_CASES.items():
        phases = tuple(
            bessel_phase(order, 1.1 * coefficient * GROUP_X_MAX)["phase"]
            for order, coefficient in zip(GROUP_ORDERS, coefficients)
        )
        u_lo, u_hi = _u_range(phases, coefficients, GROUP_X_MAX)
        cells.append(
            _Cell(
                f"GROUP_CASES {label}",
                phases,
                coefficients,
                GROUP_SIGNS,
                u_lo,
                u_hi,
            )
        )

    phases = tuple(
        bessel_phase(order, 1.1 * coefficient * GROUP_X_MAX)["phase"]
        for order, coefficient in zip(GROUP_ORDERS, K_INEXACT_CASE)
    )
    u_lo, u_hi = _u_range(phases, K_INEXACT_CASE, GROUP_X_MAX)
    cells.append(_Cell("fl(K) != K", phases, K_INEXACT_CASE, GROUP_SIGNS, u_lo, u_hi))

    phases = tuple(
        bessel_phase(order, 1.1 * coefficient * GROUP_X_MAX)["phase"]
        for order, coefficient in zip(THREE_RESIDUALS_ORDERS, THREE_RESIDUALS_CASE)
    )
    u_lo, u_hi = _u_range(phases, THREE_RESIDUALS_CASE, GROUP_X_MAX)
    cells.append(
        _Cell(
            "three non-zero residuals",
            phases,
            THREE_RESIDUALS_CASE,
            GROUP_SIGNS,
            u_lo,
            u_hi,
        )
    )

    for label, coefficients in QSI_CASES.items():
        eta_max = QSI_X_MAX / max(coefficients)
        phase_Gk = bessel_phase(QSI_ORDERS[0], 1.1 * coefficients[0] * eta_max)["phase"]
        phase_Tk = bessel_phase(
            QSI_ORDERS[1], 1.1 * max(coefficients[1], coefficients[2]) * eta_max
        )["phase"]
        phases = (phase_Gk, phase_Tk, phase_Tk)
        u_lo, u_hi = _u_range(phases, coefficients, eta_max)
        cells.append(
            _Cell(f"QSI {label}", phases, coefficients, GROUP_SIGNS, u_lo, u_hi)
        )

    for label, case in F06_CASES.items():
        k, q_cs = (float.fromhex(value) for value in case["coefficients"])
        coefficients = (k, q_cs, q_cs)
        u_lo, u_hi = (float.fromhex(value) for value in case["span"])
        phases = _build_phases(
            F06_ORDERS,
            coefficients,
            math.exp(u_hi),
            phase_atol=1.0e-12,
            amplitude_rtol=1.0e-12,
        )
        nodes = [float.fromhex(u) for u, _ in case["nodes"]]
        cells.append(
            _Cell(label, phases, coefficients, GROUP_SIGNS, u_lo, u_hi, extra_u=nodes)
        )

    _CELLS = cells
    return cells


class TestGroupPhaseRounding(unittest.TestCase):
    """Tests 1-7 of prompts/levin-phase-floor/06h section 2 (test 8 is a harness, test 9 the
    existing suites)."""

    @classmethod
    def setUpClass(cls):
        cls.cells = _cells()

    # -- the cells reproduce what they claim to ---------------------------------------------------

    def test_the_06f_cells_reproduce_06f(self):
        """The 06f cells are 06f's calls: K, the declared scalars, and the parent's theta_mod_2pi at
        the three worst nodes of each call, bit for bit as 06f recorded them."""
        for cell in self.cells:
            if cell.name not in F06_CASES:
                continue
            case = F06_CASES[cell.name]
            with self.subTest(cell=cell.name):
                self.assertEqual(cell.group.K, float.fromhex(case["K"]))
                self.assertEqual(cell.group.K_lo, 0.0)
                self.assertEqual(
                    tuple(phase.theta_abserr for phase in cell.group.phases[:2]),
                    F06_SCALAR_THETA_ABSERR,
                )
                for u_hex, theta_hex in case["nodes"]:
                    self.assertEqual(
                        cell.parent.theta_mod_2pi(float.fromhex(u_hex)).hex(),
                        float.fromhex(theta_hex).hex(),
                    )

    # -- test 1 ------------------------------------------------------------------------------------

    def test_1_the_product_rounding_is_gone(self):
        """
        In every cell, the phase sin_cos represents (atan2 of the pair, exactly) and theta_mod_2pi,
        compared as angles against the reference at the **float** x = np.exp(u), are within the
        constituents' declarations plus phi_eval. No u |K x| term.

        The parent's error at the same points, and its excess over its declaration (the
        constituents' sum), are printed. Measured when 06h was written, the parent's worst error
        and worst error/declaration were: GROUP_CASES generic 6.42e-6 rad, 2.41e9 x; fl(K) != K
        9.21e-5 rad, 3.46e10 x; three non-zero residuals 3.05e-5 rad, 1.15e10 x; QSI generic
        7.51e-9 rad, 2.82e6 x; 06f #19 4.34e-13 rad, 163 x; 06f #20 3.81e-13 rad, 1.81 x. The
        cancelling cells (K = 0, K/max ~ 1e-6 and 1e-10, QSI near-resonant), whose product is
        exact or small, were inside their declarations at both codes (at most 0.5 of it). 06h's
        worst is at most 0.499 of constituents + phi_eval on every cell.
        """
        import mpmath

        for cell in self.cells:
            with self.subTest(cell=cell.name):
                worst_new = worst_new_ratio = 0.0
                worst_parent = worst_parent_excess = 0.0
                with mpmath.workdps(REFERENCE_DPS):
                    for u, reference in zip(cell.u.tolist(), cell.reference_at_float):
                        constituents, _, phi_eval = cell.group._declared_error_parts(u)
                        bound = constituents + phi_eval
                        pair_error = _pair_angle_error(
                            cell.group.sin_cos(u), reference, mpmath
                        )
                        mod_error = _angle_error(
                            cell.group.theta_mod_2pi(u), reference, mpmath
                        )
                        for label, error in (
                            ("sin_cos pair", pair_error),
                            ("theta_mod_2pi", mod_error),
                        ):
                            self.assertLessEqual(
                                abs(error),
                                bound,
                                msg=(
                                    f"{cell.name}: {label} error {error:.6e} at u = {u!r}"
                                    f" (K x = {cell.group.K * math.exp(u):.6e}) exceeds the"
                                    f" constituents' {constituents:.6e} plus phi_eval"
                                    f" {phi_eval:.6e}"
                                ),
                            )
                        worst_new = max(worst_new, abs(pair_error), abs(mod_error))
                        worst_new_ratio = max(
                            worst_new_ratio,
                            max(abs(pair_error), abs(mod_error)) / bound,
                        )
                        parent_error = _pair_angle_error(
                            cell.parent.sin_cos(u), reference, mpmath
                        )
                        worst_parent = max(worst_parent, abs(parent_error))
                        worst_parent_excess = max(
                            worst_parent_excess,
                            abs(parent_error) / cell.parent.theta_abserr(u),
                        )
                print(
                    f"\n@@ 06h test 1, {cell.name}: |delta theta| at the float x: 06h worst"
                    f" {worst_new:.4e} ({worst_new_ratio:.3g} of constituents + phi_eval);"
                    f" parent worst {worst_parent:.4e} ({worst_parent_excess:.3g} x its"
                    f" declaration)"
                )

    # -- test 2 ------------------------------------------------------------------------------------

    def test_2_the_rounding_of_K_is_gone(self):
        """
        On the fl(K) != K cell, against the reference with the exact K (the constituents at the
        float coefficients, composed in mpmath, carry K = sum_i e_i m_i exactly): 06h is within the
        constituents' declarations plus phi_eval. K_lo is K - fl(K) correctly rounded, and K_lo x
        reaches 8.3e-5 rad on the cell. The parent's error carries |K_lo| x: measured when 06h was
        written, 9.21e-5 rad at x = 1e12, where |K_lo| x = 8.33e-5 and u |fl(K) x| = 1.78e-4,
        against 06h's worst of 1.71e-15.
        """
        import mpmath

        (cell,) = [c for c in self.cells if c.name == "fl(K) != K"]
        group = cell.group
        exact = sum(
            Fraction(sign) * Fraction(m)
            for sign, m in zip(group.signs, group.coefficients)
        )
        self.assertNotEqual(Fraction(group.K), exact)
        self.assertEqual(group.K_lo, float(exact - Fraction(group.K)))
        self.assertGreater(abs(group.K_lo) * GROUP_X_MAX, 1.0e-5)

        worst_new = worst_parent = 0.0
        at_parent = None
        with mpmath.workdps(REFERENCE_DPS):
            for u, reference in zip(cell.u.tolist(), cell.reference_at_float):
                constituents, _, phi_eval = group._declared_error_parts(u)
                error = _pair_angle_error(group.sin_cos(u), reference, mpmath)
                self.assertLessEqual(abs(error), constituents + phi_eval)
                worst_new = max(worst_new, abs(error))
                parent_error = _pair_angle_error(
                    cell.parent.sin_cos(u), reference, mpmath
                )
                if abs(parent_error) > worst_parent:
                    worst_parent, at_parent = abs(parent_error), u
        x_top = math.exp(at_parent)
        print(
            f"\n@@ 06h test 2: K = {group.K!r}, K_lo = {group.K_lo:.6e}; 06h worst"
            f" {worst_new:.4e}; parent worst {worst_parent:.4e} at x = {x_top:.4e}, where"
            f" |K_lo| x = {abs(group.K_lo) * x_top:.4e} and u |fl(K) x| = {U * abs(group.K) * x_top:.4e}"
        )

    # -- test 3 ------------------------------------------------------------------------------------

    def test_3_the_declaration_bounds_the_phase_error_as_a_function_of_u(self):
        """
        In every cell, at every point, |theta_mod_2pi(u) - theta(e^u)| <= theta_abserr(u), the
        reference at the exact e^u. The parent's declaration is scored on the same points and
        printed; it fails on 06f's #19 cell (asserted, as the cell's reproduction of 06f's F4).
        Measured when 06h was written: the parent violates at 29 of #19's 53 points, worst 277 x,
        and at 5 of #20's 53, worst 3.14 x; on every cell whose |K| x is large (GROUP_CASES
        generic 3.5e9 x, fl(K) != K 1.7e10 x, three non-zero residuals 4.4e10 x, QSI generic
        6.0e6 x) and on K/max ~ 1e-6 (1.0e4 x, the abscissa alone, its product being exact) and
        ~1e-10 (1.47 x). 06h's worst on any cell is 0.499 of its declaration.
        """
        import mpmath

        parent_violations = {}
        for cell in self.cells:
            # the parent first, over every point, so that its record does not depend on whether
            # the 06h assertions below stop part way
            worst_parent_ratio = 0.0
            violations = 0
            with mpmath.workdps(REFERENCE_DPS):
                for u, reference in zip(cell.u.tolist(), cell.reference_at_exact):
                    parent_error = _angle_error(
                        cell.parent.theta_mod_2pi(u), reference, mpmath
                    )
                    parent_ratio = abs(parent_error) / cell.parent.theta_abserr(u)
                    worst_parent_ratio = max(worst_parent_ratio, parent_ratio)
                    violations += parent_ratio > 1.0
            parent_violations[cell.name] = violations
            with self.subTest(cell=cell.name):
                worst_ratio = 0.0
                with mpmath.workdps(REFERENCE_DPS):
                    for u, reference in zip(cell.u.tolist(), cell.reference_at_exact):
                        declared = cell.group.theta_abserr(u)
                        error = _angle_error(
                            cell.group.theta_mod_2pi(u), reference, mpmath
                        )
                        constituents, abscissa, phi_eval = (
                            cell.group._declared_error_parts(u)
                        )
                        self.assertLessEqual(
                            abs(error),
                            declared,
                            msg=(
                                f"{cell.name}: |theta_mod_2pi(u) - theta(e^u)| = {abs(error):.6e}"
                                f" at u = {u!r} (x = {math.exp(u):.6e}, K x ="
                                f" {cell.group.K * math.exp(u):.6e}) exceeds theta_abserr"
                                f" {declared:.6e} = constituents {constituents:.6e} + abscissa"
                                f" {abscissa:.6e} + phi_eval {phi_eval:.6e}"
                            ),
                        )
                        worst_ratio = max(worst_ratio, abs(error) / declared)
                print(
                    f"\n@@ 06h test 3, {cell.name}: worst |delta theta|/theta_abserr at the exact"
                    f" e^u: 06h {worst_ratio:.4g}; parent {worst_parent_ratio:.4g}, violating at"
                    f" {violations} of {len(cell.u)} points"
                )
        self.assertGreater(parent_violations["06f g-1e-7-0-12689 #19 (J4)"], 0)

    # -- test 4 ------------------------------------------------------------------------------------

    def test_4_the_declaration_is_not_loose_by_orders_where_its_new_terms_dominate(
        self,
    ):
        """
        Descriptive, not asserted: measured over declared at the exact e^u, split by the part of
        the declaration that dominates (constituents, abscissa or phi_eval), over every cell.
        """
        import mpmath

        by_part = {"constituents": [], "abscissa": [], "phi_eval": []}
        with mpmath.workdps(REFERENCE_DPS):
            for cell in self.cells:
                for u, reference in zip(cell.u.tolist(), cell.reference_at_exact):
                    parts = dict(
                        zip(
                            ("constituents", "abscissa", "phi_eval"),
                            cell.group._declared_error_parts(u),
                        )
                    )
                    error = _angle_error(cell.group.theta_mod_2pi(u), reference, mpmath)
                    dominant = max(parts, key=parts.get)
                    by_part[dominant].append(abs(error) / cell.group.theta_abserr(u))
        lines = [
            "\n@@ 06h test 4: measured/declared at the exact e^u, by dominant part"
        ]
        for part, ratios in by_part.items():
            if not ratios:
                lines.append(f"   {part:12s} points 0")
                continue
            r = np.array(ratios)
            lines.append(
                f"   {part:12s} points {len(r):4d}; median {np.median(r):.3e}; 90 %"
                f" {np.quantile(r, 0.9):.3e}; max {np.max(r):.3e}; min {np.min(r):.3e}"
            )
        print("\n".join(lines))

    # -- test 5 ------------------------------------------------------------------------------------

    def test_5_eps_exp_basis(self):
        """
        eps_exp is measured (the library documents no accuracy): on the seeded sample of
        measure_exp_relerr.py, 10^6 points in each range, the worst relative error of np.exp --
        on a 0-d argument and on an array, which are bit-identical -- against mpmath at 30 digits
        is at most EXP_RELERR / EXP_RELERR_MULTIPLE. Measured 1.1146e-16 = 1.004 u.
        """
        import mpmath

        self.assertGreaterEqual(tb.EXP_RELERR_MULTIPLE, 2.0)
        self.assertGreaterEqual(
            tb.EXP_RELERR, tb.EXP_RELERR_MULTIPLE * tb.EXP_RELERR_MEASURED_WORST
        )
        worst = 0.0
        for index, (name, (lo, hi)) in enumerate(EXP_RANGES.items()):
            rng = np.random.default_rng([EXP_SEED, index])
            u = rng.uniform(lo, hi, EXP_POINTS)
            self.assertEqual(
                hashlib.sha256(u.tobytes()).hexdigest(), EXP_SAMPLE_SHA256[name]
            )
            x_array = np.exp(u)
            x_scalar = np.array([np.exp(value) for value in u.tolist()])
            self.assertTrue(
                np.array_equal(x_array.view(np.int64), x_scalar.view(np.int64))
            )
            with mpmath.workdps(REFERENCE_DPS):
                for value, x in zip(u.tolist(), x_array.tolist()):
                    e = mpmath.exp(mpmath.mpf(value))
                    worst = max(worst, float(abs(mpmath.mpf(x) - e) / e))
        print(
            f"\n@@ 06h test 5: worst relative error of np.exp {worst:.6e} = {worst / U:.4f} u;"
            f" EXP_RELERR {tb.EXP_RELERR:.4e} = {tb.EXP_RELERR_MULTIPLE} x"
            f" {tb.EXP_RELERR_MEASURED_WORST:.6e}, rounded up"
        )
        self.assertLessEqual(worst, tb.EXP_RELERR / tb.EXP_RELERR_MULTIPLE)

    # -- test 6 ------------------------------------------------------------------------------------

    def test_6_scalar_and_array_agree_bit_for_bit(self):
        """
        sin_cos, theta_mod_2pi and theta_abserr on an array of u are bit-identical to their scalar
        results at every point of every cell; TwoProduct is exact (big + p_lo = fl(K) x in exact
        arithmetic) and its scalar (math.fma) and array (Dekker) forms agree; and
        _detect_vectorized still finds the group's callables vectorising.
        """
        for cell in self.cells:
            group = cell.group
            u = cell.u
            with self.subTest(cell=cell.name):
                sin_a, cos_a = group.sin_cos(u)
                pairs = [group.sin_cos(value) for value in u.tolist()]
                for name, array, scalar in (
                    ("sin", sin_a, [p[0] for p in pairs]),
                    ("cos", cos_a, [p[1] for p in pairs]),
                    (
                        "theta_mod_2pi",
                        group.theta_mod_2pi(u),
                        [group.theta_mod_2pi(value) for value in u.tolist()],
                    ),
                    (
                        "theta_abserr",
                        group.theta_abserr(u),
                        [group.theta_abserr(value) for value in u.tolist()],
                    ),
                ):
                    self.assertTrue(
                        np.array_equal(
                            np.asarray(array, dtype=float).view(np.int64),
                            np.asarray(scalar, dtype=float).view(np.int64),
                        ),
                        msg=f"{cell.name}: {name} array and scalar results differ",
                    )

                x = np.exp(u)
                big = group.K * x
                p_lo_array = tb._two_product_error(group.K, x, big)
                p_lo_scalar = [
                    tb._two_product_error(group.K, xi, bi)
                    for xi, bi in zip(x.tolist(), big.tolist())
                ]
                self.assertTrue(
                    np.array_equal(
                        p_lo_array.view(np.int64),
                        np.asarray(p_lo_scalar, dtype=float).view(np.int64),
                    )
                )
                for xi, bi, pi in zip(x.tolist(), big.tolist(), p_lo_scalar):
                    self.assertEqual(
                        Fraction(bi) + Fraction(pi),
                        Fraction(group.K) * Fraction(xi),
                        msg=f"{cell.name}: TwoProduct not exact at x = {xi!r}",
                    )

                grid = np.linspace(u[0], u[1], 25)
                for name in ("theta_mod_2pi", "theta_deriv", "theta_abserr"):
                    self.assertTrue(
                        _detect_vectorized(getattr(group, name), grid),
                        msg=f"{cell.name}: {name} no longer detected as vectorising",
                    )

                # descriptive: the parent, whose sums used sum()
                differ = {}
                for name in ("theta_mod_2pi", "theta_abserr"):
                    array = np.asarray(getattr(cell.parent, name)(u), dtype=float)
                    scalar = np.array(
                        [getattr(cell.parent, name)(value) for value in u.tolist()]
                    )
                    differ[name] = int(
                        np.sum(array.view(np.int64) != scalar.view(np.int64))
                    )
                print(
                    f"\n@@ 06h test 6, {cell.name}: 06h bit-identical at all {len(u)} points;"
                    f" parent's array and scalar results differ at {differ}"
                )

    # -- test 7 ------------------------------------------------------------------------------------

    def test_7_the_method_check(self):
        """
        The constituent reference reproduces, at one argument per order, the error bessel_phase.py
        documents there, and the two reference formulas agree to 1e-25 (as do 30 and 50 digits):
        - nu = 1/2 at x = 1e3: phase-pair error at most 1.11e-16 (EVALUATION_FLOOR's comment);
        - nu = 3/2 at x = 1e3: at most 1.1e-16 (sin_cos_theta's docstring, the angle-addition
          route at 1e3, 1e7, 1e12, 1e15);
        - nu = 5/2 at x = 1e7: at most the 8.9e-16 theta_abserr_at's docstring says the tail
          declares there.
        """
        import mpmath

        documented = {
            0.5: (1.0e3, 1.11e-16),
            1.5: (1.0e3, 1.1e-16),
            2.5: (1.0e7, 8.9e-16),
        }
        for nu, (x, bound) in documented.items():
            with self.subTest(nu=nu):
                phase = bessel_phase(nu, 1.1 * x)["phase"]
                s, c = phase.sin_cos_theta(x)
                with mpmath.workdps(REFERENCE_DPS):
                    y = mpmath.mpf(x)
                    t30 = _constituent_phase(nu, y, mpmath)
                    h30 = _constituent_phase_hankel(nu, y, mpmath)
                    pair = max(
                        abs(mpmath.mpf(float(s)) - mpmath.sin(t30)),
                        abs(mpmath.mpf(float(c)) - mpmath.cos(t30)),
                    )
                    formulas = abs(_wrap(t30 - h30, mpmath))
                with mpmath.workdps(CHECK_DPS):
                    y = mpmath.mpf(x)
                    t50 = _constituent_phase(nu, y, mpmath)
                    digits = abs(_wrap(mpmath.mpf(t30) - t50, mpmath))
                print(
                    f"\n@@ 06h test 7, nu = {nu} at x = {x:g}: phase-pair error {float(pair):.4e}"
                    f" (documented {bound:.3g}); |atan2(J,-Y) - (arg H1 + pi/2)|"
                    f" {float(formulas):.2e}; |30 - 50 digits| {float(digits):.2e}"
                )
                self.assertLessEqual(float(pair), bound)
                self.assertLess(float(formulas), 1.0e-25)
                self.assertLess(float(digits), 1.0e-25)


if __name__ == "__main__":
    unittest.main()
