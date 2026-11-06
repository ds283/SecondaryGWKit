"""
The signed trust yardstick in the driver (prompts/levin-phase-floor/06m-signed-yardstick-in-the-driver.md,
section 2 tests 1-5).

Since 06m, _check_leaf()'s noise clause trusts a leaf with r < 1 if delta1 <= nu_B, the yardstick of
_trust_noise_yardstick() (route B, as prompts/levin-phase-floor/06l-signed-trust-yardstick.md measured it):

    nu_B = phi_P + sum over Levin halves of phi_h
                 + [Clenshaw-Curtis parent only: sum over CC halves of phi^-_h + |sum over CC halves of E_h|],

in that order. From 06g to 06m it compared delta1 with phi_P alone.

Three modules are compared:
  * the commit's, AdaptiveLevin.levin_quadrature;
  * the parent, the dispatch commit's module (DISPATCH_SHA, 06g's code), read with `git show` and executed
    under another name;
  * 06l's variant, docs/levin-phase-floor/06l/scripts/variants/levin_quadrature_routeB.py, executed under
    another name after its SHA-256 is checked. It computes nu_B inline in _check_leaf() and records every
    leaf it trusts that 06g's test does not (a "firing") under the result key "route_06l", 06l's instrument.

Tests 2-5 run on real integrands; none needs a stand-in. Two families make the noise-level leaves 06j found
in A3 v2's census, rare on clean integrands (no firing on any of 06g's cells):
  * "noisy": amplitudes computed as (A + g(x)) - A, which carry an absolute rounding noise of about ulp(A)
    at every node, the census's second population (06j: the integrand's input sensitivity at the nodes);
  * "far": a weak phase on [3e5, 3e5 + 4], where a node's rounding, ulp(3e5) ~ 6e-11, sets the
    Clenshaw-Curtis node error E (06e's node term), the census's first population.
The numbers quoted are docs/levin-phase-floor/06m/calibration/out/cells.txt's.

No Ray and no datastore.
"""

import hashlib
import math
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import AdaptiveLevin.levin_quadrature as lq

DISPATCH_SHA = "5ff8a5e44be775df299af0f1754123578905b86e"
_PARENT_NAME = "levin_quadrature_parent_5ff8a5e"
_REPO = Path(__file__).resolve().parents[2]
VARIANT_PATH = (
    _REPO
    / "docs"
    / "levin-phase-floor"
    / "06l"
    / "scripts"
    / "variants"
    / "levin_quadrature_routeB.py"
)
VARIANT_SHA256 = "5aa6178a2298fb4cc32620cf6ae81538e873bf6a3177e8db6f3a5274d9bb4d9f"
_VARIANT_NAME = "levin_quadrature_routeB_06l"


def _exec_module(name, source, file_label):
    module = types.ModuleType(name)
    module.__file__ = file_label
    sys.modules[name] = module
    exec(compile(source, file_label, "exec"), module.__dict__)
    return module


def load_parent():
    """The dispatch commit's module, read with `git show` and executed under another name, once per process."""
    module = sys.modules.get(_PARENT_NAME)
    if module is None:
        source = subprocess.run(
            [
                "git",
                "-C",
                str(_REPO),
                "show",
                f"{DISPATCH_SHA}:AdaptiveLevin/levin_quadrature.py",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        module = _exec_module(
            _PARENT_NAME,
            source,
            f"<git show {DISPATCH_SHA[:7]}:AdaptiveLevin/levin_quadrature.py>",
        )
    return module


def load_variant():
    """06l's route B variant, its SHA-256 checked, executed under another name, once per process."""
    module = sys.modules.get(_VARIANT_NAME)
    if module is None:
        raw = VARIANT_PATH.read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        if got != VARIANT_SHA256:
            raise AssertionError(
                f"{VARIANT_PATH} has SHA-256 {got}, not 06l's {VARIANT_SHA256}"
            )
        module = _exec_module(_VARIANT_NAME, raw.decode(), str(VARIANT_PATH))
    return module


# ---------------------------------------------------------------------------------------------
# Bit-for-bit comparison of results.


def canon(x):
    """x canonicalised for a bit-for-bit comparison: floats as hex, arrays and sequences as lists."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x).hex()
    if isinstance(x, np.ndarray):
        return [canon(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [canon(v) for v in x]
    if isinstance(x, dict):
        return {str(k): canon(v) for k, v in x.items()}
    if x is None or isinstance(x, str):
        return x
    return repr(x)


# every returned region's span, depth, value, errors, trust and differences
REGION_FIELDS = (
    "start",
    "end",
    "depth",
    "type",
    "value",
    "value_kind",
    "value_own",
    "abserr",
    "relerr",
    "phase_err",
    "phase_err_roundoff",
    "phase_err_declared",
    "phase_err_less_node",
    "node_error_signed",
    "abserr_truncation",
    "declared_resolution",
    "total_err",
    "budget_err",
    "phase_limited",
    "trusted",
    "delta1",
    "delta2",
    "convergence_ratio",
)


def region_record(region):
    return {name: canon(getattr(region, name)) for name in REGION_FIELDS}


def result_record(data, exclude=("elapsed",)):
    """Every result key but the excluded ones, the regions field by field."""
    out = {}
    for key, value in data.items():
        if key in exclude:
            continue
        if key == "regions":
            out[key] = [region_record(r) for r in value]
        else:
            out[key] = canon(value)
    return out


# ---------------------------------------------------------------------------------------------
# The cells. Each: (name, span, f, theta, kwargs), run through adaptive_levin_sincos().


def _noisy_cell(A, k, atol):
    return (
        f"noisy(A={A:g},k={k:g})",
        (1.0, 60.0),
        [lambda x: (A + 1.0 / x) - A, lambda x: (A + np.cos(x)) - A],
        {"theta": lambda x: k * x + x * x, "theta_deriv": lambda x: k + 2.0 * x},
        dict(atol=atol, rtol=1e-30, depth_max=16),
    )


def far_cell():
    X0 = 3.0e5
    return (
        "far(3e5)",
        (X0, X0 + 4.0),
        [lambda x: 1e3 * np.cos(0.7 * (x - X0)), lambda x: np.exp(-(x - X0))],
        {
            "theta": lambda x: 200.0 * (x - X0) + 3.0 * (x - X0) ** 2,
            "theta_deriv": lambda x: 200.0 + 6.0 * (x - X0),
        },
        dict(atol=1e-30, rtol=1e-6, depth_max=12),
    )


def noisy_512_cell():
    return _noisy_cell(512.0, 60.0, 1e-12)


def test2_cells():
    """
    Test 2's nine cells: six of 06g's (test_levin_leaf_values), smooth and oscillatory, Levin and
    fallback, on which route B never fires; and three noise-level cells on which it does.
    """
    third = 1.0 / 3.0
    seven_thirds = 7.0 / 3.0
    return [
        # Levin, one leaf
        (
            "exp_sin_1e3",
            (third, seven_thirds),
            [lambda x: math.exp(-x), lambda x: 0.0],
            {"theta": lambda x: 1e3 * x, "theta_deriv": lambda x: 1e3 + 0.0 * x},
            dict(atol=1e-20, rtol=1e-12),
        ),
        # Levin, leaves at depth 2
        (
            "si_1000_exp",
            (0.0, 20.0),
            [lambda x: 1.0, lambda x: 0.0],
            {
                "theta": lambda x: 1000.0 * np.exp(x),
                "theta_deriv": lambda x: 1000.0 * np.exp(x),
            },
            dict(atol=1e-15, rtol=1e-7),
        ),
        # Levin, oscillatory: 06c's Gaussian bump, leaves to depth 5
        (
            "gaussian_bump",
            (0.0, 3.0),
            [lambda x: math.exp(-400.0 * (x - 1.1) ** 2), lambda x: 0.0],
            {"theta": lambda x: 3.0e4 * x},
            dict(atol=1e-10),
        ),
        # fallback, smooth amplitude with a near-singular derivative, to depth 4
        (
            "sqrt_fallback",
            (0.0, 1.0),
            [lambda x: math.sqrt(x + 1e-3), lambda x: 0.0],
            {"theta": lambda x: 2.0 * x, "theta_deriv": lambda x: 2.0 + 0.0 * x},
            dict(atol=1e-20, rtol=1e-10),
        ),
        # fallback, a Runge amplitude on a weak phase
        (
            "runge_fallback",
            (-1.0, 1.0),
            [lambda x: 1.0 / (1.0 + 400.0 * x * x), lambda x: 0.0],
            {"theta": lambda x: x + 1.0, "theta_deriv": lambda x: 1.0 + 0.0 * x},
            dict(atol=1e-20, rtol=1e-8),
        ),
        # Levin and fallback leaves, to depth 12: 06c's test 12b, SincIntegral at order 12
        (
            "sinc_12b",
            (1.0, 100.0),
            [lambda x: 1.0 / x, lambda x: 0.0],
            {"theta": lambda x: 100.0 * x},
            dict(atol=1e-15, rtol=1e-10, chebyshev_order=12),
        ),
        # noise-level cells: a Levin-parent firing; Levin- and CC-parent firings; CC-parent firings
        _noisy_cell(128.0, 60.0, 1e-12),
        noisy_512_cell(),
        far_cell(),
    ]


def sincos(module, cell):
    _, span, f, theta, kwargs = cell
    return module.adaptive_levin_sincos(span, f, theta, **kwargs)


def run_recorded(module, cell):
    """sincos(), with every used_interval the driver makes recorded, those later bisected included."""
    made = []
    original = module.used_interval

    class Recorded(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    with mock.patch.object(module, "used_interval", Recorded):
        data = sincos(module, cell)
    return data, made


def checked_leaf(made, key):
    """The one region with (start, end, depth) == key that the trust check was taken on."""
    found = [
        r for r in made if (r.start, r.end, r.depth) == key and r.delta1 is not None
    ]
    if len(found) != 1:
        raise AssertionError(f"{len(found)} checked regions at {key}")
    return found[0]


def returned(data, key):
    return any((r.start, r.end, r.depth) == key for r in data["regions"])


def leaf_solves(cell, key):
    """
    The leaf's own solve and its halves', solved afresh as the driver solves them (the break point
    (end + start) / 2), with the commit's _adaptive_levin_subregion().
    """
    _, _, f, theta, kwargs = cell
    basis = lq._Basis_SinCos(theta)
    order = kwargs.get("chebyshev_order", lq.DEFAULT_LEVIN_CHEBSHEV_ORDER)
    rtol = kwargs.get("rtol", lq.DEFAULT_LEVIN_RELTOL)
    a, b, _ = key
    c = (b + a) / 2.0

    def solve(span):
        return lq._adaptive_levin_subregion(
            span, f, basis, None, chebyshev_order=order, rtol=rtol
        )

    return solve((a, b)), solve((a, c)), solve((c, b))


def nu1(phi_P, data_L, data_R):
    """Route 1's yardstick (06k), phi_P + phi_L + phi_R, in 06l's order."""
    return (phi_P + (data_L.get("phase_err", 0.0) or 0.0)) + (
        data_R.get("phase_err", 0.0) or 0.0
    )


# ---------------------------------------------------------------------------------------------
# Test 1's dicts.


def levin_half(phi):
    return {"is_direct": False, "phase_err": phi}


def cc_half(phi_minus, E, rest=0.0):
    """A Clenshaw-Curtis half: its floor less the node term phi^-, its E, and its whole floor phi^- + |E| + rest."""
    return {
        "is_direct": True,
        "phase_err": phi_minus + abs(E) + rest,
        "phase_err_less_node": phi_minus,
        "node_error_signed": E,
    }


class TestYardstickExactly(unittest.TestCase):
    """
    Test 1. _trust_noise_yardstick() on each of section 0.1's four cases and on a Levin parent with one
    Clenshaw-Curtis half on either side, each bit for bit against the formula written out here in
    section 0.1's order, with nu_B <= nu_1 in each.
    """

    # phi_P, the halves' phi (Levin) and phi^- (Clenshaw-Curtis), E of both signs; one set is chosen
    # so that adding |E_L + E_R| before the phi^- terms changes the sum's last bit
    PHI_P = 7.01546366168602e-11
    PHI_L = 3.3307440498366427e-11
    PHI_R = 2.180193461337113e-11
    PHIM_L = 1.771150605405849e-11
    PHIM_R = 1.645661928464921e-11
    E_POS = 1.5101380514788435e-09
    E_NEG = -1.990870174183882e-09
    REST = 3.1e-12

    def assertBits(self, got, expected, label):
        self.assertIsInstance(got, float, label)
        self.assertEqual(float(got).hex(), float(expected).hex(), label)

    def _cases(self):
        P, L, R = self.PHI_P, self.PHI_L, self.PHI_R
        mL, mR = self.PHIM_L, self.PHIM_R
        Ep, En, rest = self.E_POS, self.E_NEG, self.REST
        return [
            # (label, parent_is_direct, data_L, data_R, nu_B written out in section 0.1's order)
            ("Levin / Levin + Levin", False, levin_half(L), levin_half(R), (P + L) + R),
            ("Levin / CC + Levin", False, cc_half(mL, En, rest), levin_half(R), P + R),
            ("Levin / Levin + CC", False, levin_half(L), cc_half(mR, Ep, rest), P + L),
            (
                "Levin / CC + CC",
                False,
                cc_half(mL, En, rest),
                cc_half(mR, Ep, rest),
                P,
            ),
            (
                "CC / CC + CC, E of opposite signs",
                True,
                cc_half(mL, Ep, rest),
                cc_half(mR, En, rest),
                ((P + mL) + mR) + abs(Ep + En),
            ),
            (
                "CC / CC + CC, E both negative",
                True,
                cc_half(mL, En, rest),
                cc_half(mR, -3.0e-10, rest),
                ((P + mL) + mR) + abs(En + -3.0e-10),
            ),
            (
                "CC / CC + Levin",
                True,
                cc_half(mL, En, rest),
                levin_half(R),
                ((P + R) + mL) + abs(En),
            ),
            (
                "CC / Levin + CC",
                True,
                levin_half(L),
                cc_half(mR, Ep, rest),
                ((P + L) + mR) + abs(Ep),
            ),
            ("CC / Levin + Levin", True, levin_half(L), levin_half(R), (P + L) + R),
        ]

    def test_cases_bit_for_bit(self):
        for label, parent_is_direct, data_L, data_R, expected in self._cases():
            with self.subTest(case=label):
                got = lq._trust_noise_yardstick(
                    self.PHI_P, parent_is_direct, data_L, data_R
                )
                self.assertBits(got, expected, label)
                self.assertLessEqual(got, nu1(self.PHI_P, data_L, data_R), label)

    def test_order_matters_and_is_kept(self):
        """
        The case chosen so that another order changes the result's last bit: |E_L + E_R| added before
        the phi^- terms differs from section 0.1's order.
        """
        P, mL, mR = self.PHI_P, self.PHIM_L, self.PHIM_R
        S = abs(self.E_POS + self.E_NEG)
        expected = ((P + mL) + mR) + S
        self.assertNotEqual(((P + S) + mL) + mR, expected)
        got = lq._trust_noise_yardstick(
            P, True, cc_half(mL, self.E_POS), cc_half(mR, self.E_NEG)
        )
        self.assertBits(got, expected, "order")

    def test_signed_sum_is_below_the_absolute_sum(self):
        """|E_L + E_R| enters, not |E_L| + |E_R|: on opposite signs the yardstick is below their sum."""
        P, mL, mR = self.PHI_P, self.PHIM_L, self.PHIM_R
        got = lq._trust_noise_yardstick(
            P, True, cc_half(mL, self.E_POS), cc_half(mR, self.E_NEG)
        )
        self.assertLess(got, ((P + mL) + mR) + (abs(self.E_POS) + abs(self.E_NEG)))

    def test_levin_half_without_a_floor_reads_zero(self):
        half = {"is_direct": False, "phase_err": None}
        got = lq._trust_noise_yardstick(self.PHI_P, False, half, levin_half(self.PHI_R))
        self.assertBits(got, (self.PHI_P + 0.0) + self.PHI_R, "None read as 0.0")

    def test_raises_on_a_cc_parent_whose_cc_half_lacks_its_parts(self):
        whole = cc_half(self.PHIM_L, self.E_NEG)
        for missing in ("phase_err_less_node", "node_error_signed"):
            for side in ("L", "R"):
                for how in ("absent", "None"):
                    broken = dict(whole)
                    if how == "absent":
                        del broken[missing]
                    else:
                        broken[missing] = None
                    halves = (broken, whole) if side == "L" else (whole, broken)
                    with self.subTest(missing=missing, side=side, how=how):
                        with self.assertRaises(RuntimeError):
                            lq._trust_noise_yardstick(self.PHI_P, True, *halves)
                        # on a Levin parent a Clenshaw-Curtis half's parts are not read
                        self.assertBits(
                            lq._trust_noise_yardstick(self.PHI_P, False, *halves),
                            self.PHI_P,
                            "Levin parent",
                        )


class TestCodeIsTheVariants(unittest.TestCase):

    def test_bit_identical_to_the_variant(self):
        """
        Test 2. On nine cells (test2_cells()), run through adaptive_levin_sincos() under the commit,
        06l's variant and the parent: every result key the commit returns, "elapsed" aside, is
        bit-identical to the variant's, and so is every returned region's span, depth, value, errors,
        trusted, delta1, delta2 and convergence_ratio (REGION_FIELDS). On every cell where the variant
        records no firing, the commit is bit-identical to the parent too. The variant fires
        (cells.txt) on noisy(A=128) once on a Levin parent; on noisy(A=512) twice on a Levin parent
        and three times on a Clenshaw-Curtis parent; on far(3e5) 40 times, each on a Clenshaw-Curtis
        parent; and on none of 06g's six cells.
        """
        parent = load_parent()
        variant = load_variant()
        firings = {"cc": 0, "levin": 0}
        saw = {"fallback": False, "levin": False, "depth2": False}
        for cell in test2_cells():
            name = cell[0]
            with self.subTest(cell=name):
                new = sincos(lq, cell)
                var = sincos(variant, cell)
                old = sincos(parent, cell)
                self.assertNotIn("route_06l", new)
                self.assertEqual(set(new), set(var) - {"route_06l"}, name)
                self.assertEqual(
                    result_record(new),
                    result_record(var, exclude=("elapsed", "route_06l")),
                    name,
                )

                fired = [
                    rec for rec in var["route_06l"]["records"] if rec["kind"] == "fired"
                ]
                self.assertEqual(var["route_06l"]["n_withdrawn"], 0, name)
                firings["cc"] += sum(1 for rec in fired if rec["is_direct"])
                firings["levin"] += sum(1 for rec in fired if not rec["is_direct"])
                if not fired:
                    self.assertEqual(set(new), set(old), name)
                    self.assertEqual(result_record(new), result_record(old), name)

                saw["fallback"] |= any(
                    r.type == lq.types[lq.INTERVAL_TYPE_DIRECT] for r in new["regions"]
                )
                saw["levin"] |= any(
                    r.type == lq.types[lq.INTERVAL_TYPE_LEVIN] for r in new["regions"]
                )
                saw["depth2"] |= new["max_depth"] >= 2
        self.assertGreaterEqual(firings["cc"], 1)
        self.assertGreaterEqual(firings["levin"], 1)
        self.assertTrue(all(saw.values()), saw)


class TestLeafCases(unittest.TestCase):

    def _leaf(self, cell, key):
        """
        The leaf under the commit and the parent, and its solves afresh; the solves are checked to be
        the driver's: their delta1 is the commit's to the last bit.
        """
        new, made_new = run_recorded(lq, cell)
        old, made_old = run_recorded(load_parent(), cell)
        leaf_new = checked_leaf(made_new, key)
        leaf_old = checked_leaf(made_old, key)
        data_P, data_L, data_R = leaf_solves(cell, key)
        self.assertEqual(
            abs(data_P["value"] - (data_L["value"] + data_R["value"])), leaf_new.delta1
        )
        self.assertEqual(data_P["phase_err"], leaf_new.phase_err)
        # the same leaf, the same differences, under both codes
        for name in ("delta1", "delta2", "convergence_ratio", "phase_err"):
            self.assertEqual(getattr(leaf_new, name), getattr(leaf_old, name), name)
        return new, old, leaf_new, leaf_old, data_P, data_L, data_R

    def test_trusted_on_cancelling_node_errors(self):
        """
        Test 3. far(3e5), the leaf [300003.640625, 300003.65625] at depth 8: a Clenshaw-Curtis parent
        with two Clenshaw-Curtis halves, r = 0.4190, delta1 = 7.163e-10 above phi_P = 5.629e-10. The
        halves' E, -1.969e-9 and +1.744e-9, are 4.0 and 3.5 times E_P = 4.92e-10 and nearly equal and
        opposite: |E_L + E_R| = 2.24e-10 against |E_L| + |E_R| = 3.71e-9. nu_B = 9.145e-10 covers
        delta1 (nu_1 = 4.40e-9). Under the commit the leaf is trusted and returned, with Q. At the
        parent (06g's phi_P alone; cells.txt) it is untrusted, with the same r, delta1 and phi_P, and
        refined: the call returns 146 regions there, 107 under the commit.
        """
        key = (300003.640625, 300003.65625, 8)
        new, old, leaf_new, leaf_old, P, L, R = self._leaf(far_cell(), key)
        self.assertTrue(P["is_direct"] and L["is_direct"] and R["is_direct"])
        E_P, E_L, E_R = (
            P["node_error_signed"],
            L["node_error_signed"],
            R["node_error_signed"],
        )
        self.assertLess(E_L * E_R, 0.0)
        self.assertGreater(min(abs(E_L), abs(E_R)), 2.0 * abs(E_P))
        self.assertLess(abs(E_L + E_R), 0.1 * (abs(E_L) + abs(E_R)))

        r, delta1, phi_P = (
            leaf_new.convergence_ratio,
            leaf_new.delta1,
            leaf_new.phase_err,
        )
        nu_B = lq._trust_noise_yardstick(phi_P, True, L, R)
        self.assertLess(abs(r - 0.5), 0.1)
        self.assertLess(phi_P, delta1)
        self.assertLessEqual(delta1, nu_B)
        self.assertAlmostEqual(r, 0.4190, places=4)

        self.assertIs(leaf_new.trusted, True)
        self.assertTrue(returned(new, key))
        self.assertEqual(leaf_new.value_kind, "quarters")
        self.assertTrue(new["converged"])

        self.assertIs(leaf_old.trusted, False)
        self.assertFalse(returned(old, key))

    def test_levin_parent_keeps_cc_halves_out(self):
        """
        Test 4. noisy(A=512), the leaf [22.89453125, 23.125] at depth 8: a Levin parent with two
        Clenshaw-Curtis halves, r = 0.4856, delta1 = 6.484e-16 above phi_P = 1.645e-16. The halves'
        floors are 1.536e-14 and 1.302e-14, 93 and 79 times phi_P, so route 1's nu_1 = 2.854e-14
        would trust it; nu_B = phi_P, the halves adding nothing, and under the commit it is
        untrusted, as at the parent.
        """
        key = (22.89453125, 23.125, 8)
        new, old, leaf_new, leaf_old, P, L, R = self._leaf(noisy_512_cell(), key)
        self.assertFalse(P.get("is_direct", False))
        self.assertTrue(L["is_direct"] and R["is_direct"])

        r, delta1, phi_P = (
            leaf_new.convergence_ratio,
            leaf_new.delta1,
            leaf_new.phase_err,
        )
        nu_B = lq._trust_noise_yardstick(phi_P, False, L, R)
        nu_1 = nu1(phi_P, L, R)
        self.assertEqual(nu_B, phi_P)
        self.assertLess(abs(r - 0.5), 0.1)
        self.assertLess(nu_B, delta1)
        self.assertLessEqual(delta1, nu_1)  # route 1 would trust it
        self.assertAlmostEqual(r, 0.4856, places=4)

        self.assertIs(leaf_new.trusted, False)
        self.assertIs(leaf_old.trusted, False)

    def test_levin_over_levin_far_above_noise_stays_untrusted(self):
        """
        Test 5, 06h's kind of leaf (06l's f02 the model, delta1 / nu_B = 4.7e6). far(3e5), the leaf
        [300002.0, 300002.5] at depth 3: a Levin parent with two Levin halves, r = 0.6330,
        delta1 = 1.159e-10 against nu_B = nu_1 = phi_P + phi_L + phi_R = 8.11e-14, a ratio of
        1.43e3. Untrusted under both codes.
        """
        key = (300002.0, 300002.5, 3)
        new, old, leaf_new, leaf_old, P, L, R = self._leaf(far_cell(), key)
        self.assertFalse(P.get("is_direct", False))
        self.assertFalse(L.get("is_direct", False) or R.get("is_direct", False))

        r, delta1, phi_P = (
            leaf_new.convergence_ratio,
            leaf_new.delta1,
            leaf_new.phase_err,
        )
        nu_B = lq._trust_noise_yardstick(phi_P, False, L, R)
        self.assertEqual(nu_B, nu1(phi_P, L, R))
        self.assertLess(r, 1.0)
        self.assertGreaterEqual(r, 0.25)
        self.assertGreater(delta1, 1e3 * nu_B)

        self.assertIs(leaf_new.trusted, False)
        self.assertIs(leaf_old.trusted, False)
        self.assertFalse(returned(new, key))
        self.assertFalse(returned(old, key))


if __name__ == "__main__":
    unittest.main()
