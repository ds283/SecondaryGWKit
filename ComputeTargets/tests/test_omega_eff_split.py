"""
Tests for the leading/correction split of the two Liouville-Green frequency modules
(``prompts/GkTk-remedial``, prompt 05 §1 item 1, §2).

``Gk_omegaEff_sq`` and ``Tk_omegaEff_sq`` return ``A + B + C`` with ``A`` -- ``(k/H)^2`` and
``w (k/H)^2`` respectively -- between 1e12 and 1e24 times the other two over the production
range. The phase residual needs ``B + C`` on its own, and forming it as ``omega^2 - A`` would
lose every digit (``RECONCILIATION.md`` §2 item 3), so each module now exposes

    *_omegaEff_sq_leading(model, k, z)      -> A
    *_omegaEff_sq_correction(model, k, z)   -> B + C     (independent of k)
    *_omegaEff_sq(model, k, z)              -> A + B + C, unchanged

**What "unchanged" is tested to mean.** ``omega_WKB_sq`` is a stored column
(``GkWKBValue``/``TkWKBValue``), so the refactor must not move the returned value by a bit --
README §2 (f), and standing note 4 of ``IMPLEMENTATION_STATE.md``. That is tested here as exact
equality against a verbatim copy of the pre-refactor expression (``_reference_Gk_omegaEff_sq``,
``_reference_Tk_omegaEff_sq`` below), which is the strongest form of the claim.

It is *not* tested as exact equality against ``leading + correction``, because floating-point
addition is not associative and ``(A + B) + C != A + (B + C)`` on 13-29 % of the sample points
below, by one or two ulp of ``A``. Prompt 05 §2 asks for the second form; the first is what
actually protects the stored column, and the two cannot both hold. The split is therefore scored
against the re-association bound instead -- see ``test_*_split_sums_to_the_return_value``, which
records the measured maximum.
"""

import unittest
from math import fabs, ulp

import numpy as np

from ComputeTargets.WKB_Gk import (
    Gk_omegaEff_sq,
    Gk_omegaEff_sq_correction,
    Gk_omegaEff_sq_leading,
    _Gk_correction_terms,
)
from ComputeTargets.WKB_Tk import (
    Tk_omegaEff_sq,
    Tk_omegaEff_sq_correction,
    Tk_omegaEff_sq_leading,
    _Tk_correction_terms,
)
from ComputeTargets.tests.wkb_reference import (
    LambdaCDMModel,
    QCDModel,
    RadiationModel,
    REFERENCE_K_VALUES,
    load_references,
    production_source_grid,
)

# prompt 05 §2: 200 log-spaced redshifts across the production grid, three wavenumbers
NUM_SAMPLE_POINTS = 200

# the re-association bound: |(A+B)+C - (A+(B+C))| <= 2 ulp of the largest of A, B, C. Two, not
# one, because each association rounds once and the two roundings need not agree. The scale is
# the largest *term*, not the largest of A and B+C: deep outside the horizon, where
# 3 eps/2 - eps^2/4 - 2 nearly vanishes, B and C are ten times |B + C| and it is their own
# rounding that the re-association moves.
REASSOCIATION_ULPS = 2.0


def _reference_Gk_omegaEff_sq(model, k: float, z: float) -> float:
    """``ComputeTargets/WKB_Gk.py:Gk_omegaEff_sq`` verbatim, as it stood before prompt 05."""
    one_plus_z = 1.0 + z
    one_plus_z_2 = one_plus_z * one_plus_z

    H = model.functions.Hubble(z)
    eps = model.functions.epsilon(z)
    epsPrime = model.functions.d_epsilon_dz(z)

    k_over_H = k / H
    k_over_H_2 = k_over_H * k_over_H

    A = k_over_H_2
    B = -epsPrime / 2.0 / one_plus_z
    C = (3.0 * eps / 2.0 - eps * eps / 4.0 - 2.0) / one_plus_z_2

    return A + B + C


def _reference_Tk_omegaEff_sq(model, k: float, z: float) -> float:
    """``ComputeTargets/WKB_Tk.py:Tk_omegaEff_sq`` verbatim, as it stood before prompt 05."""
    one_plus_z = 1.0 + z
    one_plus_z_2 = one_plus_z * one_plus_z

    H = model.functions.Hubble(z)
    w = model.functions.wPerturbations(z)
    wPrime = model.functions.d_wPerturbations_dz(z)
    eps = model.functions.epsilon(z)
    epsPrime = model.functions.d_epsilon_dz(z)

    k_over_H = k / H
    k_over_H_2 = k_over_H * k_over_H

    A = w * k_over_H_2
    B = (3.0 / 2.0 * wPrime - epsPrime / 2.0) / one_plus_z
    C = (
        3.0 / 2.0 * (1.0 + eps) * (1.0 + w)
        - eps * (3.0 + eps / 2.0) / 2.0
        - 9.0 / 4.0 * (1.0 + w) * (1.0 + w)
    ) / one_plus_z_2

    return A + B + C


class _Shared:
    """The three stand-ins and the sample redshifts, built once."""

    references = None
    z_values = None
    models = None

    @classmethod
    def build(cls):
        if cls.references is not None:
            return
        cls.references = load_references()

        grid_block = cls.references["models"]["LambdaCDMModel"]["grid"]
        z_top = float(grid_block["z_init"])
        z_end = float(grid_block["z_end"])
        grid = production_source_grid(z_top)

        cls.z_values = np.logspace(
            np.log10(z_end), np.log10(z_top), num=NUM_SAMPLE_POINTS
        )
        cls.models = [RadiationModel(), LambdaCDMModel(), QCDModel(grid)]


class TestReturnValueUnchanged(unittest.TestCase):
    """
    The refactor must not move ``omega_WKB_sq`` by a bit: it is a stored column (README §2 (f)).
    """

    @classmethod
    def setUpClass(cls):
        _Shared.build()
        cls.s = _Shared

    def test_Gk_omegaEff_sq_is_bit_identical_to_the_pre_refactor_expression(self):
        for model in self.s.models:
            for k in REFERENCE_K_VALUES:
                for z in self.s.z_values:
                    z = float(z)
                    self.assertEqual(
                        Gk_omegaEff_sq(model, k, z),
                        _reference_Gk_omegaEff_sq(model, k, z),
                        msg=f"Gk_omegaEff_sq moved at {model.name}, k={k:.6e}, z={z:.10g}",
                    )

    def test_Tk_omegaEff_sq_is_bit_identical_to_the_pre_refactor_expression(self):
        for model in self.s.models:
            for k in REFERENCE_K_VALUES:
                for z in self.s.z_values:
                    z = float(z)
                    self.assertEqual(
                        Tk_omegaEff_sq(model, k, z),
                        _reference_Tk_omegaEff_sq(model, k, z),
                        msg=f"Tk_omegaEff_sq moved at {model.name}, k={k:.6e}, z={z:.10g}",
                    )


class TestSplitSumsToTheReturnValue(unittest.TestCase):
    """
    ``leading + correction`` reproduces ``*_omegaEff_sq`` up to the re-association rounding, and
    no further: the difference is bounded by a couple of ulp of the largest term.
    """

    @classmethod
    def setUpClass(cls):
        _Shared.build()
        cls.s = _Shared

    def _check(self, label, total_fn, leading_fn, correction_fn, terms_fn):
        worst_ulps, worst_where, exact, total = 0.0, None, 0, 0
        for model in self.s.models:
            for k in REFERENCE_K_VALUES:
                for z in self.s.z_values:
                    z = float(z)
                    value = total_fn(model, k, z)
                    leading = leading_fn(model, k, z)
                    correction = correction_fn(model, k, z)
                    split = leading + correction

                    total += 1
                    if split == value:
                        exact += 1
                        continue

                    # the private (B, C) accessor is used only to set the scale of the bound
                    B, C = terms_fn(model, k, z)
                    scale = max(fabs(leading), fabs(B), fabs(C), fabs(value))
                    ulps = fabs(split - value) / ulp(scale)
                    if ulps > worst_ulps:
                        worst_ulps, worst_where = ulps, (model.name, k, z)

        print(
            f"\n[{label}] leading + correction == {label}_omegaEff_sq exactly on "
            f"{exact}/{total} samples; worst re-association {worst_ulps:.3f} ulp at "
            f"{worst_where}"
        )
        self.assertLessEqual(worst_ulps, REASSOCIATION_ULPS)

    def test_Gk_split_sums_to_the_return_value(self):
        self._check(
            "Gk",
            Gk_omegaEff_sq,
            Gk_omegaEff_sq_leading,
            Gk_omegaEff_sq_correction,
            _Gk_correction_terms,
        )

    def test_Tk_split_sums_to_the_return_value(self):
        self._check(
            "Tk",
            Tk_omegaEff_sq,
            Tk_omegaEff_sq_leading,
            Tk_omegaEff_sq_correction,
            _Tk_correction_terms,
        )


class TestCorrectionIsKIndependent(unittest.TestCase):
    """
    Neither correction involves ``k``. This is why one residual table per ``k`` is cheap and why
    the leading term is a per-model table (review §6 (d), §12.2).
    """

    @classmethod
    def setUpClass(cls):
        _Shared.build()
        cls.s = _Shared

    def test_Gk_correction_does_not_depend_on_k(self):
        for model in self.s.models:
            for z in self.s.z_values:
                z = float(z)
                reference = Gk_omegaEff_sq_correction(model, REFERENCE_K_VALUES[0], z)
                for k in REFERENCE_K_VALUES[1:]:
                    self.assertEqual(
                        Gk_omegaEff_sq_correction(model, k, z),
                        reference,
                        msg=f"{model.name}: Gk correction moved with k at z={z:.10g}",
                    )

    def test_Tk_correction_does_not_depend_on_k(self):
        for model in self.s.models:
            for z in self.s.z_values:
                z = float(z)
                reference = Tk_omegaEff_sq_correction(model, REFERENCE_K_VALUES[0], z)
                for k in REFERENCE_K_VALUES[1:]:
                    self.assertEqual(
                        Tk_omegaEff_sq_correction(model, k, z),
                        reference,
                        msg=f"{model.name}: Tk correction moved with k at z={z:.10g}",
                    )


class TestRadiationCorrections(unittest.TestCase):
    """
    The exact-radiation controls on the two corrections, which fix the size of the two residuals.
    """

    @classmethod
    def setUpClass(cls):
        _Shared.build()
        cls.s = _Shared
        cls.radiation = _Shared.models[0]

    def test_Gk_correction_is_exactly_zero_in_radiation(self):
        """
        With ``eps = 2`` and ``eps' = 0``: ``B = 0`` and
        ``C = (3*2/2 - 2*2/4 - 2)/s^2 = (3 - 1 - 2)/s^2 = 0``, with no rounding at any ``z``.
        This is why ``rho_G`` is bit-exactly zero on the radiation control (review §6).
        """
        for k in REFERENCE_K_VALUES:
            for z in self.s.z_values:
                z = float(z)
                self.assertEqual(
                    Gk_omegaEff_sq_correction(self.radiation, k, z),
                    0.0,
                    msg=f"Gk correction is not zero in radiation at z={z:.10g}",
                )

    def test_Tk_correction_is_minus_two_over_s_squared_in_radiation(self):
        """
        With ``eps = 2``, ``w = 1/3``, ``eps' = w' = 0``: ``B = 0`` and
        ``C = [(3/2)(3)(4/3) - (2)(4)/2 - (9/4)(16/9)]/s^2 = [6 - 4 - 4]/s^2 = -2/s^2``.
        Unlike the Green's function's, it does **not** vanish, which is why ``rho_T`` is
        ~-0.09 rad and must be carried (review §12.2, §12.4).
        """
        worst, worst_z = 0.0, None
        for z in self.s.z_values:
            z = float(z)
            s = 1.0 + z
            expected = -2.0 / (s * s)
            got = Tk_omegaEff_sq_correction(self.radiation, REFERENCE_K_VALUES[0], z)
            err = fabs(got - expected) / fabs(expected)
            if err > worst:
                worst, worst_z = err, z
        print(
            f"\n[Tk] radiation correction vs -2/s^2: worst {worst:.3e} relative at "
            f"z = {worst_z}"
        )
        self.assertLessEqual(worst, 4.0e-16)


if __name__ == "__main__":
    unittest.main()
