"""
Symbolic derivation and verification of ComputeTargets/phase_groups.py.

Re-runnable:  PYTHONPATH=. ./venv/bin/python ComputeTargets/tests/sympy_phase_groups.py

Starting from `QuadSource.source_function` (the code's own source kernel, not a transcription
of it) and the Liouville-Green forms T_i = M_i sin theta_i, G = A_G sin theta_G, this script
lets sympy do the product-to-sum expansion itself and checks that

  1. `smooth_source` (alpha, beta form with D = (1+z) d/dz) is identically `source_function`;
  2. D[M sin theta] = a sin theta + b cos theta with a = (1+z) M d ln M/dz, b = (1+z) M omega;
  3. every named coefficient (c_SS, c_SC, c_CS, c_CC; c_S, c_C) that the module's coefficient
     functions imply equals the coefficient sympy extracts from the expanded kernel;
  4. for every one of the seven oscillatory regimes, the module's own `phase_group_terms`
     reassembles G * f identically:  sum_groups [f_sin sin Psi + f_cos cos Psi] - G f == 0.

Every residual is reduced with expand(expand_trig(...)) -- the product-to-sum identities are
polynomial identities in sin/cos, so an exact zero is expected and anything else is a failure.
Exit status is non-zero if any residual is not zero.
"""

import itertools
import os
import sys

import sympy as sp

sys.path.insert(0, os.getcwd())  # run from the repository root

from ComputeTargets.QuadSource import source_function
from ComputeTargets.phase_groups import (
    source_coefficients,
    smooth_source,
    both_oscillatory_coefficients,
    one_oscillatory_coefficients,
    phase_group_terms,
    signs_label,
)

failures = 0


def report(name, residual):
    global failures
    # expand_trig turns sin/cos of composed phases into polynomials in sin/cos of the individual
    # phases; cancel then puts the rational-in-w coefficients over a common denominator, which a
    # bare expand does not (it leaves e.g. 4X/(6w+6) - 2X/(3w+3) uncombined)
    residual = sp.cancel(sp.expand(sp.expand_trig(sp.expand(residual))))
    status = "zero" if residual == 0 else f"NON-ZERO: {residual}"
    if residual != 0:
        failures += 1
    print(f"  {name:<40s} residual = {status}")


# --- symbols -----------------------------------------------------------------------------

w, z = sp.symbols("w z", positive=True)
Tq, Tr, Tqp, Trp = sp.symbols("T_q T_r Tp_q Tp_r")  # smooth values and plain dT/dz
Mq, Mr, dlnMq, dlnMr, omq, omr = sp.symbols("M_q M_r dlnM_q dlnM_r omega_q omega_r")
thq, thr, thG = sp.symbols("theta_q theta_r theta_G")
AG, Gs = sp.symbols("A_G G_s")  # sine amplitude of oscillatory G; value of smooth G

one_plus_z = 1 + z
alpha, beta = source_coefficients(w)

# --- 1. the kernel -------------------------------------------------------------------------

print("1. smooth_source vs QuadSource.source_function")
f_code = source_function(Tq, Tr, Tqp, Trp, z, w)["source"]
# source_function is written with float literals (5.0, 3.0, 2.0); rationalise them so that the
# comparison can be exact
f_code = sp.nsimplify(f_code, rational=True)
f_module = smooth_source(alpha, beta, Tq, one_plus_z * Tqp, Tr, one_plus_z * Trp)
report("source_function - smooth_source", sp.simplify(f_code - f_module))
report("alpha - (5+3w)/(3(1+w))", alpha - (5 + 3 * w) / (3 * (1 + w)))
report("beta - 2/(3(1+w))", beta - sp.Rational(2) / (3 * (1 + w)))

# --- 2. the derivative decomposition -----------------------------------------------------

print("2. D[M sin theta] = a sin theta + b cos theta")
zz = sp.symbols("zz")
Mf = sp.Function("M")(zz)
thf = sp.Function("theta")(zz)
DT = (1 + zz) * sp.diff(Mf * sp.sin(thf), zz)
a = (1 + zz) * Mf * (sp.diff(Mf, zz) / Mf)
b = (1 + zz) * Mf * sp.diff(thf, zz)
report(
    "D[M sin theta] - (a S + b C)",
    sp.simplify(DT - (a * sp.sin(thf) + b * sp.cos(thf))),
)

# --- helpers for the LG substitution -----------------------------------------------------


def osc_T(M, dlnM, om):
    """(value, dT/dz) of T = M sin theta with M' = M dlnM, theta' = omega -- as symbols."""
    return M, dlnM, om


aq = one_plus_z * Mq * dlnMq
bq = one_plus_z * Mq * omq
ar = one_plus_z * Mr * dlnMr
br = one_plus_z * Mr * omr

Sq, Cq, Sr, Cr = sp.symbols("S_q C_q S_r C_r")


def kernel_with(q_osc: bool, r_osc: bool):
    """
    source_function with T_i = M_i sin theta_i substituted for each oscillatory factor
    (dT/dz = M dlnM sin theta + M omega cos theta), expressed in sin/cos of the *individual*
    phases.
    """
    Tq_val = Mq * sp.sin(thq) if q_osc else Tq
    Tqp_val = Mq * dlnMq * sp.sin(thq) + Mq * omq * sp.cos(thq) if q_osc else Tqp
    Tr_val = Mr * sp.sin(thr) if r_osc else Tr
    Trp_val = Mr * dlnMr * sp.sin(thr) + Mr * omr * sp.cos(thr) if r_osc else Trp
    return f_code.subs(
        {Tq: Tq_val, Tqp: Tqp_val, Tr: Tr_val, Trp: Trp_val}, simultaneous=True
    )


def as_polynomial(expr):
    """Replace sin/cos of the individual phases by symbols, so coefficients can be read off."""
    return sp.expand(
        expr.subs(
            {
                sp.sin(thq): Sq,
                sp.cos(thq): Cq,
                sp.sin(thr): Sr,
                sp.cos(thr): Cr,
            }
        )
    )


# --- 3. named coefficients -----------------------------------------------------------------

print("3a. both T oscillatory: coefficients of {S_q, C_q} x {S_r, C_r}")
poly_both = sp.Poly(as_polynomial(kernel_with(True, True)), Sq, Cq, Sr, Cr)
P_plus, Q_plus, P_minus, Q_minus = both_oscillatory_coefficients(
    alpha, beta, Mq, aq, bq, Mr, ar, br
)
# invert the module's product-to-sum: c_CC = P+ + P-, c_SS = P- - P+, c_SC = Q+ + Q-, c_CS = Q+ - Q-
implied = {
    "c_SS": (P_minus - P_plus, Sq * Sr),
    "c_SC": (Q_plus + Q_minus, Sq * Cr),
    "c_CS": (Q_plus - Q_minus, Cq * Sr),
    "c_CC": (P_plus + P_minus, Cq * Cr),
}
for name, (value, monomial) in implied.items():
    report(
        f"{name} (module) - {name} (sympy)", value - poly_both.coeff_monomial(monomial)
    )
# and nothing else: every other monomial of the expanded kernel must vanish
other = sum(
    poly_both.coeff_monomial(m)
    for m in poly_both.monoms()
    if sp.Mul(*[g**e for g, e in zip((Sq, Cq, Sr, Cr), m)])
    not in (Sq * Sr, Sq * Cr, Cq * Sr, Cq * Cr)
)
report("sum of stray monomials", other)

print("3b. one T oscillatory (T_q; T_r smooth): coefficients of S_q, C_q")
poly_one = sp.Poly(as_polynomial(kernel_with(True, False)), Sq, Cq)
P_one, Q_one = one_oscillatory_coefficients(
    alpha, beta, Mq, aq, bq, Tr, one_plus_z * Trp
)
report("c_S (module) - c_S (sympy)", Q_one - poly_one.coeff_monomial(Sq))
report("c_C (module) - c_C (sympy)", P_one - poly_one.coeff_monomial(Cq))
report("constant term", poly_one.coeff_monomial(1))

print("3c. one T oscillatory (T_r; T_q smooth): coefficients of S_r, C_r")
poly_one_r = sp.Poly(as_polynomial(kernel_with(False, True)), Sr, Cr)
P_one_r, Q_one_r = one_oscillatory_coefficients(
    alpha, beta, Mr, ar, br, Tq, one_plus_z * Tqp
)
report("c_S (module) - c_S (sympy)", Q_one_r - poly_one_r.coeff_monomial(Sr))
report("c_C (module) - c_C (sympy)", P_one_r - poly_one_r.coeff_monomial(Cr))
report("constant term", poly_one_r.coeff_monomial(1))

# --- 4. every regime, reassembled by the module's own phase_group_terms ------------------

print("4. G f - sum_groups [f_sin sin Psi + f_cos cos Psi], per regime")
phases = (thG, thq, thr)
for regime in itertools.product((True, False), repeat=3):
    if not any(regime):
        continue
    G_osc, q_osc, r_osc = regime

    G_value = AG * sp.sin(thG) if G_osc else Gs
    G_arg = AG if G_osc else Gs
    Tq_arg = (Mq, aq, bq) if q_osc else (Tq, one_plus_z * Tqp)
    Tr_arg = (Mr, ar, br) if r_osc else (Tr, one_plus_z * Trp)

    direct = G_value * kernel_with(q_osc, r_osc)

    terms = phase_group_terms(regime, alpha, beta, G_arg, Tq_arg, Tr_arg)
    reassembled = 0
    for signs, f_sin, f_cos in terms:
        Psi = sum(s * th for s, th in zip(signs, phases))
        reassembled += f_sin * sp.sin(Psi) + f_cos * sp.cos(Psi)

    labels = ",".join(signs_label(signs) for signs, _, _ in terms)
    name = f"regime G={int(G_osc)} q={int(q_osc)} r={int(r_osc)} [{labels}]"
    report(name, reassembled - direct)

print()
if failures:
    print(f"FAILED: {failures} non-zero residual(s)")
    sys.exit(1)
print("all residuals are identically zero")
