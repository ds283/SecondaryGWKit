import os
"""QI_01: symbolic check of the analytic_integral prefactor against spec 04 R14 and spec 05 R31.

Checks:
  (a) code F = -B*C*D*E  (QuadSourceIntegral.py:868-874) == spec 04 R14 prefactor with a_0 -> 1
      (i.e. the a_0^2 form written in the code's variables k/a_0, a_0*eta).
  (b) ratio (spec04 R14 prefactor) / (spec05 R31 prefactor) == -1/c^2, c = (2+b)/(3+2b).
  (c) same ratio for the source function f:  f_code (spec04 R11 final) / f_spec05_R28 == 1/c^2.
  (d) w <-> b identities used by the code's cs^2 = (1-b)/(3(1+b)).
"""

import sympy as sp

b = sp.symbols("b", real=True)
a0 = sp.symbols("a_0", positive=True)
q, r, k, eta, cs = sp.symbols("q r k eta c_s", positive=True)

c = (2 + b) / (3 + 2 * b)  # spec 03 sec 0.1: c^2 = (2+b)^2/(3+2b)^2

# ---- (a) code prefactor, transcribed from QuadSourceIntegral.py:868-874 -------------------
cs_sq = (1 - b) / (1 + b) / 3
B = sp.pi / 2
C = 2 ** (3 + 2 * b) / (3 + 2 * b) / (2 + b)
D = sp.gamma(sp.Rational(5, 2) + b) ** 2
E = (q * r * cs_sq * eta) ** (-sp.Rational(1, 2) - b)
F_code = -B * C * D * E

# spec 04 R14 prefactor, with the signed-off a_0^2 (sec 0 item 4) and cs kept symbolic
F_R14 = (
    -(a0**2) * sp.pi / 2
    * 2 ** (3 + 2 * b) / ((3 + 2 * b) * (2 + b))
    * sp.gamma(sp.Rational(5, 2) + b) ** 2
    * (q * r * cs**2 * eta) ** (-sp.Rational(1, 2) - b)
)
ratio_a = sp.simplify((F_code / F_R14).subs({cs: sp.sqrt(cs_sq)}))
print("(a) code prefactor / R14 prefactor =", sp.simplify(ratio_a))

# ---- (b) R14 prefactor vs spec 05 R31 prefactor -------------------------------------------
F_R31 = (
    sp.pi * 2 ** (2 + 2 * b)
    * (2 + b) / (3 + 2 * b) ** 3
    * sp.gamma(sp.Rational(5, 2) + b) ** 2
    * (cs**2 * q * r * eta) ** (-sp.Rational(1, 2) - b)
)
ratio_b = sp.simplify(F_R14 / F_R31)
print("(b) R14 prefactor / R31 prefactor =", sp.simplify(ratio_b))
print("    -a_0^2/c^2 =", sp.simplify(-(a0**2) / c**2))
print("    equal?", sp.simplify(ratio_b + a0**2 / c**2) == 0)

# also the code (a_0-free) version
ratio_b_code = sp.simplify(F_code.subs(cs_sq, cs**2) / F_R31)
print("    code prefactor / R31 prefactor =", sp.simplify(ratio_b_code),
      "  == -1/c^2 ?", sp.simplify(ratio_b_code + 1 / c**2) == 0)

# ---- (c) f normalisation ------------------------------------------------------------------
# spec 04 R11 final ("so f =", the one the code's analytic branch implements)
pref_f_R11 = 2 ** (3 + 2 * b) / ((3 + 2 * b) * (2 + b)) * sp.gamma(sp.Rational(5, 2) + b) ** 2
# spec 05 R28 (MAIN 14 completed square)
pref_f_R28 = 2 ** (3 + 2 * b) * (2 + b) / (3 + 2 * b) ** 3 * sp.gamma(sp.Rational(5, 2) + b) ** 2
print("(c) f_R11 / f_R28 =", sp.simplify(pref_f_R11 / pref_f_R28),
      "  == 1/c^2 ?", sp.simplify(pref_f_R11 / pref_f_R28 - 1 / c**2) == 0)

# and the (1+w*)/(5+3w*) chain: c_* = 3(1+w)/(5+3w) equals (2+b)/(3+2b) * 3 ... check
w = sp.symbols("w", real=True)
b_of_w = (1 - 3 * w) / (1 + 3 * w)
c_star = 3 * (1 + w) / (5 + 3 * w)
print("(c') c_*(w) expressed in b:", sp.simplify(c_star.subs(w, sp.solve(sp.Eq(b, b_of_w), w)[0])))

# ---- (d) cs^2 identity --------------------------------------------------------------------
print("(d) (1-b)/(3(1+b)) in terms of w =", sp.simplify(((1 - b) / (3 * (1 + b))).subs(b, b_of_w)))
