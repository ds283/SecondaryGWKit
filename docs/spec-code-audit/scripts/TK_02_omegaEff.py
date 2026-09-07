import os
"""
TK_02: check ComputeTargets/WKB_Tk.py against spec 01 R23/R24/R27/R29/R30.

(a) Build P from R23 and evaluate the R27 *definition* of omega_eff^2; compare with the
    R27/R29 compact form and with Tk_omegaEff_sq (WKB_Tk.py:4-25).
(b) Differentiate Tk_omegaEff_sq symbolically and compare with the numerator of
    Tk_d_ln_omegaEff_dz (WKB_Tk.py:28-69), i.e. with corrected R30.
"""

import sympy as sp

z, k, a0 = sp.symbols("z k a0", positive=True)
H = sp.Function("H")(z)
w = sp.Function("w")(z)  # c_s^2 = wPerturbations
opz = 1 + z

# eps and its derivatives are *independent* functions of z here (eps is fixed by H, but we
# want the expressions the code actually evaluates, which take eps, eps', eps'' from the model)
epsf = sp.Function("eps")(z)

# H is constrained:  dH/dz = H eps/(1+z)   (spec 01 R15)
subsH = {sp.Derivative(H, z): H * epsf / opz}


def dz(expr):
    return sp.expand(sp.diff(expr, z).subs(subsH).doit())


# ---------------------------------------------------------------- (a) omega_eff^2
# R23:  2 P'/P = -(1/(1+z)){eps - 3(1+cs^2)}   ->  work with L = ln P
Lp = -sp.Rational(1, 2) * (epsf - 3 * (1 + w)) / opz  # P'/P
# P''/P = L'' + (L')^2
Lpp = dz(Lp)
Ppp_over_P = Lpp + Lp**2

# R24 check:  P''/P - (P'/P)^2 = (1/2)(eps-3(1+cs^2))/(1+z)^2 - (1/2)(eps' - 6 cs cs')/(1+z)
#             with 2 cs cs' = w'  (R28)  ->  6 cs cs' = 3 w'
R24_spec = sp.Rational(1, 2) * (epsf - 3 * (1 + w)) / opz**2 - sp.Rational(1, 2) * (
    sp.Derivative(epsf, z) - 3 * sp.Derivative(w, z)
) / opz
print("R24 check:", sp.simplify(sp.expand(Ppp_over_P - Lp**2 - R24_spec).doit()))

# R27 definition
omega_def = (
    Ppp_over_P
    + Lp * (epsf - 3 * (1 + w)) / opz
    + (3 * (1 + w) - 2 * epsf) / opz**2
    # the k^2 term of R21/R27 is c_s^2 k^2/(a^2H^2) inside the (1+z)^-2 bracket; with
    # a = a0/(1+z) that is w k^2 (1+z)^2/(a0^2 H^2), so after the 1/(1+z)^2 it is
    # w k^2/(a0^2 H^2) with no (1+z) left (spec 01 R29 / Q8)
    + w * k**2 / (a0**2 * H**2)
)

# R27/R29 compact form
omega_compact = (
    (3 * sp.Derivative(w, z) - sp.Derivative(epsf, z)) / 2 / opz
    + (
        sp.Rational(3, 2) * (1 + epsf) * (1 + w)
        - epsf / 2 * (3 + epsf / 2)
        - sp.Rational(9, 4) * (1 + w) ** 2
    )
    / opz**2
    + w * k**2 / (a0**2 * H**2)
)
print("R27 definition - R27/R29 compact:", sp.simplify(sp.expand((omega_def - omega_compact).doit())))

# code: WKB_Tk.Tk_omegaEff_sq   (k passed in is k_code = k/a0)
k_code = k / a0
code_omega_sq = (
    w * (k_code / H) ** 2
    + (sp.Rational(3, 2) * sp.Derivative(w, z) - sp.Derivative(epsf, z) / 2) / opz
    + (
        sp.Rational(3, 2) * (1 + epsf) * (1 + w)
        - epsf * (3 + epsf / 2) / 2
        - sp.Rational(9, 4) * (1 + w) ** 2
    )
    / opz**2
)
print("code Tk_omegaEff_sq - R29:", sp.simplify(sp.expand((code_omega_sq - omega_compact).doit())))

# ---------------------------------------------------------------- (b) d omega^2/dz
deriv_true = dz(code_omega_sq)

# code: numerator of Tk_d_ln_omegaEff_dz  (should equal d(omega^2)/dz = 2 omega omega')
wp = sp.Derivative(w, z)
wpp = sp.Derivative(w, z, 2)
ep = sp.Derivative(epsf, z)
epp = sp.Derivative(epsf, z, 2)

A = wp * (k_code / H) ** 2
B = (
    sp.Rational(3, 2) * wpp - epp / 2 - 2 * epsf * w * (k_code / H) ** 2
) / opz
C = (
    ep / 2 * (3 * w - epsf + 1) + sp.Rational(3, 2) * wp * (epsf - 3 * (1 + w))
) / opz**2
D = -(
    3 * (1 + epsf) * (1 + w)
    - epsf * (3 + epsf / 2)
    - sp.Rational(9, 2) * (1 + w) ** 2
) / opz**3
code_numerator = A + B + C + D

print(
    "\ncode numerator - d(omega_eff^2)/dz:",
    sp.simplify(sp.expand((code_numerator - deriv_true).doit())),
)

# and the *pre-fix* version, for the record
C_old = (
    ep / 2 * (3 * w - epsf + 1)
    + sp.Rational(3, 2) * wp * (epsf - sp.Rational(3, 2) * (1 + w))
) / opz**2
print(
    "pre-641bb51 numerator - truth     :",
    sp.simplify(sp.expand((A + B + C_old + D - deriv_true).doit())),
)
