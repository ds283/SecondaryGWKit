import os
"""QS_01: sympy check that ComputeTargets/QuadSource.py:35-45 source_function
reproduces spec 03 R22 / §0.3 f exactly.

spec 03 R22 (NUM 03 p.4):
  f = T_q T_r + 2/(3(1+w)) * (T - (1+z) dT/dz)_q * (T - (1+z) dT/dz)_r

code (QuadSource.py:35-44):
  undiff = (5+3w)/(3(1+w)) * Tq*Tr
  diff   = 2/(3(1+w)) * ( -(1+z) Tq Tr' - (1+z) Tr Tq' + (1+z)^2 Tq' Tr' )
"""

import sympy as sp

Tq, Tr, Tqp, Trp, z, w = sp.symbols("Tq Tr Tqp Trp z w", real=True)

one_plus_z = 1 + z

# --- spec form ---
spec_f = Tq * Tr + sp.Rational(2, 3) / (1 + w) * (Tq - one_plus_z * Tqp) * (
    Tr - one_plus_z * Trp
)

# --- code form ---
undiff = (5 + 3 * w) / (3 * (1 + w)) * Tq * Tr
diff = (
    2
    / (3 * (1 + w))
    * (
        -one_plus_z * Tq * Trp
        - one_plus_z * Tr * Tqp
        + one_plus_z**2 * Tqp * Trp
    )
)
code_f = undiff + diff

d = sp.simplify(sp.together(sp.expand(spec_f - code_f)))
print("spec f - code f =", d)
assert d == 0, d
print("QS_01: spec 03 R22 f  ==  QuadSource.source_function  EXACTLY")

# also confirm the undiff/diff split coefficient identity 1 + 2/(3(1+w)) = (5+3w)/(3(1+w))
print(
    "1 + 2/(3(1+w)) - (5+3w)/(3(1+w)) =",
    sp.simplify(1 + 2 / (3 * (1 + w)) - (5 + 3 * w) / (3 * (1 + w))),
)

# symmetry under q <-> r
sym = sp.simplify(code_f - code_f.subs({Tq: Tr, Tr: Tq, Tqp: Trp, Trp: Tqp}, simultaneous=True))
print("code f - code f(q<->r) =", sym)
assert sym == 0
print("QS_01: source_function is exactly symmetric under (q,q') <-> (r,r')")

# What the *code's* split calls "undiff"/"diff" vs the spec's natural split:
# the spec's own T_qT_r term has coefficient 1, the code's "undiff" has (5+3w)/(3(1+w)).
# The difference 2/(3(1+w)) T_qT_r has been moved from "diff" to "undiff".
moved = sp.simplify(undiff - Tq * Tr)
print("code undiff - spec T_qT_r =", sp.simplify(moved), " (= 2/(3(1+w)) TqTr, a regrouping only)")
