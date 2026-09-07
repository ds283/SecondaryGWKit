import os
"""QS_04:
(a) exercise compute_quad_source's z-grid alignment loop (ComputeTargets/QuadSource.py:82-102)
    with mock Tq/Tr whose z_sample is the *actual* shape main.py gives it -- shorter than
    z_sample at BOTH ends -- and record what happens.
(b) sympy: confirm the change of variable used by QuadSourceIntegral.numeric_quad_integral
    (ComputeTargets/QuadSourceIntegral.py:954-975) reproduces spec 03 R28's
    int dz' Gbar (1+z)/(1+z') f/H^2  up to the deferred Q_s/a_0^2.
"""

import sys, types

sys.path.insert(0, os.getcwd())  # run from the repository root

# ------------------------------------------------------------------ (a)
# stand-ins with just the attributes the loop touches
class Z:
    def __init__(self, z, sid):
        self.z = z
        self.store_id = sid


class TkVal:
    T = 1.0
    Tprime = 0.0
    analytic_T_rad = 1.0
    analytic_Tprime_rad = 0.0
    analytic_T_w = 1.0
    analytic_Tprime_w = 0.0


class MockTk:
    def __init__(self, zs):
        self.z_sample = zs

    def __getitem__(self, i):
        return self.z_sample[i] and TkVal()


class MockModelFns:
    def wBackground(self, z):
        return 1.0 / 3.0


class MockModel:
    functions = MockModelFns()


class MockProxy:
    def get(self):
        return MockModel()


from ComputeTargets.QuadSource import compute_quad_source

# undo the @ray.remote wrapper so we can call the function body directly
fn = compute_quad_source._function

# full source grid: 12 redshifts
full = [Z(10.0 ** (5 - 0.5 * i), 100 + i) for i in range(12)]

# Tk grid as main.py:505-507 builds it: truncated at the top (z > z_exit_suph_e5 dropped)
# AND at the bottom (z < 0.85*z_exit_subh_e6 dropped; plus "stop" mode shortens it further,
# main.py:528).  Here: drop the first 2 and the last 4.
tk = MockTk(full[2:8])

print("(a) len(z_sample) =", len(full), "  len(Tq.z_sample) =", len(tk.z_sample))
try:
    out = fn(MockProxy(), full, tk, tk)
    print("    returned", len(out["source"]), "source values (expected", len(full), ")")
except Exception as e:
    print(f"    RAISED {type(e).__name__}: {e}")

# control: Tk grid missing only leading (high-z) entries -- the case the code handles
tk2 = MockTk(full[2:])
print("\n    control, Tk missing only the 2 leading high-z samples:")
try:
    out = fn(MockProxy(), full, tk2, tk2)
    print("      OK,", len(out["source"]), "values; first two use the T=1,T'=0 default:",
          out["source"][:3])
except Exception as e:
    print(f"      RAISED {type(e).__name__}: {e}")

# ------------------------------------------------------------------ (b)
print("\n(b) Jacobian of the source integral")
import sympy as sp

z, zp = sp.symbols("z zp", positive=True)
G, f, Hf = sp.Function("G"), sp.Function("f"), sp.Function("H")
u = sp.Symbol("u", positive=True)  # u = log(1+z')

spec = G(z, zp) * (1 + z) / (1 + zp) * f(zp) / Hf(zp) ** 2  # integrand of R28 (dz')
# substitute z' = e^u - 1, dz' = e^u du = (1+z') du
code = ((1 + z) * G(z, zp) * f(zp) / Hf(zp) ** 2).subs(zp, sp.exp(u) - 1)
spec_in_u = sp.simplify(spec.subs(zp, sp.exp(u) - 1) * sp.exp(u))
print("   spec integrand * dz'/du =", spec_in_u)
print("   code integrand (Green*f/H^2) * (1+z_response) =", sp.simplify(code))
print("   difference =", sp.simplify(spec_in_u - code))
