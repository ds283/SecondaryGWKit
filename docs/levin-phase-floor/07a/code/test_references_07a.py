"""
levin-phase-floor prompt 07a, Part B: the references of the new tests that are not computed in the tests
themselves (AdaptiveLevin/tests/test_levin_phase_variation.py test 2, AdaptiveLevin/tests/test_levin_stationary.py
test 14 (d)). Each is a closed form, evaluated in mpmath at 40 and at 60 digits; the two readings' relative
agreement is recorded beside each value, and the value used by the tests is the 60-digit reading rounded to
the nearest double (float hex).

- test 2: theta = e^{alpha x} on [0, 1], f = (1, 0), alpha = log1p(0.98 * 6 pi) as a double:
      I = int_0^1 sin(e^{alpha x}) dx = (Si(e^alpha) - Si(1)) / alpha.
- test 14 (d): theta = theta_0 + (c/2)(x - x_s)^2 on [0, 1], theta_0 = 1e8, c = 1e6, x_s = 0.3819660112501051,
  f = (1, 0): with k = sqrt(c / pi), the Fresnel integrals C(z) = int_0^z cos(pi t^2 / 2) dt and S likewise,
      C_q = sqrt(pi / c) (C(k (1 - x_s)) - C(-k x_s)),  S_q = sqrt(pi / c) (S(k (1 - x_s)) - S(-k x_s)),
      I = sin(theta_0) C_q + cos(theta_0) S_q.
Every parameter is taken as the double the test passes, converted exactly to mpmath.

Run from the repository root:  ./venv/bin/python docs/levin-phase-floor/07a/code/test_references_07a.py
It writes docs/levin-phase-floor/07a/code/out/test_references_07a.json. No Ray and no datastore.
"""

import hashlib
import json
import math
from pathlib import Path

import mpmath as mp

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "test_references_07a.json"

ALPHA_T2 = math.log1p(0.98 * 6.0 * math.pi)
T14D_THETA0, T14D_C, T14D_XS = 1.0e8, 1.0e6, 0.3819660112501051


def reference_t2():
    alpha = mp.mpf(ALPHA_T2)
    return (mp.si(mp.exp(alpha)) - mp.si(1)) / alpha


def reference_t14d():
    t0, c, xs = mp.mpf(T14D_THETA0), mp.mpf(T14D_C), mp.mpf(T14D_XS)
    k = mp.sqrt(c / mp.pi)
    scale = mp.sqrt(mp.pi / c)
    C = scale * (mp.fresnelc(k * (1 - xs)) - mp.fresnelc(-k * xs))
    S = scale * (mp.fresnels(k * (1 - xs)) - mp.fresnels(-k * xs))
    return mp.sin(t0) * C + mp.cos(t0) * S


def main():
    out = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mpmath_version": mp.__version__,
        "alpha_t2_hex": float(ALPHA_T2).hex(),
        "references": {},
    }
    for name, fn in (("t2", reference_t2), ("t14d", reference_t14d)):
        readings = {}
        for dps in (40, 60):
            mp.mp.dps = dps
            readings[dps] = fn()
        mp.mp.dps = 60
        agreement = abs(readings[40] - readings[60]) / abs(readings[60])
        out["references"][name] = {
            "value_60": mp.nstr(readings[60], 45),
            "value_40": mp.nstr(readings[40], 35),
            "relative_agreement": mp.nstr(agreement, 5),
            "float_hex": float(readings[60]).hex(),
        }
        print(name, out["references"][name])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
