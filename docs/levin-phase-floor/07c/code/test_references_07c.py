"""
levin-phase-floor prompt 07c, Part B: the references of section 3's tests in AdaptiveLevin/tests/test_levin_chase_stop.py.

Each test problem is I(A, kappa) = int_0^1.5 [e^{-x} sin(A e^{kappa x}) + cos(x) cos(A e^{kappa x})] dx, the integrand
without the tests' deterministic noise (the tests bound the noise's contribution separately: at most eta times
int_0^1.5 (e^{-x} + |cos x|) dx = eta (1 - e^{-1.5} + sin 1.5)). Computed by mpmath's tanh-sinh quadrature on the
intervals between consecutive zeros of sin(theta) (theta = A e^{kappa x} crossing each multiple of pi; for kappa = 0
none), at 30 and at 45 significant digits; the two must agree to far below a double's resolution, and their
difference is recorded as the reference's own error.

  (b), (c), (d), (e):  A = 20, kappa = 4
  (c3):                A = 200, kappa = 1
  (f2):                A = 2, kappa = 4

Run from the repository root:  ./venv/bin/python docs/levin-phase-floor/07c/code/test_references_07c.py
Writes docs/levin-phase-floor/07c/code/out/test_references_07c.json. No Ray and no datastore.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import mpmath as mp

HERE = Path(__file__).resolve().parent
PROBLEMS = {"b": (20, 4), "c3": (200, 1), "f2": (2, 4)}
L = "1.5"


def integral(A, kappa, dps):
    mp.mp.dps = dps
    A, kappa, span = mp.mpf(A), mp.mpf(kappa), mp.mpf(L)
    theta = lambda x: A * mp.exp(kappa * x)
    g = lambda x: mp.exp(-x) * mp.sin(theta(x)) + mp.cos(x) * mp.cos(theta(x))
    first = int(mp.floor(theta(0) / mp.pi)) + 1
    last = int(mp.floor(theta(span) / mp.pi))
    cuts = [mp.log(j * mp.pi / A) / kappa for j in range(first, last + 1)]
    points = [mp.mpf(0)] + [c for c in cuts if 0 < c < span] + [span]
    return (
        mp.fsum(mp.quad(g, [a, b]) for a, b in zip(points[:-1], points[1:])),
        len(points) - 1,
    )


def main():
    out = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE
        ).stdout.strip(),
        "mpmath": mp.__version__,
        "problems": {},
    }
    for name, (A, kappa) in PROBLEMS.items():
        v30, n = integral(A, kappa, 30)
        v45, _ = integral(A, kappa, 45)
        mp.mp.dps = 45
        diff = abs(v30 - v45)
        value = float(v45)
        out["problems"][name] = {
            "A": A,
            "kappa": kappa,
            "span": [0, 1.5],
            "pieces": n,
            "value_30": mp.nstr(v30, 30),
            "value_45": mp.nstr(v45, 45),
            "agreement": mp.nstr(diff, 5),
            "agreement_relative": mp.nstr(diff / abs(v45), 5),
            "value_double_hex": value.hex(),
            "value_double": repr(value),
        }
        print(name, out["problems"][name])
    mp.mp.dps = 45
    out["noise_integral"] = mp.nstr(1 - mp.exp(-mp.mpf(L)) + mp.sin(mp.mpf(L)), 20)
    (HERE / "out" / "test_references_07c.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
