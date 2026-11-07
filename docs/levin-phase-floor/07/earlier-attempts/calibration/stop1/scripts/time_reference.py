"""levin-phase-floor 07 section 2, the orchestrator's item 3: time the prompt's reference on the
largest case of each family, before anything else is run.

The prompt's reference is ``mpmath.quad`` at 30 digits, split at every zero of theta' and at x_0.
On x in [0, 1], with f_sin = exp(-x) and f_cos = cos(3x):
- (R) theta = (c/2)(x - x_s)^2 (omega = 0), largest case c = 1e8, x_s = 0.7236067977499790;
- (C) theta = a(x - x_0) + (b/3)(x - x_0)^3, x_0 = 0.3819660112501051, largest case b = 1e8,
  lambda = a^(3/2)/b^(1/2) = 1024.

Two measurements per case:
1. **As written:** one ``mpmath.quad`` call over the split interval, with ``error=True``, timed.
   Its own error estimate says whether it converged.
2. **The same, made to converge by subdivision:** ``mpmath.quad`` on consecutive pieces of at most
   pi rad of phase each, timed on 200 consecutive pieces near x = 0.9, and extrapolated to the
   case's total variation of theta divided by pi.

Writes ``out/time_reference.json``.
"""

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mpmath as mp  # noqa: E402

from prov import OUT, header  # noqa: E402

mp.mp.dps = 30

X0 = mp.mpf("0.3819660112501051")
XS_R = mp.mpf("0.7236067977499790")


def case_R(c):
    c = mp.mpf(c)

    def theta(x):
        return c / 2 * (x - XS_R) ** 2

    def rate(x):
        return c * (x - XS_R)

    tv = c / 2 * (XS_R**2 + (1 - XS_R) ** 2)
    return theta, rate, [0, XS_R, 1], tv


def case_C(b, lam):
    b = mp.mpf(b)
    a = mp.cbrt(mp.mpf(lam) ** 2 * b)

    def theta(x):
        u = x - X0
        return a * u + b / 3 * u**3

    def rate(x):
        u = x - X0
        return a + b * u**2

    tv = theta(mp.mpf(1)) - theta(mp.mpf(0))
    return theta, rate, [0, X0, 1], tv


def measure(name, theta, rate, split, tv):
    def g(x):
        t = theta(x)
        return mp.exp(-x) * mp.sin(t) + mp.cos(3 * x) * mp.cos(t)

    t0 = time.perf_counter()
    value, err = mp.quad(g, split, error=True)
    literal_s = time.perf_counter() - t0

    n_probe = 200
    a = mp.mpf("0.9")
    t0 = time.perf_counter()
    for _ in range(n_probe):
        b = a + mp.pi / abs(rate(a))
        mp.quad(g, [a, b])
        a = b
    per_piece = (time.perf_counter() - t0) / n_probe
    n_pieces = float(tv / mp.pi)
    rec = {
        "case": name,
        "total_variation_rad": float(tv),
        "as_written": {
            "value": mp.nstr(value, 30),
            "mpmath_error_estimate": mp.nstr(err, 5),
            "wall_s": literal_s,
        },
        "subdivided": {
            "probe_pieces": n_probe,
            "seconds_per_piece": per_piece,
            "pieces_needed": n_pieces,
            "projected_wall_s": per_piece * n_pieces,
            "projected_wall_h": per_piece * n_pieces / 3600.0,
        },
    }
    print(json.dumps(rec, indent=1), flush=True)
    return rec


def main():
    out = {"header": header(Path(__file__), "07 section 2: timing the reference")}
    out["cases"] = [
        measure("R c=1e8 x_s=0.7236067977499790", *case_R(1e8)),
        measure("C b=1e8 lambda=1024", *case_C(1e8, 1024)),
    ]
    path = OUT / "time_reference.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
