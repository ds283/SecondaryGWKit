"""levin-phase-floor prompt 06i Sec 1, Gate R0: the method check on closed forms. New in 06i.

Two integrands on [L, 2L], L = W * 2 pi / omega (W wavelengths):
  (1) g(y) = y^3 exp(-y/L): int g exp(i omega y) dy in closed form, the antiderivative
      exp(b y) sum_k (-1)^k 3!/(3-k)! y^(3-k) / b^(k+1), b = -1/L + i omega, at 50 digits;
  (2) g(y) = exp(i beta / y), beta = 4 L (beta/y runs from 4 to 2 rad): mpmath.quad at 50 digits,
      Gauss-Legendre on pieces of two wavelengths.
omega in {10, 1e3, 1e6, 1e9}; W in {1, 10, 1e3, 1e5} for (1) and {1, 10, 1e3, 1e4} for (2).
Each implementation at its two settings (FCC: N = 32 on 4 panels, N = 48 on 7 panels; QAWO:
P_max = 2^12 rad with epsrel 1e-13, P_max = 2^9 rad with epsrel 1.2e-14). g is evaluated in double
in both, as the samples' amplitudes are.
Cell criterion (prompt Sec 1): |I_impl - I_ref| <= 1e-13 max|g| / omega.
Usage: python -B r0_closed_forms.py   (writes ../out/r0_closed_forms.json and .txt)
"""

import cmath
import json
import math
import multiprocessing as mp_proc
import sys
import time
from fractions import Fraction
from pathlib import Path

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import filon  # noqa: E402
import prov  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"
OMEGAS = [10.0, 1e3, 1e6, 1e9]
W1 = [1, 10, 1000, 100000]
W2 = [1, 10, 1000, 10000]
FCC_SETTINGS = [(32, 4), (48, 7)]
QAWO_SETTINGS = [(2.0**12, 1e-13), (2.0**9, 1.2e-14)]


def reference(test, omega, W):
    with mp.workdps(50):
        om = mp.mpf(omega)
        L = W * 2 * mp.pi / om
        if test == 1:
            b = mp.mpc(-1 / L, om)

            def A(y):
                return mp.exp(b * y) * mp.fsum(
                    (-1) ** k
                    * mp.factorial(3)
                    / mp.factorial(3 - k)
                    * y ** (3 - k)
                    / b ** (k + 1)
                    for k in range(4)
                )

            return +(A(2 * L) - A(L))
        beta = 4 * L
        pts = mp.linspace(L, 2 * L, max(2, int(W // 2) + 2))
        return mp.quad(
            lambda y: mp.expj(om * y + beta / y), pts, method="gauss-legendre"
        )


def cell(args):
    test, omega, W = args
    filon.set_precision()
    t0 = time.time()
    ref = reference(test, omega, W)
    t_ref = time.time() - t0
    Ld = W * 2 * math.pi / omega
    L = mp.mpf(W) * 2 * mp.pi / mp.mpf(omega)
    if test == 1:
        gd = lambda y: (y**3) * math.exp(-y / Ld) + 0j  # noqa: E731
        gmax = (2 * Ld) ** 3 * math.exp(-2.0)
    else:
        beta = 4 * Ld
        gd = lambda y: cmath.exp(1j * beta / y)  # noqa: E731
        gmax = 1.0
    tol = 1e-13 * gmax / omega
    out = {
        "test": test,
        "omega": omega,
        "W": W,
        "ref": mp.nstr(ref, 30),
        "tol": tol,
        "t_ref": t_ref,
    }
    gmp = lambda y: mp.mpc(gd(float(y)))  # noqa: E731
    for N, npan in FCC_SETTINGS:
        t0 = time.time()
        edges = [L + (L * j) / npan for j in range(npan + 1)]
        v = filon.fcc_integrate(gmp, edges, mp.mpf(omega), N)
        out[f"fcc_N{N}_p{npan}"] = {
            "value": mp.nstr(v, 30),
            "err": float(abs(v - ref)),
            "err_over_tol": float(abs(v - ref)) / tol,
            "wall": time.time() - t0,
        }
    for Pmax, er in QAWO_SETTINGS:
        t0 = time.time()
        h_max = min(Pmax / (2 * omega), Ld / 16)
        panels = filon.dyadic_tiling(L, 2 * L, h_max)
        v, st = filon.qawo_integrate(
            gd,
            panels,
            Fraction(omega),
            lambda c, h: 1e-17 * gmax * min(2 * h, 2 / omega),
            er,
        )
        out[f"qawo_P{int(Pmax)}"] = {
            "value": mp.nstr(v, 30),
            "err": float(abs(v - ref)),
            "err_over_tol": float(abs(v - ref)) / tol,
            "wall": time.time() - t0,
            **st,
        }
    return out


def main():
    jobs = [(1, o, W) for o in OMEGAS for W in W1] + [
        (2, o, W) for o in OMEGAS for W in W2
    ]
    t0 = time.time()
    with mp_proc.get_context("spawn").Pool(6) as pool:
        rows = pool.map(cell, jobs, chunksize=1)
    keys = [f"fcc_N{N}_p{p}" for N, p in FCC_SETTINGS] + [
        f"qawo_P{int(P)}" for P, _ in QAWO_SETTINGS
    ]
    worst = {k: max(r[k]["err_over_tol"] for r in rows) for k in keys}
    met = all(r[k]["err_over_tol"] <= 1 for r in rows for k in keys)
    res = {
        "header": prov.header(),
        "wall_total": time.time() - t0,
        "all_cells_met": met,
        "worst_err_over_tol": worst,
        "cells": rows,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "r0_closed_forms.json").write_text(json.dumps(res, indent=1))
    lines = [
        json.dumps(res["header"], indent=1),
        "",
        "test omega W | " + " | ".join(keys) + "   (|err| / (1e-13 max|g|/omega))",
    ]
    for r in rows:
        lines.append(
            f"{r['test']} {r['omega']:.0e} {r['W']:>6} | "
            + " | ".join(f"{r[k]['err_over_tol']:.3g}" for k in keys)
        )
    lines.append(f"\nworst: {worst}\nall cells met by both implementations: {met}")
    (OUT / "r0_closed_forms.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[2:]))


if __name__ == "__main__":
    main()
