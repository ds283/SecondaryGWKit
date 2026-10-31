"""levin-phase-floor prompt 06h, section 0.2 item 2: the measurement of eps_exp, the relative error
of the `exp` that LiouvilleGreen.three_bessel_integrals.BesselPhaseGroup calls (`np.exp`, on a 0-d
argument and on an array).

The platform library's accuracy is not documented: macOS's exp(3) manual page states none, and
numpy 2.2.4 on aarch64 calls the platform's exp for float64 (this script shows its scalar and array
results bit-identical to math.exp on the whole sample). So eps_exp is measured, as the prompt
allows: on a seeded sample of 10^6 arguments over each of two ranges of u,

  production   u = log(eta) in [-30, 5]: 06f's 20 M2 calls (the analytic block's
               _three_bessel_Levin) span u in [-17.38, -1.79], widened by a margin on each side;
  LiouvilleGreen  u = log(x) in [-2, log(1e12)]: quad_JJJ/quad_YJJ's Levin part runs from
               min_cut = x_cut/min(k, q, s) (above e^-2 for every draw of test_3bessel_analytic.py)
               to max_x <= 1e12;

the relative error |fl(exp(u)) - e^u| / e^u against mpmath at 30 digits, for np.exp on a Python
float, on an np.float64, and on the whole sample as one array; and math.exp, for comparison. The
worst is reported with the point that attains it, and in ulps of the result.

Descriptive, in support of the angle-addition and atan2 terms of phi_eval (which are derived from
an assumed accuracy of one ulp for sin, cos and atan2): the worst error in ulps of math.sin/cos and
np.sin/cos on seeded large arguments (log-uniform in [1e-2, 1e13]) and small ones (uniform in
[-20, 20]), and of math.atan2/np.arctan2 on seeded pairs, each against mpmath at 30 digits.

Usage, from the repository root: PYTHONPATH=. ./venv/bin/python <this script>. Writes
../out/exp_relerr.txt. No Ray, no datastore.
"""

import hashlib
import math
import sys
import time
from pathlib import Path

import mpmath
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prov  # noqa: E402

OUT = prov.OUT / "exp_relerr.txt"

SEED = 20261031
N = 1_000_000
RANGES = {
    "production": (-30.0, 5.0),
    "LiouvilleGreen": (-2.0, math.log(1.0e12)),
}
N_TRIG = 100_000


def sample(name):
    """The seeded sample of u for one range; the unit test test_group_phase_rounding draws the same."""
    lo, hi = RANGES[name]
    rng = np.random.default_rng([SEED, list(RANGES).index(name)])
    return rng.uniform(lo, hi, N)


def rel_errors(u, x):
    """|x - e^u| / e^u and |x - e^u| / ulp(x), at 30 digits, for every point."""
    rel = np.empty(u.shape)
    ulps = np.empty(u.shape)
    with mpmath.workdps(30):
        for i, (ui, xi) in enumerate(zip(u.tolist(), x.tolist())):
            e = mpmath.exp(mpmath.mpf(ui))
            d = abs(mpmath.mpf(xi) - e)
            rel[i] = float(d / e)
            ulps[i] = float(d / mpmath.mpf(float(np.spacing(xi))))
    return rel, ulps


def ulp_error(values, references):
    out = np.empty(len(values))
    with mpmath.workdps(30):
        for i, (v, r) in enumerate(zip(values, references)):
            spacing = float(np.spacing(abs(float(r))))
            out[i] = float(abs(mpmath.mpf(float(v)) - r) / spacing)
    return out


def main():
    lines = prov.header(
        __file__, "section 0.2 item 2, eps_exp measured", inputs=(prov.THREE_BESSEL,)
    )
    lines.append(f"# numpy {np.__version__}, mpmath {mpmath.__version__}")
    lines.append(f"# seed {SEED}; {N} points per range; ranges {RANGES}")
    lines.append("")
    start = time.perf_counter()
    overall = 0.0
    for name in RANGES:
        u = sample(name)
        digest = hashlib.sha256(u.tobytes()).hexdigest()
        x_array = np.exp(u)
        x_float = np.array([np.exp(v) for v in u.tolist()])
        x_np64 = np.array([np.exp(np.float64(v)) for v in u.tolist()])
        x_math = np.array([math.exp(v) for v in u.tolist()])
        lines.append(f"== range {name} {RANGES[name]}: sample sha256 {digest}")
        lines.append(
            "   bit-identical to the array path: np.exp(float)"
            f" {np.array_equal(x_float.view(np.int64), x_array.view(np.int64))},"
            f" np.exp(np.float64) {np.array_equal(x_np64.view(np.int64), x_array.view(np.int64))},"
            f" math.exp {np.array_equal(x_math.view(np.int64), x_array.view(np.int64))}"
        )
        for label, x in (("np.exp array", x_array), ("np.exp scalar", x_float)):
            rel, ulps = rel_errors(u, x)
            i = int(np.argmax(rel))
            j = int(np.argmax(ulps))
            overall = max(overall, float(rel[i]))
            lines.append(
                f"   {label:14s} worst relative error {rel[i]:.6e} ({rel[i] / 2.0**-53:.6f} u) at"
                f" u = {float(u[i]).hex()} ({u[i]!r}); worst in ulps of the result {ulps[j]:.6f}"
                f" at u = {u[j]!r}; median relative {np.median(rel):.3e}; points above u/2:"
                f" {int(np.sum(rel > 2.0**-54))}"
            )
    lines.append("")
    lines.append(
        f"worst relative error over both ranges and both paths: {overall:.6e}"
        f" = {overall / 2.0**-53:.6f} u  (u = 2^-53)"
    )
    lines.append(f"   repr {overall!r}, hex {overall.hex()}")

    lines.append("")
    lines.append(
        "== descriptive: sin, cos and atan2 in ulps of the result, against mpmath at 30 digits"
    )
    rng = np.random.default_rng([SEED, 99])
    big = 10.0 ** rng.uniform(-2.0, 13.0, N_TRIG)
    small = rng.uniform(-20.0, 20.0, N_TRIG)
    pairs = rng.uniform(-1.0, 1.0, (N_TRIG, 2))
    for label, args in (("large args", big), ("small args", small)):
        with mpmath.workdps(30):
            sin_ref = [mpmath.sin(mpmath.mpf(a)) for a in args.tolist()]
            cos_ref = [mpmath.cos(mpmath.mpf(a)) for a in args.tolist()]
        for fname, values, ref in (
            ("math.sin", [math.sin(a) for a in args.tolist()], sin_ref),
            ("np.sin", np.sin(args), sin_ref),
            ("math.cos", [math.cos(a) for a in args.tolist()], cos_ref),
            ("np.cos", np.cos(args), cos_ref),
        ):
            err = ulp_error(values, ref)
            lines.append(
                f"   {label:10s} {fname:8s} worst {np.max(err):.4f} ulp; points above 0.5 ulp"
                f" {int(np.sum(err > 0.5))}"
            )
        same = np.array_equal(
            np.array([math.sin(a) for a in args.tolist()]).view(np.int64),
            np.sin(args).view(np.int64),
        ) and np.array_equal(
            np.array([math.cos(a) for a in args.tolist()]).view(np.int64),
            np.cos(args).view(np.int64),
        )
        lines.append(f"   {label:10s} math and np bit-identical: {same}")
    with mpmath.workdps(30):
        atan_ref = [
            mpmath.atan2(mpmath.mpf(s), mpmath.mpf(c)) for s, c in pairs.tolist()
        ]
    for fname, values in (
        ("math.atan2", [math.atan2(s, c) for s, c in pairs.tolist()]),
        ("np.arctan2", np.arctan2(pairs[:, 0], pairs[:, 1])),
    ):
        err = ulp_error(values, atan_ref)
        lines.append(
            f"   pairs      {fname:10s} worst {np.max(err):.4f} ulp; points above 0.5 ulp"
            f" {int(np.sum(err > 0.5))}"
        )

    lines.append("")
    lines.append(f"# wall time {time.perf_counter() - start:.1f} s")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
