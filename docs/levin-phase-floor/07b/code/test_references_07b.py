"""
levin-phase-floor prompt 07b, Part B: the references of AdaptiveLevin/tests/test_levin_rate_rules.py that are not
computed in the tests themselves. Each is a closed form, evaluated in mpmath at 40 and at 60 digits; the two
readings' relative agreement is recorded beside each value, and the value the tests use is the 60-digit reading
rounded to the nearest double (float hex).

- test (b), the edge mechanism: on [0, 1], theta = (c/2)(x - x_s)^2, c = 1.19e9, x_s = 0.8186383592305085,
  f_sin = e^{g x}, g = 0.744, f_cos = cos(h x), h = 109.28. With J(alpha) = int_0^1 e^{alpha x} e^{i theta} dx,
      I = Im J(g) + Re[J(i h) + J(-i h)] / 2,
  and J by completing the square (07's docs/levin-phase-floor/07/calibration/scripts/reference.py, closed_R, with
  omega = 0): with beta = -i c / 2, u = x - x_s,
      J(alpha) = e^{alpha x_s} e^{i alpha^2 / 2c} sqrt(pi) / (2 sqrt(beta))
                 [erf(sqrt(beta)(1 - x_s - i alpha / c)) - erf(sqrt(beta)(-x_s - i alpha / c))].
  It is also evaluated on the parent's carrying leaf and on every leaf of the parent's partition, for the
  test's attribution (out/test_references_07b.json, "edge_leaves").
- test (c), the range mechanism: test (b)'s integrand on the span of the parent's carrying leaf, [x_0, x_1] =
  [0x1.a317190ba9ba3p-2, 0x1.a317190ba9ba3p-1] (EDGE_LEAVES[1]), a call of its own; theta' = c (x - x_s) has no
  zero on it (x_1 lies 1.03e-4 below x_s), so nothing is located. The same closed form with x_0, x_1 in place of
  0, 1; its reading is "edge_leaves"[1], repeated as "range".
Every parameter is taken as the double the test passes, converted exactly to mpmath.

Run from the repository root:  ./venv/bin/python docs/levin-phase-floor/07b/code/test_references_07b.py
It writes docs/levin-phase-floor/07b/code/out/test_references_07b.json. No Ray and no datastore.
"""

import hashlib
import json
from pathlib import Path

import mpmath as mp

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "test_references_07b.json"

# test (b)
EDGE_C, EDGE_XS, EDGE_G, EDGE_H = 1.19e9, 0.8186383592305085, 0.744, 109.28
# the parent's partition of test (b)'s call (atol 1e-7, rtol 1e-7, order 24), as float hex, its carrying leaf second
EDGE_LEAVES = [
    ("0x0.0p+0", "0x1.a317190ba9ba3p-2"),
    ("0x1.a317190ba9ba3p-2", "0x1.a317190ba9ba3p-1"),
    ("0x1.a317190ba9ba3p-1", "0x1.a32491284ad21p-1"),
    ("0x1.a32491284ad21p-1", "0x1.a3320944ebe9fp-1"),
    ("0x1.a3320944ebe9fp-1", "0x1.a61878fac48aap-1"),
    ("0x1.a61878fac48aap-1", "0x1.a8fee8b09d2b6p-1"),
    ("0x1.a8fee8b09d2b6p-1", "0x1.aecbc81c4e6ccp-1"),
    ("0x1.aecbc81c4e6ccp-1", "0x1.ba6586f3b0ef8p-1"),
    ("0x1.ba6586f3b0ef8p-1", "0x1.d19904a275f50p-1"),
    ("0x1.d19904a275f50p-1", "0x1.0000000000000p+0"),
]

# test (c): the parent's carrying leaf's span
RANGE_SPAN = EDGE_LEAVES[1]


def J_quadratic(alpha, c, xs, x0, x1):
    beta = mp.mpc(0, -1) * c / 2
    sb = mp.sqrt(beta)
    shift = mp.mpc(0, 1) * alpha / c
    pref = (
        mp.exp(alpha * xs)
        * mp.exp(mp.mpc(0, 1) * alpha**2 / (2 * c))
        * mp.sqrt(mp.pi)
        / (2 * sb)
    )
    return pref * (mp.erf(sb * (x1 - xs - shift)) - mp.erf(sb * (x0 - xs - shift)))


def reference_edge(x0=0.0, x1=1.0):
    c, xs = mp.mpf(EDGE_C), mp.mpf(EDGE_XS)
    g, h = mp.mpf(EDGE_G), mp.mpf(EDGE_H)
    x0, x1 = mp.mpf(x0), mp.mpf(x1)

    def J(alpha):
        return J_quadratic(alpha, c, xs, x0, x1)

    return mp.im(J(g)) + mp.re(J(mp.mpc(0, h)) + J(mp.mpc(0, -h))) / 2


def reading(fn):
    readings = {}
    for dps in (40, 60):
        with mp.workdps(dps):
            readings[dps] = fn()
    with mp.workdps(60):
        agreement = abs(readings[40] - readings[60]) / abs(readings[60])
        return {
            "value_60": mp.nstr(readings[60], 45),
            "value_40": mp.nstr(readings[40], 35),
            "relative_agreement": mp.nstr(agreement, 5),
            "float_hex": float(readings[60]).hex(),
        }


def main():
    out = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mpmath_version": mp.__version__,
        "references": {},
    }
    out["references"]["edge"] = reading(reference_edge)
    print("edge", out["references"]["edge"])
    out["edge_leaves"] = []
    for lo_hex, hi_hex in EDGE_LEAVES:
        lo, hi = float.fromhex(lo_hex), float.fromhex(hi_hex)
        entry = {
            "span_hex": [lo_hex, hi_hex],
            **reading(lambda: reference_edge(lo, hi)),
        }
        out["edge_leaves"].append(entry)
        print("edge leaf", lo, hi, entry["float_hex"], entry["relative_agreement"])
    lo, hi = (float.fromhex(v) for v in RANGE_SPAN)
    out["references"]["range"] = {
        "span_hex": list(RANGE_SPAN),
        **reading(lambda: reference_edge(lo, hi)),
    }
    print("range", out["references"]["range"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
