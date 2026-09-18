"""Kohri & Terada (arXiv:1804.08577) as an exact oracle for the source integral.

Runs from the repository root with no arguments, no Ray and no datastore:

    PYTHONPATH=. ./venv/bin/python docs/radiation-oracle/kt_verification.py

Every number in `KOHRI-TERADA-ORACLE.md` from its section 4 down is this script's
stdout.

THREE OBJECTS APPEAR HERE AND TWO OF THEM ARE THIS SCRIPT'S OWN. Kohri & Terada
published no code; `I_RD` (eq. 22) and `I_by_quadrature` (eq. 15) are both this
script's transcription of their published equations, and they share the Green's
function sin(x - xbar), the measure xbar/x and the source f. So checks (1)-(3)
below CANNOT detect a misreading of the kernel -- both objects would be wrong
together. Only check (4), against this repository's pre-existing pipeline, can.
Read section 0 of the document before trusting any independence claim here.

  1. KT eq. (16), the general-`w` source, reproduces KT eq. (20), the explicit
     radiation-era one, at `w = 1/3`. Two separate formulas in the paper for the
     same function, so this one DOES guard the source independently.
  2. KT eq. (22), the closed form for `I_RD`, reproduces `scipy.quad` of KT
     eq. (15) built from eqs. (16), (18) and (19). This guards the transcription
     of eq. (22) given the kernel; it does not guard the kernel.
  3. The small-`x` and large-`x` limits behave as they must -- and the small-`x`
     limit is `2 x^2 / 9`, NOT the `x^2 / 2` the paper states below eq. (24).
  4. This repository's own `total`, from `evaluate_QuadSource_integral` through
     the `test_quadsource_integral` fixture at `b = 0`, equals

         total = -(9/8) * I_RD(r/k, q/k, k tau(z_resp)) / k_phys^2

     on every one of the nine `b = 0` cases. This is the only check with an
     independent object in it, and so the only evidence the kernel is right.

The `-9/8` is the whole point. Its factorisation is `-(1/2) * (9/4)`: the `9/4`
is `1/c^2` with `c = (2+b)/(3+2b) = 2/3` at `b = 0` (spec 03 section 0.1, which KT
fold into their `f`), and the `1/2` is the author's `h^us_ij = h^them_ij / 2`
relative to Kohri-Terada, which `docs/spec/05-one-loop.md` section 539 records as
transcribed and never checked and which `docs/spec/cross-spec-check.md` section 3
item 9 records as a check no spec allows. The sign is the orientation convention
of `docs/spec/04-source-integral.md` section 0 (3).
"""

import math

from scipy.integrate import quad
from scipy.special import sici

SQRT3 = math.sqrt(3.0)
EULER_GAMMA = 0.5772156649015328606

#: The reference tolerance pair. Far tighter than production; the code's own
#: quadrature error is then orders below the agreement being reported.
REF_ATOL, REF_RTOL = 1e-45, 1e-12

#: The measured normalisation. total = KT_NORM * I_RD / k_phys^2.
KT_NORM = -9.0 / 8.0


# --------------------------------------------------------------------------
# KT eq. (19): the radiation-era transfer function, and its derivative.
#
#   Phi(x) = (9/x^2) [ sin(x/sqrt3)/(x/sqrt3) - cos(x/sqrt3) ],  Phi(0) = 1
#
# The closed form is 0/0 at x = 0 and has lost most of its digits well before
# it, so both carry a series branch.
# --------------------------------------------------------------------------

_SERIES_CUTOFF = 1e-2


def Phi(x: float) -> float:
    if abs(x) < _SERIES_CUTOFF:
        x2 = x * x
        return 1.0 - x2 / 30.0 + x2 * x2 / 2520.0
    y = x / SQRT3
    return 9.0 / (x * x) * (math.sin(y) / y - math.cos(y))


def dPhi(x: float) -> float:
    """d Phi / d x."""
    if abs(x) < _SERIES_CUTOFF:
        return -x / 15.0 + x * x * x / 630.0
    y = x / SQRT3
    g = math.sin(y) / y - math.cos(y)
    gp = (math.cos(y) / y - math.sin(y) / (y * y) + math.sin(y)) / SQRT3
    return 9.0 * gp / (x * x) - 18.0 * g / (x * x * x)


# --------------------------------------------------------------------------
# KT eq. (16): the general-w source. "xbar d_etabar Phi" is read as the
# dimensionless etabar d_etabar = xbar d_xbar; check (1) is what justifies that
# reading.
# --------------------------------------------------------------------------


def f_general_w(v: float, u: float, x: float, w: float = 1.0 / 3.0) -> float:
    c1 = 6.0 * (w + 1.0) / (3.0 * w + 5.0)
    c2 = 6.0 * (1.0 + 3.0 * w) * (w + 1.0) / (3.0 * w + 5.0) ** 2
    c3 = 3.0 * (1.0 + 3.0 * w) ** 2 * (1.0 + w) / (3.0 * w + 5.0) ** 2

    Pv, Pu = Phi(v * x), Phi(u * x)
    xdPv, xdPu = x * v * dPhi(v * x), x * u * dPhi(u * x)
    return c1 * Pv * Pu + c2 * (xdPv * Pu + xdPu * Pv) + c3 * xdPv * xdPu


# --------------------------------------------------------------------------
# KT eq. (20): the explicit radiation-era source.
# --------------------------------------------------------------------------


def f_RD(v: float, u: float, x: float) -> float:
    cu, su = math.cos(u * x / SQRT3), math.sin(u * x / SQRT3)
    cv, sv = math.cos(v * x / SQRT3), math.sin(v * x / SQRT3)
    x2 = x * x
    bracket = (
        18.0 * u * v * x2 * cu * cv
        + (54.0 - 6.0 * (u * u + v * v) * x2 + u * u * v * v * x2 * x2) * su * sv
        + 2.0 * SQRT3 * u * x * (v * v * x2 - 9.0) * cu * sv
        + 2.0 * SQRT3 * v * x * (u * u * x2 - 9.0) * su * cv
    )
    return 12.0 / (u**3 * v**3 * x**6) * bracket


# --------------------------------------------------------------------------
# KT eq. (22), the closed form, with the resonance regularised.
#
# As written, eq. (22) contains -Ci(|1 - (u+v)/sqrt3| x) and
# + log|(3 - (u+v)^2)/(3 - (u-v)^2)|. Both diverge as u + v -> sqrt3 and their
# SUM does not. Writing Cin(z) = gamma + ln z - Ci(z), which is analytic and
# starts at z^2/4,
#
#   -Ci(|c|x) + log|...| = -gamma - ln x + Cin(|c|x)
#                          + log(1 + (u+v)/sqrt3) - log|a b|
#
# with c = 1 - (u+v)/sqrt3, a = 1 - (v-u)/sqrt3, b = 1 + (v-u)/sqrt3. The ln|c|
# cancels analytically and the result is regular AT the resonance, where the
# unregularised form returns +-inf.
# --------------------------------------------------------------------------


def Si(x: float) -> float:
    return sici(x)[0]


def Ci(x: float) -> float:
    return sici(abs(x))[1]


def Cin(z: float) -> float:
    """gamma + ln z - Ci(z) = int_0^z (1 - cos t)/t dt, stable at small z."""
    z = abs(z)
    if z < 1e-2:
        z2 = z * z
        return z2 / 4.0 - z2 * z2 / 96.0 + z2 * z2 * z2 / 4320.0
    return EULER_GAMMA + math.log(z) - sici(z)[1]


def I_RD(v: float, u: float, x: float) -> float:
    """KT eq. (22). Regular at u + v = sqrt(3)."""
    d = u * u + v * v - 3.0
    cu, su = math.cos(u * x / SQRT3), math.sin(u * x / SQRT3)
    cv, sv = math.cos(v * x / SQRT3), math.sin(v * x / SQRT3)

    rational = (
        u * v * d * x**3 * math.sin(x)
        - 6.0 * u * v * x * x * cu * cv
        + 6.0 * SQRT3 * u * x * cu * sv
        + 6.0 * SQRT3 * v * x * su * cv
        - 3.0 * (6.0 + d * x * x) * su * sv
    )

    a = 1.0 - (v - u) / SQRT3
    b = 1.0 + (v - u) / SQRT3
    c = 1.0 - (v + u) / SQRT3
    e = 1.0 + (v + u) / SQRT3

    # sin x bracket, with -Ci(|c|x) + log|...| combined as above
    regular = (
        -EULER_GAMMA
        - math.log(x)
        + Cin(abs(c) * x)
        + math.log(e)
        - math.log(abs(a * b))
    )
    sin_bracket = Ci(a * x) + Ci(b * x) - Ci(e * x) + regular
    cos_bracket = -Si(a * x) - Si(b * x) + Si(c * x) + Si(e * x)

    special = d * d * (math.sin(x) * sin_bracket + math.cos(x) * cos_bracket)
    return 3.0 / (4.0 * u**3 * v**3 * x) * (-4.0 / x**3 * rational + special)


def I_RD_asymptotic(v: float, u: float, x: float) -> float:
    """KT eq. (25), the x -> infinity limit."""
    d = u * u + v * v - 3.0
    log_arg = abs((3.0 - (u + v) ** 2) / (3.0 - (u - v) ** 2))
    theta = 1.0 if (v + u - SQRT3) > 0.0 else 0.0
    return (
        3.0
        * d
        / (4.0 * u**3 * v**3 * x)
        * (
            math.sin(x) * (-4.0 * u * v + d * math.log(log_arg))
            - math.pi * d * theta * math.cos(x)
        )
    )


# --------------------------------------------------------------------------
# KT eq. (15) by quadrature: the independent reference for check (2).
# --------------------------------------------------------------------------


def I_by_quadrature(v, u, x, x_min=0.0, source=f_general_w):
    """int_{x_min}^{x} dxbar (xbar/x) sin(x - xbar) f(v,u,xbar).

    The source defaults to eq. (16), NOT eq. (20). Eq. (20) is unusable as the
    integrand of this quadrature: its bracket cancels to O(x^6) against an x^6
    denominator, so below x ~ 0.1 it returns noise -- at x = 0.03 the integral
    it produces is wrong by nine orders. Eq. (16) built from a series-guarded
    Phi has no such cancellation, and check (1) is what licenses substituting
    one for the other.
    """

    def integrand(xb):
        return (xb / x) * math.sin(x - xb) * source(v, u, xb)

    # break at every oscillation the integrand carries, so quad never has to
    # discover one: the source carries (u +- v)/sqrt3, u/sqrt3, v/sqrt3, the
    # Green's function carries 1.
    pts = set()
    for fr in (1.0, (u + v) / SQRT3, abs(u - v) / SQRT3, u / SQRT3, v / SQRT3):
        if fr <= 0.0:
            continue
        period = 2.0 * math.pi / fr
        n = 1
        while n * period < x and len(pts) < 4000:
            if n * period > x_min:
                pts.add(n * period)
            n += 1

    edges = [x_min] + sorted(pts) + [x]
    total, err = 0.0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        val, e = quad(integrand, lo, hi, limit=200, epsabs=1e-16, epsrel=1e-13)
        total += val
        err += e
    return total, err


def rel(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale > 0.0 else 0.0


def emit(line: str = "") -> None:
    print(line)


# --------------------------------------------------------------------------


def check_1_source():
    emit("## 4. KT eq. (16) at `w = 1/3` against KT eq. (20)")
    emit()
    emit(
        "The general-`w` source against the explicit radiation-era one. They share no code."
    )
    emit()
    emit("| v | u | x | eq. (16) | eq. (20) | rel |")
    emit("|---|---|---|---|---|---|")
    worst = 0.0
    for v, u in ((0.7, 1.1), (1.0, 1.0), (0.3, 1.6), (1.2, 0.5), (0.9, 0.87)):
        for x in (0.5, 2.0, 7.3, 31.0, 100.0):
            a, b = f_general_w(v, u, x), f_RD(v, u, x)
            worst = max(worst, rel(a, b))
            emit(f"| {v} | {u} | {x} | {a:+.12e} | {b:+.12e} | {rel(a, b):.2e} |")
    emit()
    emit(
        f"**Worst relative disagreement: {worst:.3e}.** The general-`w` transcription is right, "
        f"and with it the reading of `xbar d_etabar Phi` as `xbar d_xbar Phi`."
    )
    emit()
    return worst


def check_2_closed_form():
    emit("## 5. KT eq. (22) against `scipy.quad` of KT eq. (15)")
    emit()
    emit("The load-bearing check. The closed form knows nothing of the quadrature.")
    emit()
    emit("| v | u | x | eq. (22) | quadrature | rel | quad's own error |")
    emit("|---|---|---|---|---|---|---|")
    worst = 0.0
    cases = [
        (0.7, 1.1, 1.0),
        (0.7, 1.1, 5.0),
        (0.7, 1.1, 20.0),
        (0.7, 1.1, 60.0),
        (1.0, 1.0, 10.0),
        (0.3, 1.6, 12.0),
        (1.2, 0.5, 25.0),
        (0.9, 0.87, 40.0),
        (0.86, 0.87, 15.0),
        (0.9, 0.9, 15.0),
    ]
    for v, u, x in cases:
        c = I_RD(v, u, x)
        q, qerr = I_by_quadrature(v, u, x)
        worst = max(worst, rel(c, q))
        emit(
            f"| {v} | {u} | {x} | {c:+.12e} | {q:+.12e} | {rel(c, q):.2e} | {qerr:.1e} |"
        )
    emit()
    emit(f"**Worst relative disagreement: {worst:.3e}.**")
    emit()
    return worst


def check_3_limits():
    emit("## 6. The limits, and the paper's small-`x` remark")
    emit()
    emit(
        "KT state below eq. (24) that `I_RD ~ x^2/2` for small `x`. Both their own closed form "
        "and the quadrature give `2 x^2 / 9`, a factor `4/9` smaller. The formula is right; the "
        "remark is not."
    )
    emit()
    emit("| x | eq. (22) | quadrature | ratio to `x^2/2` | ratio to `2x^2/9` |")
    emit("|---|---|---|---|---|")
    for x in (0.3, 0.1, 0.03, 0.01):
        c = I_RD(0.7, 1.1, x)
        q, _ = I_by_quadrature(0.7, 1.1, x)
        emit(
            f"| {x} | {c:+.10e} | {q:+.10e} | {c / (x * x / 2.0):.6f} | "
            f"{c / (2.0 * x * x / 9.0):.10f} |"
        )
    emit()
    emit(
        "And the large-`x` limit against eq. (25), which should approach it like `1/x`:"
    )
    emit()
    emit("| v | u | x | eq. (22) | eq. (25) | rel |")
    emit("|---|---|---|---|---|---|")
    for v, u in ((0.7, 1.1), (1.2, 0.5), (0.3, 1.6)):
        for x in (200.0, 2000.0, 20000.0):
            c, a = I_RD(v, u, x), I_RD_asymptotic(v, u, x)
            emit(f"| {v} | {u} | {x} | {c:+.10e} | {a:+.10e} | {rel(c, a):.2e} |")
    emit()


def check_4_resonance():
    emit("### 6.1 The resonance `u + v = sqrt(3)`")
    emit()
    emit(
        "Eq. (22) as written returns `+-inf` exactly at the resonance and loses accuracy "
        "approaching it: `Ci` and the `log` each diverge and their sum does not. With the `Cin` "
        "regularisation of this module the closed form is finite AT the resonance and matches the "
        "quadrature there."
    )
    emit()
    emit("| u + v - sqrt(3) | eq. (22), regularised | quadrature | rel |")
    emit("|---|---|---|---|")
    for eps in (1e-2, 1e-4, 1e-6, 1e-8, 0.0):
        u = v = (SQRT3 + eps) / 2.0
        c = I_RD(v, u, 15.0)
        q, _ = I_by_quadrature(v, u, 15.0)
        emit(f"| {2.0 * u - SQRT3:+.1e} | {c:+.12e} | {q:+.12e} | {rel(c, q):.2e} |")
    emit()


def check_5_against_the_code():
    from ComputeTargets.tests.test_quadsource_integral import (
        SHAPES,
        X_RESP_VALUES,
        Case,
    )

    emit("## 7. The code's `total` against the oracle")
    emit()
    emit(
        "From spec 04 section 0 (2), `G_code(z,z') = -a0 H(z') Gr_k`, and KT's `G_k` IS `Gr_k`; "
        "with `dlog(1+z') = -H a0 deta'/(1+z')` and `(1+z_resp)/(1+z') = a(etabar)/a(eta)` the "
        "two signs and the two powers of `H` cancel, leaving"
    )
    emit()
    emit("        total = N * I_RD(r/k, q/k, k tau(z_resp)) / k_phys^2")
    emit()
    emit(
        "with `N` a pure number. `a0^2/k^2 = 1/k_phys^2` passes the spec's `a0`-covariance test. "
        "`I_trunc` integrates between the code's OWN limits so that the code's finite "
        "`z_source_max` is not charged to `N`."
    )
    emit()
    emit(
        "| shape | x_resp | u = q/k | v = r/k | x = k tau | code `total` | `I_trunc` | N |"
    )
    emit("|---|---|---|---|---|---|---|---|")

    Ns = []
    direct = []
    for shape in SHAPES:
        for x_resp in X_RESP_VALUES:
            case = Case(b=0.0, shape=shape, x_resp=x_resp, exact=True)
            out = case.run(atol=REF_ATOL, rtol=REF_RTOL)
            total = float(out["total"])

            tau = case.model.functions.tau
            k = shape.k
            u, v = shape.q / k, shape.r / k
            x = k * tau(case.z_resp)
            x_min = k * tau(case.z_source_max)

            I_tr, _ = I_by_quadrature(v, u, x, x_min=x_min)
            N = total / (I_tr / (k * k))
            Ns.append(N)
            emit(
                f"| {shape.name} | {x_resp:g} | {u:.4f} | {v:.4f} | {x:.4e} | "
                f"{total:+.7e} | {I_tr:+.6e} | **{N:.12f}** |"
            )

            # The same case scored against eq. (22) ITSELF rather than against a
            # quadrature of the integrand. Nothing else in this script closes
            # that link: section 5 compares the closed form to the quadrature and
            # the table above compares the code to the quadrature, so without
            # this the code and eq. (22) are never put side by side. The head
            # 0 -> x_min carries the OUTER x in the kernel, so it is its own
            # integral and not I_RD(x_min).
            head, _ = quad(
                lambda xb: (xb / x) * math.sin(x - xb) * f_general_w(v, u, xb),
                0.0,
                x_min,
                limit=200,
                epsabs=1e-18,
                epsrel=1e-13,
            )
            I_direct = I_RD(v, u, x) - head
            direct.append(
                (
                    shape.name,
                    x_resp,
                    total,
                    I_direct,
                    head,
                    total / (I_direct / (k * k)),
                )
            )

    emit()
    lo, hi = min(Ns), max(Ns)
    mean = sum(Ns) / len(Ns)
    emit(
        f"**N is constant at {KT_NORM} = -9/8**: min {lo:.12f}, max {hi:.12f}, "
        f"spread {(hi - lo) / abs(mean):.2e} about the mean {mean:.12f}. "
        f"Worst deviation from -9/8 over the nine cases: {max(abs(n - KT_NORM) for n in Ns):.2e}."
    )
    emit()
    emit(
        "`-9/8` factorises as `-(1/2) * (9/4)`. The `9/4` is `1/c^2` with `c = (2+b)/(3+2b) = 2/3` "
        "at `b = 0` (spec 03 section 0.1), which KT fold into their `f`. The `1/2` is the author's "
        "`h^us_ij = h^them_ij / 2` relative to Kohri-Terada. The sign is the orientation "
        "convention of spec 04 section 0 (3)."
    )
    emit()

    emit("### 7.1 The same nine cases against eq. (22) itself")
    emit()
    emit(
        "The table above scores the code against a QUADRATURE of KT's integrand. This one scores "
        "it against the closed form, with the head `0 -> x_min` subtracted. It is the only place "
        "the code and eq. (22) are put side by side, and it is three orders looser -- not because "
        "the agreement is worse, but because the head must be quadratured and then subtracted from "
        "a number about 1e5 times larger."
    )
    emit()
    emit("| shape | x_resp | code `total` | eq. (22) - head | head / eq. (22) | N |")
    emit("|---|---|---|---|---|---|")
    for name, x_resp, total, I_direct, head, N in direct:
        emit(
            f"| {name} | {x_resp:g} | {total:+.14e} | {I_direct:+.14e} | "
            f"{abs(head / (I_direct + head)):.2e} | **{N:.12f}** |"
        )
    emit()
    worst = max(abs(row[5] - KT_NORM) for row in direct)
    emit(
        f"**Worst deviation from -9/8 against the closed form: {worst:.2e}**, against "
        f"{max(abs(n - KT_NORM) for n in Ns):.2e} against the quadrature. Both give -9/8."
    )
    emit()
    return Ns


if __name__ == "__main__":
    emit("<!-- generated by docs/radiation-oracle/kt_verification.py -->")
    emit()
    check_1_source()
    check_2_closed_form()
    check_3_limits()
    check_4_resonance()
    check_5_against_the_code()
