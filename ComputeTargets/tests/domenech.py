"""
Domenech's general-b kernel for the source time integral, as an oracle beside kohri_terada.py.

Guillermo Domenech, "Scalar Induced Gravitational Waves Review", Universe 7 (2021) 398,
arXiv:2109.01398v2 (cited as "the review"), and "Induced gravitational waves in a general
cosmological background", Int. J. Mod. Phys. D 29 (2020) 2050028, arXiv:1912.05583v3 (cited as
"the 2020 paper"). THE VERSIONS MATTER -- the review's Green's function (4.7) changed between v1
and v2; docs/handover/sources/SOURCES.md pins both papers and records the hazard. He published no
code; everything here is a transcription of published equations, following the reconnaissance
docs/handover/DOMENECH-KERNEL-RECON.md, whose sections are cited as "recon section N".

    I_target(v, u, x, b)      the corrected (4.10) with the x -> infinity coefficients (3.3)/(3.4)
                              substituted and J_{b+1/2}(x), Y_{b+1/2}(x) kept exact
    I_asymptotic(v, u, x, b)  (4.12) WITH ITS cos SIGN FLIPPED -- for large-x regression only
    I_quadrature(v, u, x, b)  the corrected (4.10) with the finite-x integrals (4.11) by
                              scipy.quad: no asymptotics anywhere, and the decisive instrument
    A_coefficient/B_/C_       the cos and sin coefficients of recon section 9, separately callable
    total_from_I_rev(...)     the mapping onto this repository's `total`, with the head

THE REVIEW CARRIES TWO SIGN ERRORS, each recorded where it bites:

  1. (4.10) prints (J_{b+1/2} I_Y - Y_{b+1/2} I_J); its own (4.7) and (4.9) give the OPPOSITE
     order. A faithful transcription of the page is -1 times the truth, and is smooth, correctly
     scaling and entirely plausible -- see I_target. Recon section 11 traces it to a v1 -> v2
     correction of (4.7) that was never propagated.
  2. (4.12)'s cos term has the wrong sign relative to its own sin terms, so it is consistent with
     neither sign of (4.10) -- see I_asymptotic.

The 2020 paper's (3.1) and (3.6)-(3.8) are self-consistent and carry the correct order; so does
docs/spec/05-one-loop.md R31, transcribed from the author's own notes four years earlier. Anyone
comparing this file against the review will find it disagrees in those two places. The review is
wrong there; do not "fix" this file back to it.

CONVENTIONS, AND WHOSE OBJECT IS WHOSE.

    I_rev(x, u, v)   int_0^x dxbar G_x(x, xbar) f(xbar, u, v) with the review's (4.7) and (4.9),
                     dimensionless (review.tex:535 carries a 1/k^2; this does not). EVERYTHING IN
                     THIS MODULE IS I_rev.
    I_2020 = I_KT    = 2 c^2 I_rev, c = (2+b)/(3+2b): the 2020 paper folds 2 c^2 into its
                     DEFINITION of I and the review does not (recon sections 4.1 and 8). At b = 0,
                     2 c^2 = 8/9, so I_rev = (9/8) I_KT and kohri_terada.I_RD is I_KT.
    b                the code's b; w = c_s^2 = (1 - b)/(3(1 + b)) is `wPerturbations`.
    u = q/k, v = r/k, x = k tau with tau = a0 eta. Every argument is a0-invariant: a0 is
                     absorbed, not set to one (docs/spec/04-source-integral.md section 0 (1)).

WHAT ITS AGREEMENT WITH ITSELF DOES AND DOES NOT PROVE, in the shape of
docs/radiation-oracle/KOHRI-TERADA-ORACLE.md section 0. I_target and I_quadrature share the
prefactor N (c_s^2 u v x)^{-b-1/2} and the outer Bessels, so their agreement checks the
Gervois-Navelet coefficients (3.3)/(3.4) against the finite-x integrals (4.11) -- which is a real
cross-check, because the two come from different pages by different routes -- but it cannot detect
a common error in the weight xbar^{1/2-b}, in the source's rho = (b+2)/(b+1) or in the overall
sign. Two things can:

  * at b = 0, I_quadrature must equal (9/8) kohri_terada.I_RD and I_target must equal
    (9/8) kohri_terada.I_RD_asymptotic. kohri_terada is a transcription of a DIFFERENT paper made
    by a different campaign, and I_RD is a closed form with no quadrature in it at all;
  * at b = 0.2, this repository's own `total` must be -I_rev/k_phys^2 on the nine fixture cases of
    test_quadsource_integral. That is the only object in the chain that nobody transcribing a
    paper wrote.

WHERE THIS LIVES, AND WHAT IT IS NOT. An oracle for tests, not a pipeline component: nothing in
main.py or under ComputeTargets/ outside tests/ calls it.

REGIME. J_{b+1/2} and Y_{b+1/2} come from `scipy.special`, whose Amos implementation has a silent
accuracy boundary above order ~86 (docs/OPEN_ISSUES.md, [01-scipy-jv-yv-high-order-boundary]). The
orders here are b + 1/2 and b + 5/2 with -1/2 <= b < 1, i.e. at most 2.5, so that boundary is not
approached; the arguments used by this module's tests reach x ~ 1.6e3. For production x -- 1e7 and
above -- the outer Bessels should come from the repository's own two-region amplitude/residual
machinery instead (`prompts/handover/README.md` section 2 (l)); I_quadrature would in any case
need a Levin-type evaluation there (KT section 8, Table 8.2).

NO TOLERANCE MOVES THE ANSWER. I_target, I_asymptotic and the coefficients are closed forms; the
only tolerances in the module are scipy.quad's inside I_quadrature, which returns its own declared
error beside the value so that every comparison can be scored against it.
"""

import math
import sys
from collections import namedtuple

import mpmath as mp
from scipy.integrate import quad
from scipy.special import jv, yv

_EPS = sys.float_info.epsilon

#: working precision for the two genuine hypergeometrics in B and C. Fixed here, and applied with
#: `mp.workdps`, so that the module's answer does not depend on the caller's `mp.dps` -- an oracle
#: that moves with global state is not an oracle. 30 digits is what recon section 3 verified the
#: Legendre conventions at.
MPMATH_DPS = 30

#: the one value of b at which B's general form has to be replaced. R_nu is a difference of two
#: terms that are individually infinite at b = 0 (cot(b pi) and Gamma(b)), so there is no
#: small-|b| branch to widen this into: it is exactly b = 0 that is replaced, and a nearby b is
#: evaluated by the general form at MPMATH_DPS digits, which handles it (10^-4 away from zero it
#: agrees with the b = 0 limit to 4 parts in 10^4, i.e. linearly in b, as it must). The constant
#: exists to name the branch, not to be tuned. C needs no such branch: see _S_nu.
_B_ZERO = 0.0


# ------------------------------------------------------------------------------------------
# background and parametrisation
# ------------------------------------------------------------------------------------------


def w_of_b(b: float) -> float:
    """w = c_s^2 = (1 - b)/(3(1 + b)), this repository's `wPerturbations` (spec 01)."""
    return (1.0 - b) / (3.0 * (1.0 + b))


def cs_of_b(b: float) -> float:
    """c_s = sqrt(w). b -> 1 sends c_s -> 0 and the x -> infinity forms meaningless."""
    return math.sqrt(w_of_b(b))


def rho_of_b(b: float) -> float:
    """rho = (b+2)/(b+1) = 3(1+w)/2, the relative weight of the second source term in (4.9)."""
    return (b + 2.0) / (b + 1.0)


def two_c_squared(b: float) -> float:
    """
    2 c^2 with c = (2+b)/(3+2b) = 3(1+w)/(5+3w): the constant the 2020 paper and Kohri & Terada
    fold into their definition of I and the review does not, so I_KT = 2 c^2 I_rev at every b
    (recon sections 4.1 and 8). 8/9 at b = 0.
    """
    return 2.0 * ((b + 2.0) / (2.0 * b + 3.0)) ** 2


def kt_convention_norm(b: float) -> float:
    """
    N(b) = k_phys^2 `total` / I_KT = -1/(2c^2) = -(3+2b)^2 / (2(2+b)^2), recon section 8's boxed
    result. -9/8 at b = 0 (which is kohri_terada.KT_NORM), -1.194214876... at b = 0.2. Against
    I_rev, which is what this module computes, the same statement is N = -1 exactly.

    IT IS A DERIVATION, NOT A MEASUREMENT. What a test must show is that N is CONSTANT over
    (u, v, x); a wrong kernel drifts, a wrong normalisation is constant but wrong
    (KOHRI-TERADA-ORACLE.md section 0).
    """
    return -((2.0 * b + 3.0) ** 2) / (2.0 * (b + 2.0) ** 2)


def kernel_constant(b: float) -> float:
    """
    K(b) = 4^b Gamma^2[b+3/2] (2b+3)/(b+2), the constant of the target object of recon section
    2.4. The exact kernel (4.10) carries N = pi K; the coefficients (3.3)/(3.4) contribute the
    remaining sqrt(pi/2) and sqrt(2/pi) separately. K(0) = 3 pi / 8.
    """
    return 4.0**b * math.gamma(b + 1.5) ** 2 * (2.0 * b + 3.0) / (b + 2.0)


# ------------------------------------------------------------------------------------------
# kinematics
# ------------------------------------------------------------------------------------------

Kinematics = namedtuple(
    "Kinematics", ("cs", "delta", "one_plus_y", "one_minus_y", "y", "on_cut")
)


def kinematics(v: float, u: float, b: float) -> Kinematics:
    """
    (4.13) `eq:y`, with the small differences formed directly (recon section 5.3):

        delta = c_s(u + v) - 1,
        1 + y = delta (2 + delta) / (2 c_s^2 u v),
        1 - y = (1 - c_s(u - v))(1 + c_s(u - v)) / (2 c_s^2 u v),

    never `1 + y` after computing y, which costs eps/|1+y| at the resonance -- the same
    discipline kohri_terada applies to its c = 1 - (v+u)/sqrt3. y itself is recovered from
    whichever of the two is the smaller, so it is exact to rounding at both ends.

    `on_cut` is delta >= 0, i.e. y >= -1: the Ferrers branch of (3.3)/(3.4), Theta_+. delta < 0
    is y < -1, the off-cut branch, Theta_-. The resonance delta = 0 (y = -1 exactly) is taken
    on-cut, where recon section 5.2's regularised forms are finite for b > 0 and the two branches
    agree, B(-1) = -C(1).

    y > 1 (c_s|u - v| >= 1, i.e. `one_minus_y` <= 0) is OUTSIDE both Gervois-Navelet cases the
    papers quote; see A_coefficient.
    """
    cs = cs_of_b(b)
    delta = cs * (u + v) - 1.0
    diff = cs * (u - v)
    two_cs2uv = 2.0 * cs * cs * u * v
    one_plus_y = delta * (2.0 + delta) / two_cs2uv
    one_minus_y = (1.0 - diff) * (1.0 + diff) / two_cs2uv
    if abs(one_plus_y) < abs(one_minus_y):
        y = one_plus_y - 1.0
    else:
        y = 1.0 - one_minus_y
    return Kinematics(cs, delta, one_plus_y, one_minus_y, y, delta >= 0.0)


class OutsideGervoisNavelet(ValueError):
    """
    Raised for y >= 1, i.e. c_s|u - v| >= 1.

    THE PAPERS GIVE NO FORMULA THERE. Their appendices quote the Gervois-Navelet integral in two
    cases, |a-b| < c < a+b (on-cut) and c > a+b (off-cut); the third, c < |a-b|, is not given
    (recon section 6.2). No physical configuration reaches it: momentum conservation puts
    (u, v) on the triangle |u-v| <= 1 <= u+v, and c_s < 1 then makes c_s|u-v| < 1 always. The one
    shape in this repository that reaches it is `T-first` at b = 0 -- (u, v) = (10, 12),
    |q - r| = 2000 > k = 1000, not a closable triangle -- where y = 1.0042, and where Kohri &
    Terada's eq. (25) is equally inapplicable (KOHRI-TERADA-ORACLE.md section 8 item 4: with
    |v - u| > sqrt3 its cos term is absent, and with the term kept it is O(1) wrong at every x).

    RECON SECTION 6.2 OFFERS A CONTINUATION AND THIS MODULE DECLINES IT. The rule "I_J^inf = 0 and
    I_Y^inf = -[off-cut form at +y]" matched a quadrature as 1/x on three shapes, but it was
    inferred numerically and not read from a source (recon section 10 item 2), and campaign
    README section 5.1 forbids shipping a transcription that has not been scored against a source.
    There is also nothing to score it against where it would be used: at b = 0 the only reference
    is eq. (25), which does not apply there either, while eq. (22) -- finite, exact and already in
    the tree -- covers that shape completely. Refusing loudly is therefore strictly better than a
    silently inferred branch. I_quadrature is unaffected: it never forms y.
    """


def _require_defined(kin: Kinematics, v: float, u: float, b: float, what: str):
    if kin.one_minus_y <= 0.0:
        raise OutsideGervoisNavelet(
            f"{what}(v={v}, u={u}, b={b}): c_s|u - v| = {kin.cs * abs(u - v)} >= 1 gives "
            f"y = {kin.y} >= 1, outside both Gervois-Navelet cases the papers quote; "
            f"see OutsideGervoisNavelet. I_quadrature is defined there and this is not"
        )


def _require_on_cut(one_plus_y: float, one_minus_y: float, what: str):
    if one_minus_y <= 0.0:
        raise OutsideGervoisNavelet(
            f"{what}: 1 - y = {one_minus_y} <= 0, i.e. c_s|u - v| >= 1 and "
            f"y = {_y_from(one_plus_y, one_minus_y)} >= 1, outside both Gervois-Navelet cases "
            f"the papers quote; see OutsideGervoisNavelet"
        )
    if one_plus_y < 0.0:
        raise ValueError(
            f"{what}: 1 + y = {one_plus_y} < 0, i.e. c_s(u+v) < 1, is the off-cut branch, "
            f"where this coefficient does not appear"
        )


# ------------------------------------------------------------------------------------------
# the coefficients A, B, C of recon section 9
# ------------------------------------------------------------------------------------------
#
# All three take (1 + y, 1 - y) rather than y, so that the caller's accurate small difference
# survives into them: near the resonance 1 + y is the thing that must not be reconstructed
# (recon section 5.3's table -- the naive route loses a digit per decade below 1e-4 and is nan
# at the point, this one is exact to rounding at 1 + y = 0).


def A_coefficient(one_plus_y: float, one_minus_y: float, b: float) -> float:
    """
    A(y) = |1-y^2|^{b/2} [ P^{-b}_b(y) + rho P^{-b}_{b+2}(y) ], the cos coefficient of the target
    object: the ELEMENTARY closed form of recon section 3.5,

        A(y) = (2b+3)/(2^b Gamma(1+b)(b+1)) (1-y^2)^b [ 1 - (b+2)/(2(b+1)) (1-y^2) ],

    which is 3 y^2 at b = 0 (1912.tex:329). Both Ferrers P here reduce to elementary functions
    because mu + nu is 0 or 2, so the cos coefficient needs no special function at all. DLMF
    14.5.18 is the first of the two.

    Defined on the on-cut branch only (it multiplies Theta_+). At the resonance y = -1 it is 0
    for b > 0, 3 for b = 0 and divergent for b < 0, which is recon section 5.1's table.
    """
    _require_on_cut(one_plus_y, one_minus_y, "A_coefficient")
    one_minus_y2 = one_plus_y * one_minus_y
    if one_minus_y2 == 0.0 and b < 0.0:
        # y = -1 with b < 0: A diverges as (1+y)^{-|b|} (recon section 5.1). Returned rather
        # than raised, because Python's 0.0 ** negative is a ZeroDivisionError and the
        # mathematics here is an infinity, not an error.
        return math.inf
    return (
        (2.0 * b + 3.0)
        / (2.0**b * math.gamma(1.0 + b) * (b + 1.0))
        * one_minus_y2**b
        * (1.0 - (b + 2.0) / (2.0 * (b + 1.0)) * one_minus_y2)
    )


def _R_nu(nu, one_plus_y, one_minus_y, b):
    """
    (1-y^2)^{b/2} Q^{-b}_nu(y) by recon section 5.2's regularised form,

        = (pi/2) cot(b pi) (1-y^2)^{b/2} P^{-b}_nu(y)
          - 2^{b-1} Gamma(b) Gamma(nu-b+1)/Gamma(nu+b+1) 2F1(-nu-b, 1+nu-b; 1-b; (1+y)/2),

    for nu in {b, b+2}. The divergence of Q^{-b}_nu at y -> -1^+ lives entirely in P^{b}_nu,
    and DLMF 15.8.4 moves it into the elementary first term, which the (1-y^2)^{b/2} then kills
    for b > 0; the 2F1 is analytic on the whole on-cut range and equals 1 at y = -1. Nothing
    diverges and nothing cancels. Evaluated in mpmath at MPMATH_DPS.
    """
    z = one_plus_y / 2.0
    one_minus_y2 = one_plus_y * one_minus_y
    # (1-y^2)^{b/2} P^{-b}_nu(y), elementary (recon section 3.5)
    P = mp.mpf(one_minus_y2) ** b * mp.rgamma(1 + b) / mp.mpf(2) ** b
    if nu != b:  # nu = b + 2
        P = P * (1 - (2 * b + 3) / (2 * (b + 1)) * one_minus_y2)
    return mp.pi / 2 * mp.cot(b * mp.pi) * P - mp.mpf(2) ** (b - 1) * mp.gamma(
        b
    ) * mp.gamma(nu - b + 1) * mp.rgamma(nu + b + 1) * mp.hyp2f1(
        -nu - b, 1 + nu - b, 1 - b, z
    )


def _S_nu(nu, y_tilde, one_minus_z, b):
    """
    (yt^2-1)^{b/2} calQ^{-b}_nu(yt) for yt >= 1, from the 2020 paper's OWN |x| > 1 definition of
    Q^mu_nu -- the 1/x^2 hypergeometric at 1912.tex:743-747, which is DLMF 14.3.7 -- together
    with calQ = e^{-mu pi i} Q / Gamma[mu+nu+1] at 1912.tex:749-751 (DLMF 14.3.10). With
    mu = -b the (x^2-1)^{mu/2} cancels the prefactor exactly and what is left is

        (yt^2-1)^{b/2} calQ^{-b}_nu(yt)
            = sqrt(pi) 2^{-nu-1} yt^{b-nu-1} 2F1((nu-b+2)/2, (nu-b+1)/2; nu+3/2; 1/yt^2)
              / Gamma(nu+3/2).

    NOT RECON SECTION 5.2's OFF-CUT BOX, AND THIS IS DELIBERATE. That box is the same function,
    written as a difference of two 2F1 at argument (1-yt)/2, and it is CATASTROPHICALLY
    ILL-CONDITIONED away from yt = 1: at b = 0.2, yt = 150 its two terms are each 2.2426e+05 and
    their difference is 1.0e-08, a 13-order cancellation, so the 1e-17 by which a double's `b`
    differs from the b one means costs it two decimal digits of the answer. Measured against the
    raw Olver call it is 1e-15 at yt = 1.5, 9.5e-10 at yt = 40 and 8.6e-08 at yt = 150, and
    RAISING mp.dps DOES NOT HELP -- the loss is in the inputs, not the working precision. Recon
    section 10 item 5 left exactly this untested ("not tested at large |y| (~100 on q-smooth),
    where the ... (yt+-1)^b factors may cancel"); this is its answer. `q-smooth` at b = 0.2 sits
    at yt = 149.5.

    The form above has no cancellation anywhere: 1/yt^2 is in [0, 1], the series converges at 1
    itself for b > 0 (c - a - b = b), and it holds at every b including b = 0, so C needs no
    b = 0 branch. Evaluated in mpmath at MPMATH_DPS.

    `one_minus_z` = 1 - 1/yt^2 = (yt-1)(yt+1)/yt^2 is passed in from the caller's accurate
    yt - 1 = -(1+y), so that z is assembled at working precision rather than as a double: near
    the resonance the 2F1's derivative goes like (1-z)^{b-1} and a double z would cost the
    caller's small difference back again.
    """
    z = 1 - mp.mpf(one_minus_z)
    return (
        mp.sqrt(mp.pi)
        * mp.mpf(2) ** (-nu - 1)
        * mp.mpf(y_tilde) ** (b - nu - 1)
        * mp.rgamma(nu + mp.mpf(3) / 2)
        * mp.hyp2f1((nu - b + 2) / 2, (nu - b + 1) / 2, nu + mp.mpf(3) / 2, z)
    )


def B_coefficient(one_plus_y: float, one_minus_y: float, b: float) -> float:
    """
    B(y) = |1-y^2|^{b/2} [ Q^{-b}_b(y) + rho Q^{-b}_{b+2}(y) ], the ON-CUT sin coefficient, from
    recon section 9: B = R_b + rho R_{b+2} with R_nu the regularised form of _R_nu.

    At b = 0 exactly, R_nu's two terms are individually infinite (cot(b pi), Gamma(b)) and the
    form is unusable; the limit is elementary and is used instead,

        B(y) = (3y/2) ( y log((1+y)/(1-y)) - 2 ),

    which is 1912.tex:333-336 and which recon section 7 shows reduces the whole target object to
    (9/8) x Kohri & Terada's eq. (25).

    At the resonance y = -1 and b > 0 this is finite and equals
    -2^b Gamma(b)(3+2b)(1+b+b^2) / ((1+b) Gamma(2b+3)) = -B_at_resonance(b); for b <= 0 it
    diverges, as it must (recon section 5.1).
    """
    _require_on_cut(one_plus_y, one_minus_y, "B_coefficient")
    if b == _B_ZERO:
        if one_plus_y == 0.0:
            # y = -1 at b = 0: B diverges logarithmically (recon section 5.1), and
            # 1.5 y (y log(0) - 2) with y = -1 is -infinity. math.log(0.0) raises.
            return -math.inf
        y = _y_from(one_plus_y, one_minus_y)
        return 1.5 * y * (y * math.log(one_plus_y / one_minus_y) - 2.0)
    with mp.workdps(MPMATH_DPS):
        value = _R_nu(b, one_plus_y, one_minus_y, b) + rho_of_b(b) * _R_nu(
            b + 2, one_plus_y, one_minus_y, b
        )
        return float(value)


def C_coefficient(one_plus_y: float, one_minus_y: float, b: float) -> float:
    """
    C(yt) = (yt^2-1)^{b/2} [ calQ^{-b}_b(yt) + 2 rho calQ^{-b}_{b+2}(yt) ] at yt = -y, the
    OFF-CUT sin coefficient, from recon section 9: C = S_b + 2 rho S_{b+2}.

    THE FACTOR 2 ON S_{b+2} IS ASYMMETRIC AND IS CORRECT. The off-cut branch carries 2(b+2)/(b+1)
    where the on-cut branches carry (b+2)/(b+1). It is Gamma[nu-rho+1] at nu-rho = 2 in the
    Gervois-Navelet off-cut formula, i.e. Gamma[3] = 2, against Gamma[1] = 1 for the first term;
    the on-cut formula has no such factor (recon section 4.2). Both papers print it. DO NOT
    SYMMETRISE IT: with 1 in place of the 2, the target stalls at 8e-2 of the envelope instead of
    approaching the exact kernel as 1/x, and at b = 0 the coefficient reduces to
    (3 yt^2 + 1)/4 log(...) - 3 yt/2, which matches nothing.

    Unlike B, this needs no b = 0 branch: _S_nu's form is regular at b = 0 and reproduces
    1912.tex:810's elementary C(yt) = (3 yt/2)(yt log((yt+1)/(yt-1)) - 2) there.

    At yt = 1 exactly (the resonance approached from the off-cut side) it is finite for b > 0
    and equals -B_at_resonance(b), which is what makes the sin coefficient continuous across the
    resonance; for b <= 0 it diverges.
    """
    if one_plus_y >= 0.0:
        raise ValueError(
            f"C_coefficient: 1 + y = {one_plus_y} >= 0 is the on-cut branch, where this "
            f"coefficient does not appear"
        )
    y_tilde = -_y_from(one_plus_y, one_minus_y)
    # 1 - 1/yt^2 = (yt-1)(yt+1)/yt^2, from the caller's accurate yt - 1 = -(1+y)
    one_minus_z = (-one_plus_y) * one_minus_y / (y_tilde * y_tilde)
    with mp.workdps(MPMATH_DPS):
        value = _S_nu(b, y_tilde, one_minus_z, b) + 2.0 * rho_of_b(b) * _S_nu(
            b + 2, y_tilde, one_minus_z, b
        )
        return float(value)


def B_at_resonance(b: float) -> float:
    """
    B(-1) for b > 0, in closed form (recon section 5.2):

        B(-1) = -2^b Gamma(b) (3+2b)(1+b+b^2) / ((1+b) Gamma(2b+3)) = -C(+1).

    That B(-1) = -C(1) is what makes the sin coefficient B Theta_+ - C Theta_- continuous across
    the resonance, while A Theta_+ vanishes there: for b > 0 the target object is continuous in
    (u, v) at c_s(u+v) = 1. -6.2147807897686 at b = 0.2.
    """
    if b <= 0.0:
        raise ValueError(
            f"B_at_resonance: b = {b} <= 0; the coefficients diverge at y = -1 there "
            f"(logarithmically at b = 0, as (1+y)^{{-|b|}} for b < 0)"
        )
    return (
        -(2.0**b)
        * math.gamma(b)
        * (3.0 + 2.0 * b)
        * (1.0 + b + b * b)
        / ((1.0 + b) * math.gamma(2.0 * b + 3.0))
    )


def _y_from(one_plus_y: float, one_minus_y: float) -> float:
    if abs(one_plus_y) < abs(one_minus_y):
        return one_plus_y - 1.0
    return 1.0 - one_minus_y


# ------------------------------------------------------------------------------------------
# the raw mpmath Legendre calls the coefficients are checked against
# ------------------------------------------------------------------------------------------
#
# recon section 3.2's table, one function per row. These are NOT used by A, B or C -- those use
# the elementary and regularised closed forms above -- and exist so that the closed forms can be
# scored against the papers' definitions directly. mpmath's legenp/legenq take DEGREE FIRST,
# ORDER SECOND, and the order here is mu = -b.


def ferrers_P(nu: float, b: float, y: float) -> float:
    """P^{-b}_nu(y), |y| < 1: mpmath `legenp(nu, -b, y, type=2)`, no conversion factor."""
    with mp.workdps(MPMATH_DPS):
        return float(mp.legenp(nu, -b, y, type=2))


def ferrers_Q(nu: float, b: float, y: float) -> float:
    """
    Q^{-b}_nu(y), |y| < 1: mpmath `legenq(nu, -b, y, type=2)`, no conversion factor. This is
    DLMF's Ferrers function of the second kind, which is what the papers define
    (1912.tex:718-729, review.tex:1982-1992) with no change of normalisation.

    mpmath 1.3.0 fragility: at b = 1/2 this raises `hypsum() failed to converge` at y = +-0.5,
    a degenerate 2F1 connection case (recon section 3.4). b = 0 and b = +-1/2 have closed forms
    and should not be routed through here; B_coefficient never is.
    """
    with mp.workdps(MPMATH_DPS):
        return float(mp.legenq(nu, -b, y, type=2))


def olver_Q(nu: float, b: float, y_tilde: float) -> float:
    """
    calQ^{-b}_nu(yt) for yt > 1, Olver's function: mpmath `legenq(nu, -b, yt, type=3)` times the
    conversion e^{-mu pi i}/Gamma(nu+mu+1) = e^{+i b pi}/Gamma(nu-b+1), REAL PART TAKEN.

    The raw type-3 value is complex for non-integer b -- its imaginary part is |sin b pi| of its
    modulus, 0.59 at b = 0.2 -- and after the phase the residual imaginary part is ~1e-31. Do not
    take abs(): the sign matters and is carried by the real part.

    Pass yt = -y, never y. At real argument < -1 the type-3 functions carry the branch cut's
    phase and are not the papers' calQ; passing -y keeps everything on the real axis to the right
    of the cut [-1, 1], where the cut's placement is irrelevant.
    """
    with mp.workdps(MPMATH_DPS):
        raw = mp.legenq(nu, -b, y_tilde, type=3)
        return float(mp.re(mp.expjpi(b) * raw * mp.rgamma(nu - b + 1)))


# ------------------------------------------------------------------------------------------
# the target object: the corrected (4.10) with the x -> infinity coefficients substituted
# ------------------------------------------------------------------------------------------


def I_target(v: float, u: float, x: float, b: float) -> float:
    """
    The review's (4.10) `eq:Isimple` WITH ITS SIGN CORRECTED, and the 2020 paper's (3.3)/(3.4)
    `eq:IJ`/`eq:IY` substituted for the finite-x integrals, keeping J_{b+1/2}(x) and Y_{b+1/2}(x)
    exact (recon section 2.4):

        I_target = K(b) x^{-b-1/2}/(c_s^2 u v) { sqrt(pi/2) Y_{b+1/2}(x) A(y) Theta_+
                                               + sqrt(2/pi) J_{b+1/2}(x) [B(y) Theta_+
                                                                          - C(-y) Theta_-] },

    K(b) = 4^b Gamma^2[b+3/2](2b+3)/(b+2), Theta_+- = Theta(+-(c_s(u+v) - 1)).

    THE SIGN. The review prints (J_{b+1/2} I_Y - Y_{b+1/2} I_J) at review.tex:650. Its own
    Green's function (4.7) and source (4.9) give the OPPOSITE order, and so does the 2020 paper's
    (3.1) at 1912.tex:213 and docs/spec/05-one-loop.md R31. Three independent derivations agree,
    and a quadrature of the review's own G.f settles it numerically at 1e-15 on nine (b, u, v)
    cases (recon sections 2.2 and 4.1). Transcribing the page gives a smooth, correctly scaling
    function that is -1 times the truth; the small-x sign is the cheapest discriminator, since
    G > 0 and f > 0 near xbar = 0 make I_rev(x << 1) = +x^2/(2(2+b)).

    WHAT IT IS NOT. The x -> infinity coefficients are exact only up to "corrections from a
    finite upper integration limit ... suppressed by a further 1/x", AND THAT 1/x IS NOT UNIFORM:
    the control parameter is the smallest Bessel argument c_s min(u,v) x, and near the resonance
    |c_s(u+v) - 1| x, not x (recon sections 5.4 and 6.4). On the `q-smooth` shape (u = 0.01) this
    is still 5e-2 of the envelope off the exact kernel at x = 3200. I_quadrature, not this, is
    the decisive instrument at fixture x; this is for large-x regression.

    It also has no small-x limit: Y_{b+1/2}(x) ~ x^{-b-1/2} makes it ~ x^{-2b-1} as x -> 0
    (recon section 6.3). The small-x limit x^2/(2(2+b)) is a statement about the exact kernel.

    Raises OutsideGervoisNavelet for c_s|u - v| >= 1; see that class.
    """
    kin = kinematics(v, u, b)
    _require_defined(kin, v, u, b, "I_target")
    nu = b + 0.5
    pre = kernel_constant(b) * x ** (-b - 0.5) / (kin.cs * kin.cs * u * v)
    if kin.on_cut:
        A = A_coefficient(kin.one_plus_y, kin.one_minus_y, b)
        B = B_coefficient(kin.one_plus_y, kin.one_minus_y, b)
        bracket = (
            math.sqrt(math.pi / 2.0) * yv(nu, x) * A
            + math.sqrt(2.0 / math.pi) * jv(nu, x) * B
        )
    else:
        C = C_coefficient(kin.one_plus_y, kin.one_minus_y, b)
        bracket = -math.sqrt(2.0 / math.pi) * jv(nu, x) * C
    return pre * bracket


def I_asymptotic(v: float, u: float, x: float, b: float) -> float:
    """
    The DOUBLY asymptotic form: the review's (4.12) `eq:Isimple2` WITH ITS cos SIGN FLIPPED, i.e.
    I_target with J_{b+1/2}(x) and Y_{b+1/2}(x) replaced by their large-argument limits
    (recon section 2.4),

        I -> x^{-b-1} K(b)/(c_s^2 u v) { -cos(x - b pi/2) A Theta_+
                                        + (2/pi) sin(x - b pi/2) [B Theta_+ - C Theta_-] }.

    THE SECOND SIGN ERROR IN THE REVIEW. (4.12) prints +cos where the derivation gives -cos, while
    its sin terms are those of the correctly signed kernel -- so as printed it is consistent with
    NEITHER sign of (4.10), and misses a quadrature of the review's own G.f by 1.2 to 1.9 of the
    envelope at every x. The 2020 paper's (3.6)-(3.8), which it calls "the main result", are
    self-consistent and reduce at b = 0 to Kohri & Terada's eq. (25) term for term. The review's
    own results are unaffected because its (4.14) squares each term separately, which is
    presumably how both slips survived.

    For large-x regression only. It differs from I_target by the (4 nu^2 - 1)/(8x) Bessel
    correction -- 2e-3 at x = 50 falling to 1e-6 at x = 3200 for b = 0.2 -- and AT b = 0 THE TWO
    COINCIDE EXACTLY, because the order-1/2 Bessel asymptotics are exact there.
    """
    kin = kinematics(v, u, b)
    _require_defined(kin, v, u, b, "I_asymptotic")
    phase = x - b * math.pi / 2.0
    pre = kernel_constant(b) * x ** (-b - 1.0) / (kin.cs * kin.cs * u * v)
    if kin.on_cut:
        A = A_coefficient(kin.one_plus_y, kin.one_minus_y, b)
        B = B_coefficient(kin.one_plus_y, kin.one_minus_y, b)
        bracket = -math.cos(phase) * A + 2.0 / math.pi * math.sin(phase) * B
    else:
        C = C_coefficient(kin.one_plus_y, kin.one_minus_y, b)
        bracket = -2.0 / math.pi * math.sin(phase) * C
    return pre * bracket


# ------------------------------------------------------------------------------------------
# the exact kernel: (4.11) by quadrature inside the corrected (4.10)
# ------------------------------------------------------------------------------------------


def _Isimpledef_integrand(outer, v, u, b):
    """
    (4.11) `eq:Isimpledef`'s integrand with the outer Bessel `outer` (jv or yv):

        xbar^{1/2-b} Z_{b+1/2}(xbar) [ J_{b+1/2}(c_s v xbar) J_{b+1/2}(c_s u xbar)
                                       + rho J_{b+5/2}(c_s v xbar) J_{b+5/2}(c_s u xbar) ].

    Both branches vanish at xbar = 0 -- the J one like xbar^{2b+2}, the Y one like xbar, since
    Y_{b+1/2} ~ xbar^{-b-1/2} against a source ~ xbar^{2b+1} -- so the endpoint is returned as 0
    rather than as the 0 x inf the closed forms produce there. Gauss-Kronrod never evaluates it.
    """
    cs = cs_of_b(b)
    rho = rho_of_b(b)
    nu1, nu2 = b + 0.5, b + 2.5
    csv, csu = cs * v, cs * u

    def integrand(xb):
        if xb <= 0.0:
            return 0.0
        source = jv(nu1, csv * xb) * jv(nu1, csu * xb) + rho * jv(nu2, csv * xb) * jv(
            nu2, csu * xb
        )
        return xb ** (0.5 - b) * outer(nu1, xb) * source

    return integrand


def _breakpoints(v, u, b, x_lo, x_hi, cap=4000):
    """
    One breakpoint per period of every oscillation the integrand carries -- 1 from the outer
    Bessel, c_s u and c_s v from the source, and their beats c_s(u+v) and c_s|u-v| -- so that
    quad never has to discover one. The same construction as kohri_terada.I_RD_quadrature, with
    c_s in place of 1/sqrt3.
    """
    cs = cs_of_b(b)
    pts = set()
    for fr in (1.0, cs * (u + v), cs * abs(u - v), cs * u, cs * v):
        if fr <= 0.0:
            continue
        period = 2.0 * math.pi / fr
        n = 1
        while n * period < x_hi and len(pts) < cap:
            if n * period > x_lo:
                pts.add(n * period)
            n += 1
    return [x_lo] + sorted(pts) + [x_hi]


def I_quadrature(
    v: float,
    u: float,
    x: float,
    b: float,
    x_lo: float = 0.0,
    x_hi: float = None,
    epsabs: float = 1e-16,
    epsrel: float = 1e-13,
) -> tuple:
    """
    The exact kernel, returning (value, declared_error): the corrected (4.10) with (4.11)'s
    FINITE-x integrals by scipy.quad,

        I = N (c_s^2 u v x)^{-b-1/2} ( Y_{b+1/2}(x) I_J - J_{b+1/2}(x) I_Y ),   N = pi K(b),
        I_{J/Y} = int_{x_lo}^{x_hi} dxbar xbar^{1/2-b} {J or Y}_{b+1/2}(xbar) [source],

    x_hi defaulting to x, so that the default range is (4.11)'s own and the result is I_rev. The
    kernel carries the OUTER x throughout: a sub-range such as the head 0 -> x_min is its own
    integral, not I_quadrature at x_min. `declared_error` is quad's own error estimates on the two
    integrals, carried through the same prefactor.

    NO ASYMPTOTICS ANYWHERE. This is the decisive instrument, and the only one of the three that
    is meaningful at fixture x on every shape: the paper says of (4.11) "We will not be able to
    carry out this integral for general values of x", and the x -> infinity coefficients
    I_target uses in its place carry an O(1/x) that is NOT uniform in (u, v) (recon section 6.4).
    It is also the only one of the three that is finite at the resonance for every b, and the
    only one defined at c_s|u-v| >= 1, because it never forms y at all.

    ITS LIMIT IS PLAIN QUADRATURE'S. KOHRI-TERADA-ORACLE.md section 8 Table 8.2 shows scipy.quad
    of the analogous eq. (15) failing above x ~ 3e4: the breakpoint list caps out and quad's
    200-subdivision limit runs out, with a declared error of order |I|. It reports the failure
    rather than hiding it, and `declared_error` is how a caller sees that. Above that x a
    Levin-type evaluation would be needed (recon section 10 item 3; campaign README section 7 D6).
    """
    if x_hi is None:
        x_hi = x

    edges = _breakpoints(v, u, b, x_lo, x_hi)
    integrals, errors = [], []
    for outer in (jv, yv):
        integrand = _Isimpledef_integrand(outer, v, u, b)
        value, err = 0.0, 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            if hi <= lo:
                continue
            piece, piece_err = quad(
                integrand, lo, hi, limit=200, epsabs=epsabs, epsrel=epsrel
            )
            value += piece
            err += piece_err
        integrals.append(value)
        errors.append(err)

    I_J, I_Y = integrals
    err_J, err_Y = errors
    nu = b + 0.5
    Jx, Yx = jv(nu, x), yv(nu, x)
    cs = cs_of_b(b)
    pre = math.pi * kernel_constant(b) * (cs * cs * u * v * x) ** (-b - 0.5)
    return pre * (Yx * I_J - Jx * I_Y), abs(pre) * (abs(Yx) * err_J + abs(Jx) * err_Y)


def small_x_limit(x: float, b: float) -> float:
    """
    I_rev(x << 1) = +x^2 / (2(2+b)), recon sections 4.1 and 6.3. Derived from G_x -> xbar/(1+2b)
    [1 - (xbar/x)^{1+2b}] and f(0) = (3+2b)/(2+b), so it tests the harness's G, its f and above
    all the SIGN: the review's (4.10) as printed is negative there. At b = 0 it is x^2/4, which
    is (9/8) times Kohri & Terada's 2 x^2/9 (their erratum 2). It is a statement about the exact
    kernel, not about I_target, which diverges as x -> 0.
    """
    return x * x / (2.0 * (2.0 + b))


# ------------------------------------------------------------------------------------------
# the mapping onto this repository's `total`
# ------------------------------------------------------------------------------------------


def total_from_I_rev(
    k: float,
    q: float,
    r: float,
    tau_response: float,
    tau_source_max: float,
    b: float,
    epsabs: float = 1e-16,
    epsrel: float = 1e-13,
) -> dict:
    """
    What evaluate_QuadSource_integral's `total` must be at any b, from the exact kernel:

        total = -I_rev_trunc / k_phys^2,
        I_rev_trunc = (4.11) integrated between THE CODE'S OWN limits, k tau_source_max -> k tau,
                      with the outer x = k tau in the kernel throughout.

    Recon section 8 derives the -1 in three b-explicit steps: f_code = f_rev exactly
    (QuadSource.py:64-92 against review.tex:497-499, no b-dependent factor); f_KT = 2 c^2 f_rev at
    every w, which is the 2020 paper's definitional 2 c^2; and a b-independent Green's function
    and sign. Equivalently, in the Kohri-Terada convention I_KT = 2 c^2 I_rev,

        N(b) = k_phys^2 total / I_KT = -1/(2c^2) = -(3+2b)^2/(2(2+b)^2),

    which is kt_convention_norm(b), -9/8 at b = 0 and -1.194214876... at b = 0.2. THE LOAD-BEARING
    STATISTIC IS THAT N IS CONSTANT over (u, v, x), not that it equals that number: a wrong kernel
    or measure drifts with x, a wrong normalisation is constant but wrong
    (KOHRI-TERADA-ORACLE.md section 0).

    k, q, r are the code's wavenumbers k/a0 and tau is a0 eta, so x = k tau and u, v are ratios:
    every argument is a0-invariant. a0 is absorbed, not set to one, which is why the mapping is
    written in 1/k_phys^2 (docs/spec/04-source-integral.md section 0 (1)).

    THE HEAD IS COMPUTED AND IS NOT NEGLIGIBLE. Both papers integrate from xbar = 0; the code
    starts at z_source_max. `head` is 0 -> k tau_source_max with the same outer x, and
    `head_over_I` is its size relative to the full integral -- 1e-6 to 1e-4 here, and it does NOT
    shrink with x, because at large x both the head and I fall like 1/x and their ratio is set by
    how early the integral starts (campaign README section 2 (k), KT section 8 item 3). It is
    reported rather than subtracted: `I_rev_trunc` is integrated between the code's own limits
    directly, which avoids differencing two numbers that are ~1e5 times larger than the head at
    small x. The two routes agree; the direct one is the tighter reference.

    Returns a dict: `total`, `total_error` (the quadrature's own declared error through the same
    mapping), and the ingredients `u`, `v`, `x`, `x_min`, `I_rev_trunc`, `I_rev_trunc_error`,
    `head`, `head_error`, `head_over_I`.
    """
    u, v = q / k, r / k
    x = k * tau_response
    x_min = k * tau_source_max

    I_trunc, I_err = I_quadrature(
        v, u, x, b, x_lo=x_min, x_hi=x, epsabs=epsabs, epsrel=epsrel
    )
    head, head_err = I_quadrature(
        v, u, x, b, x_lo=0.0, x_hi=x_min, epsabs=epsabs, epsrel=epsrel
    )
    mapping = -1.0 / (k * k)
    return {
        "total": mapping * I_trunc,
        "total_error": abs(mapping) * I_err,
        "u": u,
        "v": v,
        "x": x,
        "x_min": x_min,
        "I_rev_trunc": I_trunc,
        "I_rev_trunc_error": I_err,
        "head": head,
        "head_error": head_err,
        "head_over_I": abs(head) / abs(I_trunc) if I_trunc != 0.0 else math.inf,
    }
