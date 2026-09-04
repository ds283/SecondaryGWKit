"""
Adaptive Levin quadrature for rapidly oscillatory integrals.

BASIS OF THE ALGORITHM
----------------------
The adaptive scheme implemented here is the algorithm of

    J. Bremer, Z. Chen, H. Yang,
    "Rapid evaluation of oscillatory integrals and their application to
     the numerical solution of the Helmholtz equation",
    arXiv:2211.13400,

specifically the adaptive Levin method of its section 5. Equation and step
numbers appearing in comments throughout this file ("Chen et al. (168)",
"Chen et al. step (4), below (173)", ...) refer to that preprint. Page
references are to **v3**; the numbering differs in v1/v2.

The underlying non-adaptive Levin rule is due to

    D. Levin, "Fast integration of rapidly oscillatory functions",
    J. Comput. Appl. Math. 67 (1996) 95-101,

whose equation numbers are cited as "Levin 96".

DEVIATIONS FROM THE REFERENCE ALGORITHM
---------------------------------------
This implementation has evolved beyond the bare algorithm of section 5. The
deliberate departures are:

  * Region acceptance (step (4)) additionally admits a *relative* tolerance;
    Chen et al.'s test is absolute only. See _adaptive_levin() for why this
    matters and how it is now gated.

  * The returned error estimate accounts for the endpoint phase-evaluation
    floor as well as the step-(4) resolution residual. See the extended
    discussion in _adaptive_levin(); the short version is that the step-(4)
    residual is a *resolution* criterion and is blind, by construction, to
    rounding of the phase at the region endpoints. Chen et al. describe that
    loss of accuracy in prose (v3 p. 30, immediately after the algorithm)
    rather than folding it into their estimate, which is legitimate for a
    paper but misleading in a library API.

  * Weakly oscillatory regions (phase span < 6*pi) are handed to ordinary
    adaptive quadrature rather than to the Levin rule.

  * A direct LU solve is used in place of the least-squares/SVD solve when the
    phase span shows the Levin super-operator to be well conditioned. Note
    that Remark 2 of Chen et al. recommends a *rank-revealing QR* in place of
    the truncated SVD throughout (they report ~5x faster with no apparent loss
    of accuracy). That suggestion has NOT been taken up here: the fast path is
    an LU solve gated on conditioning, with lstsq retained as the fallback for
    the ill-conditioned case where the minimum-norm solution is needed. An
    RRQR fallback in place of lstsq remains an unexplored optimisation.
"""

import json
import time
import uuid
from datetime import datetime
from functools import lru_cache
from math import floor, ceil
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numpy.linalg import LinAlgError
from scipy.linalg import toeplitz

from Quadrature.simple_quadrature import simple_quadrature
from utilities import format_time

# default interval at which to log progress of the integration
DEFAULT_LEVIN_NOTIFY_INTERVAL = 5 * 60

# default Chebyshev spectral order
DEFAULT_LEVIN_CHEBSHEV_ORDER = 12
_LEVIN_MINIMUM_ALLOWED_ORDER = 8

# default maximum bisection depth. 1/2^20 is roughly 1E-6
DEFAULT_LEVIN_MAX_DEPTH = 20

# default abs tolerance
DEFAULT_LEVIN_ABSTOL = 1e-15

# default rel tolerance
DEFAULT_LEVIN_RELTOL = 1e-7

# approximate machine epsilon for 64-bit floats
MACHINE_EPSILON = 1e-16

TWO_PI = 2.0 * np.pi
SIX_PI = 6.0 * np.pi

# Safety factor applied to the endpoint phase-rounding bound; see _phase_error() and the
# discussion in _adaptive_levin().
#
# Measured against closed-form oracles for int_{1/3}^{7/3} f(x) sin(w x) dx with
# f in {exp(-x), 1/x, 1} and w from 1e3 to 1e13 (18 cells), the bound with this factor set to
# 1.0 over-estimated the delivered absolute error by between 1.6x and 36x and never
# under-reported it. It is therefore reported as-is; the factor exists so the margin can be
# retuned without touching the derivation.
#
# CAVEAT on those numbers: the bound assumes the phase carries full relative rounding error,
# d(theta) ~ eps*|theta|. That is the generic case, but it is *not* attained when the endpoints
# and the frequency conspire to make theta exactly representable -- e.g. integrating over
# [0, 1] with w a power of ten, where w*x is exact and libm reduces it perfectly, so the true
# d(theta) is zero. On such intervals the bound was measured to be loose by up to ten orders of
# magnitude. This is unavoidable without a way for the phase function to report its own
# accuracy, and erring towards a bound that is loose (never optimistic) is the safe direction
# for a library that callers use to decide whether to trust a result.
_LEVIN_PHASE_ERROR_SAFETY = 1.0

# minimum phase change across a subinterval before we are prepared to invert the Levin super-operator
# with a direct LU solve rather than a least-squares (SVD) solve. Below this the operator is poorly
# conditioned and we need the minimum-norm solution; see _adaptive_levin_subregion_impl().
_LEVIN_DIRECT_SOLVE_PHASE_SPAN = 20.0 * np.pi

INTERVAL_TYPE_LEVIN = 0
INTERVAL_TYPE_DIRECT = 1
types = {0: "Levin", 1: "direct"}


def _phase_error(theta_scale: float, p_endpoint_l1: float) -> float:
    """
    Estimate the absolute error contributed to a region's Levin estimate by rounding of the
    phase function at the two region endpoints.

    The Levin estimate for a region is

        value = sum_i p_i(b) w_i(b) - sum_i p_i(a) w_i(a),        w = (sin theta, cos theta)

    i.e. it depends on theta *only* through its values at the two endpoints. If those values
    carry an absolute error d(theta) then, since |d w / d theta| <= 1 componentwise,

        |d value| <= d(theta) * ( sum_i |p_i(b)| + sum_i |p_i(a)| ).

    With d(theta) ~ eps * theta_scale this gives the bound returned here.

    Why this is needed at all: the step-(4) residual |val0 - valL - valR| cannot see this
    error. Bisecting [a, b] at c gives regions [a, c] and [c, b] whose contributions at c
    cancel in the sum, so parent and children are built from the *identical* values of
    theta(a) and theta(b). Endpoint phase rounding is therefore common-mode between the two
    sides of the step-(4) comparison and subtracts out exactly. It is invisible by
    construction, and no amount of subdivision will reveal it.

    Chen et al. (v3, p. 30) identify the same effect in prose immediately after stating the
    algorithm: they note that the condition number of the oscillatory integral grows with the
    magnitude of the phase, that the principal loss of accuracy occurs when the large-magnitude
    exponentials of their (171) -- the endpoint expression above -- are evaluated, and that
    since the integral itself typically shrinks with frequency the *absolute* error often stays
    constant or decays. So the step-(4) test is a valid resolution criterion; it is simply not
    a total-error estimate, and was never claimed to be one.

    :param theta_scale: magnitude of the phase argument handed to sin/cos at the endpoints
    :param p_endpoint_l1: sum over components of |p| at both endpoints
    """
    return _LEVIN_PHASE_ERROR_SAFETY * MACHINE_EPSILON * theta_scale * p_endpoint_l1


class used_interval:

    def __init__(
        self,
        start: float,
        end: float,
        depth: int,
        type: int,
        abserr: Optional[float] = None,
        relerr: Optional[float] = None,
        p_ratios: Optional[List[float]] = None,
        phase_err: Optional[float] = None,
        phase_limited: bool = False,
    ):
        self._start = start
        self._end = end
        self._depth = depth

        self._type = type

        # abserr/relerr are the step-(4) *resolution* residual: |val0 - valL - valR|.
        # phase_err is an estimate of the error contributed by rounding of the phase function
        # at the two region endpoints, which the resolution residual cannot see (see
        # _adaptive_levin()). The total error of the region is bounded by the larger of the two.
        self._abserr = abserr
        self._relerr = relerr

        self._phase_err = phase_err
        self._phase_limited = phase_limited

        self._p_ratios = p_ratios if p_ratios is not None else []

    def __str__(self):
        if self._abserr is not None:
            abserr_label = f"abserr={self._abserr:.8g}"
        else:
            abserr_label = "abserr=(not recorded)"

        if self._relerr is not None:
            relerr_label = f"relerr={self._relerr:.8g}"
        else:
            relerr_label = "relerr=(not recorded)"

        if self._p_ratios is not None:
            p_ratios_label = f"[{", ".join(f"{p:.4g}" for p in self._p_ratios)}]"
        else:
            p_ratios_label = "(not recorded)"

        if self._phase_err is not None:
            phase_label = f"phase_err={self._phase_err:.8g}"
            if self._phase_limited:
                phase_label = phase_label + " (PHASE LIMITED)"
        else:
            phase_label = "phase_err=(not recorded)"

        return f"({self._start:.8g}, {self._end:.8g}), depth={self._depth} | {types[self._type]}, {abserr_label}, {relerr_label}, {phase_label} | p-ratios={p_ratios_label}"

    @property
    def start(self) -> float:
        return self._start

    @property
    def end(self) -> float:
        return self._end

    @property
    def width(self) -> float:
        return np.fabs(self._end - self._start)

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def type(self) -> str:
        return types[self._type]

    @property
    def abserr(self) -> float:
        return self._abserr

    @property
    def relerr(self) -> float:
        return self._relerr

    @property
    def phase_err(self) -> Optional[float]:
        """
        Estimated absolute error contributed by rounding of the phase function at the region
        endpoints. This is *not* included in abserr; the total error of the region is
        max(abserr, phase_err).
        """
        return self._phase_err

    @property
    def phase_limited(self) -> bool:
        """
        True if this region was accepted because phase rounding, not lack of resolution, set
        the achievable accuracy. Subdividing such a region cannot improve it.
        """
        return self._phase_limited

    @property
    def total_err(self) -> Optional[float]:
        if self._abserr is None:
            return self._phase_err
        if self._phase_err is None:
            return self._abserr
        return max(self._abserr, self._phase_err)

    @property
    def p_ratios(self) -> List[float]:
        return self._p_ratios


ErrorHistoryType = dict[int, float]
PScaleHistoryType = dict[int, List[float]]


class _levin_interval:
    def __init__(
        self,
        start: float,
        end: float,
        depth: int,
        abserr_history: Optional[ErrorHistoryType] = None,
        relerr_history: Optional[ErrorHistoryType] = None,
        p_ratios_history: Optional[PScaleHistoryType] = None,
        estimate: Optional[dict] = None,
    ):
        self._start = start
        self._end = end
        self._depth = depth

        self._abserr_history = abserr_history if abserr_history is not None else {}
        self._relerr_history = relerr_history if relerr_history is not None else {}

        self._p_ratios_history = (
            p_ratios_history if p_ratios_history is not None else {}
        )

        # Levin estimate for this interval, if it has already been computed. When a region is bisected
        # its two halves have already been evaluated in order to produce the refined estimate, so we
        # carry those results forward rather than recomputing them when the children are popped.
        self._estimate = estimate

    @property
    def estimate(self) -> Optional[dict]:
        return self._estimate

    @property
    def start(self) -> float:
        return self._start

    @property
    def end(self) -> float:
        return self._end

    @property
    def width(self) -> float:
        return np.fabs(self._end - self._start)

    @property
    def break_point(self) -> float:
        return (self._end + self._start) / 2.0

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def abserr_history(self) -> ErrorHistoryType:
        return self._abserr_history

    @property
    def relerr_history(self) -> ErrorHistoryType:
        return self._relerr_history

    @property
    def p_ratios_history(self) -> PScaleHistoryType:
        return self._p_ratios_history


@lru_cache(maxsize=32)
def _chebyshev_base(N: int):
    """
    Compute the Chebyshev spectral collocation points (corresponding to the extremal points)
    and the first order spectral differentiation matrix, on the reference interval [-1, 1].

    Only the affine rescaling to an arbitrary interval depends on the endpoints, so this part of the
    construction is cached: the adaptive driver evaluates many subregions at the same Chebyshev order,
    and rebuilding these matrices each time is a significant fraction of the cost of the linear solve
    they feed.

    The returned arrays are marked read-only and shared between callers. chebyshev_matrices() below
    only ever forms new arrays from them, so this is safe; do not mutate them in place.

    Based on the implementation by Wiedeman & Reddy (http://dx.doi.org/10.1145/365723.365727)
    and pyddx (https://github.com/ronojoy/pyddx)
    :param N:
    :return:
    """
    n1 = floor(N / 2)
    n2 = ceil(N / 2)

    k = np.arange(N)
    theta = k * np.pi / (N - 1)

    # Compute the Chebyshev collocation points
    # x = np.cos(np.pi*np.linspace(N-1,0,N)/(N-1))                # obvious way
    x = np.sin(
        np.pi * ((N - 1.0) - 2.0 * np.linspace(N - 1.0, 0, N)) / (2.0 * (N - 1.0))
    )  # W&R way
    x = x[::-1]

    # Assemble the differentiation matrices
    T = np.tile(theta / 2, (N, 1))
    DX = 2 * np.sin(T.T + T) * np.sin(T.T - T)  # trigonometric identity
    DX[n1:, :] = -np.flipud(np.fliplr(DX[0:n2, :]))  # flipping trick
    DX[range(N), range(N)] = 1.0  # diagonals of D
    DX = DX.T

    C = toeplitz((-1.0) ** k)  # matrix with entries c(k)/c(j)
    C[0, :] *= 2
    C[-1, :] *= 2
    C[:, 0] *= 0.5
    C[:, -1] *= 0.5

    Z = 1.0 / DX  # Z contains entries 1/(x(k)-x(j))
    Z[range(N), range(N)] = 0.0  # with zeros on the diagonal.

    # set up differentiation matrix on [-1, 1]
    D = np.eye(N)

    D = Z * (C * np.tile(np.diag(D), (N, 1)).T - D)  # off-diagonals
    D[range(N), range(N)] = -np.sum(D, axis=1)  # negative sum trick

    x.setflags(write=False)
    D.setflags(write=False)

    return x, D


def chebyshev_matrices(x_span: Tuple[float, float], N: int):
    """
    Compute the Chebyshev spectral collocation points (corresponding to the extremal points)
    and the first order spectral differentiation matrix, rescaled to the interval x_span.

    Note the returned grid is in *descending* order: x[0] is the upper endpoint of x_span and x[-1]
    is the lower endpoint. The Levin endpoint extraction depends on this convention.
    :param x_span:
    :param N:
    :return:
    """
    x_base, D_base = _chebyshev_base(N)

    # rescale for the arbitrary interval. Both operations below form new arrays, so the cached
    # read-only base matrices are left untouched.
    a, b = x_span
    x = (a + b) / 2.0 + (b - a) / 2.0 * x_base
    D = 2.0 * D_base / (b - a)

    return x, D


class _Basis_SinCos:
    def __init__(self, theta):
        """
        :param theta: dict with a required "theta" key (the phase function) and optional
            "theta_mod_2pi" / "theta_deriv" keys; see adaptive_levin_sincos()'s docstring for the
            full contract.
        """
        if callable(theta):
            raise TypeError(
                "levin_quadrature: theta must be a dict with a 'theta' key mapping to the phase "
                "function, not a bare callable -- did you mean theta={'theta': theta}?"
            )
        if "theta" not in theta:
            raise RuntimeError(
                "levin_quadrature: Levin phase function theta must be provided as theta['theta']"
            )

        self._theta = theta["theta"]

        if "theta_mod_2pi" in theta:
            self._theta_mod_2pi = theta["theta_mod_2pi"]

        if "theta_deriv" in theta:
            self._theta_deriv = theta["theta_deriv"]

    def raw_theta(self, x):
        return self._theta(x)

    @property
    def supports_complexified_solve(self) -> bool:
        """
        True if this basis's Levin collocation system can be solved in the complexified N x N
        form (D + i diag(theta')) q = f1 + i f2 (Chen et al. (168)) instead of the realified
        2N x 2N one. This holds for the two-component (sin, cos) basis implemented here; see the
        module docstring and prompt 02's log for the derivation. A basis with a different number
        of components, or a genuinely different structure, would answer False here rather than
        the driver sniffing isinstance(BasisData, _Basis_SinCos).
        """
        return True

    def build_Levin_data(
        self, grid, Dmat, label: Optional[str] = None, need_AmatT: bool = True
    ):
        """
        Sample the phase function once and build everything derived from that sample: theta',
        the A^T super-operator block (unless need_AmatT is False), the basis vector w evaluated
        at each endpoint, and an estimate of the total phase change across the interval.

        :param label: optional identifying label for this region, used only to annotate a
            non-finite-phase-derivative warning.
        :param need_AmatT: whether to build and return the A^T block. The complexified solve
            path (see supports_complexified_solve) needs theta_prime_Cheb directly and never
            needs A^T, so the caller passes False there to skip an O(N^2) allocation that would
            just be discarded.
        :return: (AmatT, theta_prime_Cheb, w0, wk, phase_span, theta_scale). AmatT is None when
            need_AmatT is False.
        """
        # we need theta sampled on the Chebyshev grid if either (a) we have to obtain theta' by spectral
        # differentiation, or (b) we have no range-reduced phase function and therefore have to evaluate
        # the basis functions from the raw phase at the endpoints.
        # These are independent conditions, so both have to be tested; testing only the first leaves
        # theta_Cheb undefined in the endpoint block below.
        need_theta_Cheb = not hasattr(self, "_theta_deriv") or not hasattr(
            self, "_theta_mod_2pi"
        )

        theta_Cheb = None
        if need_theta_Cheb:
            # sample the phase function theta on the Chebyshev grid
            theta_Cheb = np.array([self._theta(x) for x in grid])

        if hasattr(self, "_theta_deriv"):
            # prefer an explicitly supplied theta'. At large argument the raw phase theta is a large float
            # whose absolute resolution is ~ eps*theta, so spectral differentiation of the sampled values
            # below inherits an error ~ eps*theta/(phase span across the interval). A phase function that
            # can supply theta' directly (e.g. from a range-reduced spline) does not suffer from this.
            theta_prime_Cheb = np.array([self._theta_deriv(x) for x in grid])
        else:
            # multiply theta by the spectral differentiation matrix Dmat in order to produce an estimate of theta'(x)
            # evaluated at the collocation points
            theta_prime_Cheb = np.matmul(Dmat, theta_Cheb)

        if not np.isfinite(theta_prime_Cheb).all():
            print(
                f"!! WARNING (adaptive_levin_subregion, {label}): sampled phase derivative theta' contains non-numeric values (np.nan, np.inf, or np.-inf)"
            )
            raise ValueError(
                "sampled phase derivative theta' contains non-numeric values (np.nan, np.inf, or np.-inf)"
            )

        AmatT = None
        if need_AmatT:
            # if w = (sin, cos) (regarded as a column vector), then w' = A w where A is the matrix
            #   Amat = ( 0,       theta' )
            #          ( -theta', 0      )
            # and therefore its transpose is
            #   Amat^T = ( 0,      -theta' )
            #            ( theta', 0       )
            theta_prime_I = np.diag(theta_prime_Cheb)

            zero_block = np.zeros_like(theta_prime_I)
            AmatT = np.block(
                [[zero_block, -theta_prime_I], [theta_prime_I, zero_block]]
            )

        # estimate the total phase change across the interval from quantities we have already computed.
        # This costs no further evaluations of the phase function, and is used to decide whether the
        # Levin super-operator is well enough conditioned to be inverted by a direct solve.
        phase_span = float(
            np.mean(np.fabs(theta_prime_Cheb)) * np.fabs(grid[0] - grid[-1])
        )

        # theta_scale is the magnitude of the phase argument actually handed to sin/cos at the
        # endpoints. It sets the absolute resolution of those two values, and hence the floor on
        # the accuracy of the Levin estimate for this region -- see _phase_error(). Like
        # phase_span above it costs no additional evaluations of the phase function.
        if hasattr(self, "_theta_mod_2pi"):
            theta0_mod_2pi = self._theta_mod_2pi(grid[-1])
            thetak_mod_2pi = self._theta_mod_2pi(grid[0])

            w0 = [np.sin(theta0_mod_2pi), np.cos(theta0_mod_2pi)]
            wk = [np.sin(thetak_mod_2pi), np.cos(thetak_mod_2pi)]

            # a range-reduced phase hands sin/cos an O(2pi) argument, so its representation error
            # is O(eps) no matter how large the underlying phase is. This is exactly why a
            # range-reduced phase function is worth supplying.
            #
            # CAVEAT: this accounts only for the rounding of the reduced value as a float. Any
            # error inherited from the *construction* of the reduced phase -- e.g. the fit error
            # of a phase spline, or precision lost while reducing a large product -- is invisible
            # from here, and the bound below does not include it.
            theta_scale = TWO_PI
        else:
            # note grid is in reverse order, with largest value in position 1 and smallest value in last position -1
            theta0 = theta_Cheb[-1]
            thetak = theta_Cheb[0]

            w0 = [np.sin(theta0), np.cos(theta0)]
            wk = [np.sin(thetak), np.cos(thetak)]

            # the raw phase is a large float at large argument, with absolute resolution
            # ~ eps*|theta|. sin/cos then inherit an absolute error of the same size, because
            # |d(sin)/d(theta)| <= 1. This is the dominant error at high frequency.
            theta_scale = float(np.max(np.fabs(theta_Cheb)))

        return AmatT, theta_prime_Cheb, w0, wk, phase_span, theta_scale

    def eval_basis(self, x):
        if hasattr(self, "_theta_mod_2pi"):
            return [np.sin(self._theta_mod_2pi(x)), np.cos(self._theta_mod_2pi(x))]

        return [np.sin(self._theta(x)), np.cos(self._theta(x))]

    def phase_scale(self, a: float, b: float) -> float:
        """
        Magnitude of the phase argument that is handed to sin/cos on the interval [a, b]. This
        sets the absolute resolution of the basis functions there, and hence the accuracy floor
        of any quadrature rule built on them. See _phase_error().
        """
        if hasattr(self, "_theta_mod_2pi"):
            return TWO_PI

        return float(max(np.fabs(self.raw_theta(a)), np.fabs(self.raw_theta(b))))


def _adaptive_levin_subregion(
    x_span: Tuple[float, float],
    f,
    BasisData,
    id_label: uuid,
    chebyshev_order: int = DEFAULT_LEVIN_CHEBSHEV_ORDER,
    rtol: float = DEFAULT_LEVIN_RELTOL,
    notify_label: Optional[str] = None,
    build_p_sample: bool = False,
):
    working_order = max(chebyshev_order, _LEVIN_MINIMUM_ALLOWED_ORDER)
    num_order_changes = 0

    # to handle possible SVD failures, allow the working Chebyshev order to be stepped down.
    # this changes the matrices that we need to invert, so gives another change for the required SVD to converge
    finished = False
    while not finished and working_order >= _LEVIN_MINIMUM_ALLOWED_ORDER:
        data = _adaptive_levin_subregion_impl(
            x_span,
            f,
            BasisData,
            id_label,
            chebyshev_order=working_order,
            rtol=rtol,
            notify_label=notify_label,
            build_p_sample=build_p_sample,
        )
        if data["metadata"].get("SVD_failure", False):
            working_order = working_order - 2
            num_order_changes = num_order_changes + 1

            if id_label is not None:
                label = f"{notify_label} id={id_label}"
            else:
                label = f"{id_label}"
            print(
                f"!! WARNING (adaptive_levin_subregion, {label}): SVD failure - stepping down Chebyshev order to {working_order}"
            )
            continue

        finished = True

    if data["value"] is None:
        raise LinAlgError("SVD failure")

    data["metadata"]["SVD_failure"] = data["metadata"].get("SVD_failure", False)
    data["metadata"].update(
        {
            "num_order_changes": num_order_changes,
            "chebyshev_order": working_order,
        }
    )

    return data


def _adaptive_levin_subregion_impl(
    x_span: Tuple[float, float],
    f,
    BasisData,
    id_label: uuid,
    chebyshev_order: int = DEFAULT_LEVIN_CHEBSHEV_ORDER,
    rtol: float = DEFAULT_LEVIN_RELTOL,
    notify_label: Optional[str] = None,
    build_p_sample: bool = False,
):
    """
    f should be an m-vector of non-rapidly oscillating functions (Levin 96 eq. 2.1)
    theta should be an (m x m)-matrix representing the phase matrix of the system (Levin 96's A or A^t matrix)
    :param rtol:
    :param x_span:
    :param f: iterable of callables representing the integrand f-functions
    :param BasisData: callable
    :param chebyshev_order:
    :param build_p_sample: retain the sampled Levin antiderivatives p(x)? Only needed for diagnostics
    :return:
    """
    metadata = {}
    if id_label is not None:
        label = f"{notify_label} id={id_label}"
    else:
        label = f"{id_label}"

    grid, Dmat = chebyshev_matrices(x_span, chebyshev_order)

    # sample each component of f on the Chebyshev grid,
    # then assemble the result into a flattened vector in an m x k representation
    # Chen et al. around (166), (167)
    m = len(f)
    f_Cheb = np.hstack([[func(x) for x in grid] for func in f])

    if not np.isfinite(f_Cheb).all():
        print(
            f"!! WARNING (adaptive_levin_subregion, {label}): sampled amplitude f contains non-numeric values (np.nan, np.inf, or np.-inf)"
        )
        raise ValueError(
            "sampled amplitude f contains non-numeric values (np.nan, np.inf, or np.-inf)"
        )

    # Sample the phase once. build_Levin_data returns everything either solve path needs: theta'
    # itself (for the complex path's diagonal, and for the real path's A^T block), the endpoint
    # basis vectors w0/wk, and phase_span/theta_scale (needed by both regardless of solve path).
    #
    # A basis reports whether its Levin system can be solved in the complexified N x N form
    # (D + i diag(theta')) q = f1 + i f2 -- Chen et al. (168) -- instead of the realified 2N x 2N
    # one. This is asked of the basis object rather than sniffed via isinstance, so a future basis
    # with a different structure or component count simply answers False. It is also gated on
    # m == 2: the complex form only exists for a two-component (sin, cos) basis.
    use_complex_solve = m == 2 and getattr(
        BasisData, "supports_complexified_solve", False
    )

    AmatT, theta_prime_Cheb, w0, wk, phase_span, theta_scale = (
        BasisData.build_Levin_data(
            grid, Dmat, label=label, need_AmatT=not use_complex_solve
        )
    )

    if use_complex_solve:
        # Chen et al. (168) in complexified form: q = p1 + i*p2 solves (D + i diag(theta')) q =
        # f1 + i*f2, where f1/f2 are the sampled amplitudes for the sin/cos slots respectively
        # (f_Cheb is [f1(grid); f2(grid)] by construction above). D.astype(complex) always
        # allocates a fresh array (astype copies by default), so mutating its diagonal in place
        # cannot corrupt the shared, read-only base matrices from _chebyshev_base(), nor Dmat
        # itself -- see chebyshev_matrices()'s own note that it forms a new D on every call.
        LevinL = Dmat.astype(complex)
        LevinL[np.diag_indices(chebyshev_order)] += 1j * theta_prime_Cheb
        rhs = f_Cheb[:chebyshev_order] + 1j * f_Cheb[chebyshev_order:]
    else:
        # build the Levin superoperator corresponding to this system
        # Chen et al. (168), realified
        zero_block = np.zeros((chebyshev_order, chebyshev_order))

        row_list = []
        for i in range(m):
            row = [zero_block for _ in range(m)]
            row[i] = Dmat
            row_list.append(row)

        LevinL = np.block(row_list) + AmatT
        rhs = f_Cheb

    if not np.isfinite(LevinL).all():
        print(
            f"!! WARNING (adaptive_levin_subregion, {label}): Levin super-operator contains non-numeric values (np.nan, np.inf, or np.-inf)"
        )

        raise ValueError(
            "Levin super-operator contains non-numeric values (np.nan, np.inf, or np.-inf)"
        )

    # now try to invert the Levin superoperator, to find the Levin antiderivatives p(x) (or, on
    # the complex path, q = p1 + i*p2)
    # Chen et al.
    success = False

    # Fast path. As theta' -> 0 the Levin super-operator degenerates to a block-diagonal matrix of
    # spectral differentiation matrices, and Dmat is singular (it annihilates constants). So the system
    # is badly conditioned on weakly oscillatory intervals, and there we need the minimum-norm solution
    # that lstsq provides. Once the interval carries enough phase the operator becomes well conditioned
    # (empirically cond ~ 4e2 at a phase span of 20*pi, and ~13 at 100*pi), and an ordinary LU solve
    # agrees with lstsq to machine precision while being an order of magnitude cheaper. The complex
    # N x N system has identical conditioning to the realified 2N x 2N one (verified: agrees to 4
    # significant figures at every order tested), so the same gate applies unchanged.
    #
    # NOTE the gate has to be the phase span, *not* the residual of the solve. In the near-singular
    # regime the residual is small (~1e-14) even when the computed p is completely wrong, so a residual
    # test alone would silently accept garbage.
    #
    # A prior version of this branch also checked ||Lq - rhs|| / ||rhs|| against a fixed tolerance
    # before accepting the direct solve, as a secondary guard against outright failure -- but the
    # phase-span gate above is what actually protects against a wrong-but-finite answer, and the
    # residual check cost 58-72% of the solve it guarded. Measured across the full AdaptiveLevin
    # test suite plus the J000 three-Bessel oracle (221 direct-solve calls), the relative residual
    # never exceeded ~5e-15 against a 1e-10 threshold, and every call the residual check would have
    # accepted, the finiteness check below also accepted, and vice versa -- so it was removed and
    # the finiteness check on the solved vector (below) is relied on instead. See prompt 02's log
    # for the measurement.
    if phase_span > _LEVIN_DIRECT_SOLVE_PHASE_SPAN:
        try:
            sol_direct = np.linalg.solve(LevinL, rhs)
        except LinAlgError:
            pass
        else:
            if np.isfinite(sol_direct).all():
                sol = sol_direct
                success = True
                metadata["direct_solve"] = 1

    # otherwise fall back to the least-squares (SVD) solution, which handles the ill-conditioned case
    #
    # NOTE on the choice of solver. Chen et al. (arXiv:2211.13400) solve the collocation system
    # (168) with a truncated SVD, but Remark 2 of that paper records that their own implementation
    # replaces it with a *rank-revealing QR* factorization, which they report to be roughly 5x
    # faster with no observed loss of accuracy. That substitution has deliberately NOT been made
    # here: scipy exposes RRQR only via pivoted QR (scipy.linalg.qr(pivoting=True)) with manual
    # rank determination and a triangular solve, so it is a non-trivial amount of new numerical
    # code to own, and the direct-solve fast path above already removes the SVD from the
    # well-conditioned majority of regions. Swapping lstsq for RRQR in this fallback branch
    # remains an unexplored optimisation; it would need A/B benchmarking against the closed-form
    # Bessel oracles before being trusted, because this branch is exactly the ill-conditioned
    # regime where the minimum-norm property of lstsq is doing real work.
    #
    # rcond=None truncates at eps * max(M, N) * sigma_1 (the paper's own step-5 truncation). On the
    # complex path M == N == chebyshev_order rather than 2*chebyshev_order, halving that threshold;
    # this is a real behavioural difference on the ill-conditioned branch but was confirmed
    # immaterial in practice -- see prompt 02's log for the measurement.
    if not success:
        try:
            sol, residuals, rank, s = np.linalg.lstsq(LevinL, rhs, rcond=None)
        except LinAlgError as e:
            print(
                f"!! WARNING (adaptive_levin_subregion, {label}): could not solve Levin collocation system using numpy.linalg.lstsq (chebyshev_order={chebyshev_order}; will now attempt to use pseudo-inverse)"
            )
            now = datetime.now().replace(microsecond=0)
            LevinL_filename = f"LevinL_{now.isoformat()}.txt"
            f_Cheb_filename = f"f_Cheb_{now.isoformat()}.txt"
            print(
                (
                    f'   -- Levin L super-operator written to file "{LevinL_filename}", f_Cheb written to file "{f_Cheb_filename}"'
                )
            )
            np.savetxt(LevinL_filename, LevinL)
            np.savetxt(f_Cheb_filename, rhs)
            metadata["SVD_errors"] = 1
        else:
            success = True

    if not success:
        try:
            LevinL_inv = np.linalg.pinv(LevinL)
            sol = np.matmul(LevinL_inv, rhs)
        except LinAlgError as e:
            print(
                f"!! WARNING (adaptive_levin_subregion, {label}): could not solve Levin collocation system using numpy.linalg.pinv (chebyshev_order={chebyshev_order}; final failure at this order)"
            )
            metadata["SVD_failure"] = True
            return {
                "value": None,
                "p_sample": None,
                "p_ratios": None,
                "metadata": metadata,
            }

    if not np.isfinite(sol).all():
        print(
            f"!! WARNING (adaptive_levin_subregion, {label}): solved Levin antiderivative p contains non-numeric values (np.nan, np.inf, or np.-inf)"
        )
        raise ValueError(
            "solved Levin antiderivative p contains non-numeric values (np.nan, np.inf, or np.-inf)"
        )

    # P[j, i] is Levin antiderivative component j at collocation point i. On the real path p is
    # stored flattened with component j at collocation point i in position j*chebyshev_order + i,
    # so this reshape recovers P[j, i]. On the complex path q = p1 + i*p2 by construction, so its
    # real/imaginary parts stacked in slot order are exactly the same P[j, i]. Taking the mean over
    # the collocation points in numpy avoids building a Python list of m-element lists on every
    # solve; that list is only needed for diagnostics.
    if use_complex_solve:
        P = np.vstack([sol.real, sol.imag])
    else:
        P = sol.reshape(m, chebyshev_order)

    p_means = np.fabs(P).mean(axis=1)
    p_mean_max = p_means.max()
    if p_mean_max > 0.0:
        p_ratios = [float(pm / p_mean_max) for pm in p_means]
    else:
        # every component of p is identically zero -- reachable when the sampled amplitude
        # underflows to zero on this region. A zero solution genuinely contributes zero, so
        # report an all-ones ratio vector (rather than the nan that pm / 0.0 would produce) so
        # every component is kept by p_use below instead of being discarded by a divide-by-zero
        # artefact.
        p_ratios = [1.0 for _ in p_means]

    p_sample = None
    if build_p_sample:
        p_sample = [(x, [P[j, i] for j in range(m)]) for i, x in enumerate(grid)]

    # don't keep p-modes that have relative amplitude smaller than the requested rtol.
    # Presumably we cannot compute these accurately anyway (especially if they are associated with small singular values that are
    # also not being handled correctly in the singular value decomposition of the Levin operator)
    # so they just behave as a source of numerical noise that pollutes our abserr estimates,
    # and can prevent convergence of the bisection step.
    p_use = [np.isfinite(r) and r > rtol for r in p_ratios]

    # note grid is in reverse order, with P[i, 0] at the upper endpoint b and P[i, -1] at the
    # lower endpoint a. Indexing P directly (rather than a flattened p vector) means this is the
    # same expression regardless of which solve path produced P.
    lower_limit = sum(P[i, -1] * w0[i] if p_use[i] else 0.0 for i in range(m))
    upper_limit = sum(P[i, 0] * wk[i] if p_use[i] else 0.0 for i in range(m))

    # first-order bound on the error inherited from rounding of the phase at the two endpoints.
    # The same p-values and the same p_use gating are used as in the estimate itself, so this
    # costs only a handful of flops and no extra evaluations of theta or f. See _phase_error().
    p_endpoint_l1 = sum(
        (np.fabs(P[i, -1]) + np.fabs(P[i, 0])) if p_use[i] else 0.0 for i in range(m)
    )

    return {
        "value": upper_limit - lower_limit,
        "p_sample": p_sample,
        "p_ratios": p_ratios,
        "phase_span": phase_span,
        "phase_err": _phase_error(theta_scale, p_endpoint_l1),
        "metadata": metadata,
    }


def _adaptive_levin(
    x_span: Tuple[float, float],
    f,
    BasisData,
    atol: float = DEFAULT_LEVIN_ABSTOL,
    rtol: float = DEFAULT_LEVIN_RELTOL,
    chebyshev_order: int = DEFAULT_LEVIN_CHEBSHEV_ORDER,
    depth_max: int = DEFAULT_LEVIN_MAX_DEPTH,
    build_p_sample: bool = False,
    notify_interval: int = DEFAULT_LEVIN_NOTIFY_INTERVAL,
    notify_label: str = None,
    emit_diagnostics: bool = False,
):
    # Input validation (rec 4, C8, C9). This module cannot deliver a purely relative-error
    # contract: the phase-rounding floor computed by _phase_error() is absolute by nature, so a
    # caller asking for atol=0 leaves the relative-error denominator floor at :relerr_denom inert
    # and the phase_limited branch unable to fire, and was measured to subdivide an
    # identically-zero integrand to the full depth limit (256 regions / 1023 solves at
    # depth_max=8, against 1 region / 3 solves at atol=1e-15). Reject rather than silently
    # accepting a request this module cannot honour.
    if not (atol > 0):
        raise ValueError(
            f"levin_quadrature: atol must be strictly positive (received atol={atol!r}); "
            "this module cannot deliver a purely relative-error contract because its "
            "phase-rounding error floor is absolute by construction -- choose a small positive atol"
        )
    if rtol < 0:
        raise ValueError(
            f"levin_quadrature: rtol must be non-negative (received rtol={rtol!r})"
        )
    if depth_max < 0:
        raise ValueError(
            f"levin_quadrature: depth_max must be non-negative (received depth_max={depth_max!r})"
        )
    if len(x_span) != 2:
        raise ValueError(
            f"levin_quadrature: x_span must have exactly two entries (start, end); "
            f"received {len(x_span)} entries: {x_span!r}"
        )
    if not (np.isfinite(x_span[0]) and np.isfinite(x_span[1])):
        raise ValueError(
            f"levin_quadrature: x_span endpoints must be finite; received x_span={x_span!r}"
        )

    driver_start: float = time.perf_counter()
    start_time: float = time.time()
    last_notify: float = start_time
    updates_issued: int = 0

    # generate unique id to identify this calculation
    id_label = uuid.uuid4()
    if notify_label is not None:
        label = f"{notify_label}, id={id_label}"
    else:
        label = f"{id_label}"

    if chebyshev_order < _LEVIN_MINIMUM_ALLOWED_ORDER:
        print(
            f"!! WARNING (adaptive_levin, {label}): chebyshev_order={chebyshev_order} is below "
            f"the minimum allowed order {_LEVIN_MINIMUM_ALLOWED_ORDER}; every subregion solve "
            f"will be clamped up to {_LEVIN_MINIMUM_ALLOWED_ORDER}"
        )

    m = len(f)

    regions = [_levin_interval(start=x_span[0], end=x_span[1], depth=0)]

    val = 0.0
    used_regions = []
    p_points = []
    num_used_regions = 0
    num_simple_regions = 0
    num_evaluations = 0

    num_SVD_errors = 0
    num_order_changes = 0
    num_direct_solves = 0
    chebyshev_min_order = None
    max_depth = 0

    num_history_messages = 0

    while len(regions) > 0:
        now = time.time()
        if now - last_notify > notify_interval:
            updates_issued = updates_issued + 1
            _notify_progress(
                now,
                last_notify,
                start_time,
                val,
                num_used_regions,
                len(regions),
                num_simple_regions,
                max_depth,
                num_evaluations,
                num_SVD_errors,
                num_order_changes,
                chebyshev_min_order,
                updates_issued,
                id_label,
                notify_label,
            )

            # dump data every 3 notifications
            if emit_diagnostics and updates_issued % 3 == 1:
                _write_progress_data(
                    f,
                    BasisData,
                    regions,
                    chebyshev_order,
                    val,
                    id_label,
                    atol,
                    rtol,
                    notify_label,
                )

            last_notify = time.time()

        current_region = regions.pop()
        a = current_region.start
        b = current_region.end

        # updated here, unconditionally, rather than only on the branch that accepts a Levin
        # region: the direct-quadrature branch below `continue`s before reaching that branch, so
        # updating it there missed every run that terminated in direct quadrature (C6).
        if current_region.depth > max_depth:
            max_depth = current_region.depth

        if current_region.depth >= 18 and num_history_messages < 20:
            num_history_messages += 1

            print(
                f"@@ adaptive_levin ({label}): encountered subinterval of depth {current_region.depth} (notification {num_history_messages}/20 for this quadrature)"
            )

            abs_history = current_region.abserr_history
            rel_history = current_region.relerr_history
            p_scale_history = current_region.p_ratios_history

            prev_abs = None
            prev_rel = None

            for i in range(0, current_region.depth):
                this_abs = abs_history[i]
                this_rel = rel_history[i]
                this_p_ratios = p_scale_history[i]

                if i == 0:
                    print(
                        f"   -- {i+1}. abserr={this_abs :.5g}, relerr={this_rel :.5g}, p-ratios=[{', '.join(f'{p:.4g}' for p in this_p_ratios)}]"
                    )

                else:
                    abs_improvement = prev_abs / this_abs
                    rel_improvement = prev_rel / this_rel
                    print(
                        f"   -- {i+1}. abserr={this_abs :.5g} (improvement={abs_improvement:.3g}), relerr={this_rel :.5g} (improvement={rel_improvement:.3g}), p-ratios=[{', '.join(f'{p:.3g}' for p in this_p_ratios)}]"
                    )

                prev_abs = this_abs
                prev_rel = this_rel

        # if phase difference across this region is small enough that we do not have many oscillations,
        # there is likely no advantage in using the Levin rule to do the computation.
        # We can terminate the adaptive process by doing ordinary numerical quadrature
        phase_diff = np.fabs(BasisData.raw_theta(b) - BasisData.raw_theta(a))
        if np.fabs(phase_diff) < SIX_PI:

            def integrand(x):
                basis = BasisData.eval_basis(x)
                return sum(f[i](x) * basis[i] for i in range(m))

            data = simple_quadrature(
                integrand,
                a=a,
                b=b,
                atol=atol,
                rtol=rtol,
                method="quad",
            )

            # phase-rounding floor for the direct-quadrature branch. quad's own error estimate,
            # like the Levin step-(4) residual, is blind to rounding of the phase inside the
            # integrand -- refining the panel resamples the same rounded sin/cos values. Bound the
            # contribution by eps * theta_scale * integral|f|, the direct analogue of the endpoint
            # bound in _phase_error() with integral|f| in place of the sum of |p| at the endpoints.
            # integral|f| is estimated from a 3-point sample of the amplitude, which costs three
            # evaluations of f against a full adaptive quad call.
            f_scale = max(
                sum(np.fabs(f[i](x)) for i in range(m)) for x in (a, 0.5 * (a + b), b)
            )
            direct_phase_err = _phase_error(
                BasisData.phase_scale(a, b), f_scale * np.fabs(b - a)
            )

            val = val + data["value"]
            used_regions.append(
                used_interval(
                    start=a,
                    end=b,
                    depth=current_region.depth,
                    abserr=data["abserr"],
                    relerr=None,
                    type=INTERVAL_TYPE_DIRECT,
                    phase_err=direct_phase_err,
                    # as in the Levin branch: only flag it if the floor actually capped the
                    # region, not merely because the worst-case bound is large
                    phase_limited=direct_phase_err > max(data["abserr"], atol),
                )
            )
            num_used_regions = num_used_regions + 1
            num_simple_regions = num_simple_regions + 1
            continue

        # Chen et al. (172).
        # If this region was produced by bisecting a parent, its estimate was already computed as one
        # half of the parent's refined estimate, and we can reuse it. Note that the metadata bookkeeping
        # below counts each region exactly once, when it is processed here as a parent -- the metadata
        # of the comparison regions dataL/dataR has never been accumulated, so reusing them preserves
        # the existing accounting exactly.
        data = current_region.estimate
        if data is None:
            try:
                data = _adaptive_levin_subregion(
                    (a, b),
                    f,
                    BasisData,
                    id_label=id_label,
                    chebyshev_order=chebyshev_order,
                    rtol=rtol,
                    notify_label=notify_label,
                    build_p_sample=build_p_sample,
                )
                num_evaluations += 1
            except LinAlgError as e:
                print(
                    f"!! adaptive_levin ({label}): linear algebra error when estimating Levin subregion ({a}, {b}), width={current_region.width :.8g}"
                )
                raise e

        order = data["metadata"].get("chebyshev_order", None)
        if order is not None:
            if chebyshev_min_order is None or order < chebyshev_min_order:
                chebyshev_min_order = order

        num_SVD_errors = num_SVD_errors + data["metadata"].get("SVD_errors", 0)
        num_order_changes = num_order_changes + data["metadata"].get(
            "num_order_changes", 0
        )
        num_direct_solves = num_direct_solves + data["metadata"].get("direct_solve", 0)

        c = current_region.break_point
        # Chen et al. (173)
        try:
            dataL = _adaptive_levin_subregion(
                (a, c),
                f,
                BasisData,
                id_label=id_label,
                chebyshev_order=chebyshev_order,
                rtol=rtol,
                notify_label=notify_label,
                build_p_sample=build_p_sample,
            )
        except LinAlgError as e:
            print(
                f"!! adaptive_levin ({label}): linear algebra error when estimating Levin left-comparison region ({a}, {c}), parent region = ({a}, {b})"
            )
            raise e

        try:
            dataR = _adaptive_levin_subregion(
                (c, b),
                f,
                BasisData,
                id_label=id_label,
                chebyshev_order=chebyshev_order,
                rtol=rtol,
                notify_label=notify_label,
                build_p_sample=build_p_sample,
            )
        except LinAlgError as e:
            print(
                f"!! adaptive_levin ({label}): linear algebra error when estimating Levin right-comparison region ({c}, {b}), parent region = ({a}, {b})"
            )
            raise e

        num_evaluations += 2
        estimate = data["value"]
        refined_estimate = dataL["value"] + dataR["value"]

        abserr = np.fabs(estimate - refined_estimate)

        # guard the relative-error denominator. The integrand can pass through an accidental zero,
        # and there min(|estimate|, |refined_estimate|) is arbitrarily small, so relerr blows up
        # even though the region is perfectly well resolved -- driving subdivision to depth_max for
        # no gain. Below the requested absolute tolerance a relative test is meaningless anyway, so
        # floor the denominator at atol. (min() rather than max() of the two estimates is retained:
        # it is the more conservative choice.)
        relerr_denom = max(min(np.fabs(estimate), np.fabs(refined_estimate)), atol)
        relerr = abserr / relerr_denom

        # endpoint phase-rounding floor for this region, from the parent estimate: it is the
        # parent's endpoints that survive into the accumulated result. See _phase_error().
        phase_err = data.get("phase_err", 0.0) or 0.0

        # Is this region precision-limited rather than under-resolved? Both of the following must
        # hold: the phase floor exceeds what the caller asked for, AND the step-(4) resolution
        # residual has already come down to that floor. In that case bisecting cannot help -- the
        # children inherit the *same* two endpoint phase values (plus a new interior one that
        # cancels), so subdivision buys additional cost and no accuracy. Accept and flag it.
        #
        # The second condition matters: without it a region with a large but still-reducible
        # resolution residual would be accepted early merely because its phase floor sits above
        # atol, which would lose real accuracy.
        phase_limited = (
            phase_err > atol and phase_err > rtol * relerr_denom and abserr <= phase_err
        )

        # Chen et al. step (4), below (173), adapted in two ways: a relative tolerance check is
        # also admitted, and regions whose accuracy is set by phase rounding rather than by lack
        # of resolution are accepted rather than subdivided.
        # Terminate in any case if we exceed the specified number of bisections.
        resolved = abserr < atol or relerr < rtol

        # Only report the region as phase-limited if the phase floor actually capped it, i.e. the
        # tolerance was *not* otherwise met. The bound in _phase_error() is a worst case that
        # assumes d(theta) ~ eps*|theta|, and on intervals where theta happens to be exactly
        # representable (e.g. the identity phase theta(x) = x at integer endpoints) the true
        # d(theta) is zero and the bound is loose. Flagging those would produce a warning about a
        # result that in fact met its tolerance. The conservative bound is still carried into the
        # aggregate abserr either way; this only governs the diagnostic.
        phase_limited = phase_limited and not resolved

        if resolved or phase_limited or current_region.depth >= depth_max:
            val = val + estimate

            used_regions.append(
                used_interval(
                    start=a,
                    end=b,
                    depth=current_region.depth,
                    type=INTERVAL_TYPE_LEVIN,
                    abserr=abserr,
                    relerr=relerr,
                    p_ratios=data["p_ratios"],
                    phase_err=phase_err,
                    phase_limited=phase_limited,
                )
            )
            num_used_regions = num_used_regions + 1

            if build_p_sample:
                p_points.extend(data["p_sample"])

        else:
            new_abs_history = current_region.abserr_history | {
                current_region.depth: abserr
            }
            new_rel_history = current_region.relerr_history | {
                current_region.depth: relerr
            }
            new_p_ratios_history = current_region.p_ratios_history | {
                current_region.depth: data["p_ratios"]
            }
            new_depth = current_region.depth + 1

            # carry the comparison estimates forward as the children's own estimates; they have just
            # been computed on exactly these intervals, so recomputing them when the children are
            # popped would repeat a third of all the linear solves performed by this driver
            regions.extend(
                [
                    _levin_interval(
                        start=a,
                        end=c,
                        depth=new_depth,
                        abserr_history=new_abs_history,
                        relerr_history=new_rel_history,
                        p_ratios_history=new_p_ratios_history,
                        estimate=dataL,
                    ),
                    _levin_interval(
                        start=c,
                        end=b,
                        depth=new_depth,
                        abserr_history=new_abs_history,
                        relerr_history=new_rel_history,
                        p_ratios_history=new_p_ratios_history,
                        estimate=dataR,
                    ),
                ]
            )

    used_regions.sort(key=lambda x: x.start)
    if build_p_sample:
        p_points.sort(key=lambda x: x[0])

    driver_stop = time.perf_counter()
    elapsed = driver_stop - driver_start

    # Aggregate error estimate. This function previously returned no error estimate at all, which
    # left callers with no option but to trust the result or re-run at a tighter tolerance and
    # compare. Each region contributes max(resolution residual, endpoint phase floor): the two are
    # estimates of independent error sources, and the larger dominates.
    #
    # The regions are summed in absolute value rather than in quadrature. The phase-floor
    # contributions are not independent random errors -- neighbouring regions share endpoints, and
    # an inaccurate phase function produces a systematic drift common to all of them -- so a linear
    # sum is the defensible choice even though it is pessimistic when the residuals really do
    # behave randomly.
    abserr_total = 0.0
    num_phase_limited = 0
    for region in used_regions:
        contribution = region.total_err
        if contribution is not None:
            abserr_total = abserr_total + contribution
        if region.phase_limited:
            num_phase_limited = num_phase_limited + 1

    # guarded exactly as the per-region relative test is guarded, and for the same reason
    relerr_total = abserr_total / max(np.fabs(val), atol)

    # Honest-aggregate check (rec 2, C3). Region acceptance at step (4) tests each region's own
    # abserr/relerr against atol/rtol, so the *summed* abserr_total scales with the number of
    # accepted regions and was never itself compared with what the caller actually asked for.
    # Report whether the aggregate in fact meets the request; this does not change what gets
    # accepted (that is prompt 05's distribute-atol-by-length change), only what gets reported.
    requested_total = max(atol, rtol * np.fabs(val))
    converged = abserr_total <= requested_total
    if not converged:
        print(
            f"!! WARNING (adaptive_levin, {label}): the aggregate error estimate exceeds what was "
            f"requested | abserr_total={abserr_total:.3g} > requested={requested_total:.3g} "
            f"(atol={atol:.3g}, rtol={rtol:.3g})"
        )
        print(
            "   -- atol is currently a per-region tolerance, so the delivered error grows with "
            "the number of accepted regions; consider a tighter atol/rtol"
        )

    # Health check. If neither atol nor rtol is attainable -- typically because the caller has asked for
    # an accuracy the phase function cannot deliver -- the bisection runs to the depth limit and we
    # accept whatever estimate we have. The answer is usually still good, but the cost can be two orders
    # of magnitude higher than necessary, so it is worth saying so rather than letting it pass silently.
    #
    # Note we deliberately do NOT warn on a high proportion of direct-quadrature subintervals. When the
    # integration variable is log(x) (as it is for the Bessel integrals this module was written for),
    # subintervals at small x carry very little phase and are *correctly* handled by direct quadrature;
    # difference-type phase groups, whose net frequency can be small, do the same over much of the
    # range. In that geometry a direct-quadrature fraction above 90% is normal rather than pathological,
    # and no threshold separates it from genuine non-convergence. num_simple_regions is reported in the
    # returned dictionary for callers who want to look.
    if max_depth >= depth_max:
        print(
            f"!! WARNING (adaptive_levin, {label}): bisection reached the maximum depth {depth_max} "
            f"without meeting either tolerance, so some subintervals were accepted unconverged "
            f"| {num_used_regions} subintervals ({num_simple_regions} direct), "
            f"{num_evaluations} Levin solves | atol={atol:.3g}, rtol={rtol:.3g}"
        )
        print(
            "   -- consider relaxing atol/rtol, or check that the phase function is accurate "
            "enough to support the requested tolerance"
        )

    # Second health check, on the phase floor rather than on resolution. This is the case the
    # step-(4) residual cannot detect on its own, so it has to be reported explicitly: the caller
    # asked for an accuracy that the precision of the phase at the region endpoints cannot deliver.
    # Tightening atol/rtol will not help; supplying a range-reduced phase function will, because it
    # replaces an O(theta) argument to sin/cos with an O(2*pi) one.
    if num_phase_limited > 0:
        print(
            f"!! WARNING (adaptive_levin, {label}): {num_phase_limited} of {num_used_regions} "
            f"subintervals were limited by rounding of the phase at their endpoints, not by "
            f"resolution | estimated abserr={abserr_total:.3g}, relerr={relerr_total:.3g} "
            f"| atol={atol:.3g}, rtol={rtol:.3g}"
        )
        print(
            "   -- the requested tolerance is not attainable with this phase function; supplying "
            "a range-reduced phase (theta_mod_2pi) would lower the floor"
        )

    return {
        "value": float(val),
        # estimated absolute and relative error of "value". abserr already includes the endpoint
        # phase-rounding floor as well as the step-(4) resolution residual, so it does not
        # under-report at high frequency the way the resolution residual alone does.
        "abserr": float(abserr_total),
        "relerr": float(relerr_total),
        # True if the aggregate abserr actually meets max(atol, rtol*|value|). atol is currently
        # a per-region tolerance (prompt 05 changes this), so this can be False even though every
        # individual region met its own test -- see the warning printed above.
        "converged": bool(converged),
        # True if at least one accepted region was limited by phase rounding rather than by
        # resolution. When this is set, the requested tolerance was not attainable with the
        # supplied phase function and tightening atol/rtol will not help.
        "phase_limited": num_phase_limited > 0,
        "num_phase_limited_regions": num_phase_limited,
        "p_points": p_points,
        "num_regions": num_used_regions,
        "regions": used_regions,
        "num_simple_regions": num_simple_regions,
        "evaluations": int(num_evaluations),
        "elapsed": float(elapsed),
        "num_SVD_errors": num_SVD_errors,
        "num_order_changes": num_order_changes,
        "num_direct_solves": num_direct_solves,
        "chebyshev_min_order": chebyshev_min_order,
        "max_depth": max_depth,
    }


def _notify_progress(
    now: float,
    last_notify: float,
    start_time: float,
    current_val: float,
    used_regions: int,
    remain_regions: int,
    simple_regions: int,
    max_depth: int,
    num_evaluations: int,
    num_SVD_errors: int,
    num_order_changes: int,
    chebyshev_min_order: int,
    update_number: int,
    id_label,
    notify_label: str = None,
):
    since_last_notify = now - last_notify
    since_start = now - start_time

    if notify_label is not None:
        print(
            f'** STATUS UPDATE #{update_number}: Levin quadrature "{notify_label}" ({id_label}) has been running for {format_time(since_start)} ({format_time(since_last_notify)} since last notification)'
        )
    else:
        print(
            f"** STATUS UPDATE #{update_number}: Levin quadrature {id_label} has been running for {format_time(since_start)} ({format_time(since_last_notify)} since last notification)"
        )

    print(
        f"|  -- current value = {current_val:.7g}, {used_regions} used subintervals (of which {simple_regions} direct quadrature), {remain_regions} subintervals remain | {num_evaluations} integrand evaluations | max depth {max_depth}"
    )
    if num_SVD_errors > 0 or num_order_changes > 0:
        print(
            f"|  -- {num_SVD_errors} SVD errors, {num_order_changes} Chebyshev order changes | minimum used Chebyshev order = {chebyshev_min_order}"
        )


def _safe_fabs(x):
    if x is None:
        return None

    return np.fabs(x)


def _write_progress_data(
    f,
    BasisData,
    regions: List[_levin_interval],
    chebyshev_order: int,
    current_val: float,
    id_label: uuid,
    atol: float,
    rtol: float,
    notify_label: Optional[str] = None,
):
    path = Path(
        f"SlowLevinData/{id_label}/{datetime.now().replace(microsecond=0).isoformat()}"
    ).resolve()
    path.mkdir(parents=True, exist_ok=True)

    payload = {
        "current_val": current_val,
        "id_label": str(id_label),
        "user_label": notify_label,
        "atol": atol,
        "rtol": rtol,
    }

    sns.set_theme()

    m = len(f)

    region_list = []
    for reg_num, region in enumerate(regions):
        start = region.start
        end = region.end
        region_data = {
            "id": reg_num,
            "start": start,
            "end": end,
            "depth": region.depth,
            "abserr_history": region.abserr_history,
            "relerr_history": region.relerr_history,
            "p_ratios_history": region.p_ratios_history,
        }

        x_grid = np.linspace(start, end, 500)
        y_grid = []
        f_grid = [[] for _ in range(m)]
        for x in x_grid:
            basis = BasisData.eval_basis(x)
            y = sum(f[i](x) * basis[i] for i in range(m))
            y_grid.append(_safe_fabs(y))
            for i in range(m):
                f_grid[i].append(_safe_fabs(f[i](x)))

        data = _adaptive_levin_subregion(
            (start, end),
            f,
            BasisData,
            id_label=id_label,
            chebyshev_order=chebyshev_order,
            rtol=rtol,
            notify_label=notify_label,
            build_p_sample=True,
        )
        estimate = data["value"]
        region_data["estimate"] = estimate

        mid = region.break_point

        dataL = _adaptive_levin_subregion(
            (start, mid),
            f,
            BasisData,
            id_label=id_label,
            chebyshev_order=chebyshev_order,
            rtol=rtol,
            notify_label=notify_label,
            build_p_sample=True,
        )
        estimateL = dataL["value"]

        dataR = _adaptive_levin_subregion(
            (mid, end),
            f,
            BasisData,
            id_label=id_label,
            chebyshev_order=chebyshev_order,
            rtol=rtol,
            notify_label=notify_label,
            build_p_sample=True,
        )
        estimateR = dataR["value"]

        refined_estimate = estimateL + estimateR
        region_data["estimate_L"] = estimateL
        region_data["estimate_R"] = estimateR
        region_data["refined_estimate"] = refined_estimate

        relerr = np.fabs((estimate - refined_estimate)) / min(
            np.fabs(estimate), np.fabs(refined_estimate)
        )
        abserr = np.fabs(estimate - refined_estimate)

        region_data["relerr"] = relerr
        region_data["abserr"] = abserr
        region_data["abserr_status"] = "fail" if abserr >= atol else "pass"
        region_data["relerr_status"] = "fail" if relerr >= rtol else "pass"

        # only report regions that will not be accepted in the next step
        if abserr >= atol and relerr >= rtol:
            region_list.append(region_data)

            p_x_grid = []
            p_y_grid = [[] for _ in range(m)]
            for x, p_data in data["p_sample"]:
                p_x_grid.append(x)
                for i in range(m):
                    p_y_grid[i].append(_safe_fabs(p_data[i]))

            pL_x_grid = []
            pL_y_grid = [[] for _ in range(m)]
            for x, p_data in dataL["p_sample"]:
                pL_x_grid.append(x)
                for i in range(m):
                    pL_y_grid[i].append(_safe_fabs(p_data[i]))

            pR_x_grid = []
            pR_y_grid = [[] for _ in range(m)]
            for x, p_data in dataR["p_sample"]:
                pR_x_grid.append(x)
                for i in range(m):
                    pR_y_grid[i].append(_safe_fabs(p_data[i]))

            fig = plt.figure()
            ax = plt.gca()

            ax.plot(x_grid, y_grid, label="integrand", color="r")

            for i in range(m):
                ax.plot(x_grid, f_grid[i], linestyle="dashdot", label=f"Levin f{i+1}")
                ax.plot(
                    p_x_grid, p_y_grid[i], linestyle="dashed", label=f"Levin p{i+1}"
                )
                ax.plot(
                    pL_x_grid, pL_y_grid[i], linestyle="dotted", label=f"Levin p{i+1} L"
                )
                ax.plot(
                    pR_x_grid, pR_y_grid[i], linestyle="dotted", label=f"Levin p{i+1} R"
                )

            ax.set_xscale("linear")
            ax.set_yscale("log")
            ax.legend(loc="best")
            ax.grid(True)

            fig_path = path / f"region{reg_num}_start{start:.5g}_end{end:.5g}.pdf"
            fig.savefig(fig_path)
            fig.savefig(fig_path.with_suffix(".png"))

            plt.close()

    payload["regions"] = region_list
    payload_path = path / "payload.json"
    with open(payload_path, "w", newline="") as handle:
        json.dump(payload, handle, indent=4, sort_keys=True)


def adaptive_levin_sincos(
    x_span: Tuple[float, float],
    f,
    theta: dict,
    atol: float = DEFAULT_LEVIN_ABSTOL,
    rtol: float = DEFAULT_LEVIN_RELTOL,
    chebyshev_order: int = DEFAULT_LEVIN_CHEBSHEV_ORDER,
    depth_max: int = DEFAULT_LEVIN_MAX_DEPTH,
    build_p_sample: bool = False,
    notify_interval: int = DEFAULT_LEVIN_NOTIFY_INTERVAL,
    notify_label: str = None,
    emit_diagnostics=False,
):
    """
    Adaptive Levin quadrature of an integral of the form

        integral_{x_span} [ f[0](x) sin(theta(x)) + f[1](x) cos(theta(x)) ] dx

    using the adaptive Levin method of Bremer, Chen & Yang (arXiv:2211.13400, section 5), falling
    back to ordinary adaptive quadrature (scipy.integrate.quad) on subintervals that are not
    oscillatory enough for the Levin rule to offer an advantage.

    :param x_span: a 2-tuple (a, b) giving the integration limits. Must have exactly two finite
        entries.
    :param f: a 2-element sequence of callables [f_sin, f_cos]. f_sin multiplies sin(theta(x)),
        f_cos multiplies cos(theta(x)). This basis is fixed at two components (sin, cos); the
        underlying driver supports an arbitrary number of components for other bases, but this
        entry point does not expose that.
    :param theta: a dict describing the phase function, with keys:
          * "theta" (required) -- callable, the raw phase theta(x). Always used to decide
            whether a subinterval is oscillatory enough for the Levin rule (via the total phase
            change across it), and used to evaluate sin/cos directly if "theta_mod_2pi" is not
            supplied.
          * "theta_mod_2pi" (optional) -- callable, theta(x) reduced to the range [0, 2*pi) (or
            any range of length 2*pi). When supplied, sin/cos are evaluated from this
            range-reduced value instead of the raw phase, which is handed an O(2*pi) argument
            rather than an O(theta) one and so has O(eps) absolute rounding error instead of
            O(eps*theta). This is the single most effective way to lower the reported error floor
            at high frequency (see phase_err below).
          * "theta_deriv" (optional) -- callable, theta'(x). When supplied, it is used directly
            instead of differentiating a sampled theta(x) with the spectral differentiation
            matrix; this avoids inheriting rounding error from the magnitude of the raw phase
            into the derivative estimate.
    :param atol: requested absolute tolerance. Must be strictly positive: this module cannot
        deliver a purely relative-error contract because its phase-rounding error floor is
        absolute by construction. NOTE: atol is currently applied as a *per-region* tolerance,
        and the returned "abserr" is a sum over accepted regions, so the delivered accuracy
        degrades with the number of regions the driver needs. The returned "converged" flag
        reports whether the aggregate in fact met max(atol, rtol*|value|); a future change will
        distribute atol across regions so that "converged" is usually true by construction.
    :param rtol: requested relative tolerance. Must be non-negative.
    :param chebyshev_order: spectral order used for each Levin subregion collocation grid.
        Values below 8 are clamped up, with a warning.
    :param depth_max: maximum bisection depth. Must be non-negative.
    :param build_p_sample: if True, retain and return the sampled Levin antiderivatives p(x) on
        every accepted region (diagnostics only; costs additional memory).
    :param notify_interval: seconds between progress notifications for long-running calls.
    :param notify_label: optional label included in progress and warning messages.
    :param emit_diagnostics: if True, periodically write plots and a JSON payload describing
        slow-to-converge regions to disk (see _write_progress_data()).

    :return: a dict with (at least) the following keys:
          * "value" -- the estimated value of the integral.
          * "abserr" -- estimated absolute error of "value". This is a sum, over accepted
            regions, of max(step-(4) resolution residual, endpoint phase-rounding floor); it is
            an estimate, not a proven bound.
          * "relerr" -- "abserr" divided by max(|value|, atol).
          * "converged" -- True if "abserr" <= max(atol, rtol*|value|), i.e. whether the request
            was actually met in aggregate (see the atol caveat above).
          * "phase_limited" -- True if at least one accepted region's achievable accuracy was set
            by phase rounding rather than by lack of resolution; tightening atol/rtol will not
            help such regions, only a more accurate phase function will.
          * "num_phase_limited_regions" -- count of such regions.
          * "num_regions" -- number of accepted subintervals (Levin + direct quadrature).
          * "num_simple_regions" -- of those, how many were handled by direct quadrature rather
            than the Levin rule.
          * "regions" -- list of used_interval objects describing each accepted subinterval.
          * "p_points" -- sampled Levin antiderivatives, if build_p_sample was True.
          * "evaluations" -- number of Levin subregion solves performed.
          * "elapsed" -- wall-clock time in seconds.
          * "num_SVD_errors", "num_order_changes", "num_direct_solves", "chebyshev_min_order",
            "max_depth" -- diagnostic counters; see the source for exact semantics.
    """
    if len(f) != 2:
        raise ValueError(
            f"levin_quadrature: adaptive_levin_sincos requires exactly two amplitude functions "
            f"f=[f_sin, f_cos] (the sin/cos basis _Basis_SinCos is fixed at two components); "
            f"received {len(f)}"
        )

    A = _Basis_SinCos(theta)

    return _adaptive_levin(
        x_span,
        f,
        A,
        atol=atol,
        rtol=rtol,
        chebyshev_order=chebyshev_order,
        depth_max=depth_max,
        build_p_sample=build_p_sample,
        notify_interval=notify_interval,
        notify_label=notify_label,
        emit_diagnostics=emit_diagnostics,
    )
