"""
The constant-w Bessel form of the tensor Green's function, spec 02 R2-R9 (docs/spec/02-greens-function.md),
as a comparison aid for the numeric and WKB solutions.

Spec 02 R2 writes the constant-w background as a(eta) ∝ eta^(1+b) with b = (1 - 3w)/(1 + 3w), and R9
gives the retarded Green's function with Bessel order b + 1/2. That derivation needs 1 + 3w > 0: there
a ∝ eta^(2/(1+3w)) with eta > 0 and growing. As w -> -1/3 from above the order diverges, and for
1 + 3w <= 0 the form does not describe the background at all. In LambdaCDM w(z) crosses -1/3 at late
times, where the form evaluated at the background's own w(z) reached 7.57e+243 in the A3 v2 store
(prompts/handover/IMPLEMENTATION_STATE.md, [08-the-gkwkb-reader-fails-plotting-…]). Both functions
therefore return None outside the domain, and None where the evaluated form is not finite.
"""

from math import isfinite, pi, sqrt
from typing import Optional

from scipy.special import jv, yv


def constant_w_form_applies(w: float) -> bool:
    """
    True when the constant-w form of spec 02 R2 describes a background of equation of state w,
    i.e. when 1 + 3w > 0.

    The test is exactly the derivation's, with no tolerance band: a margin would be a tuning choice
    nobody has measured. It is written with the same expression as the denominator of b in the two
    functions below, so the form is evaluated only where that denominator is strictly positive.
    """
    return 1.0 + 3.0 * w > 0.0


def compute_analytic_G(k, w, tau_source, tau, H_source) -> Optional[float]:
    """
    The constant-w Green's function of spec 02 R9, in the code's normalisation (spec 02 §0.1 item (2):
    G_code(z, z') = -H(z') Gr_k(eta, eta')), at wavenumber k, response time tau and source time
    tau_source.

    Returns None when 1 + 3w <= 0 (``constant_w_form_applies``), before any Bessel function is
    evaluated: spec 02 R2's derivation needs 1 + 3w > 0, and outside it the form has no meaning.
    Also returns None when the form, inside its domain, evaluates to a value that is not finite
    (Bessel overflow, or a 0 * inf product). A value of exactly 0 is finite and is returned. Every
    other value is the expression below, unchanged.
    """
    if not constant_w_form_applies(w):
        return None

    b = (1.0 - 3.0 * w) / (1.0 + 3.0 * w)
    k_tau = k * tau
    k_tau_source = k * tau_source

    # The main numerical calculation obtains G_us(z,z') defined in DS' analytical calculation, which is
    # for the source -delta(z-z') expressed in redshift.
    # The analytic result is expressed in terms of conformal time, as G_them(eta, eta') with source
    # delta(eta-eta').
    # The delta function transforms with the (absolute value of) the Jacobian of the transformation.
    # This means we get G_us(z,z') = -H(z') G_them(eta, eta').
    # The multiplication of -H_source here accounts for this minus sign and the coordinate Jacobian
    n0 = 0.5 + b

    A = -H_source * pi / 2.0
    B = sqrt(tau * tau_source)
    C = jv(n0, k_tau_source) * yv(n0, k_tau)
    D = jv(n0, k_tau) * yv(n0, k_tau_source)

    G = A * B * (C - D)
    if not isfinite(G):
        return None

    return G


def compute_analytic_Gprime(k, w, tau_source, tau, H_source, H) -> Optional[float]:
    """
    The derivative with respect to the response redshift z of ``compute_analytic_G``, from the same
    constant-w form (spec 02 R2-R9), with H the Hubble rate at the response point.

    Returns None when 1 + 3w <= 0, before any Bessel function is evaluated, and None when the form,
    inside its domain, evaluates to a value that is not finite, for the reasons
    ``compute_analytic_G`` gives. A value of exactly 0 is returned. Every other value is the
    expression below, unchanged.
    """
    if not constant_w_form_applies(w):
        return None

    b = (1.0 - 3.0 * w) / (1.0 + 3.0 * w)
    k_tau = k * tau
    k_tau_source = k * tau_source

    n0 = 0.5 + b
    n1 = 1.5 + b

    A = pi / 2.0 * (H_source / H)
    B = sqrt(tau_source / tau)
    C1 = k_tau * jv(n1, k_tau) - (b + 1.0) * jv(n0, k_tau)
    C2 = yv(n0, k_tau_source)
    D1 = (b + 1.0) * yv(n0, k_tau) - k_tau * yv(n1, k_tau)
    D2 = jv(n0, k_tau_source)

    Gprime = A * B * (C1 * C2 + D1 * D2)
    if not isfinite(Gprime):
        return None

    return Gprime
