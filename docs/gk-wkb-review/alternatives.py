"""Additional controls and alternative numerical constructions. Offline only."""

import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import importlib
import numpy as np
import mpmath as mp
from scipy.integrate import solve_ivp
from numpy.polynomial.legendre import leggauss
from measure import Model, key, grid, radiation_ref, phase_error, production

mp.mp.dps = 60
phase_module = importlib.import_module("Quadrature.integrators.WKB_phase_function")


def direct_comparison():
    out = []
    for constant in [False, True]:
        k = 1e7
        z = grid(100, 1)
        v = np.log1p(z)
        f = (lambda z: k) if constant else (lambda z: k / (1 + z) ** 2)
        exact = (
            [-mp.mpf(k) * (99 - mp.mpf(float(q))) for q in z]
            if constant
            else radiation_ref(k, 99, z)
        )
        for coord in ["z", "log1pz"]:
            t = z if coord == "z" else v
            fun = (
                (lambda t, y: [f(t)])
                if coord == "z"
                else (lambda t, y: [np.exp(t) * f(np.expm1(t))])
            )
            for rtol, atol in [(1e-8, 1e-10), (1e-12, 1e-14)]:
                sol = solve_ivp(
                    fun,
                    (t[0], t[-1]),
                    [0.0],
                    method="DOP853",
                    t_eval=t,
                    rtol=rtol,
                    atol=atol,
                )
                err = [float(mp.mpf(float(a)) - b) for a, b in zip(sol.y[0], exact)]
                out.append(
                    dict(
                        constant=constant,
                        coord=coord,
                        rtol=rtol,
                        max_phase_error=max(abs(np.array(err))),
                        nfev=sol.nfev,
                    )
                )
    return out


def local_quadrature():
    # Exact radiation, source curve, k/s from 1e7 down to 1e4, same geometry as spline probe.
    k = 1e8
    s = np.geomspace(10, 1e4, 301)
    v = np.log(s)
    out = []
    for order in [2, 4, 8]:
        roots, weights = leggauss(order)
        increments = []
        errors = []
        local_errors = []
        # Integrate whole intervals, then reconstruct midpoint from left interval anchor.
        for a, b in zip(v[:-1], v[1:]):

            def integ(a, b):
                t = (a + b) / 2 + (b - a) * roots / 2
                return float((b - a) / 2 * np.dot(weights, -k * np.exp(-t)))

            mid = (a + b) / 2
            approx = math_fsum(increments) + integ(a, mid)
            ref = mp.mpf(k) * (
                mp.exp(-mp.mpf(float(mid))) - mp.exp(-mp.mpf(float(v[0])))
            )
            localref = mp.mpf(k) * (
                mp.exp(-mp.mpf(float(mid))) - mp.exp(-mp.mpf(float(a)))
            )
            errors.append(float(mp.mpf(approx) - ref))
            local_errors.append(float(mp.mpf(integ(a, mid)) - localref))
            increments.append(integ(a, b))
        out.append(
            dict(
                order=order,
                global_midpoint_error=max(abs(np.array(errors))),
                local_midpoint_error=max(abs(np.array(local_errors))),
            )
        )
    return out


from math import fsum as math_fsum


def edge_cases():
    from CosmologyConcepts import redshift, redshift_array

    model = Model()
    k = key(1e14)
    # Mock only unit compatibility, not any numerical operation.
    zinit = 10.0
    ztarget = zinit - 1e-11
    samples = redshift_array([redshift(0, ztarget)])
    with patch.object(phase_module, "check_units", lambda *a: None):
        res = phase_module.WKB_phase_function._function(
            NS(get=lambda: model),
            k,
            zinit,
            samples,
            phase_module_om2,
            phase_module_dlom,
            atol=1e-10,
            rtol=1e-8,
        )
    exact = radiation_ref(1e14, zinit, [ztarget])[0]
    coords = []
    for zi in [99.0, 1e6, 1e8, 1e12]:
        z = grid(zi + 1, 1)
        u = zi - z
        recovered = zi - u
        coords.append(
            dict(
                zinit=zi,
                max_redshift_roundtrip_error=float(max(abs(recovered - z))),
                ulp_u=float(np.spacing(zi)),
            )
        )
    # A genuine terminal phase event at the final sample; no missing work.
    z = []
    m = []
    d = []
    breaks = []
    try:
        phase_module.stage_1_evolution(
            model,
            key(1),
            10000.0,
            0.0,
            [10000.0, 5000.0, 0.0],
            breaks,
            lambda m, k, z: 1.0,
            lambda m, k, z: 0.0,
            z,
            m,
            d,
            0,
            1e-10,
            1e-8,
            "edge",
            "G",
        )
        endpoint = "completed"
    except Exception as e:
        endpoint = type(e).__name__ + ": " + str(e)
    import contextlib, io

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        try:
            production(1.1e19, 1e8 + 1)
            large_start = "completed"
        except Exception as e:
            large_start = type(e).__name__ + ": " + str(e)
    return dict(
        large_start_actual_run=large_start,
        short_interval=dict(
            delta_z=zinit - ztarget,
            exact_phase=float(exact),
            returned_phase=res["theta_mod_2pi_sample"],
            metadata=res["metadata"],
        ),
        coordinate_roundtrip=coords,
        stage1_endpoint=endpoint,
    )


from ComputeTargets.WKB_Gk import (
    Gk_omegaEff_sq as phase_module_om2,
    Gk_d_ln_omegaEff_dz as phase_module_dlom,
)


def varying_background():
    # Smooth localized change in epsilon, with mutually consistent exact H and derivatives.
    from scipy.special import erf
    from scipy.integrate import quad

    p0 = 1.8
    a = 0.12
    center = 3.0
    width = 0.3
    k = 1e6

    def H(z):
        v = np.log1p(z)
        t = (v - center) / width
        return np.exp(p0 * v + a * width * np.sqrt(np.pi) / 2 * erf(t))

    def eps(z):
        t = (np.log1p(z) - center) / width
        return p0 + a * np.exp(-t * t)

    def ep(z):
        t = (np.log1p(z) - center) / width
        return -2 * a * t * np.exp(-t * t) / (width * (1 + z))

    def epp(z):
        s = 1 + z
        t = (np.log(s) - center) / width
        ev = -2 * a * t * np.exp(-t * t) / width
        evv = a * np.exp(-t * t) * (4 * t * t - 2) / width**2
        return (evv - ev) / s**2

    model = NS(functions=NS(Hubble=H, epsilon=eps, d_epsilon_dz=ep, d2_epsilon_dz2=epp))
    z = grid(100, 2)
    d = []
    m = []
    zz = []
    phase_module.stage_2_evolution(
        model,
        key(k),
        99.0,
        1.0,
        list(z),
        phase_module_om2,
        phase_module_dlom,
        zz,
        m,
        d,
        0.0,
        0,
        1e-10,
        1e-8,
        "varying",
        "G",
    )
    # Independent Gauss-Kronrod quadrature, in log-redshift with explicit transition breakpoint.
    ref = []
    quaderr = []
    for q in z:
        lo = np.log1p(q)
        hi = np.log(100.0)
        value, error = quad(
            lambda v: np.exp(v) * np.sqrt(phase_module_om2(model, k, np.expm1(v))),
            lo,
            hi,
            epsabs=1e-9,
            epsrel=2e-14,
            points=[center] if lo < center < hi else None,
        )
        ref.append(-value)
        quaderr.append(error)
    e = phase_error(d, m, [mp.mpf(t) for t in ref])
    return dict(
        max_phase_error=float(max(abs(e))),
        reference_quad_error_estimate=max(quaderr),
        phase_span=abs(ref[-1]),
    )


if __name__ == "__main__":
    with patch.object(phase_module, "DEFAULT_OMEGA_WKB_SQ_MAX", float("inf")):
        stage1_only = production(1e7, 100)
    result = dict(
        stage1_only=stage1_only,
        direct=direct_comparison(),
        local_quadrature=local_quadrature(),
        edges=edge_cases(),
        varying_background=varying_background(),
    )
    print(json.dumps(result, indent=2))
    Path(__file__).with_name("alternatives.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
