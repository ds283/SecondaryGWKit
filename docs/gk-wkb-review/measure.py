"""Offline numerical probes, no Ray cluster or datastore, no production mutations.
Run: PYTHONPATH=. ./venv/bin/python docs/gk-wkb-review/measure.py
"""

import json
import math
from types import SimpleNamespace as NS
from pathlib import Path
import mpmath as mp

mp.mp.dps = 60
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import make_interp_spline
from scipy.special import jv, yv
from ComputeTargets.WKB_Gk import Gk_omegaEff_sq as om2, Gk_d_ln_omegaEff_dz as dlom
from CosmologyConcepts import redshift, redshift_array
from Quadrature.integrators.WKB_phase_function import (
    integrate_phase_function,
    stage_2_evolution,
)
from LiouvilleGreen.phase_spline import phase_spline
from LiouvilleGreen.WKBtools import WKB_mod_2pi


class Model:
    def __init__(self, p=2.0):
        self.p = p
        self.functions = NS(
            Hubble=lambda z: (1 + z) ** p,
            epsilon=lambda z: p,
            d_epsilon_dz=lambda z: 0.0,
            d2_epsilon_dz2=lambda z: 0.0,
        )


def key(k):
    return NS(k=NS(k=k, k_inv_Mpc=k, store_id=0))


def grid(si, sf, n=100):
    return np.geomspace(si, sf, int(np.ceil(abs(np.log10(si / sf)) * n)) + 1) - 1


def phase_error(d, m, exact):
    # mpmath is needed: longdouble is float64 on this Apple Silicon host.
    return np.array(
        [
            float(mp.mpf(int(di)) * 2 * mp.pi + mp.mpf(float(mi)) - ex)
            for di, mi, ex in zip(d, m, exact)
        ]
    )


def radiation_ref(k, zi, z):
    return [
        mp.mpf(float(k)) * (1 / (1 + mp.mpf(float(zi))) - 1 / (1 + mp.mpf(float(v))))
        for v in z
    ]


def production(k, si, sf=1, n=100, rtol=1e-8, atol=1e-10):
    model = Model()
    zs = grid(si, sf, n)
    samples = redshift_array([redshift(i, float(z)) for i, z in enumerate(zs)])
    metadata = {}
    out = integrate_phase_function(
        model,
        key(k),
        float(si - 1),
        samples,
        om2,
        dlom,
        om2(model, k, si - 1),
        atol,
        rtol,
        metadata,
        "probe",
        "G",
    )
    err = phase_error(
        out["theta_div_2pi_sample"],
        out["theta_mod_2pi_sample"],
        radiation_ref(k, si - 1, zs),
    )
    return dict(
        k=k,
        si=si,
        sf=sf,
        rtol=rtol,
        atol=atol,
        max_phase_error=float(np.max(abs(err))),
        end_phase_error=float(err[-1]),
        metadata=metadata,
        stage1_nfev=out["stage_1_data"].RHS_evaluations if out["stage_1_data"] else 0,
        stage2_nfev=out["stage_2_data"].RHS_evaluations if out["stage_2_data"] else 0,
    )


def constant_frequency(k, rtol):
    zs = grid(100, 1)
    z = []
    d = []
    m = []
    out = stage_2_evolution(
        Model(),
        key(k),
        99.0,
        0.0,
        list(zs),
        lambda m, k, z: k * k,
        lambda m, k, z: 0.0,
        z,
        m,
        d,
        0.0,
        0,
        1e-10,
        rtol,
        "probe",
        "constant",
    )
    e = phase_error(d, m, [-mp.mpf(k) * (99 - mp.mpf(float(v))) for v in zs])
    return dict(
        omega=k,
        rtol=rtol,
        max_phase_error=float(max(abs(e))),
        end_phase_error=float(e[-1]),
        nfev=out["data"].RHS_evaluations,
    )


def spline_probe(k, density, chunklog):
    # Fixed response at s=1, independently sampled sources s=10..10000.
    # Subtract a common integer cycle count as GkSource does. All sources inside horizon for k>=1e6.
    s = np.geomspace(10, 1e4, 3 * density + 1)
    z = s - 1
    exact = [mp.mpf(k) * (1 / (1 + mp.mpf(float(v))) - 1) for v in z]
    pairs = [WKB_mod_2pi(float(v)) for v in exact]
    d, m = map(list, zip(*pairs))
    base = d[0]
    d = [v - base for v in d]
    spl = phase_spline(
        z,
        d,
        m,
        x_is_redshift=True,
        increasing=False,
        chunk_step=None,
        chunk_logstep=chunklog,
    )
    sm = np.sqrt(s[:-1] * s[1:])
    zm = sm - 1
    # Exact phase relative to the same common offset; compare raw for interpolation, angular for sin/cos.
    ref = [mp.mpf(k) * (1 / (1 + mp.mpf(float(v))) - 1) - base * 2 * mp.pi for v in zm]
    err = np.array(
        [float(mp.mpf(float(spl.raw_theta(float(v)))) - ex) for v, ex in zip(zm, ref)]
    )
    angular = np.array([spl.theta_mod_2pi(float(v)) for v in zm])
    osc = np.max(
        abs(np.exp(1j * angular) - np.array([complex(mp.exp(1j * v)) for v in ref]))
    )
    widths = [max(sp._y_points) - min(sp._y_points) for sp in spl._splines.values()]
    max_y = max(max(abs(np.asarray(sp._y_points))) for sp in spl._splines.values())
    node_roundoff = max(
        abs(np.exp(1j * spl.theta_mod_2pi(float(v))) - np.exp(1j * mi))
        for v, mi in zip(z, m)
    )
    from scipy.optimize import brentq

    jumps = []
    derivative_jumps = []
    scan = np.linspace(np.log(s[0]), np.log(s[-1]), 3001)
    previous = spl._match_chunk(scan[0])[0]
    for left, right in zip(scan[:-1], scan[1:]):
        current = spl._match_chunk(right)[0]
        if current is not previous:

            def metric(sp, t):
                return (0.5 - (t - sp.min_log_x) / (sp.max_log_x - sp.min_log_x)) ** 2

            root = brentq(
                lambda t: metric(previous, t) - metric(current, t),
                left,
                right,
                xtol=1e-14,
            )
            a = previous.theta_mod_2pi(root, x_is_log=True, warn_unsafe=False)
            b = current.theta_mod_2pi(root, x_is_log=True, warn_unsafe=False)
            jumps.append(abs(np.angle(np.exp(1j * (a - b)))))
            ap = previous.theta_deriv(
                root, x_is_log=True, log_derivative=True, warn_unsafe=False
            )
            bp = current.theta_deriv(
                root, x_is_log=True, log_derivative=True, warn_unsafe=False
            )
            derivative_jumps.append(abs(ap - bp) / (k * np.exp(-root)))
        previous = current
    # Amplitude for directly initialized radiation Green's function = H_source/k = s_source**2/k.
    amp = make_interp_spline(np.log(s), s * s / k)
    ae = max(abs(amp(np.log(sm)) / (sm * sm / k) - 1))
    return dict(
        k=k,
        density=density,
        chunk_logstep=chunklog,
        chunks=spl.num_chunks,
        max_chunk_span=max(widths),
        max_abs_spline_ordinate=float(max_y),
        node_roundoff=float(node_roundoff),
        max_switch_phase_jump=float(max(jumps, default=0)),
        max_switch_derivative_relative_jump=float(max(derivative_jumps, default=0)),
        interior_error=float(max(abs(err[3:-3]))),
        max_error=float(max(abs(err))),
        osc_error=float(osc),
        amplitude_relative_error=float(ae),
    )


def truncation(p, xs):
    # x=k*tau, tau=s**(1-p)/(p-1). Hold k=1e4, source x=xs; only WKB propagation, Gs=0,Gs'=1.
    # Omega dz = -sqrt(1+a/x**2) dx. Integrate only small correction with quad.
    k = 1e4
    c = 1.5 * p - p * p / 4 - 2
    a = c / (p - 1) ** 2
    x = np.linspace(xs, 1000, 20001)
    si = (k / ((p - 1) * xs)) ** (1 / (p - 1))
    s = (k / ((p - 1) * x)) ** (1 / (p - 1))

    def correction(t):
        return a / t**2 / (np.sqrt(1 + a / t**2) + 1)

    sol = solve_ivp(
        lambda t, y: [-correction(t)],
        (xs, 1000),
        [0.0],
        rtol=2e-13,
        atol=1e-14,
        t_eval=x,
    )
    theta = -(x - xs) + sol.y[0]
    omega = np.sqrt((k / s**p) ** 2 + c / s**2)
    oi = np.sqrt((k / si**p) ** 2 + c / si**2)
    amp = np.sqrt(si**p / s**p / (omega * oi))
    g = amp * np.sin(theta)
    nu = 1 / (p - 1) - 0.5
    exact = (
        -(si**p)
        * np.pi
        / (2 * k)
        * np.sqrt(x * xs)
        * (jv(nu, xs) * yv(nu, x) - jv(nu, x) * yv(nu, xs))
    )
    # Exact phase difference from scaled Hankel ratio; exact amplitude from Bessel magnitudes.
    from scipy.special import hankel1e

    hs = hankel1e(nu, xs)
    h = hankel1e(nu, x)
    residual = -np.angle(h / hs)
    phase_err = np.angle(np.exp(1j * (sol.y[0] - residual)))
    exactamp = si**p * np.pi / (2 * k) * np.sqrt(x * xs) * abs(hs) * abs(h)
    return dict(
        p=p,
        x_source=xs,
        max_G_error_over_exact_envelope=float(max(abs(g - exact) / exactamp)),
        max_phase_error=float(max(abs(phase_err))),
        max_amplitude_relative_error=float(max(abs(amp / exactamp - 1))),
        initial_WKB_criterion=float(
            abs(dlom(Model(p), k, si - 1)) / np.sqrt(om2(Model(p), k, si - 1))
        ),
    )


if __name__ == "__main__":
    result = {}
    result["radiation"] = [
        production(k, si, rtol=r)
        for k, si in [(1e4, 1e4 / 30), (1e7, 1e7 / 30), (1e7, 100), (1e9, 100)]
        for r in [1e-8, 1e-12]
    ]
    result["tight_radiation"] = [
        production(1e7, 100, rtol=r, atol=a)
        for r, a in [(1e-12, 1e-14), (5e-14, 1e-16)]
    ]
    result["constant_frequency"] = [
        constant_frequency(k, r) for k in [1e3, 1e7] for r in [1e-8, 1e-12]
    ]
    result["splines"] = [
        spline_probe(k, n, ch)
        for k in [1e6, 1e8]
        for n in [100, 300]
        for ch in [None, 125, 2]
    ]
    result["truncation"] = [
        truncation(p, xs) for p in [2, 1.8, 1.5] for xs in [30, 100]
    ]
    print(json.dumps(result, indent=2))
    Path(__file__).with_name("measurements.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
