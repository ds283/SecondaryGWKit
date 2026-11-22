"""
levin-phase-floor prompt 05a section 6, with section 9's T1: the phase probe. New in 05a (05's
harness has no probe); run by worker.py in each tree's rtol 1e-7 unit, after the unit's timed
section, on the transfer-function objects the unit's own ``build_partition`` call built (captured
by worker.py). It reads phase objects only and runs no quadrature.

For every Levin sub-interval of the row (``method == "Levin"``) and every transfer-function factor
oscillatory there (``regime[1]`` for T_q, ``regime[2]`` for T_r), one **window** at the
sub-interval's low-z end ``z_min`` (where |theta_T| is largest):

- **the ladder** (section 9). Level n has relative width ``w_n = 1e-6 / 2**n`` in z: the window is
  ``[a, b]`` with ``a = z_min`` and ``b = a + a * w_n``, so it lies inside the sub-interval. Its 65
  samples are ``numpy.linspace(a, b, 65)``, ends included, as prompt 05a section 3 test 3
  (``test_t_phase_reanchor._sigma_theta``) takes them. Each tree computes **every** level from
  n = 0 until the 65 samples stop being distinct doubles (``numpy.diff > 0`` fails), and records
  each level. Which level is *used* depends on both trees (section 9: "halve while either tree's
  largest unwrapped step is >= pi/2"), so it is chosen by analyse.py from the two trees' ladders:
  the first level at which both trees' largest unwrapped step (sampled in z) is below pi/2. A
  window with no such level before the samples stop being distinct is **unresolved**.
- **at every level, in this tree:**
  - sampled in z: ``theta_mod_2pi(z, x_is_log=False)`` at the 65 doubles; ``numpy.unwrap``; a
    degree-4 least-squares polynomial in t in [-1, 1]; sigma_theta, the RMS residual; and the
    largest |step| of the unwrapped samples. This is test 3's ``_sigma_theta``, unchanged;
  - sampled in u (descriptive): the same with ``theta_mod_2pi(u, x_is_log=True)`` at
    ``numpy.linspace(log1p(a), log1p(b), 65)``, the same window in log(1+z) (section 9: "the
    u-sampled sigma_theta uses the same width, in log(1+z)"). If those 65 u doubles are not
    distinct, the u numbers of that level are None;
  - ``p_z`` at each z sample, from the factor's ``cs_tau.end_partial`` (the phase's leading
    primitive), with its node; max |p_z| (section 9 asks for it under t; it is recorded under both);
  - every sample: the abscissae and the values, as doubles (in the unit's pickle).
- **per window:** z = a, k, z_init, |theta_T| = |raw_theta(a)|, |d theta_T/du| =
  |theta_deriv(a, log_derivative=True)|, |d theta_T/dz| (so that the true step per sample,
  |d theta/dz| (b - a)/64, can be set beside the unwrapped one), the abscissa term
  eps z/(1+z) |d theta_T/du|, ``exact_remainder`` of the phase, and the sub-interval.

``EPS`` is 2**-52, ``numpy.finfo(float).eps``, as the tests use.
"""

import math

import numpy as np

EPS = float(np.finfo(np.float64).eps)
SIGMA_POINTS = 65
SIGMA_DEGREE = 4
W0 = 1.0e-6
MAX_STEP = 0.5 * math.pi
MAX_LEVELS = 80  # a guard only; the ladder ends when the samples stop being distinct


def _fit(x, vals, a, b):
    """test_t_phase_reanchor._sigma_theta's fit, on samples already taken."""
    th = np.unwrap(vals)
    t = (2.0 * x - (a + b)) / (b - a)
    coef = np.polynomial.polynomial.polyfit(t, th, SIGMA_DEGREE)
    res = th - np.polynomial.polynomial.polyval(t, coef)
    return float(np.sqrt(np.mean(res * res))), float(np.max(np.abs(np.diff(th))))


def _distinct(x):
    return bool(np.all(np.diff(x) > 0.0))


def probe_factor(phase, z_min):
    """The ladder of one window. Returns (summary, samples): summary is JSON-able (floats as
    repr strings where exactness matters is not needed: the samples carry the doubles)."""
    leading = phase._leading
    a = float(z_min)
    theta_a = float(phase.raw_theta(a))
    dth_du = float(phase.theta_deriv(a, log_derivative=True))
    dth_dz = float(phase.theta_deriv(a))
    levels = []
    samples = []
    for n in range(MAX_LEVELS):
        w = W0 / 2.0**n
        b = a + a * w
        x = np.linspace(a, b, SIGMA_POINTS)
        if not _distinct(x):
            levels.append({"n": n, "w": w, "a": a.hex(), "b": float(b).hex(), "distinct": False})
            samples.append(None)
            break
        th_z = np.array([phase.theta_mod_2pi(float(z), x_is_log=False) for z in x])
        sig_z, step_z = _fit(x, th_z, a, b)
        parts = [leading.end_partial(float(z)) for z in x]
        nodes = np.array([int(j) for j, _p in parts], dtype=np.int64)
        p_z = np.array([float(p) for _j, p in parts])
        ua, ub = math.log1p(a), math.log1p(float(b))
        u = np.linspace(ua, ub, SIGMA_POINTS)
        if _distinct(u):
            th_u = np.array([phase.theta_mod_2pi(float(v), x_is_log=True) for v in u])
            sig_u, step_u = _fit(u, th_u, ua, ub)
        else:
            th_u, sig_u, step_u = None, None, None
        levels.append(
            {
                "n": n,
                "w": w,
                "a": a.hex(),
                "b": float(b).hex(),
                "distinct": True,
                "sigma_z": sig_z,
                "step_z": step_z,
                "sigma_u": sig_u,
                "step_u": step_u,
                "u_distinct": th_u is not None,
                "max_abs_p_z": float(np.max(np.abs(p_z))),
                "nodes": sorted({int(j) for j in nodes}),
                "true_step_z_est": abs(dth_dz) * (float(b) - a) / (SIGMA_POINTS - 1),
            }
        )
        samples.append(
            {"z": x, "theta_z": th_z, "p_z": p_z, "node": nodes, "u": u, "theta_u": th_u}
        )
    summary = {
        "z": a,
        "z_hex": a.hex(),
        "k": float(phase.k),
        "z_init": float(phase.z_anchor),
        "theta": theta_a,
        "abs_theta": abs(theta_a),
        "eps_abs_theta": EPS * abs(theta_a),
        "abs_dtheta_du": abs(dth_du),
        "abs_dtheta_dz": abs(dth_dz),
        "abscissa_term": EPS * a / (1.0 + a) * abs(dth_du),
        "exact_remainder": bool(getattr(phase, "exact_remainder", None)),
        "levels": levels,
    }
    return summary, samples


def probe_partition(subintervals):
    """Every window of a row. ``subintervals`` is build_partition's list (each with z_max, z_min,
    regime, method, Tq, Tr). Returns (windows, samples), in sub-interval order, T_q before T_r."""
    windows = []
    all_samples = []
    for i, item in enumerate(subintervals):
        if item["method"] != "Levin":
            continue
        regime = tuple(bool(x) for x in item["regime"])
        for label, osc, fn in (("Tq", regime[1], item["Tq"]), ("Tr", regime[2], item["Tr"])):
            if not osc:
                continue
            summary, samples = probe_factor(fn.phase, item["z_min"])
            summary.update(
                {
                    "subinterval": i,
                    "factor": label,
                    "regime": list(regime),
                    "sub_z_max": float(item["z_max"]),
                    "sub_z_min": float(item["z_min"]),
                    "window_inside_subinterval_at_level0": float(
                        item["z_min"] + item["z_min"] * W0
                    )
                    <= float(item["z_max"]),
                }
            )
            windows.append(summary)
            all_samples.append(samples)
    return windows, all_samples
