"""
levin-phase-floor prompt 06 section 6: the phase probe. New in 06 (05a's probe.py, which measured
sigma_theta of T's remainder in narrow z windows, is copied unchanged and unused). Run by
worker.py in each tree's rtol 1e-7 unit of 04's rows R1-R7, after the unit's timed section, with
the dispatcher uninstalled, on the objects the unit's own ``build_partition`` call was handed and
built (captured by worker.py). It reads phase objects, the GkSource's values and, under ``s``,
the re-run producers' ``deltaTheta``; it runs no quadrature.

Prompt 06 section 6: "for every factor of every Levin sub-interval: 04's ``window_rms`` of phi
over windows of 9 consecutive samples, centred inside the sub-interval, under t5 and under s.
Record F = eps k eta_R for each row, and each window's |theta|."

- **The factors.** For each sub-interval with ``method == "Levin"``: G if ``regime[0]``, T_q if
  ``regime[1]``, T_r if ``regime[2]``. G's phase is ``GkPolicy.functions.phase`` and T's
  ``TkSourceFunctions.phase``, the ``PrimitivePhase`` objects the unit's Levin calls used. Its
  samples are ``phase.z_samples`` (ascending) and ``phase.phi_samples``, in ``u = log1p(z)``.
- **The windows.** 04's ``window_rms`` (docs/levin-phase-floor/04/scripts/check_phases.py:106),
  copied unchanged: the RMS residual about a degree-4 fit in ``t`` in [-1, 1] over each window
  of 9 consecutive samples; a window's centre is its 5th sample. A window is the sub-interval's
  when its centre lies in the sub-interval's ``x_span = (log(1 + z_min), log(1 + z_max))``, as
  04 formed it.
- **Per window:** the start index, the 9 z and phi as ``float.hex``, the RMS, ``|theta|`` at the
  centre (``|raw_theta(z_c)|``) and the largest over the 9, ``theta_deriv`` (log derivative) at
  the centre (descriptive: "theta_deriv's change at the Levin windows").
- **G's windows** also carry each of the 9 samples' ``z_init`` from the GkSource value it was
  read from (``v.WKB.z_init``: None for a pure-WKB object), and the tag **pure-WKB** if all 9 are
  None, **numerically seeded** otherwise. Under ``s`` they carry each sample's ``deltaTheta`` from
  the re-run producer of its source redshift (rerun.py), and ``window_rms`` of it on the same fit:
  ``dtheta_rms_unwrapped`` after ``numpy.unwrap`` (the wraps of deltaTheta between neighbours are
  whole cycles, which the phi unwrap removes too), ``dtheta_rms_raw`` without.
- **T's windows** carry the re-run ``deltaTheta`` of the T object under ``s`` (a constant).

``EPS`` is 2**-52. F is formed in the analysis (eps k eta(z_response), 04's F).
"""

import math

import numpy as np

PHI_WINDOW = 9
PHI_DEG = 4


def window_rms(u, y):
    """RMS residual about a degree-4 fit over each window of 9 consecutive samples, in u.
    (docs/levin-phase-floor/04/scripts/check_phases.py:106, unchanged.)"""
    out = []
    centres = []
    for i in range(0, len(u) - PHI_WINDOW + 1):
        uu = u[i : i + PHI_WINDOW]
        yy = y[i : i + PHI_WINDOW]
        t = (2.0 * uu - (uu[0] + uu[-1])) / (uu[-1] - uu[0])
        coef = np.polynomial.polynomial.polyfit(t, yy, PHI_DEG)
        res = yy - np.polynomial.polynomial.polyval(t, coef)
        out.append(float(np.sqrt(np.mean(res * res))))
        centres.append(float(uu[PHI_WINDOW // 2]))
    return np.array(out), np.array(centres)


def _hex(v):
    return [None if x is None else float(x).hex() for x in v]


def probe_row(partition, GkPolicy, Tq_f, Tr_f, dtheta_G=None, dtheta_T=None):
    """Every window of a row. Returns (windows, samples): windows is JSON-able; samples holds
    each factor's full z and phi arrays (for the pickle). ``dtheta_G`` (under ``s``) maps a
    z_source serial to rerun.py's record of its re-run GkWKBIntegration (``deltaTheta``,
    ``pure_WKB``, ...); ``dtheta_T`` maps "Tq"/"Tr" to that T object's re-run ``deltaTheta``."""
    subs = [
        (i, item)
        for i, item in enumerate(partition["subintervals"])
        if item["method"] == "Levin"
    ]
    factors = {}
    need = {"G": False, "Tq": False, "Tr": False}
    for _i, item in subs:
        reg = tuple(bool(x) for x in item["regime"])
        need["G"] |= reg[0]
        need["Tq"] |= reg[1]
        need["Tr"] |= reg[2]

    G_vals = None
    if need["G"]:
        phase = GkPolicy.functions.phase
        source = GkPolicy._source_proxy.get()
        G_vals = {v.z_source.z: v for v in source.values if v.has_WKB}
        factors["G"] = phase
    if need["Tq"]:
        factors["Tq"] = Tq_f.phase
    if need["Tr"]:
        factors["Tr"] = Tr_f.phase

    T_dtheta = dtheta_T or {}

    samples = {}
    rms_by_factor = {}
    for name, phase in factors.items():
        zs = np.asarray(phase.z_samples, dtype=float)
        phi = np.asarray(phase.phi_samples, dtype=float)
        if not np.all(np.diff(zs) > 0):
            raise RuntimeError(f"probe: {name}'s z samples are not ascending")
        u = np.log1p(zs)
        w, centres = window_rms(u, phi)
        rms_by_factor[name] = (u, w, centres)
        samples[name] = {
            "z": zs,
            "phi": phi,
            "k": float(phase.k),
            "sign": int(phase.sign),
            "z_anchor": float(phase.z_anchor),
        }

    windows = []
    for i, item in subs:
        reg = tuple(bool(x) for x in item["regime"])
        x_span = (math.log(1.0 + item["z_min"]), math.log(1.0 + item["z_max"]))
        for name, flag in (("G", reg[0]), ("Tq", reg[1]), ("Tr", reg[2])):
            if not flag:
                continue
            phase = factors[name]
            u, w, centres = rms_by_factor[name]
            zs = samples[name]["z"]
            phi = samples[name]["phi"]
            for i0 in range(len(w)):
                c = centres[i0]
                if not (x_span[0] <= c <= x_span[1]):
                    continue
                zz = zs[i0 : i0 + PHI_WINDOW]
                pp = phi[i0 : i0 + PHI_WINDOW]
                zc = float(zz[PHI_WINDOW // 2])
                thetas = [abs(float(phase.raw_theta(float(z)))) for z in zz]
                win = {
                    "sub": i,
                    "regime": list(reg),
                    "x_span": [x_span[0].hex(), x_span[1].hex()],
                    "factor": name,
                    "i0": int(i0),
                    "z_centre": zc,
                    "u_centre": float(c),
                    "z": _hex(zz),
                    "phi": _hex(pp),
                    "rms": float(w[i0]),
                    "abs_theta_centre": abs(float(phase.raw_theta(zc))),
                    "abs_theta_max": max(thetas),
                    "theta_deriv_u_centre": float(phase.theta_deriv(zc, log_derivative=True)),
                    "k": samples[name]["k"],
                }
                if name == "G":
                    vals = [G_vals[float(z)] for z in zz]
                    z_inits = [v.WKB.z_init for v in vals]
                    win["z_source_serials"] = [int(v.z_source.store_id) for v in vals]
                    win["z_init"] = [None if x is None else float(x) for x in z_inits]
                    win["pure_WKB"] = all(x is None for x in z_inits)
                    if dtheta_G is not None:
                        dts = [dtheta_G[int(v.z_source.store_id)]["deltaTheta"] for v in vals]
                        uu = np.log1p(np.asarray(zz))
                        dt = np.asarray(dts, dtype=float)
                        r_un, _ = window_rms(uu, np.unwrap(dt))
                        r_raw, _ = window_rms(uu, dt)
                        win["deltaTheta"] = _hex(dts)
                        win["dtheta_rms_unwrapped"] = float(r_un[0])
                        win["dtheta_rms_raw"] = float(r_raw[0])
                        win["rerun_pure_WKB"] = [
                            bool(dtheta_G[int(v.z_source.store_id)]["pure_WKB"]) for v in vals
                        ]
                else:
                    win["deltaTheta_T"] = (
                        None if T_dtheta.get(name) is None else float(T_dtheta[name]).hex()
                    )
                windows.append(win)
    return windows, samples
