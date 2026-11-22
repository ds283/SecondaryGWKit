"""levin-phase-floor prompt 06g section 5, the references: do 06i's 13 captured groups at g equal those
06i read? New in 06g.

The brief: 06i's 13 keep their references "if the captured groups at g equal those 06i read:
coefficients, orders, signs and spans, bit for bit". For each of samples 1, 2, 3, 8, 9, 10, 13, 14,
15, 22, 23, 25 and 27, 06i's input (var/levin-phase-floor-06e/census/references/inputs/sNN.pkl,
read in place; captured by 06c's depth20-tiling at 06c's census tree, and read by 06i at 06e's tree)
and the capture at g (state/inputs/zNN.pkl) are both loaded here, at g, with ray.cloudpickle (the
census harness on PYTHONPATH, for ``harness._rebuild_lru_cache``), and compared:

**The criterion as written** (every item bit for bit; floats by ``float.hex``):
- the route (``tools.filon_reference.route_of``) and the span ``x_span``;
- Route A (a ``BesselPhaseGroup``, ``theta["theta"].__self__``): ``signs``, ``coefficients`` and the
  orders ``[p.nu for p in phases]``;
- Route B (a cosmological group; the active factors are the closure the tool walks,
  ``theta["theta"].__closure__[0]``): the signs of the active factors and, as their coefficients,
  each factor's phase spline (knots ``t``, coefficients ``c``, degree ``k``) and its leading
  table's nodes and breaks, by bytes.

**The integrand, as the tool reads it** (also bit for bit; a difference here is reported as a stop
too, since the reference would then not be for g's integrand):
- both: ``f_sin`` and ``f_cos`` at 65 points uniform in the span (``numpy.linspace``, endpoints
  included);
- Route A: the tool's own ingredients, ``RouteA(call).K_exact`` (a Fraction), ``C_mp()`` at 40 digits,
  ``u_breaks()``, and the group's ``residual`` at ``eta = exp(u)`` at the 65 points;
- Route B: ``theta``, ``theta_mod_2pi`` and ``theta_deriv`` at the 65 points, and
  ``production_edges(call)`` (the edges and the number of tables).

Descriptive only: the call's keyword arguments and the phase dict's keys (they steer the driver,
not the integral); and, Route A, ``theta`` and ``theta_mod_2pi`` at the 65 points. 06h changed the Bessel
group's ``sin_cos`` (the small angle carries the product and K roundings exactly), so these may
differ, and Route A uses neither (it uses K exactly, C and the residual).

Writes out/compare13.json; exits 1 if any sample differs in the criterion or the integrand. Reads
only.

Usage: PYTHONPATH=<tree g>:<this directory> python -B compare13.py INPUTS_JSON
"""

import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402

N_POINTS = 65


def load(path):
    import ray.cloudpickle as cp

    return cp.loads(Path(path).read_bytes())


def hx(v):
    return float(v).hex()


def arr_bytes(a):
    a = np.asarray(a)
    return (str(a.dtype), list(a.shape), a.tobytes().hex())


def points(call):
    a, b = (float(v) for v in call["x_span"])
    return [float(x) for x in np.linspace(a, b, N_POINTS)]


def kw_repr(kw):
    out = {}
    for k in sorted(kw):
        v = kw[k]
        out[k] = hx(v) if isinstance(v, float) else repr(v)
    return out


def describe(call, F):
    route = F.route_of(call)
    d = {"route": route, "x_span": [hx(v) for v in call["x_span"]]}
    crit = {"route": route, "x_span": d["x_span"]}
    integ = {}
    desc = {"kw": kw_repr(call.get("kw") or {}), "theta_keys": sorted(call["theta"])}
    us = points(call)
    f_sin, f_cos = call["f"]
    integ["f_sin"] = [hx(f_sin(u)) for u in us]
    integ["f_cos"] = [hx(f_cos(u)) for u in us]
    th = call["theta"]
    if route == "A":
        g = th["theta"].__self__
        crit["signs"] = [repr(s) for s in g.signs]
        crit["coefficients"] = [hx(c) for c in g.coefficients]
        crit["orders"] = [repr(Fraction(p.nu)) for p in g.phases]
        A = F.RouteA(call)
        integ["K_exact"] = str(A.K_exact)
        import mpmath as mp

        with mp.workdps(F.DPS):
            integ["C_40"] = mp.nstr(A.C_mp(), 40)
        integ["u_breaks"] = [hx(v) for v in A.u_breaks()]
        integ["residual"] = [hx(g.residual(float(np.exp(u)))) for u in us]
        # descriptive: 06i's input, loaded at g, holds a group pickled before 06h (no K_lo), on
        # which g's sin_cos may raise; recorded, not compared
        for name in ("theta", "theta_mod_2pi"):
            try:
                desc[name] = [hx(th[name](u)) for u in us]
            except Exception as exc:
                desc[name] = f"raised {type(exc).__name__}: {exc}"
        desc["K"] = hx(g.K) if hasattr(g, "K") else None
        desc["has_K_lo"] = hasattr(g, "K_lo")
    else:
        active = th["theta"].__closure__[0].cell_contents
        crit["signs"] = [repr(s) for s, _ in active]
        facs = []
        for _, fac in active:
            ph = fac.phase
            sp = ph._spline
            tab = ph._leading._table
            facs.append(
                {
                    "spline_t": arr_bytes(sp.t),
                    "spline_c": arr_bytes(sp.c) if hasattr(sp, "c") else None,
                    "spline_k": int(sp.k) if hasattr(sp, "k") else None,
                    "table_u_asc": arr_bytes(tab._u_asc),
                    "table_breaks": (
                        None if tab._breaks is None else arr_bytes(tab._breaks)
                    ),
                }
            )
        crit["factors"] = facs
        for name in ("theta", "theta_mod_2pi", "theta_deriv"):
            integ[name] = [hx(th[name](u)) for u in us]
        edges, n_tables = F.production_edges(call)
        integ["production_edges"] = [hx(v) for v in edges]
        integ["n_tables"] = n_tables
    return crit, integ, desc


def first_diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}/{k}: present in one only"
            r = first_diff(a[k], b[k], f"{path}/{k}")
            if r:
                return r
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: lengths {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            r = first_diff(x, y, f"{path}[{i}]")
            if r:
                return r
        return None
    return None if a == b else f"{path}: {str(a)[:80]} vs {str(b)[:80]}"


def main():
    inputs = json.loads(Path(sys.argv[1]).read_text())
    import tools.filon_reference as F

    ins = [Path(F.__file__)]
    ins += [RC.I06_INPUTS / f"s{n:02d}.pkl" for n in RC.I06_SAMPLES]
    ins += [Path(inputs["captures"][f"z{n:02d}"]["pickle"]) for n in RC.I06_SAMPLES]
    for l in header(
        Path(__file__).resolve(),
        "levin-phase-floor 06g: 06i's 13 groups, 06i's inputs against the captures at g",
        ins,
    ):
        print(l)
    out = []
    n_bad = 0
    for n in RC.I06_SAMPLES:
        p06i = RC.I06_INPUTS / f"s{n:02d}.pkl"
        pg = Path(inputs["captures"][f"z{n:02d}"]["pickle"])
        crit_g, integ_g, desc_g = describe(load(pg), F)
        try:
            crit_i, integ_i, desc_i = describe(load(p06i), F)
        except Exception:
            import traceback

            err = traceback.format_exc()
            print(err)
            crit_i = integ_i = desc_i = {
                "error loading or reading 06i's input at g": err[-2000:]
            }
        dc = first_diff(crit_i, crit_g)
        di = first_diff(integ_i, integ_g)
        dd = first_diff(desc_i, desc_g)
        row = {
            "sample": n,
            "route": crit_g["route"],
            "criterion_equal": dc is None,
            "criterion_first_difference": dc,
            "integrand_equal": di is None,
            "integrand_first_difference": di,
            "descriptive_equal": dd is None,
            "descriptive_first_difference": dd,
            "compared": {
                "criterion": sorted(crit_g),
                "integrand": sorted(integ_g),
                "descriptive": sorted(desc_g),
            },
            "06i": {"criterion": crit_i, "integrand": integ_i, "descriptive": desc_i},
            "g": {"criterion": crit_g, "integrand": integ_g, "descriptive": desc_g},
        }
        bad = dc is not None or di is not None
        n_bad += bad
        out.append(row)
        print(
            f"s{n:02d} route {crit_g['route']}: criterion {'EQUAL' if dc is None else 'DIFFERS ' + dc}; "
            f"integrand {'EQUAL' if di is None else 'DIFFERS ' + di}; descriptive {'equal' if dd is None else 'differs ' + dd}"
        )
    dst = RC.ROOT / "out" / "compare13.json"
    if dst.exists():
        raise SystemExit(f"{dst} exists; not overwritten")
    dst.write_text(json.dumps(out, indent=1))
    print(
        f"{13 - n_bad} of 13 groups equal 06i's (criterion and integrand); wrote {dst} sha256 {RC.sha256(dst)}"
    )
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
