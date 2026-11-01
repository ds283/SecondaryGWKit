"""levin-phase-floor prompt 06i Sec 2: Route A, one reading of one analytic-block sample. New in 06i.

    I = Re int_{e^{u_a}}^{e^{u_b}} g(eta) exp(i K eta) d eta,  g = F(log eta) exp(i (C + R(eta))) / eta

(samples.py's module docstring). K exact, C exact, limits exact.

--impl fcc: panels uniform in u between the constituents' representation switches (near-region
    panel edges and x_star), at most du wide; FCC with N + 1 points per panel.
      res 1: du = 0.25, N = 32;   res 2: du = 0.17, N = 48.
--impl qawo: the same switches, segments at most du wide, each tiled with dyadic panels whose
    phase span 2 |K| h is at most P_max and whose width is at most half the segment's;
      res 1: du = 0.25, P_max = 2^12, epsrel = 1e-13;   res 2: du = 0.125, P_max = 2^9, epsrel = 1.2e-14.
Work is split over at most 6 processes. Writes ../out/readings/sNN-<impl>-r<res>.json.
Usage (census tree as cwd, see run.sh): python -B route_a.py N --impl fcc|qawo --res 1|2 [--limit-panels K]
"""

import argparse
import json
import math
import multiprocessing as mpp
import sys
import time
from fractions import Fraction
from pathlib import Path

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import filon  # noqa: E402
import prov  # noqa: E402
import samples as S  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out" / "readings"
FCC = {1: dict(du=0.25, N=32), 2: dict(du=0.17, N=48)}
QAWO = {
    1: dict(du=0.25, Pmax=2.0**12, epsrel=1e-13),
    2: dict(du=0.125, Pmax=2.0**9, epsrel=1.2e-14),
}

_W = {}


def _init(n):
    filon.set_precision()
    rec = S.load(n)
    A = S.RouteA(rec)
    _W["A"] = A
    _W["gmp"] = A.g_mp()
    C = A.C_mp()
    _W["gd"] = A.g_double(float(S.rem2pi(C)))
    _W["K"] = A.K_mp()


def _fcc_job(args):
    panels, N = args
    total = mp.mpc(0)
    for l, r in panels:
        l, r = mp.mpf(l), mp.mpf(r)
        total += filon.fcc_panel(_W["gmp"], (l + r) / 2, (r - l) / 2, _W["K"], N)
    return mp.nstr(total.real, 45), mp.nstr(total.imag, 45)


def _qawo_job(args):
    panels, epsrel = args
    A = _W["A"]
    gd = _W["gd"]
    K = A.K_exact
    Kabs = abs(float(K))

    def epsabs(c, h):
        return 1e-17 * abs(gd(float(c))) * min(2 * h, 2 / Kabs)

    pan = [(mp.mpf(c), h) for c, h in panels]
    total, st = filon.qawo_integrate(gd, pan, K, epsabs, epsrel)
    return mp.nstr(total.real, 45), mp.nstr(total.imag, 45), st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--impl", choices=("fcc", "qawo"), required=True)
    ap.add_argument("--res", type=int, choices=(1, 2), required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--limit-panels", type=int, default=0, help="testing only: first K panels"
    )
    a = ap.parse_args()
    assert a.workers <= 6 and a.n in S.ROUTE_A
    filon.set_precision()
    t0 = time.time()
    rec = S.load(a.n)
    A = S.RouteA(rec)
    K = A.K_mp()
    if a.impl == "fcc":
        cfg = FCC[a.res]
        ue = A.u_segments(cfg["du"])
        ee = [mp.exp(u) for u in ue]
        panels = [(mp.nstr(l, 45), mp.nstr(r, 45)) for l, r in zip(ee[:-1], ee[1:])]
        if a.limit_panels:
            panels = panels[: a.limit_panels]
        chunks = [panels[i :: a.workers * 2] for i in range(a.workers * 2)]
        jobs = [(c, cfg["N"]) for c in chunks if c]
    else:
        cfg = QAWO[a.res]
        ue = A.u_segments(cfg["du"])
        ee = [mp.exp(u) for u in ue]
        Kabs = abs(float(A.K_exact))
        panels = []
        for l, r in zip(ee[:-1], ee[1:]):
            h_max = min(cfg["Pmax"] / (2 * Kabs), float(r - l) / 2)
            panels += [(mp.nstr(c, 45), h) for c, h in filon.dyadic_tiling(l, r, h_max)]
        if a.limit_panels:
            panels = panels[: a.limit_panels]
        nch = a.workers * 8
        chunks = [panels[i::nch] for i in range(nch)]
        jobs = [(c, cfg["epsrel"]) for c in chunks if c]
    t_setup = time.time() - t0
    with mpp.get_context("spawn").Pool(
        a.workers, initializer=_init, initargs=(a.n,)
    ) as pool:
        res = pool.map(_fcc_job if a.impl == "fcc" else _qawo_job, jobs, chunksize=1)
    re_ = mp.fsum(mp.mpf(r[0]) for r in res)
    im_ = mp.fsum(mp.mpf(r[1]) for r in res)
    stats = {}
    if a.impl == "qawo":
        for key in ("panels", "neval", "ier_nonzero", "sum_quad_abserr"):
            stats[key] = sum(r[2][key] for r in res)
        stats["max_last"] = max(r[2]["max_last"] for r in res)
    wall = time.time() - t0
    z = S.z3_row(a.n)
    out = {
        "header": prov.header([S.pickle_path(a.n), S.Z3]),
        "sample": a.n,
        "route": "A",
        "impl": a.impl,
        "res": a.res,
        "config": cfg,
        "workers": a.workers,
        "limit_panels": a.limit_panels,
        "n_panels": len(panels),
        "u_breaks": A.u_breaks(),
        "K_exact": str(A.K_exact),
        "K_hat": A.group.K,
        "K_lo": float(A.K_lo),
        "K_lo_fsum": A.K_lo_fsum,
        "C_exact": mp.nstr(A.C_mp(), 30),
        "value_hex": float(re_).hex(),
        "value_30": mp.nstr(re_, 30),
        "imag_30": mp.nstr(im_, 30),
        "wall_s": wall,
        "setup_s": t_setup,
        "stats": stats,
        "z3_unit": z["unit"],
        "z3_index": z["index"],
    }
    tag = f"s{a.n:02d}-{a.impl}-r{a.res}" + (
        f"-TEST{a.limit_panels}" if a.limit_panels else ""
    )
    if not a.limit_panels:  # a test of the mechanics is not a reading, and is not kept
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{tag}.json").write_text(json.dumps(out, indent=1))
    print(
        tag,
        out["value_30"],
        out["value_hex"],
        f"wall {wall:.1f} s",
        out["n_panels"],
        "panels",
        stats,
    )


if __name__ == "__main__":
    main()
