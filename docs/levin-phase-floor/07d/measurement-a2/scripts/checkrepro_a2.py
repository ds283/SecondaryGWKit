"""levin-phase-floor prompt 07d Part A' (section 9.2 item 1): the reproduction of the tree's ``_check_leaf`` on one Levin
leaf, under a gate at T, outside the driver. New in Part A'.

``CallContext(LQ, rec, kw)`` sets up what ``_adaptive_levin`` sets up once per call, by the tree's own code: the basis
``_Basis_SinCos(theta)``, the vectorize cache, the phases' rates (``rate_fns``), ``guard_rates``, the location
(``_locate_stationary_points`` with the tree's constants), ``guard_sign_change_phases`` and the windows' edges.
``solve(span, T)`` is the driver's ``_solve``: ``_adaptive_levin_subregion`` with the call's arguments, the module's
``SIX_PI`` set to T for the solve only (the gate reads it at call time) and restored.

``check(a, b, T, fallback)`` is ``_check_leaf`` on a Levin leaf [a, b] that is not forced direct, line for line:
V_P by the leaf's own solve; its halves at c = (b + a)/2 and its quarters at (c + a)/2, (b + c)/2, each solved through
the gate at T; residual = |V_P - (V_L + V_R)| (its region's abserr, ``_make_leaf``); delta1, delta2, r as the tree forms
them; nu_B = ``_trust_noise_yardstick(phase_err, False, dataL, dataR)``; trusted = r < ``_LEVIN_TRUST_MAX_RATIO`` or
(r < 1 and delta1 <= nu_B); then, on a trusted leaf with rates, ``_leaf_rate_rule`` with the ends that are window edges;
the declaration max(residual, delta1)/(1 - r) and ``total_err`` = max(declaration, phase_err) + truncation; Q, the
quarters' sum; and the depth-cap form beside it. A leaf whose own solve takes the Clenshaw-Curtis branch, or is guarded,
is not a checked Levin leaf and is reported as such.

``fallback`` = "cc47": a Clenshaw-Curtis sub-solve takes the tree's value (CC_{2N-1}); "cc93": it takes CC_{4N-3},
formed as the V93 variant forms it: ``np.dot(_cc_weights(span, 4N - 3), g)`` on ``chebyshev_matrices(span, 4N - 3)``'s
grid, g = sum_i f_i b_i with f by ``_sample_vectorized`` and b by ``eval_basis_and_arguments`` (the tree's expressions).
On a Levin parent nu_B reads no Clenshaw-Curtis half, so the fallback moves V_L + V_R and Q only.
"""

import math

import numpy as np


def H(x):
    return None if x is None else float(x).hex()


class CallContext:
    def __init__(self, LQ, rec, kw):
        self.LQ = LQ
        self.f = rec["f"]
        self.x_span = rec["x_span"]
        self.A = LQ._Basis_SinCos(rec["theta"])
        self.cache = {}
        self.N = int(kw.get("chebyshev_order", LQ.DEFAULT_LEVIN_CHEBSHEV_ORDER))
        self.rtol = float(kw["rtol"])
        self.notify_label = rec.get("notify_label")
        self.id_label = LQ._LazyUUID()
        A = self.A
        phases_method = getattr(A, "independent_phases", None)
        rates_method = getattr(A, "independent_phase_rates", None)
        phase_fns = phases_method() if callable(phases_method) else None
        rate_fns = rates_method() if callable(rates_method) else None
        if phase_fns is None or rate_fns is None or len(rate_fns) != len(phase_fns):
            rate_fns = None
        self.rate_fns = rate_fns
        self.guard_rates = rate_fns if rate_fns is not None and len(rate_fns) > 1 else None
        location = None
        if rate_fns is not None and self.x_span[1] != self.x_span[0]:
            location = LQ._locate_stationary_points(
                self.x_span, phase_fns, rate_fns, self.cache, window_half_phase=LQ._LEVIN_WINDOW_HALF_PHASE,
                coalescence_lambda=LQ._LEVIN_COALESCENCE_LAMBDA, piece_phase=LQ._LEVIN_WINDOW_PIECE_PHASE)
        self.guard_sign_change_phases = None
        if self.guard_rates is not None and location is not None:
            self.guard_sign_change_phases = frozenset(s["phase"] for s in location["skipped_phases"])
        self.window_edges = frozenset(float(e) for w in (location["windows"] if location is not None else [])
                                      for e in (w["lo"], w["hi"]))
        self.n_solves = 0
        self.n_cc93 = 0

    def solve(self, span, T, force_direct=False):
        LQ = self.LQ
        six = LQ.SIX_PI
        try:
            LQ.SIX_PI = T
            d = LQ._adaptive_levin_subregion(
                span, self.f, self.A, id_label=self.id_label, chebyshev_order=self.N, rtol=self.rtol,
                notify_label=self.notify_label, build_p_sample=False, diagnostics_path=LQ.DEFAULT_LEVIN_DIAGNOSTICS_PATH,
                vectorize_cache=self.cache, force_direct=force_direct, guard_rates=self.guard_rates,
                guard_sign_change_phases=self.guard_sign_change_phases)
        finally:
            LQ.SIX_PI = six
        self.n_solves += 1
        return d

    def cc_value(self, span, order):
        """CC at ``order`` points as the tree forms CC_{2N-1} (np.dot of its weights with g on its grid)."""
        LQ = self.LQ
        grid, _ = LQ.chebyshev_matrices(span, order)
        f_fine = np.vstack([LQ._sample_vectorized(fn, grid, self.cache, ("f", id(fn))) for fn in self.f])
        basis = np.array([v for v, _ in (self.A.eval_basis_and_arguments(x) for x in grid)]).T
        g = np.sum(f_fine * basis, axis=0)
        return float(np.dot(LQ._cc_weights(span, order), g)), grid, g

    def cc93(self, span):
        self.n_cc93 += 1
        return self.cc_value(span, 4 * self.N - 3)[0]

    def check(self, a, b, T, fallback="cc47"):
        LQ = self.LQ
        P = self.solve((a, b), T)
        out = {"a": H(a), "b": H(b), "T": T, "fallback": fallback, "own_route": "cc" if P.get("is_direct") else "levin",
               "own_guarded": bool(P.get("guarded", False)), "phase_span": P.get("phase_span")}
        if P.get("is_direct") or P.get("guarded", False):
            out["checked"] = False
            return out
        c = (b + a) / 2.0
        cL = (c + a) / 2.0
        cR = (b + c) / 2.0
        subs = [self.solve(s, T) for s in ((a, c), (c, b), (a, cL), (cL, c), (c, cR), (cR, b))]
        vals = []
        for d, s in zip(subs, ((a, c), (c, b), (a, cL), (cL, c), (c, cR), (cR, b))):
            if d.get("is_direct") and fallback == "cc93":
                vals.append(self.cc93(s))
            else:
                vals.append(d["value"])
        dataL, dataR = subs[0], subs[1]
        own_value = P["value"]
        refined = vals[0] + vals[1]
        residual = np.fabs(own_value - refined)
        phase_err = P.get("phase_err", 0.0) or 0.0
        trunc = P.get("abserr_truncation")
        halves_sum = vals[0] + vals[1]
        quarters_sum = (vals[2] + vals[3]) + (vals[4] + vals[5])
        delta1 = float(np.fabs(own_value - halves_sum))
        delta2 = float(np.fabs(halves_sum - quarters_sum))
        if not (np.isfinite(delta1) and np.isfinite(delta2)):
            ratio = float("inf")
        elif delta1 == 0.0:
            ratio = 0.0 if delta2 == 0.0 else float("inf")
        else:
            ratio = delta2 / delta1
        nu = LQ._trust_noise_yardstick(phase_err, False, dataL, dataR)
        trusted = ratio < LQ._LEVIN_TRUST_MAX_RATIO or (ratio < 1.0 and delta1 <= nu)
        refused, rule = False, None
        if trusted and self.rate_fns is not None:
            lo, hi = min(a, b), max(a, b)
            refused, rule, _ = LQ._leaf_rate_rule((lo, hi), self.N, self.rate_fns,
                                                   tuple(e for e in (lo, hi) if e in self.window_edges))
            if refused:
                trusted = False
        else:
            rule = None
        decl = max(residual, delta1) / (1.0 - ratio) if ratio < 1.0 else None
        cap = (max(residual, delta1) / (1.0 - ratio)) if ratio < 1.0 else max(residual, delta1 + delta2)

        def combine(res):
            base = max(res, phase_err)
            return base if trunc is None else base + trunc

        out.update({
            "checked": True, "V_P": H(own_value), "halves_sum": H(halves_sum), "Q": H(quarters_sum),
            "sub_routes": ["cc" if d.get("is_direct") else "levin" for d in subs],
            "sub_guarded": [bool(d.get("guarded", False)) for d in subs],
            "residual": H(residual), "phase_err": H(phase_err), "trunc": H(trunc), "delta1": H(delta1), "delta2": H(delta2),
            "r": ratio, "nu_B": H(nu), "trusted": bool(trusted), "refused_by_rate_rule": bool(refused),
            # v2: "rule" as the tree records it (the rule that refused, else None); the rule applied kept beside
            "rule": rule if refused else None, "rule_applied": rule,
            "own_decl": H(combine(residual)), "declared_resolution": H(decl) if trusted else None,
            "total_err_trusted": H(combine(decl)) if trusted else None, "total_err_depth_cap": H(combine(cap)),
            "value_returned": H(quarters_sum if trusted else own_value),
        })
        return out
