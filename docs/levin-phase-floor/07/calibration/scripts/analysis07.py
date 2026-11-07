"""levin-phase-floor 07 section 2: what every section 2 script records of one Levin call, and section 2.5's
account of a call that does not finish. New in 07; shared by calibrate.py, replay.py and the LiouvilleGreen
harness's call records.

``Recorder(module)`` subclasses the driver module's ``used_interval`` so that every region the driver makes is
snapshotted when it is made (those later bisected included): its span, depth, type, window and guard flags, its
own value and its error e' as made (``budget_err`` before any trust check widens it), its residual and its floor.
The driver makes one used_interval per leaf, in the order the leaves are made: the seed leaves first, then each
bisection's left and right child. ``install()`` puts it on the module; ``remove()`` restores the original.

``summary(result, ref=None)`` is the call's record: value, abserr (float hex too), the true error and
|v - ref| / abserr where a reference is given, converged, node_limited, regions, window pieces, maximum depth,
guard counts, and the located points with their windows.

``section_2_5(result, made, request)`` is section 2.5's account, for a call that does not converge or reaches
depth_max:
- the returned leaves by class (window, guarded, ordinary Levin, ordinary Clenshaw-Curtis): count, summed e',
  at depth_max, untrusted;
- the leaves holding the unmet error: the returned leaves by e', largest first, until they hold 90 % of sum e'
  (at most 20), and every class's leaves at depth_max summarised;
- for each of those leaves its ancestry, depth by depth, from the snapshots: e' as made, the residual, the
  floor, and whether e' fell from the parent (by at least half: "falls with h");
- QUADPACK's counters (dqage): over the whole call, in bisection order, iroff1 counts a bisection whose
  children's values change the parent's by at most 1e-5 relatively while their summed error is at least 0.99 of
  the parent's, and iroff2 counts one whose children's summed error exceeds the parent's once the partition has
  more than 10 leaves; ier = 2 at iroff1 >= 6 or iroff2 >= 20, and the bisection at which it would have been
  reached. Along each listed leaf's ancestry, the same two counts. The errors compared are e' as made; QUADPACK
  stops the whole call and marks no leaf.
"""

import math

import numpy as np

SNAPSHOT_FIELDS = (
    "start",
    "end",
    "depth",
    "type",
    "value_own",
    "budget_err",
    "abserr",
    "phase_err",
    "window",
    "guarded",
    "guard_at_depth_max",
)


class Recorder:
    def __init__(self, module):
        self.module = module
        self.original = module.used_interval
        self.made = []
        recorder = self

        class RecordedInterval(self.original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                recorder.made.append(
                    {
                        "start": float(self.start),
                        "end": float(self.end),
                        "depth": int(self.depth),
                        "type": self.type,
                        "value_own": _f(self.value_own),
                        "budget_err": _f(self.budget_err),
                        "abserr": _f(self.abserr),
                        "phase_err": _f(self.phase_err),
                        "window": bool(getattr(self, "window", False)),
                        "guarded": bool(getattr(self, "guarded", False)),
                        "guard_at_depth_max": bool(
                            getattr(self, "guard_at_depth_max", False)
                        ),
                    }
                )

        self.cls = RecordedInterval

    def install(self):
        self.made.clear()
        self.module.used_interval = self.cls

    def remove(self):
        self.module.used_interval = self.original


def _f(x):
    return None if x is None else float(x)


def _hex(x):
    return None if x is None else float(x).hex()


def leaf_class(region):
    if getattr(region, "window", False):
        return "window"
    if getattr(region, "guarded", False):
        return "guarded"
    if region.type == "Levin":
        return "Levin"
    return "Clenshaw-Curtis"


def summary(result, ref=None, depth_max=20):
    rec = {
        "value": result["value"],
        "value_hex": _hex(result["value"]),
        "abserr": result["abserr"],
        "abserr_hex": _hex(result["abserr"]),
        "converged": bool(result["converged"]),
        "node_limited": bool(result.get("node_limited", False)),
        "regions": result["num_regions"],
        "simple_regions": result["num_simple_regions"],
        "max_depth": result["max_depth"],
        "at_depth_max": sum(1 for r in result["regions"] if r.depth >= depth_max),
        "num_phase_limited_regions": result["num_phase_limited_regions"],
        "num_untrusted_regions": result.get("num_untrusted_regions"),
        "evaluations": result["evaluations"],
        "elapsed": result["elapsed"],
    }
    if ref is not None:
        err = abs(result["value"] - ref)
        rec["ref"] = ref
        rec["true_error"] = err
        rec["error_over_abserr"] = (
            err / result["abserr"] if result["abserr"] > 0 else float("inf")
        )
        rec["honest"] = bool(err <= result["abserr"])
    if "stationary_points" in result:
        rec["window_pieces"] = result["num_window_pieces"]
        rec["guarded_regions"] = result["num_guarded_regions"]
        rec["guard_at_depth_max"] = result["num_guard_at_depth_max"]
        rec["window_leaves_phase_limited"] = sum(
            1 for r in result["regions"] if r.window and r.phase_limited
        )
        rec["stationary_points"] = [
            {
                "x": p["x"],
                "kind": p["kind"],
                "found_as": p["found_as"],
                "phase": p["phase"],
                "theta_deriv": p["theta_deriv"],
                "lambda": p.get("lambda"),
                "window": p["window"],
                "window_pieces": p["window_pieces"],
                "own_window": p["own_window"],
                "own_window_edges": p["own_window_edges"],
            }
            for p in result["stationary_points"]
        ]
        rec["stationary_windows"] = [
            {
                "lo": w["lo"],
                "hi": w["hi"],
                "pieces": len(w["cuts"]) - 1,
                # section 1.3's rule (after the section 2 stop): the window's iroff1 count, and,
                # if it stopped, what was true of its leaves then
                **{
                    k: w[k]
                    for k in (
                        "iroff1_count",
                        "stopped",
                        "stopped_after_split",
                        "leaves_at_stop",
                        "leaves_falling_at_stop",
                        "max_depth_at_stop",
                    )
                    if k in w
                },
            }
            for w in result["stationary_windows"]
        ]
        rec["windows_stopped"] = sum(
            1 for w in result["stationary_windows"] if w.get("stopped")
        )
        rec["windows_stopped_with_falling_leaves"] = sum(
            1
            for w in result["stationary_windows"]
            if w.get("stopped") and w.get("leaves_falling_at_stop")
        )
        rec["unlocated_minima"] = [
            {"x": m["x"], "lambda": m["lambda"]} for m in result["unlocated_minima"]
        ]
    return rec


def _key(span):
    a, b = span
    return (min(a, b), max(a, b))


def _ancestry(made_by_span, leaf_span, depth):
    """The snapshots of a leaf's ancestors, depth 0 to the leaf, found by span containment."""
    lo, hi = _key(leaf_span)
    chain = []
    for snap in made_by_span.values():
        s_lo, s_hi = _key((snap["start"], snap["end"]))
        if s_lo <= lo and hi <= s_hi and snap["depth"] <= depth:
            chain.append(snap)
    chain.sort(key=lambda s: s["depth"])
    return chain


def _quadpack_global(made, n_seeds):
    """iroff1 / iroff2 over the call, in bisection order, from the snapshots (see the module docstring)."""
    by_span = {}
    for snap in made[:n_seeds]:
        by_span[_key((snap["start"], snap["end"]))] = snap
    iroff1 = iroff2 = 0
    ier2_at = None
    leaves = n_seeds
    bisections = 0
    i = n_seeds
    while i + 1 < len(made):
        left, right = made[i], made[i + 1]
        parent_key = (
            min(
                _key((left["start"], left["end"]))[0],
                _key((right["start"], right["end"]))[0],
            ),
            max(
                _key((left["start"], left["end"]))[1],
                _key((right["start"], right["end"]))[1],
            ),
        )
        parent = by_span.get(parent_key)
        by_span[_key((left["start"], left["end"]))] = left
        by_span[_key((right["start"], right["end"]))] = right
        i += 2
        if parent is None:
            continue
        bisections += 1
        leaves += 1
        area12 = (left["value_own"] or 0.0) + (right["value_own"] or 0.0)
        erro12 = (left["budget_err"] or 0.0) + (right["budget_err"] or 0.0)
        errmax = parent["budget_err"] or 0.0
        if (
            abs((parent["value_own"] or 0.0) - area12) <= 1.0e-5 * abs(area12)
            and erro12 >= 0.99 * errmax
        ):
            iroff1 += 1
        if leaves > 10 and erro12 > errmax:
            iroff2 += 1
        if ier2_at is None and (iroff1 >= 6 or iroff2 >= 20):
            ier2_at = bisections
    return {
        "bisections": bisections,
        "iroff1": iroff1,
        "iroff2": iroff2,
        "ier2": bool(iroff1 >= 6 or iroff2 >= 20),
        "ier2_reached_at_bisection": ier2_at,
    }


def section_2_5(result, made, request, n_seeds, depth_max=20, max_listed=20):
    regions = result["regions"]
    classes = {}
    for r in regions:
        c = leaf_class(r)
        entry = classes.setdefault(
            c,
            {"count": 0, "sum_budget_err": 0.0, "at_depth_max": 0, "untrusted": 0},
        )
        entry["count"] += 1
        entry["sum_budget_err"] += r.budget_err
        entry["at_depth_max"] += int(r.depth >= depth_max)
        entry["untrusted"] += int(r.trusted is False)
    total = sum(r.budget_err for r in regions)
    order = sorted(regions, key=lambda r: -r.budget_err)
    listed = []
    held = 0.0
    for r in order:
        if len(listed) >= max_listed or (total > 0 and held >= 0.9 * total):
            break
        listed.append(r)
        held += r.budget_err

    made_by_span = {}
    for snap in made:
        made_by_span[(_key((snap["start"], snap["end"])), snap["depth"])] = snap
    leaves = []
    for r in listed:
        chain = _ancestry(made_by_span, (r.start, r.end), r.depth)
        steps = []
        iroff1 = iroff2 = 0
        prev = None
        for snap in chain:
            step = {
                "depth": snap["depth"],
                "span": [snap["start"], snap["end"]],
                "type": snap["type"],
                "window": snap["window"],
                "guarded": snap["guarded"],
                "e_prime": snap["budget_err"],
                "residual": snap["abserr"],
                "floor": snap["phase_err"],
            }
            if prev is not None:
                step["falls_by_half_from_parent"] = bool(
                    snap["budget_err"] is not None
                    and prev["budget_err"] is not None
                    and snap["budget_err"] <= 0.5 * prev["budget_err"]
                )
                # the sibling: the other half of prev
                p_lo, p_hi = _key((prev["start"], prev["end"]))
                s_lo, s_hi = _key((snap["start"], snap["end"]))
                other = (p_lo, s_lo) if s_lo > p_lo else (s_hi, p_hi)
                sib = made_by_span.get((other, snap["depth"]))
                if sib is not None:
                    area12 = (snap["value_own"] or 0.0) + (sib["value_own"] or 0.0)
                    erro12 = (snap["budget_err"] or 0.0) + (sib["budget_err"] or 0.0)
                    errmax = prev["budget_err"] or 0.0
                    step["halves_e_prime_sum"] = erro12
                    if (
                        abs((prev["value_own"] or 0.0) - area12) <= 1.0e-5 * abs(area12)
                        and erro12 >= 0.99 * errmax
                    ):
                        iroff1 += 1
                    if erro12 > errmax:
                        iroff2 += 1
            steps.append(step)
            prev = snap
        leaves.append(
            {
                "span": [r.start, r.end],
                "depth": r.depth,
                "class": leaf_class(r),
                "e_prime": r.budget_err,
                "trusted": r.trusted,
                "convergence_ratio": r.convergence_ratio,
                "phase_limited": r.phase_limited,
                "ancestry": steps,
                "ancestry_iroff1": iroff1,
                "ancestry_iroff2_rises": iroff2,
            }
        )
    return {
        "request": request,
        "sum_budget_err": total,
        "abserr": result["abserr"],
        "abserr_node_signed": result.get("abserr_node_signed"),
        "node_limited": bool(result.get("node_limited", False)),
        "classes": classes,
        "listed_hold_fraction": held / total if total > 0 else None,
        "listed_leaves": leaves,
        "quadpack_global": _quadpack_global(made, n_seeds),
    }


def n_seed_leaves(result):
    """The number of leaves the call started from: 1, or the window pieces and the ranges between windows."""
    windows = result.get("stationary_windows") or []
    if not windows:
        return 1
    lo = min(r.start if r.start <= r.end else r.end for r in result["regions"])
    hi = max(r.end if r.start <= r.end else r.start for r in result["regions"])
    n = 0
    cursor = lo
    for w in windows:
        if w["lo"] > cursor:
            n += 1
        n += sum(1 for a, b in zip(w["cuts"][:-1], w["cuts"][1:]) if b > a)
        cursor = max(cursor, w["hi"])
    if hi > cursor:
        n += 1
    return n


def needs_section_2_5(result, depth_max=20):
    return (not result["converged"]) or result["max_depth"] >= depth_max


def request_of(result, atol, rtol):
    return max(atol, rtol * abs(result["value"]))
