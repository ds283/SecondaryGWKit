"""levin-phase-floor 07 section 2.3: the real calls -- z00, z12 and V1's twelve -- replayed from 06g's captures,
each call in its own process. New in 07.

**The captures** are var/levin-phase-floor-06g-fcc-laptop/inputs/{z00,z12,w00..w11,r55}.pkl, copies of the Mac
Studio's made for 06g's laptop FCC batch: each a dict (x_span, f, theta, kw) pickled by Ray's cloudpickle, read
here with ``ray.cloudpickle.loads`` (Ray is imported for its pickler and never initialised), with
06g's harness directory (var/levin-phase-floor-06g-fcc-laptop/harness/, read only) on sys.path after the tree,
since the pickles name its lru_cache reducer. Each pickle's SHA-256
is checked against 06g's record of it: references/out/capture.json for z00, z12 and r55,
references/v1/out/v1_capture.json ("g", R24's capture) for w00-w11.

**The modes** (one process per call and mode):
- ``g06``: 06g's code, a scratch worktree at 3ae6f3a4 given by ``--tree``, first on sys.path; the main
  checkout is not on it;
- ``parent``: the dispatch commit's AdaptiveLevin/levin_quadrature.py (06m's code), read with git show into a
  temporary file and installed by variant_loader.py in place of the tree's module before anything imports
  AdaptiveLevin; everything else is the main checkout's;
- ``commit``: the main checkout's tree (section 1's code) with ``_LEVIN_WINDOW_HALF_PHASE`` set to the rung;
- ``barlifted``: as ``commit``, with the bar on the floor-limited exit for window and guarded leaves removed
  (calibrate.py's variant), run only where the commit's call left a window leaf at depth_max (section 2.5).
In every mode the call is ``adaptive_levin_sincos(x_span, f, theta, **kw)`` of the module in force.

**The replay check** (section 2.3): under ``g06`` each replay reproduces its 06g record bit for bit -- value and
abserr by float.hex, regions, maximum depth and converged: for the w calls V1's R24 in
references/out/scores_stage3/scores.json (value, abserr, converged) with references/v1/out/v1_capture.json's
"g" (num_regions, max_depth); for z00, z12 and r55 references/out/capture.json's census record. Under
``parent``, z00 and z12 are compared with route B's records in docs/levin-phase-floor/06l/out/partC/h1_calls.csv
(ratio |v - ref| / abserr, converged, regions), which 06m's census equals; r55 and the w calls have no parent
record there and are reported. A replay that does not reproduce is a stop.

**The references** are 06g's (scores_stage3/scores.json: ``ref_hex``, FCC at resolution 2): ``V1`` for w00-w11,
``rows`` for z00 and z12; r55 is a control (no stationary point), referenced as rows gives it, if it does.

**Recorded for each call and rung:** value, abserr, |v - ref| / abserr, converged, node_limited, regions,
maximum depth, the located points and their windows; and section 2.5's account for a call that does not
converge or reaches depth_max (analysis07.py).

A call that does not finish within the unit limit is recorded as such, re-run at bounded depth_max (8
and 12) with section 2.5's account, and once with the bar lifted.

Writes out/replay.json and out/replay_tables.md, and appends each unit as it arrives to out/replay_units.jsonl.
"""

import argparse
import concurrent.futures
import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prov  # noqa: E402

INPUTS = prov.REPO / "var" / "levin-phase-floor-06g-fcc-laptop" / "inputs"
HARNESS = prov.REPO / "var" / "levin-phase-floor-06g-fcc-laptop" / "harness"
REFS = prov.REPO / "docs" / "levin-phase-floor" / "06g" / "references"
SCORES = REFS / "out" / "scores_stage3" / "scores.json"
CAPTURE = REFS / "out" / "capture.json"
V1_CAPTURE = REFS / "v1" / "out" / "v1_capture.json"
H1_CALLS = (
    prov.REPO / "docs" / "levin-phase-floor" / "06l" / "out" / "partC" / "h1_calls.csv"
)
SIDS = ("z12", "z00") + tuple(f"w{i:02d}" for i in range(12)) + ("r55",)
PHI_LADDER_PI = (2.0, 8.0, 32.0, 128.0, 512.0)
DEPTH_MAX = 20
BOUNDED_DEPTHS = (8, 12)
G06_SHA = "3ae6f3a4"


def records():
    """{sid: {"pickle_sha256", "g06": {...}, "ref_hex", "parent": {...} or None}} from 06g's and 06l's files."""
    scores = json.loads(SCORES.read_text())
    capture = {e["sid"]: e for e in json.loads(CAPTURE.read_text()) if e.get("sid")}
    v1cap = {e["wid"]: e for e in json.loads(V1_CAPTURE.read_text())}
    v1 = {e["wid"]: e for e in scores["V1"]}
    rows = {e["sid"]: e for e in scores["rows"] if e.get("sid")}
    h1 = {}
    header = None
    for line in H1_CALLS.read_text().splitlines():
        if line.startswith("#"):
            continue
        cols = line.split(",")
        if header is None:
            header = cols
            continue
        rec = dict(zip(header, cols))
        h1[rec["rid"]] = rec
    out = {}
    for sid in SIDS:
        if sid.startswith("w"):
            g = v1cap[sid]["g"]
            out[sid] = {
                "pickle_sha256": g["sha256"],
                "g06": {
                    "value": v1[sid]["R24"]["value"],
                    "abserr": v1[sid]["R24"]["abserr"],
                    "converged": v1[sid]["R24"]["converged"],
                    "num_regions": g["num_regions"],
                    "max_depth": g["max_depth"],
                    "source": "scores_stage3 V1 R24; v1_capture g",
                },
                "ref_hex": v1[sid]["ref_hex"],
                "ref_30": v1[sid]["ref_30"],
                "tol": v1[sid]["tol"],
            }
        else:
            c = capture[sid]
            out[sid] = {
                "pickle_sha256": c["sha256"],
                "g06": {
                    "value": c["census_value"],
                    "abserr": c["census_abserr"],
                    "converged": c["census_converged"],
                    "num_regions": c["census_num_regions"],
                    "max_depth": c["census_max_depth"],
                    "source": "references/out/capture.json census record",
                },
                "ref_hex": rows.get(sid, {}).get("ref_hex"),
                "ref_30": rows.get(sid, {}).get("ref_30"),
                "tol": rows.get(sid, {}).get("tol"),
            }
        if sid in h1:
            r = h1[sid]
            out[sid]["parent"] = {
                "ref_30": out[sid]["ref_30"],
                "ratio": r["routeB_ratio"],
                "converged": r["routeB_converged"] == "True",
                "num_regions": int(r["routeB_regions"]),
                "source": "06l partC h1_calls.csv route B",
            }
        else:
            out[sid]["parent"] = None
    return out


# ------------------------------------------------------------------------------------------------
# one unit


def do_unit(spec):
    import tempfile

    mode = spec["mode"]
    evidence = {"mode": mode}
    if mode == "g06":
        tree = Path(spec["tree"]).resolve()
        sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != prov.REPO]
        sys.path.insert(0, str(tree))
        evidence["tree"] = str(tree)
    else:
        sys.path.insert(0, str(prov.REPO))
        if mode == "parent":
            import variant_loader

            source = prov.git(
                "show", f"{prov.PARENT_SHA}:AdaptiveLevin/levin_quadrature.py"
            )
            path = (
                Path(tempfile.mkdtemp(prefix="lpf07-parent-"))
                / "levin_quadrature_parent.py"
            )
            path.write_text(source)
            variant_loader.install(path, prov.sha256(path), "parent")
            evidence["variant"] = variant_loader.evidence()
        elif mode == "barlifted":
            import variant_loader

            import calibrate

            source = prov.LEVIN.read_text()
            if source.count(calibrate.BAR_TEXT) != 1:
                raise RuntimeError("replay: the bar's text is not found exactly once")
            path = (
                Path(tempfile.mkdtemp(prefix="lpf07-barlifted-"))
                / "levin_quadrature_barlifted.py"
            )
            path.write_text(
                source.replace(calibrate.BAR_TEXT, calibrate.BAR_LIFTED_TEXT)
            )
            variant_loader.install(path, prov.sha256(path), "barlifted")
            evidence["variant"] = variant_loader.evidence()

    import AdaptiveLevin.levin_quadrature as lq

    evidence["module_file"] = lq.__file__
    evidence["module_sha256"] = prov.sha256(lq.__file__)
    if mode in ("commit", "barlifted"):
        lq._LEVIN_WINDOW_HALF_PHASE = spec["phi_w"]
        evidence["window_half_phase"] = spec["phi_w"]
        evidence["coalescence_lambda"] = lq._LEVIN_COALESCENCE_LAMBDA

    # the captures were pickled by 06g's census harness (its _GroupPickler, whose lru_cache reducer
    # _rebuild_lru_cache the pickles name), so its module must be importable to read them: the laptop copy's
    # harness/ directory, read only, after the tree on sys.path, as 06g's laptop FCC batch had it
    sys.path.insert(1, str(HARNESS))
    import ray.cloudpickle as cp

    pickle_path = INPUTS / f"{spec['sid']}.pkl"
    blob = pickle_path.read_bytes()
    import hashlib

    evidence["pickle_sha256"] = hashlib.sha256(blob).hexdigest()
    call = cp.loads(blob)
    import ray

    evidence["ray_initialized"] = bool(ray.is_initialized())

    sys.path.insert(0, str(HERE))
    import analysis07 as A

    recorder = A.Recorder(lq)
    recorder.install()
    t0 = time.perf_counter()
    kw = dict(call["kw"])
    if spec.get("depth_max") is not None:
        kw["depth_max"] = spec["depth_max"]
    result = lq.adaptive_levin_sincos(call["x_span"], call["f"], call["theta"], **kw)
    wall = time.perf_counter() - t0
    recorder.remove()
    ref = float.fromhex(spec["ref_hex"]) if spec.get("ref_hex") else None
    depth_max = kw.get("depth_max", DEPTH_MAX)
    rec = A.summary(result, ref=ref, depth_max=depth_max)
    rec["depth_max"] = depth_max
    rec["wall_s"] = wall
    rec["x_span"] = [float(call["x_span"][0]).hex(), float(call["x_span"][1]).hex()]
    rec["kw"] = {
        k: (v if not isinstance(v, float) else v) for k, v in call["kw"].items()
    }
    atol = call["kw"].get("atol", lq.DEFAULT_LEVIN_ABSTOL)
    rtol = call["kw"].get("rtol", lq.DEFAULT_LEVIN_RELTOL)
    rec["request"] = A.request_of(result, atol, rtol)
    rec["abserr_over_request"] = result["abserr"] / rec["request"]
    if "stationary_points" in result:
        rec["window_leaves_at_depth_max"] = sum(
            1 for r in result["regions"] if r.window and r.depth >= depth_max
        )
    if A.needs_section_2_5(result, depth_max):
        rec["section_2_5"] = A.section_2_5(
            result,
            recorder.made,
            rec["request"],
            A.n_seed_leaves(result) if "stationary_windows" in result else 1,
            depth_max,
        )
    rec["evidence"] = evidence
    return rec


# ------------------------------------------------------------------------------------------------
# the run


def _execute(specs, processes, timeout, units_path, log):
    def one(spec):
        t0 = time.perf_counter()
        cmd = [sys.executable, "-B", str(Path(__file__)), "unit", json.dumps(spec)]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, cwd=str(HERE)
            )
        except subprocess.TimeoutExpired:
            return {
                "spec": spec,
                "did_not_finish": True,
                "wall_s": time.perf_counter() - t0,
            }
        if proc.returncode != 0:
            return {
                "spec": spec,
                "error": proc.stderr[-4000:],
                "wall_s": time.perf_counter() - t0,
            }
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("UNIT ")]
        return {"spec": spec, "record": json.loads(lines[-1][5:])}

    out = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=processes) as pool:
        futures = [pool.submit(one, s) for s in specs]
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            out.append(res)
            with open(units_path, "a") as handle:
                handle.write(json.dumps(res, default=str) + "\n")
            s, r = res["spec"], res.get("record")
            if r is None:
                log(
                    f"  {s['sid']} {s['mode']} {s.get('phi_w_pi')}: "
                    f"{'DID NOT FINISH' if res.get('did_not_finish') else 'ERROR ' + res.get('error', '')[-300:]}"
                )
            else:
                log(
                    f"  {s['sid']} {s['mode']} {s.get('phi_w_pi')}: ratio {r.get('error_over_abserr', float('nan')):.4g} "
                    f"converged {r['converged']} node_limited {r['node_limited']} regions {r['regions']} "
                    f"depth {r['max_depth']} wall {r['wall_s']:.1f} s"
                )
    return out


def _check(sid, rec, expected, kind):
    """The replay check: rec against the 06g record (kind "g06") or route B's (kind "parent")."""
    if kind == "g06":
        got = {
            "value": rec["value_hex"],
            "abserr": rec["abserr_hex"],
            "converged": rec["converged"],
            "num_regions": rec["regions"],
            "max_depth": rec["max_depth"],
        }
        want = {k: expected[k] for k in got}
    else:
        # 06l's ratio is |v - ref| / abserr with ref the 30-digit reading (ref_30), in mpmath, rounded to a
        # double at the end; the double nearest the reference (ref_hex) gives a ratio that differs from it in
        # the tenth digit (found after the run, which first compared that one; deviation in the log)
        import mpmath

        with mpmath.workdps(40):
            ratio_30 = float(
                abs(
                    mpmath.mpf(float.fromhex(rec["value_hex"]))
                    - mpmath.mpf(expected["ref_30"])
                )
                / mpmath.mpf(float.fromhex(rec["abserr_hex"]))
            )
        got = {
            "ratio": repr(ratio_30),
            "converged": rec["converged"],
            "num_regions": rec["regions"],
        }
        want = {k: expected[k] for k in ("ratio", "converged", "num_regions")}
    diff = {k: [got[k], want[k]] for k in got if got[k] != want[k]}
    return {"reproduces": not diff, "differences": diff, "got": got, "want": want}


def do_run(processes, timeout, tree, sids):
    log = lambda s: print(s, flush=True)  # noqa: E731
    recs = records()
    inputs = [
        SCORES,
        CAPTURE,
        V1_CAPTURE,
        H1_CALLS,
        HERE / "analysis07.py",
        HERE / "variant_loader.py",
        HERE / "calibrate.py",
    ] + [INPUTS / f"{sid}.pkl" for sid in sids]
    out = {
        "header": prov.json_header(
            __file__, "section 2.3, the real calls replayed (run)", inputs
        ),
        "tree_g06": str(tree),
        "tree_g06_head": subprocess.run(
            ["git", "-C", str(tree), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "tree_g06_status": subprocess.run(
            ["git", "-C", str(tree), "status", "--short"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout,
        "parent_sha": prov.PARENT_SHA,
        "phi_ladder_pi": PHI_LADDER_PI,
        "processes": processes,
        "unit_timeout_s": timeout,
        "records": {sid: recs[sid] for sid in sids},
    }
    out["pickle_sha256_matches_06g_record"] = {
        sid: prov.sha256(INPUTS / f"{sid}.pkl") == recs[sid]["pickle_sha256"]
        for sid in sids
    }
    if not all(out["pickle_sha256_matches_06g_record"].values()):
        raise SystemExit(
            f"a capture's sha256 differs from 06g's record: {out['pickle_sha256_matches_06g_record']}"
        )
    units_path = prov.OUT / "replay_units.jsonl"
    units_path.write_text("")
    t_start = time.perf_counter()

    def spec(sid, mode, phi_pi=None):
        return {
            "sid": sid,
            "mode": mode,
            "tree": str(tree),
            "phi_w_pi": phi_pi,
            "phi_w": None if phi_pi is None else phi_pi * math.pi,
            "ref_hex": recs[sid]["ref_hex"],
        }

    specs = []
    for sid in sids:
        specs.append(spec(sid, "g06"))
        specs.append(spec(sid, "parent"))
        for phi in PHI_LADDER_PI:
            specs.append(spec(sid, "commit", phi))
    log(f"stage 1: {len(specs)} units")
    results = _execute(specs, processes, timeout, units_path, log)
    bar = []
    for r in results:
        rec = r.get("record")
        if (
            r["spec"]["mode"] == "commit"
            and rec
            and rec.get("window_leaves_at_depth_max")
        ):
            s = dict(r["spec"])
            s["mode"] = "barlifted"
            bar.append(s)
    for r in results:
        if r.get("did_not_finish") and r["spec"]["mode"] == "commit":
            for dm in BOUNDED_DEPTHS:
                s = dict(r["spec"])
                s["mode"] = "commit"
                s["depth_max"] = dm
                bar.append(s)
            s = dict(r["spec"])
            s["mode"] = "barlifted"
            bar.append(s)
    log(
        f"stage 2 (bar lifted; and bounded depth where a call did not finish): {len(bar)} units"
    )
    results += _execute(bar, processes, timeout, units_path, log)

    table = {}
    for r in results:
        s = r["spec"]
        key = s["mode"] if s["phi_w_pi"] is None else f"{s['mode']} {s['phi_w_pi']:g}pi"
        if s.get("depth_max") is not None:
            key = f"{key} depth_max {s['depth_max']}"
        table.setdefault(s["sid"], {})[key] = r
    out["results"] = table
    checks = {}
    for sid in sids:
        g = table[sid]["g06"].get("record")
        checks[sid] = {
            "g06": None if g is None else _check(sid, g, recs[sid]["g06"], "g06")
        }
        p = table[sid]["parent"].get("record")
        if recs[sid]["parent"] is not None and p is not None:
            checks[sid]["parent"] = _check(sid, p, recs[sid]["parent"], "parent")
        else:
            checks[sid]["parent"] = None
    out["replay_check"] = checks
    out["replay_check_all_reproduce"] = all(
        c["g06"] is not None
        and c["g06"]["reproduces"]
        and (c["parent"] is None or c["parent"]["reproduces"])
        for c in checks.values()
    )
    out["wall_s"] = time.perf_counter() - t_start
    out["uptime_at_end"] = prov.uptime()
    (prov.OUT / "replay.json").write_text(json.dumps(out, indent=1, default=str))
    (prov.OUT / "replay_tables.md").write_text(tables(out))
    log(f"replay check, all reproduce: {out['replay_check_all_reproduce']}")
    log(f"wrote {prov.OUT / 'replay.json'} and replay_tables.md")


def _fmt(x, spec=".3g"):
    if x is None:
        return "--"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return format(x, spec)


def tables(out):
    lines = (
        ["# 07 section 2.3: the real calls replayed", ""]
        + [f"    {x}" for x in out["header"]]
        + [""]
    )
    lines += [
        f"06g's tree: {out['tree_g06']} HEAD {out['tree_g06_head']} (status --short: "
        f"{'empty' if not out['tree_g06_status'].strip() else out['tree_g06_status']})",
        "",
    ]
    lines += [
        "## The replay check",
        "",
        "| call | 06g (3ae6f3a4) reproduces | differences | parent (route B) reproduces | differences |",
        "|---|---|---|---|---|",
    ]
    for sid, c in out["replay_check"].items():
        g, p = c["g06"], c["parent"]
        lines.append(
            f"| {sid} | {None if g is None else g['reproduces']} | {'' if g is None else g['differences']} | "
            f"{'no record' if p is None else p['reproduces']} | {'' if p is None else p['differences']} |"
        )
    lines += ["", f"**All reproduce:** {out['replay_check_all_reproduce']}", ""]
    lines += [
        "## Every call and rung",
        "",
        "| call | mode | |v - ref| / abserr | honest | converged | node_limited | regions | max depth | "
        "abserr/request | window pieces | located (kind x) | windows (lo, hi, pieces) | wall s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for sid, modes in out["results"].items():
        for key in sorted(
            modes, key=lambda k: (k.split()[0] != "g06", k.split()[0] != "parent", k)
        ):
            r = modes[key].get("record")
            if r is None:
                lines.append(f"| {sid} | {key} | did not finish or error |" + " |" * 10)
                continue
            pts = "; ".join(
                f"{p['kind']} {p['x']:.10g}" for p in r.get("stationary_points", [])
            )
            wins = "; ".join(
                f"{w['lo']:.8g}, {w['hi']:.8g}, {w['pieces']}"
                for w in r.get("stationary_windows", [])
            )
            lines.append(
                f"| {sid} | {key} | {_fmt(r.get('error_over_abserr_30', r.get('error_over_abserr')), '.4g')} | "
                f"{r.get('honest_30', r.get('honest'))} | {r['converged']} | "
                f"{r['node_limited']} | {r['regions']} | {r['max_depth']} | {_fmt(r['abserr_over_request'])} | "
                f"{r.get('window_pieces', '--')} | {pts} | {wins} | {r['wall_s']:.1f} |"
            )
    lines += [
        "",
        "## Section 2.5 on these calls",
        "",
        "| call | mode | converged | node_limited | max depth | leaves by class: count, sum e', at depth_max | "
        "QUADPACK iroff1, iroff2, ier=2 at bisection |",
        "|---|---|---|---|---|---|---|",
    ]
    for sid, modes in out["results"].items():
        for key, res in sorted(modes.items()):
            r = res.get("record")
            if r is None or "section_2_5" not in r:
                continue
            a = r["section_2_5"]
            cls = "; ".join(
                f"{k} {v['count']}, {v['sum_budget_err']:.2e}, {v['at_depth_max']}"
                for k, v in sorted(a["classes"].items())
            )
            q = a["quadpack_global"]
            lines.append(
                f"| {sid} | {key} | {r['converged']} | {r['node_limited']} | {r['max_depth']} | {cls} | "
                f"{q['iroff1']}, {q['iroff2']}, {q['ier2_reached_at_bisection']} |"
            )
    lines += [
        "",
        f"Wall time {out['wall_s']:.0f} s; uptime at end: {out['uptime_at_end']}",
        "",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "unit", "tables"))
    parser.add_argument("spec", nargs="?")
    parser.add_argument(
        "--tree", default=None, help="the scratch worktree at 3ae6f3a4 (06g's code)"
    )
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--unit-timeout", type=float, default=4 * 3600.0)
    parser.add_argument("--sids", default=",".join(SIDS))
    args = parser.parse_args()
    if args.action == "unit":
        rec = do_unit(json.loads(args.spec))
        print("UNIT " + json.dumps(rec, default=str), flush=True)
    elif args.action == "run":
        do_run(
            args.processes,
            args.unit_timeout,
            Path(args.tree).resolve(),
            args.sids.split(","),
        )
    else:
        out = json.loads((prov.OUT / "replay.json").read_text())
        recheck(out)
        (prov.OUT / "replay.json").write_text(json.dumps(out, indent=1, default=str))
        (prov.OUT / "replay_tables.md").write_text(tables(out))


def recheck(out):
    """
    After the run: the replay check recomputed from the stored records, with the parent's ratio in 06l's
    arithmetic (the 30-digit reference, in mpmath), and every record's |v - ref| / abserr likewise
    ("error_over_abserr_30", "honest_30"). The run's own check is kept as "replay_check_as_run". No call is
    re-run; the header of this recomputation is "recheck_header".
    """
    import mpmath

    recs = records()
    out["recheck_header"] = prov.json_header(
        __file__,
        "section 2.3, the replay check recomputed from the run's records",
        (prov.OUT / "replay_units.jsonl", SCORES, CAPTURE, V1_CAPTURE, H1_CALLS),
    )
    out.setdefault("replay_check_as_run", out["replay_check"])
    out.setdefault(
        "replay_check_all_reproduce_as_run", out["replay_check_all_reproduce"]
    )
    for sid, modes in out["results"].items():
        ref_30 = recs[sid].get("ref_30")
        for res in modes.values():
            r = res.get("record")
            if r is None or ref_30 is None:
                continue
            with mpmath.workdps(40):
                ratio = float(
                    abs(mpmath.mpf(float.fromhex(r["value_hex"])) - mpmath.mpf(ref_30))
                    / mpmath.mpf(float.fromhex(r["abserr_hex"]))
                )
            r["error_over_abserr_30"] = ratio
            r["honest_30"] = bool(ratio <= 1.0)
    checks = {}
    for sid in out["results"]:
        g = out["results"][sid]["g06"].get("record")
        p = out["results"][sid]["parent"].get("record")
        checks[sid] = {
            "g06": None if g is None else _check(sid, g, recs[sid]["g06"], "g06"),
            "parent": (
                _check(sid, p, recs[sid]["parent"], "parent")
                if recs[sid]["parent"] is not None and p is not None
                else None
            ),
        }
    out["replay_check"] = checks
    out["replay_check_all_reproduce"] = all(
        c["g06"] is not None
        and c["g06"]["reproduces"]
        and (c["parent"] is None or c["parent"]["reproduces"])
        for c in checks.values()
    )


if __name__ == "__main__":
    main()
