"""levin-phase-floor 07 section 3: the deliberate breakages (i)-(ix), each with the test that must catch it.
New in 07.

Each breakage is a text substitution in a copy of AdaptiveLevin/levin_quadrature.py (or, for (viii), of
ComputeTargets/QuadSourceIntegral.py), each substitution checked to occur exactly once. The tree is never edited:
- a levin_quadrature.py breakage is written to a temporary file and installed in place of the module by
  variant_loader.py (06m's), before anything imports AdaptiveLevin, in a fresh process;
- the QuadSourceIntegral breakage is executed into the imported module's namespace (its functions redefined
  from the changed source) before the test module is imported, in a fresh process.
The named test is then run by unittest, and the breakage is caught if it fails or errors. As a control, every
named test is also run once against the unbroken tree. Writes out/run2/breakages.json and .txt.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prov  # noqa: E402

LQ = prov.REPO / "AdaptiveLevin" / "levin_quadrature.py"
QSI = prov.REPO / "ComputeTargets" / "QuadSourceIntegral.py"
T = "AdaptiveLevin.tests.test_levin_stationary."
T11 = "ComputeTargets.tests.test_quadsource_levin_guard_counts.TestGroupRecordCounts.test_11_group_record_carries_the_four_counts"

LOCATION_ON = """    if (
        rate_fns is not None
        and not _test_only_bypass_location"""
LOCATION_OFF = """    if (
        False
        and rate_fns is not None
        and not _test_only_bypass_location"""
GUARD_ON = (
    "    guarded = _nodal_guard(theta_prime_Cheb, grid, guard_rates, vectorize_cache)"
)
GUARD_OFF = "    guarded = False"
SEED = """    if location is None or not location["windows"]:
        _add_leaf(_make_leaf(x_span[0], x_span[1], 0, None, {}, {}, {}))
    else:"""
SEED_PLAIN_SPLIT = """    if location is not None and location["windows"]:
        _split_at = location["points"][0]["x"]
        _add_leaf(_make_leaf(x_span[0], _split_at, 0, None, {}, {}, {}))
        _add_leaf(_make_leaf(_split_at, x_span[1], 0, None, {}, {}, {}))
    elif True:
        _add_leaf(_make_leaf(x_span[0], x_span[1], 0, None, {}, {}, {}))
    else:"""

BREAKAGES = {
    "i": (
        "location, windows and guard all off",
        "lq",
        [(LOCATION_ON, LOCATION_OFF), (GUARD_ON, GUARD_OFF)],
        [T + "TestOneStationaryPoint.test_1_located_windowed_and_honest"],
    ),
    "ii": (
        "location off, leaving only the guard",
        "lq",
        [(LOCATION_ON, LOCATION_OFF)],
        [T + "TestTwoClosePoints.test_2_two_points_between_adjacent_nodes"],
    ),
    "iii": (
        "the floor-limited exit allowed on window and guarded leaves",
        "lq",
        [
            (
                "        if (\n            not region.window\n            and not leaf.guarded\n            and np.isfinite(phase_err)\n",
                "        if (\n            np.isfinite(phase_err)\n",
            )
        ],
        [
            T
            + "TestFloorLimitedExitBarred.test_5_no_window_leaf_floor_limited_by_the_exit"
        ],
    ),
    "iv": (
        "minima never located (Lambda_c = 0)",
        "lq",
        [
            (
                "            coalescence_lambda=_LEVIN_COALESCENCE_LAMBDA,",
                "            coalescence_lambda=0.0,",
            )
        ],
        [T + "TestNearlyCoalescingPair.test_3_minimum_located_and_windowed"],
    ),
    "v": (
        "a guarded leaf at depth_max returned as a Levin leaf",
        "lq",
        [
            (
                "            guarded = True\n            if depth >= depth_max:",
                "            guarded = depth < depth_max\n            if False:",
            )
        ],
        [T + "TestTheGuard.test_6_the_cap"],
    ),
    "vi": (
        "the window replaced by a plain split at x_s, the leaves on either side unguarded",
        "lq",
        [(SEED, SEED_PLAIN_SPLIT), (GUARD_ON, GUARD_OFF)],
        [
            T + "TestOneStationaryPoint.test_1_located_windowed_and_honest",
            T + "TestOneStationaryPoint.test_7_window_edges_and_pieces",
        ],
    ),
    "vii": (
        "location done only for the first phase a basis reports",
        "lq",
        [
            (
                "    for phase_index, rate in enumerate(rate_fns):",
                "    for phase_index, rate in list(enumerate(rate_fns))[:1]:",
            )
        ],
        [T + "TestGenericPath.test_9_two_phases"],
    ),
    "viii": (
        "the four counts left out of the group record",
        "qsi",
        [
            (
                """                "stationary_points": len(data["stationary_points"]),
                "window_pieces": data["num_window_pieces"],
                "guarded_regions": data["num_guarded_regions"],
                "guard_at_depth_max": data["num_guard_at_depth_max"],
            }
            for group, data in results""",
                """            }
            for group, data in results""",
            )
        ],
        [T11],
    ),
    "ix": (
        "section 1.3's counter never reaching its limit",
        "lq",
        [
            (
                "window_counts[leaf.window_index] >= _LEVIN_WINDOW_IROFF1_LIMIT",
                "window_counts[leaf.window_index] >= 10**9",
            )
        ],
        [T + "TestWindowStoppingRule.test_10a_flat_window_stops"],
    ),
}

RUNNER = r"""
import json, sys, unittest
sys.path.insert(0, {repo!r})
sys.path.insert(0, {here!r})
kind, path, sha, test = {kind!r}, {path!r}, {sha!r}, {test!r}
if kind == "lq":
    import variant_loader
    variant_loader.install(path, sha, "breakage")
elif kind == "qsi":
    import importlib
    Q = importlib.import_module("ComputeTargets.QuadSourceIntegral")  # the module (the package re-exports a class of that name)
    exec(compile(open(path).read(), path, "exec"), Q.__dict__)
suite = unittest.defaultTestLoader.loadTestsFromName(test)
result = unittest.TestResult()
suite.run(result)
print("RESULT " + json.dumps({{"failures": len(result.failures), "errors": len(result.errors), "run": result.testsRun,
    "detail": [t[1][-600:] for t in result.failures + result.errors][:1]}}))
"""


def run(kind, path, sha, test, timeout=1800):
    code = RUNNER.format(
        repo=str(prov.REPO),
        here=str(HERE),
        kind=kind,
        path=str(path),
        sha=sha,
        test=test,
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-B", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.mkdtemp(prefix="lpf07-breakage-"),
        )
    except subprocess.TimeoutExpired:
        return {"timeout": True}
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    if not lines:
        return {"harness_error": proc.stderr[-1500:]}
    return json.loads(lines[-1][7:])


def main():
    out = {
        "header": prov.json_header(
            __file__,
            "section 3, the deliberate breakages",
            (
                LQ,
                QSI,
                prov.REPO / "AdaptiveLevin" / "tests" / "test_levin_stationary.py",
                prov.REPO
                / "ComputeTargets"
                / "tests"
                / "test_quadsource_levin_guard_counts.py",
                HERE / "variant_loader.py",
            ),
        ),
        "breakages": {},
    }
    txt = [f"# {x}" for x in out["header"]] + [""]
    tmp = Path(tempfile.mkdtemp(prefix="lpf07-breakages-"))
    for key, (what, kind, subs, tests) in BREAKAGES.items():
        source = (LQ if kind == "lq" else QSI).read_text()
        for old, new in subs:
            if source.count(old) != 1:
                raise SystemExit(
                    f"breakage ({key}): substitution not found exactly once: {old[:60]!r}"
                )
            source = source.replace(old, new)
        path = (
            tmp
            / f"breakage_{key}_{'levin_quadrature' if kind == 'lq' else 'QuadSourceIntegral'}.py"
        )
        path.write_text(source)
        sha = prov.sha256(path)
        rec = {"what": what, "kind": kind, "variant_sha256": sha, "tests": {}}
        caught = False
        for test in tests:
            control = run("none", "", "", test)
            broken = run(kind, path, sha, test)
            failed = bool(
                broken.get("failures") or broken.get("errors") or broken.get("timeout")
            )
            caught = caught or failed
            rec["tests"][test] = {
                "control": control,
                "broken": broken,
                "fails_under_breakage": failed,
            }
            txt.append(
                f"({key}) {what}: {test.split('.')[-1]}: control "
                f"{'passes' if control.get('run') and not control.get('failures') and not control.get('errors') else control}; "
                f"broken {'FAILS' if failed else 'passes'} "
                + (
                    f"-- {broken.get('detail', [''])[0].strip().splitlines()[-1][:200]}"
                    if broken.get("detail")
                    else ""
                )
            )
            print(txt[-1], flush=True)
        rec["caught"] = caught
        out["breakages"][key] = rec
    out["all_caught"] = all(r["caught"] for r in out["breakages"].values())
    txt.append(f"all caught: {out['all_caught']}")
    (prov.OUT / "breakages.json").write_text(json.dumps(out, indent=1))
    (prov.OUT / "breakages.txt").write_text("\n".join(txt) + "\n")
    print(txt[-1])


if __name__ == "__main__":
    main()
