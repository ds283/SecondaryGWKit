"""
handover prompt 04 (A4): the checks made before the launch, in prompt 04 section 3's order.

    python -B a4_checks.py construction   # section 2.1: every cell's objects, built, not run
    python -B a4_checks.py transparency   # section 2.2: one cell with and without the wrapper
    python -B a4_checks.py determinism    # section 3: two in-process runs per arm

Each writes `out/check_<name>.json` beside this directory's parent (`var/handover-04/out/`, or
the same place relative to wherever this file is), prints a summary, and exits 0 when the check
holds, 1 when it does not.

* **construction** (run 2: also `a4_cells.below_anchor_check`, which evaluates every phase at
  and below log(1 + z_resp), down to the lowest abscissa the Levin driver can form there, and
  requires P0's and P's G phase to evaluate at all of them) builds the `Case` of every one of the
  60 cells (every shape, rung and arm), as
  the run will, and checks the phase objects by class and `exact_remainder` (prompt 04 section
  2.1: S's G a `phase_spline`; P0's and P's `PrimitivePhase`s with `exact_remainder` false and
  true; every realistic T a `PrimitivePhase` on a `TablePrimitive` with `exact_remainder` true,
  from `_UnionTableCsTauFixture`, anchored at z_init off a node with p_R != 0).
  No integral is run.
* **transparency** runs one cell (arm S, `together` at 980: S has the most regions at that rung,
  so the most wrapper calls are crossed) twice in one process, wrapped and unwrapped, and
  compares `total` and `total_abserr` bit for bit.
* **determinism** runs `together` at 980 twice in one process per arm and compares `total`,
  `total_abserr` and every group's region count and depth bit for bit.

After landing, this file is a record of its own SHA-256.
"""

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import a4_cells as C  # noqa: E402

OUT = HERE.parent / "out"


def _hex(x):
    return float.hex(float(x))


def construction():
    from ComputeTargets.tests.test_quadsource_integral import SHAPES

    rows = []
    ok = True
    for shape_name, rungs in C.LADDER.items():
        base = [s for s in SHAPES if s.name == shape_name][0]
        for x_resp in rungs:
            for arm in C.ARMS:
                t0 = time.perf_counter()
                try:
                    case, built = C.build_case(base, x_resp, arm)
                    objects = C.phase_objects(case, built)
                    problems = C.construction_problems(arm, objects)
                    below = C.below_anchor_check(case)
                    if arm in ("P0", "P") and not below["G_ok"]:
                        problems.append(f"{arm}: G's phase fails below z_resp: {below}")
                    error = None
                except Exception as exc:  # noqa: BLE001
                    objects, problems, below = None, [], None
                    error = f"{type(exc).__name__}: {exc}"
                elapsed = time.perf_counter() - t0
                good = error is None and not problems
                ok = ok and good
                rows.append(
                    {
                        "shape": shape_name,
                        "x_resp": x_resp,
                        "arm": arm,
                        "ok": good,
                        "error": error,
                        "problems": problems,
                        "objects": objects,
                        "below_anchor": below,
                        "build_time": elapsed,
                    }
                )
                G = (objects or {}).get("G", {})
                Tq = (objects or {}).get("Tq", {})
                Tr = (objects or {}).get("Tr", {})
                print(
                    f"{shape_name:9s} {x_resp:8.0e} {arm:2s} {'ok ' if good else 'BAD'} "
                    f"G={G.get('class')}/{G.get('exact_remainder')}/{G.get('leading_class')} "
                    f"Tq={Tq.get('class')}/{Tq.get('exact_remainder')}/{Tq.get('leading_class')}"
                    f"/drop={Tq.get('drop_first_WKB_sample')} "
                    f"Tr={Tr.get('class')}/{Tr.get('exact_remainder')}/{Tr.get('leading_class')}"
                    f"/drop={Tr.get('drop_first_WKB_sample')} "
                    f"pR_q={Tq.get('p_R')!r} pR_r={Tr.get('p_R')!r} "
                    f"anchor_node_q={Tq.get('anchor_node_index')} "
                    f"anchor_node_r={Tr.get('anchor_node_index')} "
                    f"amp={(objects or {}).get('G_sin_amplitude_owner')} "
                    f"Gpath={(objects or {}).get('G_path')} "
                    f"Ggrid={(objects or {}).get('G_grid')} "
                    + (
                        ""
                        if below is None
                        else (
                            f"below: G_ok={below['G_ok']} T_ok={below['T_ok']} lowest="
                            + ",".join(
                                f"{sub['regime']}:-{sub['lowest']['below_a_ulp']:.0f}ulp"
                                for sub in below["subintervals"]
                            )
                            + " "
                        )
                    )
                    + f"{elapsed:.2f}s"
                    + (f" ERROR {error}" if error else "")
                    + (f" PROBLEMS {problems}" if problems else ""),
                    flush=True,
                )
    return ok, {"cells": rows}


def transparency():
    shape_name, x_resp, arm = "together", 980.0, "S"
    wrapped = C.run_cell(shape_name, x_resp, arm, instrument=True)
    plain = C.run_cell(shape_name, x_resp, arm, instrument=False)
    same = {
        key: _hex(wrapped[key]) == _hex(plain[key])
        for key in ("total", "total_abserr", "numeric_quad", "WKB_Levin")
    }
    same["Levin_regions_from_metadata"] = (
        wrapped["Levin_regions_from_metadata"] == plain["Levin_regions_from_metadata"]
    )
    ok = same["total"] and same["total_abserr"]
    for key in ("total", "total_abserr", "numeric_quad", "WKB_Levin"):
        print(
            f"{key:14s} wrapped {wrapped[key]!r:26s} {_hex(wrapped[key])}  "
            f"unwrapped {plain[key]!r:26s} {_hex(plain[key])}  "
            f"{'bit-identical' if same[key] else 'DIFFER'}"
        )
    print(
        f"Levin regions (metadata): wrapped {wrapped['Levin_regions_from_metadata']}, "
        f"unwrapped {plain['Levin_regions_from_metadata']}; wrapper's own count "
        f"{wrapped['Levin_regions']}"
    )
    print(
        f"integral_time wrapped {wrapped['integral_time']:.2f} s, unwrapped "
        f"{plain['integral_time']:.2f} s"
    )
    return ok, {
        "cell": [shape_name, x_resp, arm],
        "same": same,
        "wrapped": {k: v for k, v in wrapped.items() if k != "groups"},
        "wrapped_groups": wrapped["groups"],
        "unwrapped": plain,
    }


def _fingerprint(record):
    return {
        "total": _hex(record["total"]),
        "total_abserr": _hex(record["total_abserr"]),
        "groups": [
            [
                g["subinterval"],
                g["label"],
                g["num_regions"],
                g["max_depth"],
                g["max_region_depth"],
                g["num_phase_limited_regions"],
                _hex(g["raw_value"]),
                _hex(g["raw_abserr"]),
            ]
            for g in record["groups"]
        ],
    }


def determinism():
    ok = True
    arms = {}
    for arm in C.ARMS:
        first = C.run_cell("together", 980.0, arm)
        second = C.run_cell("together", 980.0, arm)
        a, b = _fingerprint(first), _fingerprint(second)
        same = a == b
        ok = ok and same
        arms[arm] = {
            "same": same,
            "first": a,
            "second": b,
            "integral_time": [first["integral_time"], second["integral_time"]],
            "Levin_regions": [first["Levin_regions"], second["Levin_regions"]],
            "max_depth": [first["max_depth"], second["max_depth"]],
        }
        print(
            f"arm {arm:2s}: total {a['total']} / {b['total']}, total_abserr "
            f"{a['total_abserr']} / {b['total_abserr']}, {len(a['groups'])} groups, regions "
            f"{first['Levin_regions']} / {second['Levin_regions']}, max depth "
            f"{first['max_depth']} / {second['max_depth']}: "
            f"{'bit-identical' if same else 'DIFFER'} "
            f"({first['integral_time']:.2f} s, {second['integral_time']:.2f} s)",
            flush=True,
        )
    return ok, {"cell": ["together", 980.0], "arms": arms}


def main():
    name = sys.argv[1]
    fn = {
        "construction": construction,
        "transparency": transparency,
        "determinism": determinism,
    }[name]
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ok, payload = fn()
    payload.update(
        {
            "check": name,
            "ok": ok,
            "a4_cells_sha256": C.SCRIPT_SHA256,
            "a4_checks_sha256": C.sha256_of(__file__),
            "started_unix": started,
            "elapsed": time.time() - started,
        }
    )
    with open(OUT / f"check_{name}.json", "w") as handle:
        json.dump(payload, handle, indent=1, default=repr)
    print(
        f"== check {name}: {'HOLDS' if ok else 'FAILS'} ({time.time() - started:.1f} s)"
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
