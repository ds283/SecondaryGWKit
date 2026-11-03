"""levin-phase-floor 06g, the FCC re-reads after the tool's b5626ec fix, on the laptop.

Units, in order:
  1. four checks: readings of the Mac Studio's parts 1-2 and rerun batch, re-read here with the tool at
     b5626ec, each to equal its recorded value_hex, value_30 and imag_30 bit for bit (a mismatch stops
     the run): w06 qawo r1 (the rerun batch, tool ff3e839), r55 fcc r1, fcc r2 and qawo r1 (part 2,
     tool 1c5336e);
  2. FCC r1 and r2 on the 14 stationary-point samples (w00-w11, z00, z12): 28 readings.
Each reading runs tools/filon_reference.reading with six spawned workers and is written to
readings/<sid>-<impl>-r<res>.json. Ray-free, no lock, no store.
Usage: python -B scripts/run_fcc.py --run-dir RUN_DIR   (launched by scripts/launch_fcc.py)
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import runregistrykit

ROOT = Path(__file__).resolve().parent.parent
TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf06g-b5626ec")
SAMPLES = [f"w{i:02d}" for i in range(12)] + ["z00", "z12"]
CHECKS = [("w06", "qawo", 1), ("r55", "fcc", 1), ("r55", "fcc", 2), ("r55", "qawo", 1)]


def units():
    out = [("check", s, i, r) for s, i, r in CHECKS]
    out += [("fcc", s, "fcc", r) for s in SAMPLES for r in (1, 2)]
    return out


def unit_name(u):
    return f"{u[0]} {u[1]} {u[2]} r{u[3]}"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    run = runregistrykit.Run(a.run_dir)
    run.heartbeat(pid=os.getpid(), stage="start")
    stop = threading.Event()

    def beat():
        while not stop.wait(60):
            run.heartbeat()

    threading.Thread(target=beat, daemon=True).start()
    import tools.filon_reference as F

    if Path(F.__file__).resolve() != (TREE / "tools" / "filon_reference.py").resolve():
        raise SystemExit(f"tool imported from {F.__file__}, not {TREE}")
    tool_sha = sha256(F.__file__)
    expected = json.loads((ROOT / "macstudio_expected.json").read_text())
    done = run.known()
    todo = units()
    n_done = sum(1 for u in todo if unit_name(u) in done)
    try:
        for u in todo:
            name = unit_name(u)
            if name in done:
                continue
            kind, sid, impl, res = u
            pkl = str(ROOT / "inputs" / f"{sid}.pkl")
            run.heartbeat(units_done=n_done, units_total=len(todo), stage=name)
            call = F._load_pickle(pkl, "ray.cloudpickle")
            t0 = time.perf_counter()
            r = F.reading(
                call,
                impl,
                res,
                workers=6,
                call_source=("tools.filon_reference", "_load_pickle", (pkl, "ray.cloudpickle")),
            )
            wall = time.perf_counter() - t0
            r = {k: v for k, v in r.items() if k != "value_mp"}
            r.update(
                sample=sid,
                impl=impl,
                res=res,
                pickle=pkl,
                pickle_sha256=sha256(pkl),
                tool=str(F.__file__),
                tool_sha256=tool_sha,
                workers=6,
                driver_wall_s=wall,
            )
            out = ROOT / "readings" / f"{sid}-{impl}-r{res}.json"
            if out.exists():
                raise SystemExit(f"{out} exists; not overwritten")
            out.write_text(json.dumps(r, indent=1, default=str))
            data = {
                "value_hex": r["value_hex"],
                "value_30": r["value_30"],
                "imag_30": r["imag_30"],
                "wall_s": wall,
                "reading": str(out),
                "reading_sha256": sha256(out),
                "direct_segments": r.get("tmap", {}).get("direct_segments"),
            }
            if kind == "check":
                exp = expected[f"check:{sid} {impl} {res}"]
                data["expected"] = exp[:3]
                data["expected_tool_sha256"] = exp[3]
                data["bit_identical"] = [r["value_hex"], r["value_30"], r["imag_30"]] == exp[:3]
            run.record(name, data)
            n_done += 1
            print(name, r["value_hex"], data.get("bit_identical", ""), round(wall, 1), flush=True)
            if kind == "check" and not data["bit_identical"]:
                stop.set()
                run.finish("failed", exit_code=2)
                print("check mismatch: stopped", flush=True)
                return 2
        stop.set()
        run.heartbeat(units_done=n_done, units_total=len(todo), stage="done")
        run.finish("done", exit_code=0)
        return 0
    except BaseException:
        stop.set()
        run.finish("failed", exit_code=1)
        raise


if __name__ == "__main__":
    sys.exit(main())
