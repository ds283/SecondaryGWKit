"""Y1 references: run refquad.py's three steps for a list of calls, one call at a time.

For each (row, index): ``panels`` and ``integrate`` in R18's tree (e5764a6) on R18's pickle, then
``compare`` in R17's tree (fef401c) on R17's pickle against R18's sample. With --dedupe, a call
whose row has another call in this list with the same label pair G+q-r / G-q+r and the same
span is first compared (``compare``, in R18's tree) against that call's sample; if every sampled
value is bit-identical the earlier call's reference is reused and recorded as such, and the R17
comparison is still made against the earlier call's sample.

Every step's own output goes to state/ref/<row>-<idx>.<step>.log; this script prints a summary.
At most six worker processes at any time (one step at a time, each with --workers 6).

Usage: python -B batch.py CALLS (e.g. Y01:0,Y01:1) [--dedupe]
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from yprov import PYTHON, T17, T18, Y1, header  # noqa: E402

STATE = Y1 / "state"
REF = STATE / "ref"


def run(tree, args, log):
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{tree}:{HERE}"
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    t0 = time.perf_counter()
    with open(log, "w") as fh:
        rc = subprocess.run([str(PYTHON), "-B", str(HERE / "refquad.py")] + args, cwd=str(STATE), env=env,
                            stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT).returncode
    return rc, time.perf_counter() - t0


def main():
    calls = [(c.split(":")[0], int(c.split(":")[1])) for c in sys.argv[1].split(",")]
    dedupe = "--dedupe" in sys.argv[2:]
    for l in header(Path(__file__).resolve(), f"Y1 references: batch of {len(calls)} calls", [HERE / "refquad.py"]):
        print(l, flush=True)
    REF.mkdir(parents=True, exist_ok=True)
    done = {}
    for row, idx in calls:
        tag = f"{row}-{idx:02d}"
        p18 = f"inputs/{row}-R18-{idx:02d}.pkl"
        p17 = f"inputs/{row}-R17-{idx:02d}.pkl"
        rec = {"call": tag}
        twin = None
        if dedupe:
            lab = json.loads((STATE / "moved.json").read_text())
            me = next(e for e in lab if e["row"] == row and e["index"] == idx)
            for (r2, i2), t2 in done.items():
                o = next(e for e in lab if e["row"] == r2 and e["index"] == i2)
                if r2 == row and {o["label"], me["label"]} == {"G+q-r", "G-q+r"} and o["x_span"] == me["x_span"] and t2 == "ok":
                    twin = (r2, i2)
        if twin is not None:
            ttag = f"{twin[0]}-{twin[1]:02d}"
            rc, s = run(T18, ["compare", p18, f"ref/{ttag}.sample.npz", f"ref/{tag}.twin18.json"], REF / f"{tag}.twin18.log")
            ok = rc == 0 and json.loads((REF / f"{tag}.twin18.json").read_text())["all_bit_identical"]
            print(f"{tag}: twin of {ttag}? R18 compare rc {rc}, all bit-identical {ok} ({s:.0f} s)", flush=True)
            if ok:
                rc, s = run(T17, ["compare", p17, f"ref/{ttag}.sample.npz", f"ref/{tag}.cmp17.json"], REF / f"{tag}.compare.log")
                ok17 = rc == 0 and json.loads((REF / f"{tag}.cmp17.json").read_text())["all_bit_identical"]
                print(f"{tag}: R17 compare against {ttag}'s sample rc {rc}, all bit-identical {ok17} ({s:.0f} s)", flush=True)
                (REF / f"{tag}.twin.json").write_text(json.dumps({"twin_of": ttag, "r18_identical": ok, "r17_identical": ok17}))
                done[(row, idx)] = "twin"
                continue
        for step, tree, args in (
            ("panels", T18, ["panels", p18, f"ref/{tag}"]),
            ("integrate", T18, ["integrate", p18, f"ref/{tag}"]),
            ("compare", T17, ["compare", p17, f"ref/{tag}.sample.npz", f"ref/{tag}.cmp17.json"]),
        ):
            rc, s = run(tree, args, REF / f"{tag}.{step}.log")
            rec[step] = (rc, s)
            print(f"{tag}: {step} rc {rc} in {s:.0f} s", flush=True)
            if rc != 0:
                print(f"{tag}: STOPPED at {step}; see {REF / f'{tag}.{step}.log'}", flush=True)
                break
        if all(rec.get(s, (1,))[0] == 0 for s in ("panels", "integrate", "compare")):
            r = json.loads((REF / f"{tag}.ref.json").read_text())
            c = json.loads((REF / f"{tag}.cmp17.json").read_text())
            print(f"{tag}: ref {r['ref']} conv_diff {r['conv_diff']} panels {r['npanels']} nodes {r['nodes']} "
                  f"R17 sample identical {c['all_bit_identical']} ({c['columns']['f_sin']['nodes']} nodes)", flush=True)
            done[(row, idx)] = "ok"
        else:
            done[(row, idx)] = "failed"
    print("batch done:", {f"{r}-{i:02d}": v for (r, i), v in done.items()}, flush=True)


if __name__ == "__main__":
    main()
