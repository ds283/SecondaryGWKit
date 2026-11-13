"""Why do 05's depth-20 rows stay unconverged once the cap has gone? A planning measurement.

Run by the orchestrator on 2026-10-26 while writing prompt 05a, on the Mac Studio, read-only and
Ray-free. It reads prompt 05's wider re-measurement, the run
`levin-phase-floor-05-wider-20261026T020005`: its `checkpoint.jsonl` (one record per unit, with
each Levin call's group value, abserr, atol, rtol, converged and phase_limited) and its
`unit_*.pkl` (each call's regions, with the abserr each region was accepted on). It reads no
store and writes only its own output.

    scp convergence_kinds.py macstudio-tunnel:Documents/Code/SecondaryGWKit/var/levin-phase-floor-measure/scripts/
    ssh macstudio-tunnel 'cd ~/Documents/Code/SecondaryGWKit && ./venv/bin/python -I \\
        var/levin-phase-floor-measure/scripts/convergence_kinds.py \\
        var/runs/levin-phase-floor-05-wider-20261026T020005'

`--tree` names the tree whose units are read (default `c`, 05's code). A row is "capped" when its
unit of `--capped-tree` at `rtol = 1e-7` in `--capped-run` reaches depth 20 (default: tree `p`,
B1's code without 05's change, in the same run). Prompt 05a runs this on its own run with
`--tree t --capped-run <05's run>`, so that its rows are read against 05's capped set.

For each Levin call (one phase group of one sub-interval) of the tree that is not converged, three
numbers:
- E, the group's declared abserr, which is the sum of each region's `total_err`, the larger of its
  resolution residual (or a Clenshaw-Curtis region's nested-pair estimate) and its eq. (151)
  round-off bound, plus its truncation term (`levin_quadrature.py`, `used_interval`);
- R, the sum of the regions' own abserr, the resolution residual or nested-pair estimate alone;
- req = max(atol, rtol |value|), what the group's `converged` tests E against
  (`levin_quadrature.py:2308-2309`).

Every unconverged call falls in one of four kinds:
- "cap": a region at depth >= 20;
- "phase-limited": the driver accepted a region because its round-off bound exceeded the request;
- "resolution": R > req with neither of the above. Each region passed rtol against its own value,
  so the regions' values cancel in the group's total by at least R / req;
- "bound only": R <= req < E. The resolution residual meets the request; the excess is the
  round-off bound or the truncation term, which these records do not separate.
"""

import argparse
import collections
import hashlib
import json
import os
import pickle
import platform
import subprocess
from datetime import datetime, timezone


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fval(x):
    if isinstance(x, dict):
        return float.fromhex(x["hex"]) if "hex" in x else float(x["repr"])
    return float(x)


def kind(call, regions):
    if int(regions["depth"].max()) >= 20:
        return "cap"
    if call["phase_limited"]:
        return "phase-limited"
    req = max(call["atol"], call["rtol"] * abs(fval(call["value"])))
    if float(regions["abserr"].sum()) > req:
        return "resolution"
    return "bound only"


def read_checkpoint(run_dir):
    checkpoint = os.path.join(run_dir, "checkpoint.jsonl")
    print(f"# input {checkpoint} sha256 {sha256(checkpoint)}")
    recs = {}
    with open(checkpoint) as f:
        for line in f:
            r = json.loads(line)
            recs[r["unit"]] = r
    with open(os.path.join(run_dir, "manifest.json")) as f:
        print(f"# run {run_dir}: git_head {json.load(f).get('git_head')}")
    return recs


def main(run_dir, tree, capped_run, capped_tree):
    script = os.path.abspath(__file__)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(script),
    ).stdout.strip()

    print(
        f"# convergence_kinds.py, run {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    )
    print(
        f"# host {platform.node()}; python {platform.python_version()}; git HEAD {head}"
    )
    print(f"# script sha256 {sha256(script)}")
    print(f"# tree {tree}; capped: tree {capped_tree} at rtol 1e-7 in {capped_run}")

    recs = read_checkpoint(run_dir)
    capped_recs = (
        recs
        if os.path.realpath(capped_run) == os.path.realpath(run_dir)
        else read_checkpoint(capped_run)
    )

    rows = sorted({r["data"]["row"] for r in recs.values()})
    for rtol in ("1e-7", "1e-8"):
        tally = collections.Counter()
        tally_capped = collections.Counter()
        print("")
        print(f"== tree {tree} at rtol {rtol}: every unconverged Levin call")
        print(
            f"   row            role     capped_{capped_tree}  regime/group   kind            "
            "regions maxdepth   E/req     R/req   (E-R)/E"
        )
        for row in rows:
            unit = recs.get(f"{tree}-{rtol}-{row}")
            if unit is None or unit["data"]["status"] != "done":
                continue
            data = unit["data"]
            p = capped_recs.get(f"{capped_tree}-1e-7-{row}")
            capped = (
                p is not None
                and p["data"]["status"] == "done"
                and p["data"]["result"]["WKB_Levin_data"] is not None
                and p["data"]["result"]["WKB_Levin_data"]["max_depth"] >= 20
            )
            pkl = os.path.join(run_dir, data["pickle"])
            with open(pkl, "rb") as f:
                record = pickle.load(f)
            print(f"# input {data['pickle']} sha256 {sha256(pkl)}")
            calls = {c["index"]: c for c in data["calls"]}
            for regions in record["regions"]:
                call = calls[regions["index"]]
                if call["converged"]:
                    continue
                e = fval(call["abserr"])
                r = float(regions["abserr"].sum())
                req = max(call["atol"], call["rtol"] * abs(fval(call["value"])))
                k = kind(call, regions)
                tally[k] += 1
                if capped:
                    tally_capped[k] += 1
                share = (e - r) / e if e > 0 and e != float("inf") else float("nan")
                print(
                    f"   {row:<14s} {data['role']:<8s} {str(capped):<9s} "
                    f"{call['regime'] + '/' + call['group']:<14s} {k:<15s} "
                    f"{len(regions['abserr']):>7d} {int(regions['depth'].max()):>8d} "
                    f"{e / req:>9.3g} {r / req:>9.3g} {share:>9.4f}"
                )
        print(f"   kinds, every row: {dict(tally)}")
        print(f"   kinds, rows capped under {capped_tree}: {dict(tally_capped)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir")
    parser.add_argument("--tree", default="c")
    parser.add_argument("--capped-run", default=None)
    parser.add_argument("--capped-tree", default="p")
    args = parser.parse_args()
    main(
        args.run_dir,
        args.tree,
        args.capped_run if args.capped_run is not None else args.run_dir,
        args.capped_tree,
    )
