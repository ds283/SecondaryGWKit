"""levin-phase-floor prompt 06g section 5, the references: one reading of one captured call by
``tools.filon_reference``, in a process of its own. New in 06g.

The tool's command line (``python -m tools.filon_reference PICKLE --tolerance T --unpickler
ray.cloudpickle --workers 6``) takes the four readings in one process, ``reference()``: for impl in
("qawo", "fcc"), for res in (1, 2), ``reading(call, impl, res, route=None, edges=None, workers=6,
call_source=("tools.filon_reference", "_load_pickle", (PICKLE, "ray.cloudpickle")))``, then their
scatter. This script makes exactly one of those ``reading`` calls, with exactly those arguments, so
that the driver (run_refs.py) can stop a reading at 30 minutes (part 1) or check it with its
progress (part 2, addendum 3: ``--progress``) and record each reading's wall time
(the brief); a reading does not depend on the process it runs in (the tool's determinism test, and
run_refs.py's command-line check on two of 06i's samples). The pickle is loaded as the command line
loads it (``tools.filon_reference._load_pickle(PICKLE, "ray.cloudpickle")``).

Writes OUT_JSON in 06i's form (sample, route, impl, res, config, workers, the value as
``value_hex``, ``value_30`` and ``imag_30``, the route's own terms, ``wall_s``), with a provenance
header, the call's identity, the pickle's and the tool's SHA-256, and whether Ray was initialised
(it must not be). Refuses to overwrite.

Usage: PYTHONPATH=<tree g>:<this directory> python -B one_reading.py PICKLE IMPL RES OUT_JSON
       --sid SID --unit UNIT --index INDEX --rtol RTOL [--workers 6]
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _indexed_chunk(ij):
    """Addendum 4/5: one chunk, with its index, for the out-of-order progress count."""
    import tools.filon_reference as F

    i, job = ij
    return i, F._run_chunk(job)


def _install_progress_unordered(F, path):
    """Addendum 4 section 2 and addendum 5 (the user's ruling, 2026-11-02): as
    ``_install_progress``, but the chunks come back by ``pool.imap_unordered`` (``chunksize=1``),
    each with its index, so that the count in PATH is every chunk finished, in any order; the
    results are put back in chunk order before the tool sums them, so the tool sees the list
    ``pool.map`` returns. Each chunk runs the tool's own ``_run_chunk`` in the tool's own
    initialised worker. Checked bit for bit on the batch's re-reads before any other reading.
    """
    import multiprocessing

    t0 = time.time()

    def note(done, total):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "done": done,
                    "total": total,
                    "elapsed_s": time.time() - t0,
                    "start_unix": t0,
                    "order": "unordered",
                }
            )
        )
        tmp.replace(path)

    def run_jobs(jobs, state, workers, call_source, route, edges):
        note(0, len(jobs))
        if workers <= 1:
            F._WORKER["state"] = state
            out = []
            for job in jobs:
                out.append(F._run_chunk(job))
                note(len(out), len(jobs))
            return out
        if call_source is None:
            raise ValueError("workers > 1 needs call_source = (module, function, args)")
        ctx = multiprocessing.get_context("spawn")
        got = [None] * len(jobs)
        n = 0
        with ctx.Pool(
            workers, initializer=F._worker_init, initargs=(call_source, route, edges)
        ) as pool:
            for i, r in pool.imap_unordered(
                _indexed_chunk, list(enumerate(jobs)), chunksize=1
            ):
                got[i] = r
                n += 1
                note(n, len(jobs))
        if n != len(jobs) or any(g is None for g in got):
            raise RuntimeError("a chunk did not come back")
        return got

    F._run_jobs = run_jobs


def _install_progress(F, path):
    """Addendum 3 (the user's ruling, 2026-11-01: a reading past 30 minutes is checked, with its
    progress, not stopped): ``tools.filon_reference._run_jobs`` replaced, in this process only, by
    the same thing with ``pool.imap`` (ordered, ``chunksize=1``) in place of ``pool.map``, so that
    each finished chunk is counted into PATH (``{"done", "total", "elapsed_s", "start_unix"}``) as
    it comes back. The chunks, their order, the function each runs, the workers' initialiser and
    the results list are the tool's own, so the reading's value is unchanged (checked bit for bit
    against part 1's readings before part 2 launched). The tool's file is not touched.
    """
    import multiprocessing

    t0 = time.time()

    def note(done, total):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "done": done,
                    "total": total,
                    "elapsed_s": time.time() - t0,
                    "start_unix": t0,
                }
            )
        )
        tmp.replace(path)

    def run_jobs(jobs, state, workers, call_source, route, edges):
        note(0, len(jobs))
        if workers <= 1:
            F._WORKER["state"] = state
            out = []
            for job in jobs:
                out.append(F._run_chunk(job))
                note(len(out), len(jobs))
            return out
        if call_source is None:
            raise ValueError("workers > 1 needs call_source = (module, function, args)")
        ctx = multiprocessing.get_context("spawn")
        out = []
        with ctx.Pool(
            workers, initializer=F._worker_init, initargs=(call_source, route, edges)
        ) as pool:
            for r in pool.imap(F._run_chunk, jobs, chunksize=1):
                out.append(r)
                note(len(out), len(jobs))
        return out

    F._run_jobs = run_jobs


def main():
    import refs_common as RC
    from prov import header

    ap = argparse.ArgumentParser()
    ap.add_argument("pickle")
    ap.add_argument("impl", choices=RC.IMPLS)
    ap.add_argument("res", type=int, choices=RC.RESS)
    ap.add_argument("out_json")
    ap.add_argument("--sid", required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--rtol", required=True)
    ap.add_argument("--workers", type=int, default=RC.TOOL_WORKERS)
    ap.add_argument(
        "--progress", default=None
    )  # addendum 3: chunks done, for the driver's check
    ap.add_argument(
        "--progress-order", choices=("ordered", "unordered"), default="ordered"
    )  # addendum 4/5: count chunks finished out of order
    ap.add_argument(
        "--rerun", action="store_true"
    )  # addendum 4/5: the rerun batch's marker
    a = ap.parse_args()
    if a.workers > 6:
        raise SystemExit("at most six workers")
    out = Path(a.out_json)
    if out.exists():
        raise SystemExit(f"{out} exists; not overwritten")
    t_proc = time.perf_counter()
    import ray
    import tools.filon_reference as F

    pkl = str(Path(a.pickle).resolve())
    hdr = header(
        Path(__file__).resolve(),
        f"levin-phase-floor 06g: Filon reading {a.sid} {a.impl} res {a.res} (tools.filon_reference)",
        [Path(pkl), Path(F.__file__)],
    )
    if a.progress and a.progress_order == "unordered":
        _install_progress_unordered(F, Path(a.progress))
    elif a.progress:
        _install_progress(F, Path(a.progress))
    call = F._load_pickle(pkl, "ray.cloudpickle")
    ray_after_load = ray.is_initialized()
    route = F.route_of(call)
    r = F.reading(
        call,
        a.impl,
        a.res,
        route=None,
        edges=None,
        workers=a.workers,
        call_source=("tools.filon_reference", "_load_pickle", (pkl, "ray.cloudpickle")),
    )
    doc = {
        "header": hdr,
        "sample": a.sid,
        "unit": a.unit,
        "index": a.index,
        "rtol": a.rtol,
        "pickle": pkl,
        "pickle_sha256": RC.sha256(pkl),
        "tool": F.__file__,
        "tool_sha256": RC.sha256(F.__file__),
        "workers": a.workers,
        "route_of": route,
        **{k: v for k, v in r.items() if k != "value_mp"},
        "process_wall_s": time.perf_counter() - t_proc,
        "rerun": bool(a.rerun),
        "progress_order": a.progress_order if a.progress else None,
        "ray_initialized": [ray_after_load, ray.is_initialized()],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(
        json.dumps(
            {
                "sample": a.sid,
                "route": r["route"],
                "impl": a.impl,
                "res": a.res,
                "value_hex": r["value_hex"],
                "value_30": r["value_30"],
                "wall_s": r["wall_s"],
                "ray_initialized": doc["ray_initialized"],
            }
        )
    )
    if any(doc["ray_initialized"]):
        raise SystemExit("Ray was initialised")
    return 0


if __name__ == "__main__":
    sys.exit(main())
