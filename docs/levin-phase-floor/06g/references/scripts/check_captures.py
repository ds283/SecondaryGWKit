"""levin-phase-floor prompt 06g section 5, the references: every capture against the g census, bit for
bit, beyond x1_capture.py's tuple. New in 06g.

x1_capture.py already compares each captured call's value and abserr (hex), num_regions, max_depth,
converged and label with the g census record (``capture_matches_census``). This script adds the
regions, as the brief asks ("value, abserr and regions"):
- **the region arrays.** The capture's worker re-ran the whole unit and wrote the unit's region
  record (``regions_<worker>.pkl``, at the offset its emitted unit data gives, SHA-256 checked); the
  census wrote the same record (the run directory's ``regions_*.pkl``, at the offset its checkpoint
  record gives, SHA-256 checked). For the captured call (``regions`` for a group call,
  ``other_regions`` for an analytic block, matched by index) every key is compared: arrays by
  dtype, shape and bytes (so NaNs compare by bit pattern), anything else by equality;
- **the harness.** The capture worker's ``scripts_sha256`` (worker.py's SCRIPTS) must equal the
  census record's: the capture ran the census's harness byte for byte;
- **the checkpoint call record.** The capture's emitted record of the call against the census's,
  every key; ``elapsed`` (a wall time) is volatile and reported but not compared. Any other
  difference is reported.
A call whose value, abserr or regions differ is a stop (the brief: "A capture that does not
reproduce its census record is a stop"); the script exits 1 if any does. Reads only.

Usage: PYTHONPATH=<tree g>:<this directory> python -B check_captures.py CAPTURE_JSON SELECTION_JSON OUT_JSON
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402

VOLATILE_CALL_KEYS = {"elapsed"}


def load_blob(path, offset, length, sha):
    with open(path, "rb") as f:
        f.seek(offset)
        blob = f.read(length)
    if hashlib.sha256(blob).hexdigest() != sha:
        raise RuntimeError(f"{path}@{offset}: sha256 differs")
    import ray.cloudpickle as cp  # the records were written by harness._GroupPickler

    return cp.loads(blob)


def same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a.hex() == b.hex() if a == a or b == b else repr(a) == repr(b)
    return a == b


def census_units_data(checkpoint, units):
    """{unit: [its ``done`` records' data]} for ``units``, in one pass over the census checkpoint."""
    out = {u: [] for u in units}
    with open(checkpoint) as f:
        for line in f:
            head = line[:300]
            if not any(f'"{u}"' in head for u in units):
                continue
            r = json.loads(line)
            if r.get("unit") in out and r["data"].get("status") == "done":
                out[r["unit"]].append(r["data"])
    return out


def capture_unit_data(wdir, uid):
    for line in open(Path(wdir) / "emit.jsonl"):
        m = json.loads(line)
        if m["kind"] == "unit" and m["data"]["unit"] == uid:
            return m["data"]
    return None


def region_entry(rec, kind, index):
    key = "regions" if kind == "group" else "other_regions"
    hits = [e for e in rec[key] if e["index"] == index]
    return hits[0] if len(hits) == 1 else None


def call_entry(data, kind, index):
    key = "calls" if kind == "group" else "other_calls"
    hits = [c for c in data[key] if c["index"] == index]
    return hits[0] if len(hits) == 1 else None


def main():
    cap_p, sel_p, out_p = (Path(a) for a in sys.argv[1:4])
    run06g = RC.census_run_06g()
    ck = run06g / "checkpoint.jsonl"
    for l in header(
        Path(__file__).resolve(),
        "levin-phase-floor 06g: the references' captures against the g census, bit for bit (regions)",
        [cap_p, sel_p, ck],
    ):
        print(l)
    sel = {(e["unit"], e["index"]): e for e in json.loads(sel_p.read_text())}
    caps = json.loads(cap_p.read_text())
    gdata = census_units_data(ck, {e["g_unit"] for e in sel.values()})
    results = []
    n_stop = 0
    for c in caps:
        e = sel[(c["unit"], c["index"])]
        row = {
            "sid": e["sid"],
            "unit": c["unit"],
            "index": c["index"],
            "kind": e["kind"],
        }
        problems = []
        if not c.get("capture_matches_census"):
            problems.append(
                "x1_capture: value/abserr/num_regions/max_depth/converged/label differ, or no capture"
            )
        cu = capture_unit_data(c["dir"], c["unit"]) if c.get("dir") else None
        gu = gdata[e["g_unit"]]
        if cu is None or len(gu) != 1:
            problems.append(
                f"unit data: capture {'absent' if cu is None else 'present'}, census done records {len(gu)}"
            )
        else:
            gu = gu[0]
            # the harness the capture ran is the census's, byte for byte (worker.py's SCRIPTS hashes)
            row["scripts_sha256_equal"] = cu.get("scripts_sha256") == gu.get(
                "scripts_sha256"
            )
            if not row["scripts_sha256_equal"]:
                problems.append(
                    "the capture's harness scripts differ from the census's (scripts_sha256)"
                )
            crec = load_blob(
                Path(c["dir"]) / cu["pickle"],
                cu["pickle_offset"],
                cu["pickle_length"],
                cu["pickle_record_sha256"],
            )
            grec = load_blob(
                run06g / gu["pickle"],
                gu["pickle_offset"],
                gu["pickle_length"],
                gu["pickle_record_sha256"],
            )
            cr = region_entry(crec, e["kind"], c["index"])
            gr = region_entry(grec, e["kind"], c["index"])
            if cr is None or gr is None:
                problems.append("region entry missing")
            else:
                diff = sorted(
                    k
                    for k in set(cr) | set(gr)
                    if k not in cr or k not in gr or not same(cr[k], gr[k])
                )
                row["region_keys_compared"] = sorted(set(cr) | set(gr))
                row["region_keys_differing"] = diff
                row["n_regions"] = int(len(gr["start"]))
                if diff:
                    problems.append(f"region arrays differ: {diff}")
            cc = call_entry(cu, e["kind"], c["index"])
            gc = call_entry(gu, e["kind"], c["index"])
            if cc is None or gc is None:
                problems.append("call record missing")
            else:
                diff = sorted(
                    k
                    for k in set(cc) | set(gc)
                    if k not in VOLATILE_CALL_KEYS
                    and (k not in cc or k not in gc or not same(cc[k], gc[k]))
                )
                row["call_keys_differing"] = diff
                row["call_elapsed"] = [cc.get("elapsed"), gc.get("elapsed")]
                if diff:
                    row["call_differences"] = {k: [cc.get(k), gc.get(k)] for k in diff}
        row["problems"] = problems
        stop = bool(problems)
        n_stop += stop
        row["reproduces_census"] = not stop
        results.append(row)
        print(
            f"{row['sid']} {row['unit']} #{row['index']} {row['kind']}: "
            + (
                "REPRODUCES (value, abserr, label, converged, regions bit for bit)"
                if not stop
                else "DOES NOT REPRODUCE: " + "; ".join(problems)
            )
            + (
                f"; other call-record keys differing: {row.get('call_keys_differing')}"
                if row.get("call_keys_differing")
                else ""
            )
        )
    Path(out_p).write_text(json.dumps(results, indent=1, default=str))
    print(
        f"{len(results) - n_stop} of {len(results)} captures reproduce the g census bit for bit; wrote {out_p} sha256 {RC.sha256(out_p)}"
    )
    return 0 if n_stop == 0 and len(results) == len(sel) else 1


if __name__ == "__main__":
    sys.exit(main())
