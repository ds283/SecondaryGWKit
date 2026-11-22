"""levin-phase-floor prompt 07a section 6, the user's references: 07's census check_captures.py (a record, not edited),
changed only where marked "prompt 07a": the census is named on the command line, AGAINST ``h`` (07a's census, the
run directory RUN_H, units h-...) or ``07`` (07's census, prov.RUN07, units g-...; for the captures made with the
parent's module); the call record's keys ``route_07a`` and ``route_07a_post`` (07a's diagnostics, absent from 07's
records) are not compared, and the harness check also allows the files 07a changes (prov.py, worker.py,
run_census.py, variant_loader.py), each one's two hashes recorded; the title names 07a.
Usage: PYTHONPATH=<tree h>:<scripts> python -B check_captures_07a.py CAPTURE_JSON SELECTION_JSON OUT_JSON h|07 [RUN_H]
07's text follows.

levin-phase-floor prompt 07 section 6, V4: 06m's census copy of 06l's check_captures.py (a record), changed only
where marked "prompt 07": the census the captures are checked against is 07's census g (refs_common.CENSUS07, its
checkpoint SHA-256 checked) in place of 06g's, and the title names 07. Every call-record key but ``elapsed`` (and the
route keys) is compared, 07's ``guard_07`` included; every region-array key, 07's flags included. 06l's text follows.

levin-phase-floor prompt 06l: copied from 06k's scripts/check_captures.py (a record; its copy is in
state/scripts_06k_reference/), changed only where marked "prompt 06l": the call record's new key ``route_06l`` is
not compared, as ``route_06k`` is not; and (after run levin-phase-floor-06l-k0partL-20261105T091942, where it
failed every capture on this alone) the harness check also allows ``run_census.py``, which worker.py hashes in SCRIPTS
but does not import, and which 06k changed (``--variant`` and the backstop) after its own captures ran; its two
hashes are recorded with the others'. 06k's text follows.

levin-phase-floor prompt 06k: copied from 06j's scripts/check_captures.py (a record, not edited; its copy is in
state/scripts_06j_reference/), changed only where marked "prompt 06k": the harness check allows the SCRIPTS files
06k changes or adds (prov.py, its tree and paths; worker.py, the variant loading and the route record;
variant_loader.py, new), each one's two hashes recorded; and the call record's new key ``route_06k`` (None when no
variant is loaded, as in the captures) is not compared. 06j's text follows.

levin-phase-floor prompt 06j: copied from 06g's references scripts/check_captures.py (a record, not edited; its
copy is in state/scripts_06g_reference/), changed only where marked "prompt 06j": the harness check compares
worker.py's SCRIPTS hashes file by file, every one equal to the census record's except prov.py, which 06j
changes in its tree and its paths (out/scripts_diff_against_06g.txt); prov.py's two hashes are recorded.
06g's text follows.

levin-phase-floor prompt 06g section 5, the references: every capture against the g census, bit for
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
from prov import CHECKPOINT07_SHA256, RUN07  # noqa: E402  prompt 07a

VOLATILE_CALL_KEYS = {"elapsed", "route_06k", "route_06l", "route_07a", "route_07a_post"}  # prompt 07a: route_07a*  # prompt 06k: route_06k, the new key; prompt 06l: route_06l


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
    against = sys.argv[4]  # prompt 07a
    if against == "07":
        run06g = RUN07
        ck = run06g / "checkpoint.jsonl"
        if RC.sha256(ck) != CHECKPOINT07_SHA256:
            raise SystemExit("07's census checkpoint has another SHA-256")
        ukey = "g_unit"
    elif against == "h":
        run06g = Path(sys.argv[5])
        ck = run06g / "checkpoint.jsonl"
        ukey = "unit"
    else:
        raise SystemExit("AGAINST is h or 07")
    for l in header(
        Path(__file__).resolve(),
        f"levin-phase-floor 07a: the reference captures against the census {against}, bit for bit (regions)",  # prompt 07a: the title
        [cap_p, sel_p, ck],
    ):
        print(l)
    sel = {(e["unit"], e["index"]): e for e in json.loads(sel_p.read_text())}
    caps = json.loads(cap_p.read_text())
    gdata = census_units_data(ck, {e[ukey] for e in sel.values()})  # prompt 07a
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
        gu = gdata[e[ukey]]  # prompt 07a
        if cu is None or len(gu) != 1:
            problems.append(
                f"unit data: capture {'absent' if cu is None else 'present'}, census done records {len(gu)}"
            )
        else:
            gu = gu[0]
            # the harness the capture ran is the census's, byte for byte (worker.py's SCRIPTS hashes)
            # prompt 06j: every SCRIPTS file equal except prov.py (06j's tree and paths)
            cs_, gs_ = cu.get("scripts_sha256") or {}, gu.get("scripts_sha256") or {}
            row["scripts_sha256_differing"] = sorted(
                k for k in set(cs_) | set(gs_) if cs_.get(k) != gs_.get(k)
            )
            row["prov_py_sha256"] = [cs_.get("prov.py"), gs_.get("prov.py")]
            row["scripts_sha256_equal"] = set(row["scripts_sha256_differing"]) <= {"prov.py", "worker.py", "variant_loader.py", "run_census.py"}  # prompt 06k; prompt 06l: run_census.py
            row["changed_scripts_sha256_06k"] = {k: [cs_.get(k), gs_.get(k)] for k in row["scripts_sha256_differing"]}  # prompt 06k
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
