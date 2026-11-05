"""levin-phase-floor prompt 06k, Gate K0: a variant's unit records against 06g's census records, bit for
bit. New in 06k. Read only. The comparison is state/predictions_and_rules_20261104T0915Z.md section 3's.

For each unit of RUN_DIR/units.jsonl (run_k0.py's output) it loads 06g's checkpoint record (by the
offset state/index_06g.json gives, the line's SHA-256 checked) and both region-pickle records (their
SHA-256 checked), and compares every key recursively: floats by bit pattern (``float.hex``; NaN by
repr), numpy arrays by dtype, shape and bytes, everything else by equality. Keys excluded as timing or
provenance (EXCLUDED, and every key ending ``_s``) and the new key ``route_06k`` are compared too, but
reported apart and never counted against the gate.

- **Clause 1** (MODE ``identity``): every unit equal on every key not excluded.
- **Clause 2** (MODE ``route``): per call, the calls whose route record reports no firing are compared
  on everything that belongs to the call: its summary (``calls``/``other_calls``), its entries of the
  unit's ``scan``, ``deep_lines`` and ``deep_lines_06e`` lines, and every pickle list entry carrying
  its index (region arrays, checked leaves, premise checks, the floor's terms, the declared terms, the
  deep lines, W3). A unit whose calls all report none is compared whole, as clause 1.

Writes OUT_JSON with every differing path, and prints a summary. Exits 1 if the clause fails.
Usage: PYTHONPATH=<tree>:<scripts> python -B compare_k0.py MODE RUN_DIR OUT_JSON
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402

INDEX = HERE.parent / "state" / "index_06g.json"
EXCLUDED = {"wall_s", "rebuild_s", "seconds", "elapsed", "compute_time", "tree_head", "tree_status_short",
            "scripts_sha256", "worker", "worker_pid", "worker_tree", "pickle", "pickle_offset", "pickle_length",
            "pickle_record_sha256", "modules", "rows_sha256", "w2_capture_sha256", "route_06k",
            "cache_built"}  # cache_built: state/k0_comparison_correction_20261104T0911Z.md


def excluded(key):
    return isinstance(key, str) and (key in EXCLUDED or key.endswith("_s"))


def fkey(x):
    return repr(x) if x != x else float(x).hex()


def diff(a, b, path, out, excl=False):
    """Append (path, excluded, why) for every difference between a and b."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a2, b2 = np.asarray(a), np.asarray(b)
        if a2.dtype != b2.dtype or a2.shape != b2.shape or a2.tobytes() != b2.tobytes():
            out.append((path, excl, f"array differs (dtype {a2.dtype}/{b2.dtype}, shape {a2.shape}/{b2.shape})"))
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            e = excl or excluded(k)
            if k not in a or k not in b:
                out.append((f"{path}.{k}", e, "key in one record only"))
                continue
            diff(a[k], b[k], f"{path}.{k}", out, e)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append((path, excl, f"length {len(a)} vs {len(b)}"))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out, excl)
        return
    if isinstance(a, float) and isinstance(b, float):
        if fkey(a) != fkey(b):
            out.append((path, excl, f"{a!r} vs {b!r}"))
        return
    if type(a) is not type(b) and not (isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer))):
        out.append((path, excl, f"type {type(a).__name__} vs {type(b).__name__}"))
        return
    if a != b:
        out.append((path, excl, f"{str(a)[:80]!r} vs {str(b)[:80]!r}"))


def load_pickle(path, offset, length, sha):
    import ray.cloudpickle as cp

    with open(path, "rb") as f:
        f.seek(offset)
        blob = f.read(length)
    if hashlib.sha256(blob).hexdigest() != sha:
        raise SystemExit(f"{path}@{offset}: sha256 differs")
    return cp.loads(blob)


def by_index(lst):
    out = {}
    for e in lst or []:
        if isinstance(e, dict) and "index" in e:
            out.setdefault(e["index"], []).append(e)
    return out


def firings(call):
    r = call.get("route_06k")
    if r is None:
        return None
    return int(r["n_firings"])


def main():
    mode, run_dir, out_p = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    if mode not in ("identity", "route"):
        raise SystemExit("MODE is identity or route")
    run06g = RC.census_run_06g()
    ck = run06g / "checkpoint.jsonl"
    hdr = header(Path(__file__).resolve(), f"levin-phase-floor 06k: Gate K0, {run_dir.name} against 06g's census ({mode})",
                 [run_dir / "units.jsonl", INDEX])
    for l in hdr:
        print(l)
    print(f"# 06g's checkpoint {ck} (each record's line checked by its SHA-256 in the index)")
    idx = json.loads(INDEX.read_text())["units"]
    mine = [json.loads(l) for l in open(run_dir / "units.jsonl")]
    results = []
    n_fail = 0
    n_calls_compared = n_calls_fired = n_units_whole = 0
    with open(ck, "rb") as fck:
        for m in mine:
            uid, d = m["unit"], m["data"]
            ent = idx[uid]
            fck.seek(ent["offset"])
            line = fck.read(ent["length"])
            if hashlib.sha256(line).hexdigest() != ent["line_sha256"]:
                raise SystemExit(f"{uid}: 06g's checkpoint line sha256 differs")
            g = json.loads(line)["data"]
            wdir = run_dir / m["worker"]
            prec = load_pickle(wdir / d["pickle"], d["pickle_offset"], d["pickle_length"], d["pickle_record_sha256"])
            gp = ent["pickle"]
            grec = load_pickle(run06g / gp[0], gp[1], gp[2], gp[3])
            row = {"unit": uid, "worker": m["worker"], "status": d["status"]}
            if mode == "identity":
                out = []
                diff(d, g, "data", out)
                diff(prec, grec, "pickle", out)
                row["differing"] = [p for p, e, _ in out if not e]
                row["differing_excluded"] = [p for p, e, _ in out if e]
                row["detail"] = [list(x) for x in out][:200]
                ok = not row["differing"]
                n_units_whole += 1
            else:
                calls = {c["index"]: c for c in (d.get("calls") or []) + (d.get("other_calls") or [])}
                gcalls = {c["index"]: c for c in (g.get("calls") or []) + (g.get("other_calls") or [])}
                fired = {i: firings(c) for i, c in calls.items()}
                if any(v is None for v in fired.values()):
                    raise SystemExit(f"{uid}: a call has no route record")
                row["firings_by_call"] = fired
                zero = [i for i, v in fired.items() if v == 0]
                out = []
                if set(calls) != set(gcalls):
                    out.append(("data.calls", False, f"call indices {sorted(calls)} vs {sorted(gcalls)}"))
                for i in sorted(zero):
                    n_calls_compared += 1
                    diff(calls.get(i), gcalls.get(i), f"call[{i}].summary", out)
                    for key in ("scan",):
                        a_ = [e for e in (d.get(key) or []) if isinstance(e, dict) and e.get("index") == i]
                        b_ = [e for e in (g.get(key) or []) if isinstance(e, dict) and e.get("index") == i]
                        diff(a_, b_, f"call[{i}].data.{key}", out)
                    for key in ("deep_lines", "deep_lines_06e"):
                        a_ = [e for e in ((d.get(key) or {}).get("lines") or []) if e.get("index") == i]
                        b_ = [e for e in ((g.get(key) or {}).get("lines") or []) if e.get("index") == i]
                        diff(a_, b_, f"call[{i}].data.{key}.lines", out)
                    for key in sorted(set(prec) | set(grec)):
                        if key == "route_06k":
                            continue
                        a_, b_ = prec.get(key), grec.get(key)
                        if isinstance(a_, list) or isinstance(b_, list):
                            ai = by_index(a_ if isinstance(a_, list) else []).get(i, [])
                            bi = by_index(b_ if isinstance(b_, list) else []).get(i, [])
                            diff(ai, bi, f"call[{i}].pickle.{key}", out)
                n_calls_fired += sum(1 for v in fired.values() if v > 0)
                if all(v == 0 for v in fired.values()):
                    n_units_whole += 1
                    diff(d, g, "data", out)
                    diff(prec, grec, "pickle", out)
                    row["compared_whole"] = True
                else:
                    row["compared_whole"] = False
                row["differing"] = sorted({p for p, e, _ in out if not e})
                row["differing_excluded"] = sorted({p for p, e, _ in out if e})
                row["detail"] = [list(x) for x in out][:200]
                ok = not row["differing"]
            row["ok"] = ok
            n_fail += not ok
            results.append(row)
            print(f"{uid} ({m['worker']}): {'EQUAL' if ok else 'DIFFERS'}"
                  + (f"; firings {row['firings_by_call']}" if mode == "route" else "")
                  + (f"; differing {row['differing'][:6]}" if not ok else "")
                  + f"; excluded keys differing {len(row['differing_excluded'])}", flush=True)
    excl_paths = {}
    for r in results:
        for p in r["differing_excluded"]:
            key = ".".join(x.split("[")[0] for x in p.split(".")[-2:])
            excl_paths[key] = excl_paths.get(key, 0) + 1
    doc = {"header": hdr, "mode": mode, "run_dir": str(run_dir), "n_units": len(results), "n_fail": n_fail,
           "n_calls_compared_clause2": n_calls_compared, "n_calls_fired": n_calls_fired,
           "n_units_compared_whole": n_units_whole, "excluded_keys_differing_count": excl_paths, "units": results}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=1, default=str))
    print(f"units {len(results)}; failing {n_fail}; units compared whole {n_units_whole}"
          + (f"; calls with no firing compared {n_calls_compared}; calls with a firing {n_calls_fired}" if mode == "route" else ""))
    print(f"excluded keys that differ (path tail: units): {excl_paths}")
    print(f"wrote {out_p} sha256 {RC.sha256(out_p)}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
