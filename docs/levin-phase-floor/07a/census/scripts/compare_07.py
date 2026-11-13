"""levin-phase-floor prompt 07 section 6: units run by this harness against 06m's census
(levin-phase-floor-06m-census-20261106T102153, checkpoint ea5bc3ab...), bit for bit, by 06m's comparison with only
07's new keys set aside. New in 07, after 06m's compare_06m.py (a record, not edited), whose matching and evidence it
follows, and which applies 06l's compare_k0.diff unchanged: compare_k0's ``EXCLUDED`` (timing and provenance, 06m's
set; every key ending ``_s`` likewise) is extended at import by NEW_KEYS_07 below, which compare_k0.excluded() reads
at call time, so the diff itself is 06m's. Read only.

**The keys set aside, and why each** (NEW_KEYS_07): every one is a key 07 adds; none occurs in 06m's records (checked
on every unit compared: ``new_keys_in_06m_records`` must be 0).
- ``guard_07`` (each call's summary; the pickle's per-call list): prompt 07 section 1.5's diagnostics and Rule A's
  ``location_skipped_phases``, recorded by this harness;
- ``window_07``, ``guarded_07``, ``guard_at_depth_max_07`` (each returned leaf's region arrays): section 1.5's flags;
- ``scan_07``, ``scan_07_error`` (unit; pickle) and ``scan_07_s`` (timing, by the ``_s`` rule): section 6's scan;
- ``stationary_points``, ``window_pieces``, ``guarded_regions``, ``guard_at_depth_max``: the four counts section 1.6
  adds to QuadSourceIntegral's group records, which reach the pickle's ``result_full.WKB_Levin_subintervals[].groups[]``
  at g (absent at p and in 06m's records).
**Normalised, not excluded** (mode ``whole`` only, a unit run at the parent p): its ``unit`` (``p-X``) and ``tree``
(``p``), in the checkpoint record and the pickle, are compared as 06m's census names them (``g-X``, ``g``): 06m's census
ran 06m's code at a tree it named ``g``; p is that code. Every other key is compared as recorded.

Modes:
- ``whole RUN_DIR OUT_JSON``: every unit of a run (run_k0.py's ``units.jsonl``, or a census run's ``checkpoint.jsonl``)
  compared whole with 06m's unit, as 06m's Gate M0 (V0's second clause). Module evidence: each unit's
  ``AdaptiveLevin.levin_quadrature`` is its tree's file, with no foreign module and no variant; each worker's hello shows
  its tree's module SHA-256 (g ``4e5da371...``, p ``beaf0e23...``), its tree at its SHA and clean; each bye, no Ray.
- ``calls RUN_DIR OUT_JSON``: per call (V1, and stop 3's comparison): every Levin call, group (``calls``) and analytic
  (``other_calls``), in which the unit's ``guard_07`` reports no located point and no guarded leaf (``locates`` and
  ``guards`` both false) is compared with 06m's call on everything that belongs to it: its summary, its entries of the
  unit's ``scan``, ``deep_lines`` and ``deep_lines_06e`` lines, every pickle list entry carrying its index (its region
  arrays, W3, the deep lines, the floor's terms, the declared terms, its checked leaves, its premise checks), and, for a
  1e-7 group call, its entry of the pickle's ``scan`` (positional, as the worker writes it). A unit all of whose calls
  are such is also compared whole, and its ``result.total`` and ``result.total_abserr`` (hex) are checked equal. Any
  other call (located or guarded) is counted and listed, not compared.
- ``det RUN_DIR_A RUN_DIR_B OUT_JSON``: the units two run_k0.py runs share, compared whole with compare_k0's EXCLUDED
  alone (timing and provenance): 07's new keys compared too (V0's third clause).

Floats are compared by bit pattern (``float.hex``; NaN by repr), arrays by dtype, shape and bytes.
Writes OUT_JSON (provenance header, counts, inventory, per unit its counts and every differing path) and prints a
summary. Exits 1 unless every comparison it makes is equal and every evidence check holds.

Usage: PYTHONPATH=<tree g>:<scripts> python -B compare_07.py whole|calls RUN_DIR OUT_JSON [--processes N]
       PYTHONPATH=<tree g>:<scripts> python -B compare_07.py det RUN_DIR_A RUN_DIR_B OUT_JSON
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_k0 as CK  # noqa: E402  (06l's, unchanged)
from prov import CHECKPOINT06M_SHA256, LQ_SHA256_BY_TREE, RUN06M, TREES, header, sha256  # noqa: E402

EXCLUDED_06M = frozenset(CK.EXCLUDED)  # 06m's set, as compare_k0.py has it
NEW_KEYS_07 = frozenset({
    "guard_07", "window_07", "guarded_07", "guard_at_depth_max_07", "scan_07", "scan_07_error",
    "stationary_points", "window_pieces", "guarded_regions", "guard_at_depth_max",
})
INDEX = HERE.parent / "state" / "index_06m.json"


def set_excluded(with_new):
    """compare_k0.excluded() reads compare_k0.EXCLUDED at call time."""
    CK.EXCLUDED = set(EXCLUDED_06M | (NEW_KEYS_07 if with_new else frozenset()))


def norm(p):
    return re.sub(r"\[\d+\]", "[]", p)


def span_key(x_span):
    return tuple(e["hex"] if isinstance(e, dict) else float(e).hex() for e in x_span)


def call_keys(d):
    return [(kind, c["index"], c["notify_label"], span_key(c["x_span"]))
            for kind, key in (("group", "calls"), ("analytic", "other_calls")) for c in d.get(key) or []]


def region_keys(p):
    return [(kind, r["index"], span_key(r["x_span"]))
            for kind, key in (("group", "regions"), ("analytic", "other_regions")) for r in p.get(key) or []]


def count_new_keys(x):
    """How many times a NEW_KEYS_07 name occurs as a dict key in x."""
    n = 0
    if isinstance(x, dict):
        for k, v in x.items():
            n += (k in NEW_KEYS_07) + count_new_keys(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            n += count_new_keys(v)
    return n


def excluded_paths(x, path, out):
    if isinstance(x, dict):
        for k, v in x.items():
            p = f"{path}.{k}"
            if CK.excluded(k):
                out.add(p)
            excluded_paths(v, p, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            excluded_paths(v, f"{path}[]", out)


def load_06m(ent):
    with open(RUN06M / "checkpoint.jsonl", "rb") as f:
        f.seek(ent["offset"])
        line = f.read(ent["length"])
    if hashlib.sha256(line).hexdigest() != ent["line_sha256"]:
        raise SystemExit(f"06m's checkpoint line sha256 differs at offset {ent['offset']}")
    g = json.loads(line)["data"]
    gp = ent["pickle"]
    grec = CK.load_pickle(RUN06M / gp[0], gp[1], gp[2], gp[3])
    return g, grec


def load_mine(d, pdir):
    return CK.load_pickle(Path(pdir) / d["pickle"], d["pickle_offset"], d["pickle_length"], d["pickle_record_sha256"])


def uid_06m(uid):
    return "g-" + uid.split("-", 1)[1]


def by_index(lst):
    out = {}
    for e in lst or []:
        if isinstance(e, dict) and "index" in e:
            out.setdefault(e["index"], []).append(e)
    return out


def tree_of(uid):
    return uid.split("-", 1)[0]


def unit_evidence(d, tree):
    mods = d.get("modules") or {}
    lq_file = str(TREES[tree][0] / "AdaptiveLevin" / "levin_quadrature.py")
    return {
        "lq_file_is_tree": (mods.get("files") or {}).get("AdaptiveLevin.levin_quadrature") == lq_file,
        "all_in_tree": bool(mods.get("all_in_tree")),
        "no_foreign": not mods.get("foreign"),
        "no_variant_root": "variant" not in (mods.get("counts_by_root") or {}),
    }


def one_whole(args):
    """Mode whole: one unit against 06m's, whole (new keys set aside; p's unit and tree normalised)."""
    uid, line, pdir, ent = args
    set_excluded(True)
    d = json.loads(line)["data"]
    tree = tree_of(uid)
    row = {"unit": uid, "unit_06m": uid_06m(uid), "status": d["status"]}
    g, grec = load_06m(ent)
    row["status_06m"] = g["status"]
    if d["status"] != "done" or g["status"] != "done":
        row.update(ok=False, differing=["status"], excl_present=[], excl_differing=[])
        return row
    prec = load_mine(d, pdir)
    d2, prec2 = dict(d), dict(prec)
    normalised = {}
    for name, rec in (("data", d2), ("pickle", prec2)):
        if rec.get("unit") == uid and uid != uid_06m(uid):
            rec["unit"] = uid_06m(uid)
            normalised[f"{name}.unit"] = [uid, uid_06m(uid)]
    if d2.get("tree") != "g":
        normalised["data.tree"] = [d2.get("tree"), "g"]
        d2["tree"] = "g"
    out = []
    matched = call_keys(d2) == call_keys(g) and region_keys(prec2) == region_keys(grec)
    if not matched:
        out.append(("match", False, "calls or region records do not match"))
    CK.diff(d2, g, "data", out)
    CK.diff(prec2, grec, "pickle", out)
    pres = set()
    for nm, rec in (("data", d2), ("data", g), ("pickle", prec2), ("pickle", grec)):
        excluded_paths(rec, nm, pres)
    ev = unit_evidence(d, tree)
    differing = sorted({p for p, e, _ in out if not e})
    row.update(
        ok=matched and not differing and all(ev.values()),
        normalised=normalised,
        differing=differing[:200],
        detail=[list(x) for x in out if not x[1]][:50],
        excl_present=sorted(pres),
        excl_differing=sorted({norm(p) for p, e, _ in out if e}),
        new_keys_in_06m_records=count_new_keys(g) + count_new_keys(grec),
        n_calls=len(call_keys(d2)),
        n_group_calls=sum(1 for k in call_keys(d2) if k[0] == "group"),
        n_regions=len(region_keys(prec2)),
        n_regions_returned=int(sum(len(r.get("start", [])) for r in (prec.get("regions") or []) + (prec.get("other_regions") or []))),
        evidence=ev,
    )
    return row


def _call_parts(d, prec, kind, i, pos):
    """Everything that belongs to call i of a unit record d and its pickle record prec, nested by the record's own
    keys, so that compare_k0.excluded() applies to them as in a whole-unit diff."""
    key = "calls" if kind == "group" else "other_calls"
    parts = {"summary": [c for c in (d.get(key) or []) if c["index"] == i], "data": {}, "pickle": {}}
    parts["data"]["scan"] = [e for e in (d.get("scan") or []) if isinstance(e, dict) and e.get("index") == i]
    for k in ("deep_lines", "deep_lines_06e"):
        parts["data"][k] = {"lines": [e for e in ((d.get(k) or {}).get("lines") or []) if e.get("index") == i]}
    for k in sorted(prec):
        v = prec.get(k)
        if k == "scan":
            # 06a's scan in the pickle: one entry per group call, in the group calls' order (positional)
            if kind == "group" and isinstance(v, list):
                parts["pickle"]["scan"] = [v[pos]] if 0 <= pos < len(v) else ["<missing>"]
            continue
        if isinstance(v, list):
            parts["pickle"][k] = by_index(v).get(i, [])
    return parts


def one_calls(args):
    """Mode calls: per call, the calls that locate nothing and guard nothing against 06m's."""
    uid, line, pdir, ent = args
    set_excluded(True)
    d = json.loads(line)["data"]
    row = {"unit": uid, "unit_06m": uid_06m(uid), "status": d["status"], "rtol": d.get("rtol")}
    g, grec = load_06m(ent)
    row["status_06m"] = g["status"]
    if d["status"] != "done" or g["status"] != "done":
        row.update(ok=False, differing=["status"], n_calls=0, n_clean=0, n_located=0, n_guarded=0, located=[],
                   all_clean=False)
        return row
    prec = load_mine(d, pdir)
    out = []
    ck_m, ck_g = call_keys(d), call_keys(g)
    if ck_m != ck_g or region_keys(prec) != region_keys(grec):
        out.append(("match", False, f"calls {len(ck_m)}/{len(ck_g)} equal {ck_m == ck_g}; regions equal "
                                    f"{region_keys(prec) == region_keys(grec)}"))
    n_clean = 0
    located = []
    guard_errors = []
    clean_kinds = {"group": 0, "analytic": 0}
    pos_of = {c["index"]: j for j, c in enumerate(d.get("calls") or [])}
    pos_of_g = {c["index"]: j for j, c in enumerate(g.get("calls") or [])}
    for kind, i, label, _span in ck_m:
        key = "calls" if kind == "group" else "other_calls"
        s = next(c for c in d[key] if c["index"] == i)
        g7 = s.get("guard_07")
        if not isinstance(g7, dict) or "error" in g7 or not g7.get("present"):
            guard_errors.append([kind, i])
            continue
        if g7["locates"] or g7["guards"]:
            located.append({"kind": kind, "index": i, "label": label, "n_points": g7["n_points"],
                            "kinds": g7["n_points_by_kind"], "num_guarded_regions": g7["num_guarded_regions"],
                            "num_guard_at_depth_max": g7["num_guard_at_depth_max"],
                            "skipped": g7["location_skipped_phases"]})
            continue
        n_clean += 1
        clean_kinds[kind] += 1
        a = _call_parts(d, prec, kind, i, pos_of.get(i, -1))
        b = _call_parts(g, grec, kind, i, pos_of_g.get(i, -1))
        CK.diff(a, b, f"call[{kind}:{i}]", out)
    all_clean = not located and not guard_errors
    row["all_clean"] = all_clean
    if all_clean:
        CK.diff(d, g, "data", out)
        CK.diff(prec, grec, "pickle", out)
        t_m, t_g = d.get("result") or {}, g.get("result") or {}
        row["total_equal"] = all(
            (t_m.get(k) or {}).get("hex") == (t_g.get(k) or {}).get("hex") for k in ("total", "total_abserr"))
        if not row["total_equal"]:
            out.append(("data.result.total|total_abserr", False, "row totals differ"))
    else:
        t_m, t_g = d.get("result") or {}, g.get("result") or {}
        row["total"] = [(t_m.get("total") or {}).get("hex"), (t_g.get("total") or {}).get("hex")]
        row["total_abserr"] = [(t_m.get("total_abserr") or {}).get("hex"), (t_g.get("total_abserr") or {}).get("hex")]
        row["total_converged"] = [t_m.get("total_converged"), t_g.get("total_converged")]
    differing = sorted({p for p, e, _ in out if not e})
    row.update(
        ok=not differing and not guard_errors,
        differing=differing[:200],
        detail=[list(x) for x in out if not x[1]][:50],
        excl_differing=sorted({norm(p) for p, e, _ in out if e}),
        new_keys_in_06m_records=count_new_keys(g) + count_new_keys(grec),
        n_calls=len(ck_m),
        n_clean=n_clean,
        n_clean_by_kind=clean_kinds,
        n_located=sum(1 for x in located if x["n_points"]),
        n_guarded=sum(1 for x in located if x["num_guarded_regions"] or x["num_guard_at_depth_max"]),
        located=located,
        guard_errors=guard_errors,
        evidence=unit_evidence(d, tree_of(uid)),
    )
    return row


def worker_evidence(run_dir):
    out = {}
    if (run_dir / "units.jsonl").exists():
        for wdir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            hello = bye = None
            for line in open(wdir / "emit.jsonl"):
                m = json.loads(line)
                if m["kind"] == "hello":
                    hello = m["evidence"]
                elif m["kind"] == "bye":
                    bye = m
            out[wdir.name] = (hello, bye)
    else:
        for hp in sorted(run_dir.glob("worker_*_hello.json")):
            name = hp.name[len("worker_"):-len("_hello.json")]
            bp = run_dir / f"worker_{name}_bye.json"
            out[name] = (json.loads(hp.read_text())["evidence"], json.loads(bp.read_text()) if bp.exists() else None)
    checks = {}
    for name, (h, b) in out.items():
        tree = None if h is None else h.get("tree")
        want = LQ_SHA256_BY_TREE.get(tree)
        checks[name] = {
            "tree": tree,
            "hello": h is not None,
            "levin_quadrature_sha256": None if h is None else h.get("levin_quadrature_sha256"),
            "lq_is_tree_module": h is not None and want is not None and h.get("levin_quadrature_sha256") == want
            and h.get("levin_quadrature_tree_file_sha256") == want,
            "lq_file": None if h is None else h["files"].get("AdaptiveLevin.levin_quadrature"),
            "variant_none": h is not None and h.get("variant_06k") is None,
            "all_in_tree": h is not None and bool(h.get("all_in_tree")),
            "tree_head_clean": h is not None and h.get("head") == TREES[tree][1] and h.get("status_short", "x").strip() == "",
            "bye": b is not None,
            "ray_not_initialized": b is not None and b.get("ray_initialized") is False,
            "no_foreign_modules": b is not None and not b.get("modules_foreign"),
        }
    return checks


def jobs_of(run_dir, idx):
    src = run_dir / "units.jsonl"
    k0 = src.exists()
    if not k0:
        src = run_dir / "checkpoint.jsonl"
    jobs, seen, skipped = [], set(), []
    with open(src, "rb") as f:
        for line in f:
            r = json.loads(line)
            uid = r["unit"]
            if not re.fullmatch(r"[gp]-1e-[78]-\d+-\d+", uid):
                skipped.append(uid)  # e.g. an over-budget pass's record, compared apart
                continue
            if uid in seen:
                raise SystemExit(f"{uid} twice in {src}")
            seen.add(uid)
            if uid_06m(uid) not in idx:
                raise SystemExit(f"{uid_06m(uid)} is not in 06m's census")
            pdir = run_dir / r["worker"] if k0 else run_dir
            jobs.append((uid, line, str(pdir), idx[uid_06m(uid)]))
    return src, jobs, skipped


def run_det(a):
    set_excluded(False)
    ra, rb = Path(a.run_dir).resolve(), Path(a.run_dir_b).resolve()
    hdr = header(Path(__file__).resolve(), f"levin-phase-floor 07: Gate V0 (c), {ra.name} against {rb.name}",
                 [ra / "units.jsonl", rb / "units.jsonl", HERE / "compare_k0.py"])
    for l in hdr:
        print(l, flush=True)

    def load(run):
        recs = {}
        for line in open(run / "units.jsonl", "rb"):
            r = json.loads(line)
            recs[r["unit"]] = (r["data"], load_mine(r["data"], run / r["worker"]))
        return recs

    A, B = load(ra), load(rb)
    rows = []
    for uid in sorted(set(A) | set(B)):
        if uid not in A or uid not in B:
            rows.append({"unit": uid, "ok": False, "differing": ["unit in one run only"]})
            continue
        (da, pa), (db, pb) = A[uid], B[uid]
        out = []
        CK.diff(da, db, "data", out)
        CK.diff(pa, pb, "pickle", out)
        differing = sorted({p for p, e, _ in out if not e})
        g7 = [c.get("guard_07") for c in (da.get("calls") or []) + (da.get("other_calls") or [])]
        rows.append({"unit": uid, "status": [da["status"], db["status"]], "ok": not differing and da["status"] == "done",
                     "differing": differing[:200], "detail": [list(x) for x in out if not x[1]][:50],
                     "excl_differing": sorted({norm(p) for p, e, _ in out if e}),
                     "n_calls": len(g7), "n_calls_locating": sum(1 for x in g7 if isinstance(x, dict) and x.get("locates")),
                     "points": [(i, x.get("n_points_by_kind")) for i, x in enumerate(g7) if isinstance(x, dict) and x.get("locates")],
                     "wall_s": [da.get("wall_s"), db.get("wall_s")]})
    wev = {**{f"A:{k}": v for k, v in worker_evidence(ra).items()}, **{f"B:{k}": v for k, v in worker_evidence(rb).items()}}
    wev_ok = bool(wev) and all(all(v for k, v in c.items() if k not in ("levin_quadrature_sha256", "lq_file", "tree")) for c in wev.values())
    n_fail = sum(1 for r in rows if not r["ok"])
    doc = {"header": hdr, "mode": "det", "excluded": sorted(CK.EXCLUDED) + ["*_s"], "n_units": len(rows),
           "n_units_identical": len(rows) - n_fail, "worker_evidence_all_hold": wev_ok, "worker_evidence": wev, "units": rows}
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(doc, indent=1, default=str))
    for r in rows:
        print(f"{r['unit']}: {'IDENTICAL' if r['ok'] else 'DIFFERS'}; status {r.get('status')}; calls {r.get('n_calls')}; "
              f"locating {r.get('n_calls_locating')} {r.get('points')}; wall {r.get('wall_s')}; differing {r['differing'][:10]}; "
              f"excluded keys differing {r.get('excl_differing')}")
    print(f"worker evidence all hold {wev_ok}")
    print(f"wrote {a.out_json} sha256 {sha256(a.out_json)}")
    return 0 if n_fail == 0 and wev_ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("whole", "calls", "det"))
    ap.add_argument("run_dir")
    ap.add_argument("rest", nargs="+")
    ap.add_argument("--processes", type=int, default=1)
    a = ap.parse_args()
    if a.processes > 6:
        raise SystemExit("at most six processes")
    if a.mode == "det":
        if len(a.rest) != 2:
            raise SystemExit("det RUN_DIR_A RUN_DIR_B OUT_JSON")
        a.run_dir_b, a.out_json = a.rest
        return run_det(a)
    if len(a.rest) != 1:
        raise SystemExit(f"{a.mode} RUN_DIR OUT_JSON")
    a.out_json = a.rest[0]
    set_excluded(True)
    run_dir = Path(a.run_dir).resolve()
    idx_doc = json.loads(INDEX.read_text())
    if idx_doc["checkpoint_sha256"] != CHECKPOINT06M_SHA256:
        raise SystemExit("the index is not of 06m's checkpoint")
    src, jobs, skipped = jobs_of(run_dir, idx_doc["units"])
    hdr = header(Path(__file__).resolve(), f"levin-phase-floor 07: {run_dir.name} against 06m's census ({RUN06M.name}), "
                 f"mode {a.mode}", [src, INDEX, HERE / "compare_k0.py"])
    for l in hdr:
        print(l, flush=True)
    print(f"# {len(jobs)} units to compare ({len(skipped)} other records skipped: {skipped[:10]}), {a.processes} "
          f"process(es); excluded: 06m's {sorted(EXCLUDED_06M)} and every key ending _s; set aside as new: "
          f"{sorted(NEW_KEYS_07)}", flush=True)
    fn = one_whole if a.mode == "whole" else one_calls
    if a.processes > 1:
        with mp.get_context("spawn").Pool(a.processes) as pool:
            rows = list(pool.imap(fn, jobs, chunksize=8))
    else:
        rows = [fn(j) for j in jobs]
    wev = worker_evidence(run_dir)
    wev_ok = bool(wev) and all(all(v for k, v in c.items() if k not in ("levin_quadrature_sha256", "lq_file", "tree")) for c in wev.values())
    n_fail = sum(1 for r in rows if not r["ok"])
    inv = {}
    for r in rows:
        for p in r.get("excl_present", []):
            inv.setdefault(p, [0, 0])[0] += 1
        for p in r.get("excl_differing", []):
            inv.setdefault(p, [0, 0])[1] += 1
    tot_keys = ("n_calls", "n_group_calls", "n_regions", "n_regions_returned", "n_clean", "n_located", "n_guarded",
                "new_keys_in_06m_records")
    tot = {k: sum(r.get(k, 0) or 0 for r in rows) for k in tot_keys}
    ev_tot = {}
    for r in rows:
        for k, v in (r.get("evidence") or {}).items():
            ev_tot.setdefault(k, [0, 0])[0 if v else 1] += 1
    doc = {
        "header": hdr, "mode": a.mode, "run_dir": str(run_dir), "source": str(src), "source_sha256": sha256(src),
        "run06m": str(RUN06M), "checkpoint06m_sha256": CHECKPOINT06M_SHA256,
        "excluded_06m": sorted(EXCLUDED_06M) + ["*_s"], "set_aside_new_07": sorted(NEW_KEYS_07),
        "records_skipped": skipped,
        "n_units": len(rows), "n_units_equal": len(rows) - n_fail, "n_units_differing": n_fail,
        "statuses": {s: sum(1 for r in rows if r["status"] == s) for s in sorted({r["status"] for r in rows})},
        "totals": tot, "unit_evidence_true_false": ev_tot,
        "worker_evidence_all_hold": wev_ok, "worker_evidence": wev,
        "excluded_keys_inventory_units_present_differing": dict(sorted(inv.items())),
        "first_differing_units": [r for r in rows if not r["ok"]][:20],
        "units": [{k: v for k, v in r.items() if k not in ("excl_present", "detail")} | (
            {"detail": r.get("detail")} if not r["ok"] else {}) for r in rows],
    }
    if a.mode == "calls":
        doc["calls_mode"] = {
            "n_units_all_clean": sum(1 for r in rows if r.get("all_clean")),
            "n_units_all_clean_total_equal": sum(1 for r in rows if r.get("all_clean") and r.get("total_equal")),
            "n_calls_clean_by_kind": {k: sum((r.get("n_clean_by_kind") or {}).get(k, 0) for r in rows) for k in ("group", "analytic")},
            "n_guard_errors": sum(len(r.get("guard_errors") or []) for r in rows),
        }
    out_p = Path(a.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=1, default=str))
    for r in [r for r in rows if not r["ok"]][:20]:
        print(f"DIFFERS {r['unit']}: {r['differing'][:10]}", flush=True)
    print(f"units {len(rows)}; equal {len(rows) - n_fail}; differing {n_fail}; statuses {doc['statuses']}")
    print(f"totals {tot}")
    if a.mode == "calls":
        print(f"calls mode {doc['calls_mode']}")
    print(f"unit evidence (true, false): {ev_tot}")
    print(f"worker evidence: {len(wev)} workers, all hold {wev_ok}; levin_quadrature sha256 "
          f"{sorted({str(c['levin_quadrature_sha256']) for c in wev.values()})}")
    print("excluded keys (path: units present, units differing):")
    for k, (n_p, n_d) in sorted(inv.items()):
        print(f"  {k}: {n_p}, {n_d}")
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0 if n_fail == 0 and wev_ok else 1


if __name__ == "__main__":
    sys.exit(main())
