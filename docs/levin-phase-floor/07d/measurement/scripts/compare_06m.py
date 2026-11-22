"""levin-phase-floor prompt 06m section 5, Gates M0 and M: units run at the tree m (06m's code, no variant installed)
against route B's census (06l's run levin-phase-floor-06l-census-routeB-20261105T100126, checkpoint 5cb3d3c0...), bit
for bit. New in 06m, after 06l's compare_k0.py (a record, not edited), whose comparison it applies whole (its
``identity`` mode) and from which it imports, unchanged, EXCLUDED, ``excluded``, ``diff``, ``fkey`` and
``load_pickle``: so the keys left out are exactly the ones compare_k0.py leaves out. Read only.

The comparison (prompt 06m section 5), the same for M0 and M:
- **Every unit** of the run is matched by its unit id to route B's checkpoint record (by the offset in
  state/index_routeB.json, made by index_routeB_06m.py; the line's SHA-256 checked) and to route B's region-pickle
  record (its SHA-256 checked); the unit's own pickle record is loaded the same way (its SHA-256 checked). Both
  statuses must be ``done``.
- **Every call is matched** by unit, kind (group: ``data.calls``; analytic: ``data.other_calls``), index, label
  (``notify_label``) and span (``x_span``, both ends by ``float.hex``); every region record (pickle ``regions`` for the
  group calls, ``other_regions`` for the others) by kind, index and span. The matched lists must be equal in length
  and order.
- **Then the whole unit record and the whole pickle record are compared** by compare_k0.diff: every key, recursively;
  floats by bit pattern (``float.hex``; NaN by repr); numpy arrays by dtype, shape and bytes; everything else by
  equality. A key in EXCLUDED, or ending ``_s``, is compared too but reported apart and never counted against the gate
  (``route_06l``, route B's instrument, is in EXCLUDED; at m the worker writes ``None`` there).
- **The excluded keys' inventory:** every excluded key that occurs in either record, by its path with list indices
  dropped, with the number of units in which it occurs and in which it differs, so that what is left out can be stated
  key by key.
- **The module evidence** (no variant installed): in each unit record, ``modules.files["AdaptiveLevin.levin_quadrature"]``
  is the tree's file, ``modules.all_in_tree`` is true, ``modules.foreign`` is empty and no module counts under the root
  ``variant``; in each worker's ``hello`` evidence, ``levin_quadrature_sha256`` is ``beaf0e23...`` (the tree's module),
  ``variant_06k`` is None, ``all_in_tree`` holds and the tree is at 38f6a828 and clean; in each ``bye``, Ray was not
  initialised and no foreign module was loaded.

MODE ``k0``: RUN_DIR is run_k0.py's output (``units.jsonl``; each worker's directory with its ``emit.jsonl`` and region
pickle). MODE ``census``: RUN_DIR is a registered census run directory (``checkpoint.jsonl``, ``regions_<worker>.pkl``,
``worker_<worker>_hello.json`` and ``worker_<worker>_bye.json``).

Writes OUT_JSON (provenance header; the counts; the excluded keys' inventory; the evidence; per unit its counts and,
where it differs, every differing path) and prints a summary. Exits 1 unless every unit is equal and every evidence
check holds.

Usage: PYTHONPATH=<tree>:<scripts> python -B compare_06m.py MODE RUN_DIR OUT_JSON [--processes N]
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from compare_k0 import EXCLUDED, diff, excluded, fkey, load_pickle  # noqa: E402,F401  (06l's, unchanged)
from prov import CHECKPOINT06L_B_SHA256, G_SHA, G_TREE, LQ_SHA256_06M, RUN06L_B, header, sha256  # noqa: E402

INDEX = HERE.parent / "state" / "index_routeB.json"
ROUTEB_VARIANT_SHA256 = "5aa6178a2298fb4cc32620cf6ae81538e873bf6a3177e8db6f3a5274d9bb4d9f"
TREE_LQ = str(G_TREE / "AdaptiveLevin" / "levin_quadrature.py")


def span_key(x_span):
    """A call's span as two hex strings, from either a census summary ({repr, hex} pairs) or a pickle (floats)."""
    out = []
    for e in x_span:
        out.append(e["hex"] if isinstance(e, dict) else float(e).hex())
    return tuple(out)


def call_keys(d):
    keys = []
    for kind, key in (("group", "calls"), ("analytic", "other_calls")):
        for c in d.get(key) or []:
            keys.append((kind, c["index"], c["notify_label"], span_key(c["x_span"])))
    return keys


def region_keys(p):
    keys = []
    for kind, key in (("group", "regions"), ("analytic", "other_regions")):
        for r in p.get(key) or []:
            keys.append((kind, r["index"], span_key(r["x_span"])))
    return keys


def excluded_paths(x, path, out, excl=False):
    """Every excluded key that occurs in x: its path with list indices dropped."""
    if isinstance(x, dict):
        for k, v in x.items():
            e = excluded(k)
            p = f"{path}.{k}"
            if e:
                out.add(p)
            excluded_paths(v, p, out, excl or e)
    elif isinstance(x, (list, tuple)):
        for v in x:
            excluded_paths(v, f"{path}[]", out, excl)


def norm(p):
    """A diff path with list indices dropped."""
    import re

    return re.sub(r"\[\d+\]", "[]", p)


def firings_b(c):
    r = c.get("route_06l")
    return None if r is None else int(r["n_firings"])


def one_unit(args):
    """Compare one unit. args: (uid, line_of_m (bytes), pickle path of m, index entry of route B)."""
    uid, m_line, m_pickle_dir, ent = args
    rec_m = json.loads(m_line)
    d = rec_m["data"]
    with open(RUN06L_B / "checkpoint.jsonl", "rb") as fck:
        fck.seek(ent["offset"])
        line = fck.read(ent["length"])
    if hashlib.sha256(line).hexdigest() != ent["line_sha256"]:
        raise SystemExit(f"{uid}: route B's checkpoint line sha256 differs")
    g = json.loads(line)["data"]
    row = {"unit": uid, "status": d["status"], "status_B": g["status"]}
    out = []
    if d["status"] != "done" or g["status"] != "done":
        row.update(ok=False, differing=["status"], n_calls=0, n_regions=0, excl_present=[], excl_differing=[])
        return row
    prec = load_pickle(Path(m_pickle_dir) / d["pickle"], d["pickle_offset"], d["pickle_length"], d["pickle_record_sha256"])
    gp = ent["pickle"]
    grec = load_pickle(RUN06L_B / gp[0], gp[1], gp[2], gp[3])
    ck_m, ck_b = call_keys(d), call_keys(g)
    rk_m, rk_b = region_keys(prec), region_keys(grec)
    matched = ck_m == ck_b and rk_m == rk_b
    if not matched:
        out.append(("match", False, f"calls {len(ck_m)}/{len(ck_b)} equal {ck_m == ck_b}; regions {len(rk_m)}/{len(rk_b)} "
                                    f"equal {rk_m == rk_b}"))
    diff(d, g, "data", out)
    diff(prec, grec, "pickle", out)
    pres = set()
    excluded_paths(d, "data", pres)
    excluded_paths(g, "data", pres)
    excluded_paths(prec, "pickle", pres)
    excluded_paths(grec, "pickle", pres)
    # the module evidence in the unit record
    mods = d.get("modules") or {}
    ev = {
        "lq_file_is_tree": (mods.get("files") or {}).get("AdaptiveLevin.levin_quadrature") == TREE_LQ,
        "all_in_tree": bool(mods.get("all_in_tree")),
        "no_foreign": not mods.get("foreign"),
        "no_variant_root": "variant" not in (mods.get("counts_by_root") or {}),
        "B_had_variant_root": (g.get("modules") or {}).get("counts_by_root", {}).get("variant") == 1,
        "route_06l_none_at_m": all(c.get("route_06l") is None for c in (d.get("calls") or []) + (d.get("other_calls") or [])),
    }
    fired = [firings_b(c) for c in (g.get("calls") or []) + (g.get("other_calls") or [])]
    row.update(
        ok=matched and not [p for p, e, _ in out if not e] and all(v for k, v in ev.items() if k != "B_had_variant_root"),
        differing=sorted({p for p, e, _ in out if not e})[:200],
        detail=[list(x) for x in out if not x[1]][:50],
        excl_present=sorted(pres),
        excl_differing=sorted({norm(p) for p, e, _ in out if e}),
        n_calls=len(ck_m),
        n_group_calls=sum(1 for k in ck_m if k[0] == "group"),
        n_regions=len(rk_m),
        n_regions_returned=int(sum(len(r.get("start", [])) for r in (prec.get("regions") or []) + (prec.get("other_regions") or []))),
        n_calls_B_fired=sum(1 for f in fired if f),
        n_firings_B=sum(f or 0 for f in fired),
        evidence=ev,
    )
    return row


def worker_evidence(mode, run_dir):
    """Each worker's hello and bye evidence."""
    out = {}
    if mode == "k0":
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
            hello = json.loads(hp.read_text())["evidence"]
            bye = json.loads(bp.read_text()) if bp.exists() else None
            out[name] = (hello, bye)
    checks = {}
    for name, (h, b) in out.items():
        checks[name] = {
            "hello": h is not None,
            "levin_quadrature_sha256": None if h is None else h.get("levin_quadrature_sha256"),
            "lq_is_tree_module": h is not None and h.get("levin_quadrature_sha256") == LQ_SHA256_06M
            and h.get("levin_quadrature_tree_file_sha256") == LQ_SHA256_06M,
            "lq_file": None if h is None else h["files"].get("AdaptiveLevin.levin_quadrature"),
            "variant_none": h is not None and h.get("variant_06k") is None,
            "all_in_tree": h is not None and bool(h.get("all_in_tree")),
            "tree_head_clean": h is not None and h.get("head") == G_SHA and h.get("status_short", "x").strip() == "",
            "bye": b is not None,
            "ray_not_initialized": b is not None and b.get("ray_initialized") is False,
            "no_foreign_modules": b is not None and not b.get("modules_foreign"),
        }
    return checks


def b_worker_variants():
    """Route B's workers' declared variant (for the record): name and sha256 from their hello evidence."""
    out = {}
    for hp in sorted(RUN06L_B.glob("worker_*_hello.json")):
        h = json.loads(hp.read_text())["evidence"]
        v = h.get("variant_06k") or {}
        out[hp.name] = [v.get("name"), v.get("sha256"), h.get("levin_quadrature_sha256")]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("k0", "census"))
    ap.add_argument("run_dir")
    ap.add_argument("out_json")
    ap.add_argument("--processes", type=int, default=1)
    a = ap.parse_args()
    if a.processes > 6:
        raise SystemExit("at most six processes")
    run_dir = Path(a.run_dir).resolve()
    out_p = Path(a.out_json)
    src = run_dir / ("units.jsonl" if a.mode == "k0" else "checkpoint.jsonl")
    hdr = header(Path(__file__).resolve(), f"levin-phase-floor 06m: Gate {'M0' if a.mode == 'k0' else 'M'}, {run_dir.name} "
                 f"against route B's census ({RUN06L_B.name})", [src, INDEX, Path(__file__).resolve().parent / "compare_k0.py"])
    for l in hdr:
        print(l, flush=True)
    idx_doc = json.loads(INDEX.read_text())
    if idx_doc["checkpoint_sha256"] != CHECKPOINT06L_B_SHA256:
        raise SystemExit("the index is not of route B's checkpoint")
    idx = idx_doc["units"]
    jobs = []
    seen = set()
    with open(src, "rb") as f:
        for line in f:
            r = json.loads(line)
            uid = r["unit"]
            if uid in seen:
                raise SystemExit(f"{uid} twice in {src}")
            seen.add(uid)
            if uid not in idx:
                raise SystemExit(f"{uid} is not in route B's census")
            pdir = run_dir / r["worker"] if a.mode == "k0" else run_dir
            jobs.append((uid, line, str(pdir), idx[uid]))
    print(f"# {len(jobs)} units to compare, {a.processes} process(es); EXCLUDED (compare_k0.py's) = {sorted(EXCLUDED)} "
          f"and every key ending _s", flush=True)
    if a.processes > 1:
        with mp.get_context("spawn").Pool(a.processes) as pool:
            rows = list(pool.imap(one_unit, jobs, chunksize=16))
    else:
        rows = [one_unit(j) for j in jobs]
    wev = worker_evidence(a.mode, run_dir)
    wev_ok = bool(wev) and all(all(v for k, v in c.items() if k not in ("levin_quadrature_sha256", "lq_file")) for c in wev.values())
    n_fail = sum(1 for r in rows if not r["ok"])
    inv = {}
    for r in rows:
        for p in r["excl_present"]:
            inv.setdefault(p, [0, 0])[0] += 1
        for p in r["excl_differing"]:
            inv.setdefault(p, [0, 0])[1] += 1
    tot = {k: sum(r.get(k, 0) for r in rows) for k in ("n_calls", "n_group_calls", "n_regions", "n_regions_returned",
                                                          "n_calls_B_fired", "n_firings_B")}
    ev_tot = {}
    for r in rows:
        for k, v in (r.get("evidence") or {}).items():
            ev_tot.setdefault(k, [0, 0])[0 if v else 1] += 1
    doc = {
        "header": hdr, "mode": a.mode, "run_dir": str(run_dir), "source": str(src), "source_sha256": sha256(src),
        "routeB_run": str(RUN06L_B), "routeB_checkpoint_sha256": CHECKPOINT06L_B_SHA256,
        "excluded_keys_compare_k0": sorted(EXCLUDED) + ["*_s"],
        "n_units": len(rows), "n_units_equal": len(rows) - n_fail, "n_units_differing": n_fail,
        "statuses": {s: sum(1 for r in rows if r["status"] == s) for s in sorted({r["status"] for r in rows})},
        "totals": tot,
        "unit_evidence_true_false": ev_tot,
        "worker_evidence_all_hold": wev_ok, "worker_evidence": wev,
        "routeB_worker_variants": b_worker_variants(),
        "excluded_keys_inventory_units_present_differing": dict(sorted(inv.items())),
        "first_differing_units": [r for r in rows if not r["ok"]][:20],
        "units": [{k: v for k, v in r.items() if k not in ("excl_present", "detail")} | (
            {"detail": r.get("detail")} if not r["ok"] else {}) for r in rows],
    }
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=1, default=str))
    for r in rows[:: max(1, len(rows) // 20)]:
        print(f"{r['unit']}: {'EQUAL' if r['ok'] else 'DIFFERS'}; calls {r.get('n_calls')}; region records "
              f"{r.get('n_regions')}; excluded keys differing {len(r['excl_differing'])}", flush=True)
    for r in [r for r in rows if not r["ok"]][:20]:
        print(f"DIFFERS {r['unit']}: {r['differing'][:10]}", flush=True)
    print(f"units {len(rows)}; equal {len(rows) - n_fail}; differing {n_fail}; statuses {doc['statuses']}")
    print(f"totals {tot}")
    print(f"unit evidence (true, false): {ev_tot}")
    print(f"worker evidence: {len(wev)} workers, all hold {wev_ok}; levin_quadrature sha256 "
          f"{sorted({c['levin_quadrature_sha256'] for c in wev.values()})}")
    print("excluded keys (path: units present, units differing):")
    for k, (n_p, n_d) in sorted(inv.items()):
        print(f"  {k}: {n_p}, {n_d}")
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0 if n_fail == 0 and wev_ok else 1


if __name__ == "__main__":
    sys.exit(main())
