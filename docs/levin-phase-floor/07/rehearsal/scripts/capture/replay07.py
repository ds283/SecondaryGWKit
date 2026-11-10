"""levin-phase-floor prompt 07 section 5, rehearsal agent: the replay of R28's located calls, S2's capture
and the q-r captures, through 06g's references/v1/scripts/ capture harness (copied into this directory;
prov.py and worker.py changed where marked "prompt 07 rehearsal"). New, after 06g's v1_capture.py and
v1_rows.py (their run_workers, stored_form and unit_spec are taken from v1_capture.py, generalised to
several units per worker).

Subcommands (outputs under var/levin-phase-floor-07/rehearsal/; every output opens with prov.header):
- ``rows OUT``: from R27's and R28's stores (sqlite mode=ro on the real files) and out/qsi_guard_records.csv,
  every row of R28 with a nonzero guard count, P1's two rows (S2) and the w rows, each with its shard,
  serial and k exit-time serial in each store, and every Levin record of R28 with a nonzero count
  (path, label, its stored value/abserr/converged/phase_limited/regions/four counts).
- ``run SPEC_JSON``: runs the units a spec lists, one worker process per entry of ``workers`` (each a list
  of units for one tree), at most MAX_WORKERS at once; each worker writes into its own directory under
  ``state/capture/<run name>/<worker>/``; the driver writes ``<run name>.json`` there with each worker's
  return code and the path of its emit file.
- ``check RUN_DIR ROWS_JSON OUT``: pass 1 of the replay. For every row, the worker's unit must be
  ``done``, its row total and analytic_rad must equal R28's bit for bit, and every R28 Levin record with a
  nonzero count must be reproduced by exactly one call of the unit: a group record by a call whose label
  names that group and sub-interval and whose (1 + z_response) x value and x abserr equal the record's
  (one double product each, as phase_group_Levin_integral stores them), and whose converged,
  phase_limited, num_regions and four counts (from guard_07) equal it; an analytic block
  (``analytic.<rad>.Levin.<B>_data``) by the call labelled ``analytic <B>`` whose abserr, converged,
  phase_limited, num_regions and four counts equal it (the record keeps no value), the two radii being
  told apart by those fields (exactly one of the two ``analytic <B>`` calls must match). Writes the
  per-call kind and location table (CSV) and a JSON of every check.

Ray is never initialised (each worker reports it). Nothing is written in either worktree or to a store.

Usage: python -I -B replay07.py rows|run|check ...   (cwd anywhere; paths absolute)
"""

import csv
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import MAIN, PYTHON, TREES, header, sha256  # noqa: E402

REH = MAIN / "var" / "levin-phase-floor-07" / "rehearsal"
STORES = {"p": MAIN / "var" / "datastores" / "levin-phase-floor-07-r27.sqlite",
          "g": MAIN / "var" / "datastores" / "levin-phase-floor-07-r28.sqlite"}
MAX_WORKERS = 6
GUARD = ("stationary_points", "window_pieces", "guarded_regions", "guard_at_depth_max")
GUARD_RES = {"stationary_points": "num_stationary_points", "window_pieces": "num_window_pieces",
             "guarded_regions": "num_guarded_regions", "guard_at_depth_max": "num_guard_at_depth_max"}


def ro(p):
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True)


def read_store(primary):
    con = ro(primary)
    shards = [(r[0], r[1]) for r in con.execute("SELECT serial, filename FROM shards ORDER BY serial")]
    con.close()
    out = {}
    kq = ("(SELECT w.k_inv_Mpc FROM wavenumber_exit_time e JOIN wavenumber w "
          "ON w.serial = e.wavenumber_serial WHERE e.serial = q.{})")
    for shard, name in shards:
        con = ro(primary.parent / Path(name).name)
        sql = (f"SELECT {kq.format('k_wavenumber_exit_serial')}, {kq.format('q_wavenumber_exit_serial')}, "
               f"{kq.format('r_wavenumber_exit_serial')}, (SELECT z FROM redshift WHERE serial = q.z_response_serial), "
               "q.serial, q.k_wavenumber_exit_serial, q.total, q.total_abserr, q.analytic_rad, q.metadata, "
               "(SELECT log10_tol FROM tolerance WHERE serial = q.rtol_serial) FROM QuadSourceIntegral q")
        for row in con.execute(sql):
            key = tuple(float(x) for x in row[:4])
            assert key not in out
            out[key] = {"shard": shard, "serial": row[4], "k_exit_serial": row[5], "total": float(row[6]).hex(),
                        "total_abserr": float(row[7]).hex(),
                        "analytic_rad": None if row[8] is None else float(row[8]).hex(),
                        "metadata": json.loads(row[9]), "log10_rtol": row[10]}
        con.close()
    return out


def levin_records(meta):
    out = []
    for i, s in enumerate((meta.get("WKB_Levin") or {}).get("subintervals", [])):
        for j, g in enumerate(s.get("groups", [])):
            out.append(("group", f"WKB_Levin.subintervals[{i}].groups[{j}]", g, s))
    for rad in ("0pt5", "2pt5"):
        for blk, rec in (((meta.get("analytic") or {}).get(rad) or {}).get("Levin") or {}).items():
            out.append(("analytic", f"analytic.{rad}.Levin.{blk}", rec, None))
    return out


def stored_form(c, z_response):
    """As v1_capture.py: phase_group_Levin_integral stores scale * value and scale * abserr, scale = 1 + z."""
    scale = 1.0 + float(z_response)
    return ((scale * float.fromhex(c["value"]["hex"])).hex(), (scale * float.fromhex(c["abserr"]["hex"])).hex())


def unit_spec(tree, rec, capture=None):
    uid = f"{tree}-1e-7-{rec['shard']}-{rec['serial']}"
    return {"uid": uid, "tree": tree, "rtol": "1e-7", "row": f"{rec['shard']}-{rec['serial']}", "names05": None,
            "shard": rec["shard"], "serial": rec["serial"], "k_key": f"{rec['shard']}:{rec['k_exit_serial']}",
            "budget": None, "scan": False, "w3": False, "capture": capture}


def cmd_rows(out_path):
    recs_csv = REH / "out" / "qsi_guard_records.csv"
    v1_rows = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07-7460035/docs/levin-phase-floor/06g/references/v1/out/v1_rows.json")
    lines = header(Path(__file__).resolve(), "prompt 07 rehearsal: the rows to replay and capture",
                   [recs_csv, v1_rows, *sorted(STORES["p"].parent.glob("levin-phase-floor-07-r27*.sqlite")),
                    *sorted(STORES["g"].parent.glob("levin-phase-floor-07-r28*.sqlite"))])
    data = {t: read_store(p) for t, p in STORES.items()}
    touched = set()
    with open(recs_csv) as f:
        for r in csv.DictReader(f):
            touched.add((float(r["k"]), float(r["q"]), float(r["r"]), float(r["z_response"])))
    p1 = [(10 ** 5.5, 10 ** 5.5, 10 ** 5.5, 250.0), (1e5, 10 ** 5.5, 10 ** 5.5, 250.0)]
    # P1's rows by their (k, q, r, z_response): the stores' own floats, matched to these within 1e-9 relative
    def find(target):
        hits = [k for k in data["g"] if all(abs(a - b) <= 1e-9 * abs(b) for a, b in zip(k, target))]
        assert len(hits) == 1, (target, hits)
        return hits[0]
    p1_keys = [find(t) for t in p1]
    w = json.loads(v1_rows.read_text())
    w_keys = {e["wid"]: (e["k"], e["q"], e["r"], e["z_response"]) for e in w}
    for wid, k in w_keys.items():
        assert k in data["g"], (wid, k)
    rows = []
    for key in sorted(set(touched) | set(p1_keys) | set(w_keys.values())):
        e = {"key": list(key), "touched": key in touched, "p1": (p1_keys.index(key) + 1) if key in p1_keys else None,
             "wid": [wid for wid, k in w_keys.items() if k == key] or None}
        for t, tag in (("p", "R27"), ("g", "R28")):
            rec = data[t][key]
            e[tag] = {k_: rec[k_] for k_ in ("shard", "serial", "k_exit_serial", "total", "total_abserr", "analytic_rad", "log10_rtol")}
        meta28, meta27 = data["g"][key]["metadata"], data["p"][key]["metadata"]
        r27 = {name: rec for _, name, rec, _ in levin_records(meta27)}
        e["records"] = []
        for kind, name, rec, sub in levin_records(meta28):
            nonzero = any(rec[g] for g in GUARD)
            is_p1_group = e["p1"] is not None and kind == "group" and rec.get("label") == "G+q+r"
            is_w_group = e["wid"] is not None and kind == "group" and rec.get("label") == "G+q+r"
            if not (nonzero or is_p1_group or is_w_group):
                continue
            o = r27[name]
            e["records"].append({
                "kind": kind, "path": name, "label": rec.get("label"),
                "sub": None if sub is None else {"z_max": sub["z_max"], "z_min": sub["z_min"], "regime": sub["regime"]},
                "nonzero": nonzero,
                "R28": {**{k_: (float(rec[k_]).hex() if isinstance(rec.get(k_), float) else rec.get(k_))
                           for k_ in ("value", "abserr", "converged", "phase_limited", "regions")},
                        **{g: rec[g] for g in GUARD}},
                "R27": {k_: (float(o[k_]).hex() if isinstance(o.get(k_), float) else o.get(k_))
                        for k_ in ("value", "abserr", "converged", "phase_limited", "regions")},
            })
        rows.append(e)
    out = Path(out_path)
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.write_text(json.dumps({"header": lines, "rows": rows}, indent=1))
    for l in lines:
        print(l)
    n_rec = sum(len(e["records"]) for e in rows)
    n_nz = sum(1 for e in rows for r in e["records"] if r["nonzero"])
    print(f"rows {len(rows)} (touched {sum(e['touched'] for e in rows)}, P1 {[e['key'] for e in rows if e['p1']]}, "
          f"w rows {sum(1 for e in rows if e['wid'])}); records {n_rec}, nonzero {n_nz}")
    print(f"wrote {out} sha256 {sha256(out)}")


def run_workers(jobs, log):
    """jobs: [(name, tree, wdir, [units])]; at most MAX_WORKERS at once. Returns {name: rc}."""
    pending = list(jobs)
    running = {}
    out = {}
    while pending or running:
        while pending and len(running) < MAX_WORKERS:
            name, tree, wdir, units = pending.pop(0)
            wdir.mkdir(parents=True, exist_ok=False)
            spec = {"tree": tree, "name": name, "units": units, "out_dir": str(wdir), "emit_fd": None,
                    "emit_file": str(wdir / "emit.jsonl"), "rows_sha256": None, "w2_capture_sha256": None}
            (wdir / "spec.json").write_text(json.dumps(spec, indent=1))
            env = dict(os.environ, PYTHONPATH=str(TREES[tree][0]), PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", MKL_NUM_THREADS="1", GIT_OPTIONAL_LOCKS="0")
            wl = open(wdir / "worker.log", "ab")
            p = subprocess.Popen([str(PYTHON), "-B", str(HERE / "worker.py"), str(wdir / "spec.json")], cwd=str(REH / "state"),
                                 env=env, stdin=subprocess.DEVNULL, stdout=wl, stderr=subprocess.STDOUT)
            wl.close()
            running[name] = (p, wdir, time.time())
            log(f"start {name} ({tree}, {len(units)} units): pid {p.pid}")
        for name, (p, wdir, t0) in list(running.items()):
            if p.poll() is not None:
                out[name] = p.returncode
                log(f"end {name}: rc {p.returncode} after {time.time() - t0:.1f} s")
                del running[name]
        time.sleep(1.0)
    return out


def cmd_run(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    run_dir = REH / "state" / "capture" / spec["name"]
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists")
    run_dir.mkdir(parents=True)
    logf = open(run_dir / "driver.log", "a", buffering=1)

    def log(s):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {s}"
        logf.write(line + "\n")
        print(line, flush=True)

    for l in header(Path(__file__).resolve(), f"prompt 07 rehearsal: run {spec['name']}", [Path(spec_path)]):
        log(l)
    jobs = [(w["name"], w["tree"], run_dir / w["name"], w["units"]) for w in spec["workers"]]
    log(f"{len(jobs)} workers, {sum(len(j[3]) for j in jobs)} units, at most {MAX_WORKERS} at once")
    t0 = time.time()
    rcs = run_workers(jobs, log)
    res = {"name": spec["name"], "spec": str(spec_path), "spec_sha256": sha256(spec_path), "wall_s": time.time() - t0,
           "workers": {n: {"rc": rc, "emit": str(run_dir / n / "emit.jsonl")} for n, rc in rcs.items()}}
    (run_dir / f"{spec['name']}.json").write_text(json.dumps(res, indent=1))
    log(f"done in {res['wall_s']:.1f} s; return codes {dict(Counter(rcs.values()))}")
    return 0 if all(rc == 0 for rc in rcs.values()) else 1


def load_units(run_dir):
    """{uid: (unit data, bye, hello)} over every worker of a run."""
    out = {}
    for emit in sorted(Path(run_dir).glob("*/emit.jsonl")):
        hello = bye = None
        units = []
        for line in open(emit):
            m = json.loads(line)
            if m["kind"] == "hello":
                hello = m
            elif m["kind"] == "bye":
                bye = m
            elif m["kind"] == "unit":
                units.append(m["data"])
        for u in units:
            assert u["unit"] not in out, u["unit"]
            out[u["unit"]] = (u, bye, hello)
    return out


def hx(x):
    return None if x is None else x


def cmd_check(run_dir, rows_path, out_csv):
    rows = json.loads(Path(rows_path).read_text())["rows"]
    units = load_units(run_dir)
    lines = header(Path(__file__).resolve(), "prompt 07 rehearsal: the replay checked against R28, record by record",
                   [Path(rows_path), *sorted(Path(run_dir).glob("*/emit.jsonl"))])
    for l in lines:
        print(l)
    table = []
    checks = []
    n_ok = n_rec = 0
    ray_flags = Counter()
    for e in rows:
        if not e["touched"] and not e["wid"] and not e["p1"]:
            continue
        uid = f"g-1e-7-{e['R28']['shard']}-{e['R28']['serial']}"
        if uid not in units:
            checks.append({"key": e["key"], "uid": uid, "error": "unit not run"})
            continue
        u, bye, hello = units[uid]
        ray_flags[None if bye is None else str(bye.get("ray_initialized"))] += 1
        z = e["key"][3]
        row_ok = (u["status"] == "done" and u.get("result") is not None
                  and u["result"]["total"]["hex"] == e["R28"]["total"]
                  and u["result"]["analytic_rad"]["hex"] == e["R28"]["analytic_rad"])
        allc = (u.get("calls") or []) + (u.get("other_calls") or [])
        for r in e["records"]:
            n_rec += 1
            want = r["R28"]
            if r["kind"] == "group":
                sub = r["sub"]
                cands = [c for c in u.get("calls") or []
                         if c["notify_label"].endswith(f"group {r['label']}")
                         and stored_form(c, z) == (want["value"], want["abserr"])]
            else:
                blk = r["path"].split(".")[-1].replace("_data", "")
                cands = [c for c in u.get("other_calls") or [] if c["notify_label"] == f"analytic {blk}"
                         and c["abserr"]["hex"] == want["abserr"]]
            ok = False
            c = cands[0] if len(cands) == 1 else None
            fields = {}
            if c is not None:
                g = c.get("guard_07") or {}
                fields = {
                    "converged": c["converged"] == want["converged"],
                    "phase_limited": c["phase_limited"] == want["phase_limited"],
                    "regions": c["num_regions"] == want["regions"],
                    **{k_: g.get(GUARD_RES[k_]) == want[k_] for k_ in GUARD},
                }
                ok = row_ok and all(fields.values())
            n_ok += ok
            chk = {"key": e["key"], "uid": uid, "path": r["path"], "label": r["label"], "n_candidates": len(cands),
                   "row_total_and_analytic_rad_equal": row_ok, "fields_equal": fields, "reproduces": ok}
            checks.append(chk)
            # the per-call table
            g = (c or {}).get("guard_07") or {}
            var = "log(1+z)" if r["kind"] == "group" else "log(eta)"
            base = {"k": e["key"][0], "q": e["key"][1], "r": e["key"][2], "z_response": z, "kind": r["kind"],
                    "record": r["path"], "label": r["label"] or r["path"].split(".")[-1], "reproduces": ok,
                    "variable": var,
                    "x_span_a": (g.get("x_span") or [None, None])[0] and g["x_span"][0]["hex"],
                    "x_span_b": (g.get("x_span") or [None, None])[1] and g["x_span"][1]["hex"],
                    "num_stationary_points": g.get("num_stationary_points"), "num_window_pieces": g.get("num_window_pieces"),
                    "num_guarded_regions": g.get("num_guarded_regions"), "num_guard_at_depth_max": g.get("num_guard_at_depth_max"),
                    "p1": e["p1"], "wid": ";".join(e["wid"]) if e["wid"] else ""}
            if "points" in g:
                for i, p in enumerate(g["points"]):
                    x = float.fromhex(p["x"]["hex"])
                    loc = (math.exp(x) - 1.0) if r["kind"] == "group" else math.exp(x)
                    table.append({**base, "point": i, "point_kind": p["kind"], "found_as": p["found_as"], "phase": p["phase"],
                                  "x_hex": p["x"]["hex"], "x": repr(x),
                                  "location": repr(loc), "location_is": "z = exp(x) - 1" if r["kind"] == "group" else "eta = exp(x)",
                                  "theta_deriv_at_point": None if p["theta_deriv"] is None else p["theta_deriv"]["repr"],
                                  "lambda": None if p["lambda"] is None else p["lambda"]["repr"],
                                  "own_window_lo": None if p["own_window"] is None else p["own_window"][0]["repr"],
                                  "own_window_hi": None if p["own_window"] is None else p["own_window"][1]["repr"],
                                  "window_pieces_of_point": p["window_pieces"], "summary": ""})
            elif "points_summary" in g:
                s = g["points_summary"]
                xa, xb = float.fromhex(s["x_min"]["hex"]), float.fromhex(s["x_max"]["hex"])
                table.append({**base, "point": "all", "point_kind": json.dumps(s["kinds"]), "found_as": json.dumps(s["found_as"]),
                              "phase": json.dumps(s["phases"]), "x_hex": f"{s['x_min']['hex']}..{s['x_max']['hex']}",
                              "x": f"{xa!r}..{xb!r}", "location": f"{math.exp(xa) - 1.0!r}..{math.exp(xb) - 1.0!r}",
                              "location_is": "z = exp(x) - 1 (least..greatest)", "theta_deriv_at_point": "", "lambda": "",
                              "own_window_lo": "", "own_window_hi": "", "window_pieces_of_point": "",
                              "summary": json.dumps({k_: s.get(k_) for k_ in ("all_x_on_scan_grid", "n_distinct_x",
                                                                                "grid_theta_deriv_exact_zeros",
                                                                                "grid_theta_deriv_max_abs", "grid_theta_min",
                                                                                "grid_theta_max", "grid_error")})})
            else:
                table.append({**base, "point": None, "point_kind": None, "found_as": None, "phase": None, "x_hex": None,
                              "x": None, "location": None, "location_is": None, "theta_deriv_at_point": None, "lambda": None,
                              "own_window_lo": None, "own_window_hi": None, "window_pieces_of_point": None, "summary": ""})
    with open(out_csv, "w", newline="") as f:
        cols = list(table[0])
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(table)
    nz = [c for c in checks if "path" in c]
    print(f"records checked {n_rec}; reproduce bit for bit {n_ok}; not {n_rec - n_ok}")
    print(f"workers' ray_initialized at exit: {dict(ray_flags)}")
    for c in checks:
        if not c.get("reproduces"):
            print(f"  NOT REPRODUCED: {c}")
    out_json = Path(out_csv).with_suffix(".checks.json")
    out_json.write_text(json.dumps({"header": lines, "checks": checks}, indent=1))
    print(f"wrote {out_csv} sha256 {sha256(out_csv)} ({len(table)} lines) and {out_json} sha256 {sha256(out_json)}")
    return 0 if n_ok == n_rec else 1


QR_CALLS = [  # the q-r references' three calls (q = r): the worst (3,604 x R28's abserr) and two more
    (1e5, 1e6, 1e6, 6985073578.494601),
    (10 ** 5.5, 1e6, 1e6, 13627426843.66573),
    (1e5, 10 ** 5.5, 10 ** 5.5, 3046979177.3866296),
]


def _find(rows, target):
    hits = [e for e in rows if all(abs(a - b) <= 1e-9 * abs(b) for a, b in zip(e["key"], target))]
    assert len(hits) == 1, (target, len(hits))
    return hits[0]


def cmd_spec(which, rows_path, out_path, run_p=None):
    """Write a run spec. which = replay (every touched row at g, 6 workers, the q-r rows spread first),
    p1 (pass 1 at p: P1's two rows and the three q-r rows), capture (pass 2: S2's two G+q+r calls at p,
    the three q-r calls at p and at g; call indices from pass 1, ``run_p``, and the replay run at g)."""
    rows = json.loads(Path(rows_path).read_text())["rows"]
    if which == "replay":
        qr = [e for e in rows if any(r["label"] == "q-r" for r in e["records"])]
        rest = [e for e in rows if e not in qr]
        ws = [[] for _ in range(MAX_WORKERS)]
        for i, e in enumerate(qr + rest):
            ws[i % MAX_WORKERS].append(unit_spec("g", e["R28"]))
        spec = {"name": "replay", "workers": [{"name": f"replay-g-{i}", "tree": "g", "units": u} for i, u in enumerate(ws)]}
    elif which == "p1":
        es = [e for e in rows if e["p1"]] + [_find(rows, t) for t in QR_CALLS]
        spec = {"name": "pass1-p", "workers": [{"name": "pass1-p-0", "tree": "p", "units": [unit_spec("p", e["R27"]) for e in es[:3]]},
                                               {"name": "pass1-p-1", "tree": "p", "units": [unit_spec("p", e["R27"]) for e in es[3:]]}]}
    elif which == "capture":
        up = load_units(REH / "state" / "capture" / "pass1-p")
        ug = load_units(REH / "state" / "capture" / "replay")
        wp, wg = [], []
        for e, label in [(e, "G+q+r") for e in rows if e["p1"]] + [(_find(rows, t), "q-r") for t in QR_CALLS]:
            z = e["key"][3]
            rec = [r for r in e["records"] if r["kind"] == "group" and r["label"] == label]
            assert len(rec) == 1, (e["key"], label)
            rec = rec[0]
            for tree, tag, units, lst in (("p", "R27", up, wp), ("g", "R28", ug, wg)):
                if tree == "g" and label != "q-r":
                    continue
                u, _, _ = units[f"{tree}-1e-7-{e[tag]['shard']}-{e[tag]['serial']}"]
                want = rec[tag]
                hits = [c for c in u["calls"] if c["notify_label"].endswith(f"group {label}")
                        and stored_form(c, z) == (want["value"], want["abserr"])]
                assert len(hits) == 1, (e["key"], tree, len(hits))
                lst.append(unit_spec(tree, e[tag], capture=[[hits[0]["index"], hits[0]["notify_label"]]]))
        spec = {"name": "capture", "workers": [{"name": "capture-p", "tree": "p", "units": wp},
                                               {"name": "capture-g", "tree": "g", "units": wg}]}
    else:
        raise SystemExit(which)
    Path(out_path).write_text(json.dumps(spec, indent=1))
    print(f"wrote {out_path} sha256 {sha256(out_path)}: {sum(len(w['units']) for w in spec['workers'])} units")


def cmd_capcheck(rows_path, out_path):
    """Check each capture of the run ``capture``: its pickle's blob hashes as the worker recorded it, and the
    same run's record of the call equals the store's stored group (R27 at p, R28 at g) bit for bit (value and
    abserr in stored form, converged, num_regions); the row's total equals the store's. Copies each blob to
    state/inputs/<label>-<tree>-<shard>-<serial>.pkl (not unpickled here)."""
    import hashlib

    rows = json.loads(Path(rows_path).read_text())["rows"]
    run_dir = REH / "state" / "capture" / "capture"
    units = load_units(run_dir)
    lines = header(Path(__file__).resolve(), "prompt 07 rehearsal: the captures (S2 at p; q-r at p and g) checked",
                   [Path(rows_path), *sorted(run_dir.glob("*/emit.jsonl"))])
    for l in lines:
        print(l)
    res = []
    ok_all = True
    for e, label in [(e, "G+q+r") for e in rows if e["p1"]] + [(_find(rows, t), "q-r") for t in QR_CALLS]:
        rec = [r for r in e["records"] if r["kind"] == "group" and r["label"] == label][0]
        for tree, tag in (("p", "R27"), ("g", "R28")):
            if tree == "g" and label != "q-r":
                continue
            uid = f"{tree}-1e-7-{e[tag]['shard']}-{e[tag]['serial']}"
            u, bye, _ = units[uid]
            ent = (u.get("x1_inputs") or [None])[0]
            want = rec[tag]
            good = ent is not None and not ent.get("error")
            out = {"key": e["key"], "label": label, "tree": tree, "store": tag, "uid": uid, "p1": e["p1"],
                   "store_value": want["value"], "store_abserr": want["abserr"], "store_converged": want["converged"],
                   "store_regions": want["regions"], "ray_initialized": None if bye is None else bye.get("ray_initialized")}
            if good:
                wdir = run_dir / f"capture-{tree}"
                with open(wdir / ent["pickle"], "rb") as f:
                    f.seek(ent["offset"])
                    blob = f.read(ent["length"])
                if hashlib.sha256(blob).hexdigest() != ent["sha256"]:
                    raise RuntimeError(f"{uid}: blob sha256 differs")
                cc = [c for c in u["calls"] if c["index"] == ent["index"]]
                good = (len(cc) == 1 and stored_form(cc[0], e["key"][3]) == (want["value"], want["abserr"])
                        and cc[0]["converged"] == want["converged"] and cc[0]["num_regions"] == want["regions"]
                        and u["result"]["total"]["hex"] == e[tag]["total"])
                dst = REH / "state" / "inputs" / f"{label.replace('+', 'p').replace('-', 'm')}-{tree}-{e[tag]['shard']}-{e[tag]['serial']}.pkl"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    raise SystemExit(f"{dst} exists")
                dst.write_bytes(blob)
                out.update(pickle=str(dst), sha256=ent["sha256"], bytes=len(blob), index=ent["index"],
                           notify_label=cc[0]["notify_label"] if cc else None,
                           call_value=cc[0]["value"]["hex"] if cc else None, call_abserr=cc[0]["abserr"]["hex"] if cc else None,
                           x_span=cc[0]["x_span"] if cc else None)
            out["reproduces_store"] = bool(good)
            ok_all &= bool(good)
            res.append(out)
            print(f"{label} {tree} ({tag}) {e['key']}: {'REPRODUCES' if good else 'DOES NOT REPRODUCE'} the store bit for bit; "
                  f"call {out.get('call_value')} {out.get('call_abserr')}; pickle {out.get('pickle')}")
    Path(out_path).write_text(json.dumps({"header": lines, "captures": res}, indent=1))
    print(f"wrote {out_path} sha256 {sha256(out_path)}; all reproduce: {ok_all}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "spec":
        cmd_spec(sys.argv[2], sys.argv[3], sys.argv[4])
        sys.exit(0)
    if cmd == "capcheck":
        sys.exit(cmd_capcheck(sys.argv[2], sys.argv[3]))
    if cmd == "rows":
        cmd_rows(sys.argv[2])
    elif cmd == "run":
        sys.exit(cmd_run(sys.argv[2]))
    elif cmd == "check":
        sys.exit(cmd_check(sys.argv[2], sys.argv[3], sys.argv[4]))
    else:
        raise SystemExit(f"unknown command {cmd}")
