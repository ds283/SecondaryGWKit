"""levin-phase-floor prompt 07 section 5 resumed (section 10, Rule A), rehearsal agent: the replay of R29's located
calls, and the four-block check, through the stop-1 capture harness (copied into this directory; prov.py and
worker.py changed where marked "prompt 07 r29"). New, after the stop-1 replay07.py (68f985eb), whose
run_workers, load_units, stored_form, unit_spec, read_store and levin_records are taken from it unchanged but
for paths (REH, the three stores).

Subcommands (outputs under var/levin-phase-floor-07/rehearsal/r29/; every output opens with prov.header):
- ``rows OUT``: every row of R29 holding a Levin record with a nonzero count, plus P1's two rows and the w rows,
  each with its shard, serial and k exit-time serial in R27 (p), R28 (o) and R29 (g), and every Levin record of
  R29 with a nonzero count (and P1's / the w rows' G+q+r) with R29's stored fields and counts.
- ``spec replay ROWS OUT``: every row of ``rows`` at g (R29), six workers.
- ``spec blocks1 ROWS OUT``: pass 1 of the four-block check: the two rows at p, o and g.
- ``spec blocks2 ROWS OUT``: pass 2: each tree's 2pt5 J4 and Y4 calls of the two rows captured, each identified in
  pass 1 as the one ``analytic <B>`` call whose abserr, converged, phase_limited and regions (and at o and g the
  four counts) equal its store's 2pt5 record (exactly one of the two radii's calls must).
- ``run SPEC``: as stop 1.
- ``check RUN_DIR ROWS OUT_CSV``: as stop 1's, against R29: every record reproduced bit for bit (group: stored-form
  value and abserr; block: abserr; both: converged, phase_limited, regions, the four counts; the row's total and
  analytic_rad); the per-call kind and location table, with each call's location_skipped_phases.
- ``blockcheck ROWS OUT``: pass 1's and pass 2's records of the four blocks checked against each store, the
  captures' blobs extracted to state/inputs/ (not unpickled here), and each store's value per block.

Ray is never initialised (each worker reports it). Nothing is written in any worktree or to a store.
"""

import csv
import hashlib
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

REH = MAIN / "var" / "levin-phase-floor-07" / "rehearsal" / "r29"
STORES = {"p": MAIN / "var" / "datastores" / "levin-phase-floor-07-r27.sqlite",
          "o": MAIN / "var" / "datastores" / "levin-phase-floor-07-r28.sqlite",
          "g": MAIN / "var" / "datastores" / "levin-phase-floor-07-r29.sqlite"}
TAG = {"p": "R27", "o": "R28", "g": "R29"}
MAX_WORKERS = 6
GUARD = ("stationary_points", "window_pieces", "guarded_regions", "guard_at_depth_max")
GUARD_RES = {"stationary_points": "num_stationary_points", "window_pieces": "num_window_pieces",
             "guarded_regions": "num_guarded_regions", "guard_at_depth_max": "num_guard_at_depth_max"}
BLOCK_ROWS = [(10 ** 5.5, 10 ** 5.5, 10 ** 5.5, 1329131612.3012946), (1e6, 1e6, 1e6, 4017630440.8276954)]
BLOCKS = ("J4", "Y4")


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


def fieldsof(rec, counts=True):
    d = {k_: (float(rec[k_]).hex() if isinstance(rec.get(k_), float) else rec.get(k_))
         for k_ in ("value", "abserr", "converged", "phase_limited", "regions")}
    if counts:
        d.update({g: rec.get(g) for g in GUARD})
    return d


def stored_form(c, z_response):
    scale = 1.0 + float(z_response)
    return ((scale * float.fromhex(c["value"]["hex"])).hex(), (scale * float.fromhex(c["abserr"]["hex"])).hex())


def unit_spec(tree, rec, capture=None):
    uid = f"{tree}-1e-7-{rec['shard']}-{rec['serial']}"
    return {"uid": uid, "tree": tree, "rtol": "1e-7", "row": f"{rec['shard']}-{rec['serial']}", "names05": None,
            "shard": rec["shard"], "serial": rec["serial"], "k_key": f"{rec['shard']}:{rec['k_exit_serial']}",
            "budget": None, "scan": False, "w3": False, "capture": capture}


def _close(k, t):
    return all(abs(a - b) <= 1e-9 * abs(b) for a, b in zip(k, t))


def cmd_rows(out_path):
    v1_rows = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07-7460035/docs/levin-phase-floor/06g/references/v1/out/v1_rows.json")
    ins = [v1_rows] + [f for t in ("r27", "r28", "r29") for f in sorted(STORES["g"].parent.glob(f"levin-phase-floor-07-{t}*.sqlite"))]
    lines = header(Path(__file__).resolve(), "prompt 07 r29: the rows to replay and the four-block rows", ins)
    data = {t: read_store(p) for t, p in STORES.items()}
    p1 = [(10 ** 5.5, 10 ** 5.5, 10 ** 5.5, 250.0), (1e5, 10 ** 5.5, 10 ** 5.5, 250.0)]

    def find(target):
        hits = [k for k in data["g"] if _close(k, target)]
        assert len(hits) == 1, (target, hits)
        return hits[0]

    p1_keys = [find(t) for t in p1]
    blk_keys = [find(t) for t in BLOCK_ROWS]
    w = {e["wid"]: (e["k"], e["q"], e["r"], e["z_response"]) for e in json.loads(v1_rows.read_text())}
    touched = {k for k, rec in data["g"].items() if any(any(r[2].get(g) for g in GUARD) for r in levin_records(rec["metadata"]))}
    # the Levin records that differ between R28 and R29 (out/r29_vs_r28_records.csv): replayed too, for V_i
    chg_csv = REH / "out" / "r29_vs_r28_records.csv"
    lines.append(f"# input {chg_csv} sha256 {sha256(chg_csv)}")
    changed = set()
    with open(chg_csv) as f:
        for r_ in csv.DictReader(f):
            changed.add(((float(r_["k"]), float(r_["q"]), float(r_["r"]), float(r_["z_response"])), r_["record"]))
    chg_keys = {k for k, _ in changed}
    rows = []
    for key in sorted(touched | set(p1_keys) | set(w.values()) | set(blk_keys) | chg_keys):
        e = {"key": list(key), "touched": key in touched or key in chg_keys, "located": key in touched,
             "changed_vs_r28": key in chg_keys, "p1": (p1_keys.index(key) + 1) if key in p1_keys else None,
             "wid": [wid for wid, k in w.items() if k == key] or None, "block_row": key in blk_keys}
        for t in ("p", "o", "g"):
            rec = data[t][key]
            e[TAG[t]] = {k_: rec[k_] for k_ in ("shard", "serial", "k_exit_serial", "total", "total_abserr", "analytic_rad", "log10_rtol")}
        recs = {t: {name: rec for _, name, rec, _ in levin_records(data[t][key]["metadata"])} for t in ("p", "o")}
        e["records"] = []
        for kind, name, rec, sub in levin_records(data["g"][key]["metadata"]):
            nonzero = any(rec[g] for g in GUARD)
            keep = nonzero or ((e["p1"] or e["wid"]) and kind == "group" and rec.get("label") == "G+q+r") or (
                e["block_row"] and name in ("analytic.2pt5.Levin.J4_data", "analytic.2pt5.Levin.Y4_data")) or (key, name) in changed
            if not keep:
                continue
            e["records"].append({"kind": kind, "path": name, "label": rec.get("label"),
                                 "sub": None if sub is None else {"z_max": sub["z_max"], "z_min": sub["z_min"], "regime": sub["regime"]},
                                 "nonzero": nonzero, "changed_vs_r28": (key, name) in changed,
                                 "R29": fieldsof(rec), "R28": fieldsof(recs["o"][name]),
                                 "R27": fieldsof(recs["p"][name], counts=False)})
        rows.append(e)
    out = Path(out_path)
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.write_text(json.dumps({"header": lines, "rows": rows}, indent=1))
    for l in lines:
        print(l)
    print(f"changed records (R29 vs R28) {len(changed)} in {len(chg_keys)} rows, all kept: {sum(1 for e in rows for r in e['records'] if r['changed_vs_r28'])}")
    print(f"rows {len(rows)} (touched {sum(e['touched'] for e in rows)}; with a located record {sum(e['located'] for e in rows)}; P1 {sum(1 for e in rows if e['p1'])}; w {sum(1 for e in rows if e['wid'])}; "
          f"block rows {sum(e['block_row'] for e in rows)}); records {sum(len(e['records']) for e in rows)}, nonzero "
          f"{sum(1 for e in rows for r in e['records'] if r['nonzero'])} (group {sum(1 for e in rows for r in e['records'] if r['nonzero'] and r['kind'] == 'group')}, "
          f"analytic {sum(1 for e in rows for r in e['records'] if r['nonzero'] and r['kind'] == 'analytic')})")
    print(f"wrote {out} sha256 {sha256(out)}")


def run_workers(jobs, log):
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

    for l in header(Path(__file__).resolve(), f"prompt 07 r29: run {spec['name']}", [Path(spec_path)]):
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
    out = {}
    for emit in sorted(Path(run_dir).glob("*/emit.jsonl")):
        bye = hello = None
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


def block_call(u, want, blk, tree):
    """The one 'analytic <blk>' call of unit u whose fields equal the store's 2pt5 record ``want``."""
    hits = []
    for c in u.get("other_calls") or []:
        if c["notify_label"] != f"analytic {blk}":
            continue
        g = c.get("guard_07") or {}
        ok = (c["abserr"]["hex"] == want["abserr"] and c["converged"] == want["converged"]
              and c["phase_limited"] == want["phase_limited"] and c["num_regions"] == want["regions"])
        if tree != "p":
            ok = ok and all(g.get(GUARD_RES[k_]) == want[k_] for k_ in GUARD)
        if ok:
            hits.append(c)
    return hits


def cmd_spec(which, rows_path, out_path):
    rows = json.loads(Path(rows_path).read_text())["rows"]
    if which == "replay":
        es = [e for e in rows if e["touched"] or e["p1"] or e["wid"]]
        ws = [[] for _ in range(MAX_WORKERS)]
        for i, e in enumerate(es):
            ws[i % MAX_WORKERS].append(unit_spec("g", e["R29"]))
        spec = {"name": "replay", "workers": [{"name": f"replay-g-{i}", "tree": "g", "units": u} for i, u in enumerate(ws)]}
    elif which == "blocks1":
        es = [e for e in rows if e["block_row"]]
        spec = {"name": "blocks1", "workers": [{"name": f"blocks1-{t}", "tree": t, "units": [unit_spec(t, e[TAG[t]]) for e in es]}
                                               for t in ("p", "o", "g")]}
    elif which == "blocks2":
        up = load_units(REH / "state" / "capture" / "blocks1")
        ws = []
        for t in ("p", "o", "g"):
            units = []
            for e in [e for e in rows if e["block_row"]]:
                u, _, _ = up[f"{t}-1e-7-{e[TAG[t]]['shard']}-{e[TAG[t]]['serial']}"]
                cap = []
                for blk in BLOCKS:
                    rec = [r for r in e["records"] if r["path"] == f"analytic.2pt5.Levin.{blk}_data"][0]
                    hits = block_call(u, rec[TAG[t]], blk, t)
                    assert len(hits) == 1, (e["key"], t, blk, len(hits))
                    cap.append([hits[0]["index"], hits[0]["notify_label"]])
                units.append(unit_spec(t, e[TAG[t]], capture=cap))
            ws.append({"name": f"blocks2-{t}", "tree": t, "units": units})
        spec = {"name": "blocks2", "workers": ws}
    else:
        raise SystemExit(which)
    Path(out_path).write_text(json.dumps(spec, indent=1))
    print(f"wrote {out_path} sha256 {sha256(out_path)}: {sum(len(w['units']) for w in spec['workers'])} units")


def cmd_check(run_dir, rows_path, out_csv):
    rows = json.loads(Path(rows_path).read_text())["rows"]
    units = load_units(run_dir)
    lines = header(Path(__file__).resolve(), "prompt 07 r29: the replay checked against R29, record by record",
                   [Path(rows_path), *sorted(Path(run_dir).glob("*/emit.jsonl"))])
    for l in lines:
        print(l)
    table, checks = [], []
    n_ok = n_rec = 0
    ray_flags = Counter()
    skipped_calls = []
    for e in rows:
        if not (e["touched"] or e["p1"] or e["wid"]):
            continue
        uid = f"g-1e-7-{e['R29']['shard']}-{e['R29']['serial']}"
        u, bye, _ = units[uid]
        ray_flags[None if bye is None else str(bye.get("ray_initialized"))] += 1
        z = e["key"][3]
        row_ok = (u["status"] == "done" and u.get("result") is not None
                  and u["result"]["total"]["hex"] == e["R29"]["total"]
                  and u["result"]["analytic_rad"]["hex"] == e["R29"]["analytic_rad"])
        for c in (u.get("calls") or []) + (u.get("other_calls") or []):
            sk = (c.get("guard_07") or {}).get("location_skipped_phases")
            if sk:
                skipped_calls.append({"key": e["key"], "notify_label": c["notify_label"], "skipped": sk,
                                      "num_stationary_points": c["guard_07"]["num_stationary_points"],
                                      "num_guarded_regions": c["guard_07"]["num_guarded_regions"],
                                      "value": c["value"]["hex"], "abserr": c["abserr"]["hex"]})
        for r in e["records"]:
            if not (r["nonzero"] or r["kind"] == "group" or r.get("changed_vs_r28")):
                continue
            n_rec += 1
            want = r["R29"]
            if r["kind"] == "group":
                cands = [c for c in u.get("calls") or [] if c["notify_label"].endswith(f"group {r['label']}")
                         and stored_form(c, z) == (want["value"], want["abserr"])]
            else:
                blk = r["path"].split(".")[-1].replace("_data", "")
                cands = block_call(u, want, blk, "g") if r["path"].startswith("analytic.") else []
            c = cands[0] if len(cands) == 1 else None
            fields = {}
            ok = False
            if c is not None:
                g = c.get("guard_07") or {}
                fields = {"converged": c["converged"] == want["converged"], "phase_limited": c["phase_limited"] == want["phase_limited"],
                          "regions": c["num_regions"] == want["regions"], **{k_: g.get(GUARD_RES[k_]) == want[k_] for k_ in GUARD}}
                ok = row_ok and all(fields.values())
            n_ok += ok
            checks.append({"key": e["key"], "uid": uid, "path": r["path"], "label": r["label"], "n_candidates": len(cands),
                           "row_total_and_analytic_rad_equal": row_ok, "fields_equal": fields, "reproduces": ok})
            g = (c or {}).get("guard_07") or {}
            base = {"k": e["key"][0], "q": e["key"][1], "r": e["key"][2], "z_response": z, "kind": r["kind"], "record": r["path"],
                    "label": r["label"] or r["path"].split(".")[-1], "reproduces": ok,
                    "variable": "log(1+z)" if r["kind"] == "group" else "log(eta)",
                    "x_span_a": g["x_span"][0]["hex"] if g.get("x_span") else None,
                    "x_span_b": g["x_span"][1]["hex"] if g.get("x_span") else None,
                    "num_stationary_points": g.get("num_stationary_points"), "num_window_pieces": g.get("num_window_pieces"),
                    "num_guarded_regions": g.get("num_guarded_regions"), "num_guard_at_depth_max": g.get("num_guard_at_depth_max"),
                    "location_skipped_phases": json.dumps(g.get("location_skipped_phases")), "p1": e["p1"],
                    "wid": ";".join(e["wid"]) if e["wid"] else ""}
            pts = g.get("points") or []
            if pts:
                for i, p in enumerate(pts):
                    x = float.fromhex(p["x"]["hex"])
                    table.append({**base, "point": i, "point_kind": p["kind"], "found_as": p["found_as"], "phase": p["phase"],
                                  "x_hex": p["x"]["hex"], "location": repr((math.exp(x) - 1.0) if r["kind"] == "group" else math.exp(x)),
                                  "location_is": "z = exp(x) - 1" if r["kind"] == "group" else "eta = exp(x)",
                                  "theta_deriv_at_point": None if p["theta_deriv"] is None else p["theta_deriv"]["repr"],
                                  "window_pieces_of_point": p["window_pieces"]})
            else:
                table.append({**base, "point": None, "point_kind": None, "found_as": None, "phase": None, "x_hex": None,
                              "location": None, "location_is": None, "theta_deriv_at_point": None, "window_pieces_of_point": None})
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(table[0]))
        wr.writeheader()
        wr.writerows(table)
    print(f"records checked {n_rec}; reproduce bit for bit {n_ok}; not {n_rec - n_ok}")
    print(f"workers' ray_initialized at exit: {dict(ray_flags)}")
    for c in checks:
        if not c["reproduces"]:
            print(f"  NOT REPRODUCED: {c}")
    print(f"calls with a phase Rule A skipped (location_skipped_phases non-empty) in the replayed rows: {len(skipped_calls)}")
    for s in skipped_calls:
        print(f"  {s['key']} {s['notify_label'][-60:]}: skipped {s['skipped']}; points {s['num_stationary_points']}, guarded {s['num_guarded_regions']}")
    out_json = Path(out_csv).with_suffix(".checks.json")
    out_json.write_text(json.dumps({"header": lines, "checks": checks, "skipped_calls": skipped_calls}, indent=1))
    print(f"wrote {out_csv} sha256 {sha256(out_csv)} ({len(table)} lines) and {out_json} sha256 {sha256(out_json)}")
    return 0 if n_ok == n_rec else 1


def cmd_blockcheck(rows_path, out_path):
    rows = json.loads(Path(rows_path).read_text())["rows"]
    u1 = load_units(REH / "state" / "capture" / "blocks1")
    u2 = load_units(REH / "state" / "capture" / "blocks2")
    lines = header(Path(__file__).resolve(), "prompt 07 r29: the four blocks at p, o and g, checked against each store",
                   [Path(rows_path), *sorted((REH / "state" / "capture").glob("blocks*/*/emit.jsonl"))])
    for l in lines:
        print(l)
    res = []
    ok_all = True
    for e in [e for e in rows if e["block_row"]]:
        for blk in BLOCKS:
            rec = [r for r in e["records"] if r["path"] == f"analytic.2pt5.Levin.{blk}_data"][0]
            for t in ("p", "o", "g"):
                uid = f"{t}-1e-7-{e[TAG[t]]['shard']}-{e[TAG[t]]['serial']}"
                want = rec[TAG[t]]
                out = {"key": e["key"], "block": blk, "tree": t, "store": TAG[t], "uid": uid, "store_record": want}
                ok = True
                for pas, units in (("pass1", u1), ("pass2", u2)):
                    u, bye, _ = units[uid]
                    hits = block_call(u, want, blk, t)
                    row_ok = (u["status"] == "done" and u["result"]["analytic_rad"]["hex"] == e[TAG[t]]["analytic_rad"]
                              and u["result"]["total"]["hex"] == e[TAG[t]]["total"])
                    good = len(hits) == 1 and row_ok
                    ok &= good
                    out[pas] = {"n_hits": len(hits), "row_total_and_analytic_rad_equal": row_ok,
                                "ray_initialized": None if bye is None else bye.get("ray_initialized")}
                    if len(hits) == 1:
                        c = hits[0]
                        out[pas].update(index=c["index"], value=c["value"]["hex"], abserr=c["abserr"]["hex"], converged=c["converged"],
                                        phase_limited=c["phase_limited"], regions=c["num_regions"], guard_07=c.get("guard_07"))
                # the pass-2 capture of this call
                u, _, _ = u2[uid]
                ent = [x for x in (u.get("x1_inputs") or []) if x.get("index") == out["pass2"].get("index")]
                if len(ent) == 1 and not ent[0].get("error"):
                    ent = ent[0]
                    with open(REH / "state" / "capture" / "blocks2" / f"blocks2-{t}" / ent["pickle"], "rb") as f:
                        f.seek(ent["offset"])
                        blob = f.read(ent["length"])
                    if hashlib.sha256(blob).hexdigest() != ent["sha256"]:
                        raise RuntimeError(f"{uid} {blk}: blob sha256 differs")
                    dst = REH / "state" / "inputs" / f"{blk}-{t}-{e[TAG[t]]['shard']}-{e[TAG[t]]['serial']}.pkl"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists():
                        raise SystemExit(f"{dst} exists")
                    dst.write_bytes(blob)
                    out.update(pickle=str(dst), sha256=ent["sha256"], bytes=len(blob))
                else:
                    ok = False
                same_value = out["pass1"].get("value") == out["pass2"].get("value")
                ok &= same_value
                out["pass1_pass2_value_equal"] = same_value
                out["reproduces_store"] = ok
                ok_all &= ok
                res.append(out)
                print(f"{e['key']} {blk} {t} ({TAG[t]}): {'REPRODUCES' if ok else 'DOES NOT REPRODUCE'} the store's record (pass 1 and 2); "
                      f"value {out['pass1'].get('value')} abserr {out['pass1'].get('abserr')} regions {out['pass1'].get('regions')}; pickle {out.get('pickle')}")
    Path(out_path).write_text(json.dumps({"header": lines, "blocks": res}, indent=1))
    print(f"wrote {out_path} sha256 {sha256(out_path)}; all reproduce: {ok_all}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "rows":
        cmd_rows(sys.argv[2])
    elif cmd == "spec":
        cmd_spec(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "run":
        sys.exit(cmd_run(sys.argv[2]))
    elif cmd == "check":
        sys.exit(cmd_check(sys.argv[2], sys.argv[3], sys.argv[4]))
    elif cmd == "blockcheck":
        sys.exit(cmd_blockcheck(sys.argv[2], sys.argv[3]))
    else:
        raise SystemExit(f"unknown command {cmd}")
