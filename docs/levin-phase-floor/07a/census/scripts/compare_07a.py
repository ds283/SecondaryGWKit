"""levin-phase-floor prompt 07a section 6: units run by this harness against 07's census
(levin-phase-floor-07-census-20261108T171137, checkpoint 03dfe786...), bit for bit, by 07's comparison with only 07a's
new keys set aside. New in 07a, after 07's compare_07.py (a record, not edited), whose matching, evidence and diff it
follows; the diff is compare_k0.diff (06l's, unchanged), and compare_k0's ``EXCLUDED`` (timing and provenance, 07's set;
every key ending ``_s`` likewise) is extended at import by NEW_KEYS_07A below. 07's own new keys (``guard_07`` and the
rest) are in both records and are compared. Read only.

**The keys set aside, and why each** (NEW_KEYS_07A): every one is a key 07a's harness adds (worker.py, "prompt 07a");
none occurs in 07's records (checked on every unit compared: ``new_keys_in_07_records`` must be 0).
- ``route_07a`` (each call's summary; the pickle's per-call list): section 6's diagnostics;
- ``route_07a_post`` (each call's summary; the pickle): the tally of this harness's own re-solves after the timed
  section; ``route_07a_post_unit`` (unit): the same for 06h's analytic counterfactual;
- ``instrument_07a`` (unit): whether the diagnostics ran.
``phase_span`` (prompt 07a section 6: "the comparison with 07's census, which sets phase_span aside") is not a key of any
record of 07's harness, at 07 or at h; every unit compared counts its occurrences as a dict key in both records
(``phase_span_keys``: must be 0), so that setting it aside is shown to be vacuous rather than assumed.
**Normalised, not excluded** (modes ``whole``, ``calls``): a unit run at h (``h-X``, tree ``h``) is compared under 07's
names (``g-X``, ``g``), in the checkpoint record and the pickle.

Modes:
- ``whole RUN_DIR OUT_JSON``: Gate W0's second clause. Every unit of a run_k0.py run made in h with the parent's module
  installed (variant ``parent``) compared whole with 07's unit. Module evidence: each unit's
  ``AdaptiveLevin.levin_quadrature`` is the parent's file (state/variants/levin_quadrature_parent.py, 4e5da371...), counted
  under the root ``variant``, every other module from h, none foreign; each worker's hello shows the variant installed and
  bound (variant_loader's evidence all true, both production importers bound to it), the module's SHA-256 4e5da371..., h's
  own file 4ea1948b... (not executed), h at its SHA and clean; each bye, no Ray.
- ``calls RUN_DIR OUT_JSON``: per call, every Levin call (group ``calls`` and analytic ``other_calls``) of every unit of a
  run at h (run_k0.py's ``units.jsonl``, or the census's ``checkpoint.jsonl``) against 07's call. Each call is classed:
  - ``window``: a stationary-point window at 07 or at h (guard_07's ``n_windows`` > 0 in either record), with
    ``window_both``, ``window_07_only`` and ``window_h_only`` apart;
  - otherwise by section 6's diagnostics at h (route_07a): ``own`` if a solve of the class ``own`` (its span a region the
    driver made) changed its C2 route or LU choice; ``aux_only`` if only solves of the auxiliary classes did (a trust-check
    quarter, a comparison half, a fallback half, other); ``untouched`` if none did;
  - ``no_diagnostics`` if route_07a is missing or in error at h (a fault of the harness, reported).
  Every call is diffed against 07's on everything that belongs to it (compare_07.py's _call_parts: its summary, its
  entries of the unit's ``scan``, ``deep_lines``, ``deep_lines_06e``, every pickle list entry carrying its index, and a
  1e-7 group call's entry of the pickle's ``scan``), and each differing path is put in one category:
  - ``location``: guard_07's ``location_skipped_phases``, ``unlocated_minima`` and ``n_unlocated_minima`` (summary and
    pickle): the location's own report, which 07a section 1.1 item 1 changes (Rule A's entries carried a variation; 07a's
    carry a reason, and a phase Rule A skipped is now scanned for minima);
  - ``harness``: this harness's own re-solves after the timed section, which re-solve spans the driver may never have
    visited, under h's routing: the summary's ``cc_floor`` and ``declared_06h``, the unit's ``deep_lines`` and
    ``deep_lines_06e`` lines, the pickle's ``w3``, ``deep_lines``, ``deep_lines_06e``, ``cc_floor_terms`` and
    ``declared_terms_06h``;
  - ``call``: everything else (the call's value, abserr, converged, regions, leaves, its trust checks' and premise checks'
    records, its guard_07 windows and counts, its scans, ...).
  A unit all of whose calls are of the W1 classes under a reading is checked for ``result.total`` and
  ``result.total_abserr`` (hex) under that reading, and is diffed whole with its differing paths categorised (the unit's
  ``analytic_06h.parent_counterfactual`` and ``w3`` are ``harness``).
  Writes OUT_JSON (provenance, counts by class and category, per-unit summaries) and OUT_JSON's ``.calls.jsonl`` beside it
  (one line per call: unit, kind, index, class, windows at 07 and h, route_07a counts, the differing categories and the
  first differing paths, and value, abserr and converged at 07 and at h, hex).
- ``det RUN_DIR_A RUN_DIR_B OUT_JSON``: Gate W0's fourth clause. The units two run_k0.py runs share, compared whole with
  compare_k0's EXCLUDED alone (timing and provenance): every key compared, 07a's included.
- ``instr RUN_DIR_ON RUN_DIR_OFF OUT_JSON``: Gate W0's third clause. The units of a run with the diagnostics on and one
  with them off, compared whole with compare_k0's EXCLUDED and NEW_KEYS_07A set aside; the run ON must carry route_07a on
  every call and the run OFF on none.

Floats are compared by bit pattern (``float.hex``; NaN by repr), arrays by dtype, shape and bytes. Prints a summary.
Exits 1 unless every comparison the mode's verdict rests on is equal and every evidence check holds (``calls`` exits 0
whatever the classes show: it describes, and the scoring reads it).

Usage: PYTHONPATH=<tree h>:<scripts> python -B compare_07a.py whole|calls RUN_DIR OUT_JSON [--processes N]
       PYTHONPATH=<tree h>:<scripts> python -B compare_07a.py det|instr RUN_DIR_A RUN_DIR_B OUT_JSON
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
from prov import (CHECKPOINT07_SHA256, H_TREE, LQ_SHA256_07AH, LQ_SHA256_07APARENT, PARENT_VARIANT_07A, RUN07,  # noqa: E402
                  TREES, header, sha256)

EXCLUDED_07 = frozenset(CK.EXCLUDED)  # 07's set (06m's), as compare_k0.py has it
NEW_KEYS_07A = frozenset({"route_07a", "route_07a_post", "route_07a_post_unit", "instrument_07a"})
INDEX = HERE.parent / "state" / "index_07.json"
LOC_RE = re.compile(r"\.guard_07(\[\d+\])?\.(location_skipped_phases|unlocated_minima|n_unlocated_minima)\b")
HARNESS_RE = re.compile(r"\.summary\[\d+\]\.(cc_floor|declared_06h)\b|\.data\.(deep_lines|deep_lines_06e)\b|"
                        r"\.pickle\.(w3|deep_lines|deep_lines_06e|cc_floor_terms|declared_terms_06h)\b")
UNIT_HARNESS_RE = re.compile(r"^data\.(analytic_06h\.parent_counterfactual|w3|deep_lines|deep_lines_06e)\b|"
                             r"^data\.(calls|other_calls)\[\d+\]\.(cc_floor|declared_06h)\b|"
                             r"^pickle\.(w3|deep_lines|deep_lines_06e|cc_floor_terms|declared_terms_06h)\b")
UNIT_LOC_RE = re.compile(r"\.guard_07\.(location_skipped_phases|unlocated_minima|n_unlocated_minima)\b|"
                         r"^pickle\.guard_07\[\d+\]\.(location_skipped_phases|unlocated_minima)\b")


def set_excluded(with_new):
    """compare_k0.excluded() reads compare_k0.EXCLUDED at call time."""
    CK.EXCLUDED = set(EXCLUDED_07 | (NEW_KEYS_07A if with_new else frozenset()))


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


def count_keys(x, names):
    """How many times a name in names occurs as a dict key in x."""
    n = 0
    if isinstance(x, dict):
        for k, v in x.items():
            n += (k in names) + count_keys(v, names)
    elif isinstance(x, (list, tuple)):
        for v in x:
            n += count_keys(v, names)
    return n


def load_07(ent):
    with open(RUN07 / "checkpoint.jsonl", "rb") as f:
        f.seek(ent["offset"])
        line = f.read(ent["length"])
    if hashlib.sha256(line).hexdigest() != ent["line_sha256"]:
        raise SystemExit(f"07's checkpoint line sha256 differs at offset {ent['offset']}")
    g = json.loads(line)["data"]
    gp = ent["pickle"]
    grec = CK.load_pickle(RUN07 / gp[0], gp[1], gp[2], gp[3])
    return g, grec


def load_mine(d, pdir):
    return CK.load_pickle(Path(pdir) / d["pickle"], d["pickle_offset"], d["pickle_length"], d["pickle_record_sha256"])


def uid_07(uid):
    return "g-" + uid.split("-", 1)[1]


def by_index(lst):
    out = {}
    for e in lst or []:
        if isinstance(e, dict) and "index" in e:
            out.setdefault(e["index"], []).append(e)
    return out


def normalise(uid, d, prec):
    """Copies of a unit's record and pickle record with h's unit and tree under 07's names."""
    d2, prec2 = dict(d), dict(prec)
    normalised = {}
    for name, rec in (("data", d2), ("pickle", prec2)):
        if rec.get("unit") == uid and uid != uid_07(uid):
            rec["unit"] = uid_07(uid)
            normalised[f"{name}.unit"] = [uid, uid_07(uid)]
    if d2.get("tree") != "g":
        normalised["data.tree"] = [d2.get("tree"), "g"]
        d2["tree"] = "g"
    return d2, prec2, normalised


def unit_evidence(d, variant):
    mods = d.get("modules") or {}
    lq_file = str(PARENT_VARIANT_07A.resolve()) if variant else str(H_TREE / "AdaptiveLevin" / "levin_quadrature.py")
    roots = mods.get("counts_by_root") or {}
    return {
        "lq_file_is_expected": (mods.get("files") or {}).get("AdaptiveLevin.levin_quadrature") == lq_file,
        "all_in_tree": bool(mods.get("all_in_tree")),
        "no_foreign": not mods.get("foreign"),
        "variant_root_as_expected": ("variant" in roots) == bool(variant),
    }


def one_whole(args):
    """Mode whole: one unit (at the parent's module, in h) against 07's, whole (07a's new keys set aside)."""
    uid, line, pdir, ent = args
    set_excluded(True)
    d = json.loads(line)["data"]
    row = {"unit": uid, "unit_07": uid_07(uid), "status": d["status"]}
    g, grec = load_07(ent)
    row["status_07"] = g["status"]
    if d["status"] != "done" or g["status"] != "done":
        row.update(ok=False, differing=["status"], excl_present=[], excl_differing=[])
        return row
    prec = load_mine(d, pdir)
    d2, prec2, normalised = normalise(uid, d, prec)
    out = []
    matched = call_keys(d2) == call_keys(g) and region_keys(prec2) == region_keys(grec)
    if not matched:
        out.append(("match", False, "calls or region records do not match"))
    CK.diff(d2, g, "data", out)
    CK.diff(prec2, grec, "pickle", out)
    ev = unit_evidence(d, variant=True)
    differing = sorted({p for p, e, _ in out if not e})
    row.update(
        ok=matched and not differing and all(ev.values()),
        normalised=normalised,
        differing=differing[:200],
        detail=[list(x) for x in out if not x[1]][:50],
        excl_differing=sorted({norm(p) for p, e, _ in out if e}),
        new_keys_in_07_records=count_keys(g, NEW_KEYS_07A) + count_keys(grec, NEW_KEYS_07A),
        phase_span_keys=count_keys(d, {"phase_span"}) + count_keys(prec, {"phase_span"}) + count_keys(g, {"phase_span"})
        + count_keys(grec, {"phase_span"}),
        n_calls=len(call_keys(d2)),
        n_group_calls=sum(1 for k in call_keys(d2) if k[0] == "group"),
        n_regions=len(region_keys(prec2)),
        n_regions_returned=int(sum(len(r.get("start", [])) for r in (prec.get("regions") or []) + (prec.get("other_regions") or []))),
        evidence=ev,
    )
    row["ok"] = row["ok"] and row["new_keys_in_07_records"] == 0 and row["phase_span_keys"] == 0
    return row


def _call_parts(d, prec, kind, i, pos):
    """Everything that belongs to call i of a unit record d and its pickle record prec (compare_07.py's, unchanged)."""
    key = "calls" if kind == "group" else "other_calls"
    parts = {"summary": [c for c in (d.get(key) or []) if c["index"] == i], "data": {}, "pickle": {}}
    parts["data"]["scan"] = [e for e in (d.get("scan") or []) if isinstance(e, dict) and e.get("index") == i]
    for k in ("deep_lines", "deep_lines_06e"):
        parts["data"][k] = {"lines": [e for e in ((d.get(k) or {}).get("lines") or []) if e.get("index") == i]}
    for k in sorted(prec):
        v = prec.get(k)
        if k == "scan":
            if kind == "group" and isinstance(v, list):
                parts["pickle"]["scan"] = [v[pos]] if 0 <= pos < len(v) else ["<missing>"]
            continue
        if isinstance(v, list):
            parts["pickle"][k] = by_index(v).get(i, [])
    return parts


def category(path):
    if LOC_RE.search(path):
        return "location"
    if HARNESS_RE.search(path):
        return "harness"
    return "call"


def unit_category(path):
    if UNIT_LOC_RE.search(path):
        return "location"
    if UNIT_HARNESS_RE.search(path):
        return "harness"
    return "call"


def fx_of(s, k):
    v = s.get(k) if s else None
    return v.get("hex") if isinstance(v, dict) else v


def one_calls(args):
    """Mode calls: every call against 07's, classed and diffed by category."""
    uid, line, pdir, ent = args
    set_excluded(True)
    d = json.loads(line)["data"]
    row = {"unit": uid, "unit_07": uid_07(uid), "status": d["status"], "rtol": d.get("rtol")}
    g, grec = load_07(ent)
    row["status_07"] = g["status"]
    if d["status"] != "done" or g["status"] != "done":
        row.update(ok=False, problem="status", calls=[])
        return row
    prec = load_mine(d, pdir)
    d2, prec2, normalised = normalise(uid, d, prec)
    ck_m, ck_g = call_keys(d2), call_keys(g)
    row["matched"] = ck_m == ck_g and region_keys(prec2) == region_keys(grec)
    if not row["matched"]:
        row.update(ok=False, problem=f"calls {len(ck_m)}/{len(ck_g)} or region records do not match", calls=[])
        return row
    pos_of = {c["index"]: j for j, c in enumerate(d2.get("calls") or [])}
    pos_of_g = {c["index"]: j for j, c in enumerate(g.get("calls") or [])}
    calls = []
    for kind, i, label, _span in ck_m:
        key = "calls" if kind == "group" else "other_calls"
        s = next(c for c in d2[key] if c["index"] == i)
        s7 = next(c for c in g[key] if c["index"] == i)
        g7h, g77 = s.get("guard_07") or {}, s7.get("guard_07") or {}
        r7a = s.get("route_07a")
        win07 = bool(g77.get("n_windows"))
        winh = bool(g7h.get("n_windows"))
        if win07 or winh:
            cls = "window_both" if (win07 and winh) else ("window_07_only" if win07 else "window_h_only")
        elif not isinstance(r7a, dict) or "error" in r7a:
            cls = "no_diagnostics"
        elif r7a["own_differs"]:
            cls = "own"
        elif r7a["aux_differs"]:
            cls = "aux_only"
        else:
            cls = "untouched"
        out = []
        a = _call_parts(d2, prec2, kind, i, pos_of.get(i, -1))
        b = _call_parts(g, grec, kind, i, pos_of_g.get(i, -1))
        CK.diff(a, b, f"call[{kind}:{i}]", out)
        diffs = sorted({p for p, e, _ in out if not e})
        cats = {}
        for p in diffs:
            cats.setdefault(category(p), []).append(p)
        post = s.get("route_07a_post") or {}
        calls.append({
            "kind": kind, "index": i, "label": label, "class": cls, "win07": win07, "winh": winh,
            "n_windows_07": g77.get("n_windows"), "n_windows_h": g7h.get("n_windows"),
            "n_points_07": g77.get("n_points"), "n_points_h": g7h.get("n_points"),
            "guarded_07": g77.get("num_guarded_regions"), "guarded_h": g7h.get("num_guarded_regions"),
            "skipped_07": g77.get("location_skipped_phases"), "skipped_h": g7h.get("location_skipped_phases"),
            "route": None if not isinstance(r7a, dict) or "error" in r7a else {
                k: r7a[k] for k in ("n_solves", "n_route_differs", "n_lu_differs", "n_route_differs_own", "n_lu_differs_own",
                                    "n_route_differs_aux", "n_lu_differs_aux", "differs_by_class",
                                    "n_partition_nonforced", "n_cc_with_levin_half", "n_forced")},
            "post_differs": {k: [v.get("n_route_differs"), v.get("n_lu_differs")] for k, v in post.items() if v},
            "equal": not diffs,
            "categories": sorted(cats),
            "n_differing": {c_: len(v) for c_, v in cats.items()},
            "first_differing": {c_: v[:6] for c_, v in cats.items()},
            "value": [fx_of(s, "value"), fx_of(s7, "value")],
            "abserr": [fx_of(s, "abserr"), fx_of(s7, "abserr")],
            "converged": [s.get("converged"), s7.get("converged")],
            "num_regions": [s.get("num_regions"), s7.get("num_regions")],
            "max_depth": [s.get("max_depth"), s7.get("max_depth")],
        })
    # the unit, under each reading of W1: (A) every solve counts, (B) the own solves only
    w1_classes = {"A": {"untouched"}, "B": {"untouched", "aux_only"}}
    t_m, t_g = d2.get("result") or {}, g.get("result") or {}
    row["total"] = [fx_of(t_m, "total"), fx_of(t_g, "total")]
    row["total_abserr"] = [fx_of(t_m, "total_abserr"), fx_of(t_g, "total_abserr")]
    row["total_converged"] = [t_m.get("total_converged"), t_g.get("total_converged")]
    row["totals_equal"] = row["total"][0] == row["total"][1] and row["total_abserr"][0] == row["total_abserr"][1]
    row["all_w1"] = {r_: all(c["class"] in cl for c in calls) for r_, cl in w1_classes.items()}
    if row["all_w1"]["A"] or row["all_w1"]["B"]:
        out = []
        CK.diff(d2, g, "data", out)
        CK.diff(prec2, grec, "pickle", out)
        diffs = sorted({p for p, e, _ in out if not e})
        cats = {}
        for p in diffs:
            cats.setdefault(unit_category(p), []).append(p)
        row["whole"] = {"equal": not diffs, "categories": sorted(cats), "n_differing": {c_: len(v) for c_, v in cats.items()},
                        "first_differing": {c_: v[:8] for c_, v in cats.items()}}
    row.update(
        ok=True,
        normalised=normalised,
        new_keys_in_07_records=count_keys(g, NEW_KEYS_07A) + count_keys(grec, NEW_KEYS_07A),
        phase_span_keys=count_keys(d, {"phase_span"}) + count_keys(prec, {"phase_span"}) + count_keys(g, {"phase_span"})
        + count_keys(grec, {"phase_span"}),
        post_unit=d.get("route_07a_post_unit") and {k: v for k, v in d["route_07a_post_unit"].items() if k != "changed"},
        evidence=unit_evidence(d, variant=False),
        calls=calls,
    )
    return row


def worker_evidence(run_dir, variant):
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
        v = None if h is None else h.get("variant_06k")
        c = {
            "tree": tree,
            "hello": h is not None,
            "levin_quadrature_sha256": None if h is None else h.get("levin_quadrature_sha256"),
            "lq_file": None if h is None else h["files"].get("AdaptiveLevin.levin_quadrature"),
            "tree_file_is_h": h is not None and h.get("levin_quadrature_tree_file_sha256") == LQ_SHA256_07AH,
            "all_in_tree": h is not None and bool(h.get("all_in_tree")),
            "tree_head_clean": h is not None and tree in TREES and h.get("head") == TREES[tree][1]
            and h.get("status_short", "x").strip() == "",
            "bye": b is not None,
            "ray_not_initialized": b is not None and b.get("ray_initialized") is False,
            "no_foreign_modules": b is not None and not b.get("modules_foreign"),
        }
        if variant:
            c["lq_is_parent"] = h is not None and h.get("levin_quadrature_sha256") == LQ_SHA256_07APARENT
            c["variant_installed_and_bound"] = v is not None and v.get("path") == str(PARENT_VARIANT_07A.resolve()) and all(
                v.get(k) for k in ("sys_modules_is_variant", "package_attribute_is_variant", "package_function_is_variants",
                                   "QuadSourceIntegral_binds_variants", "three_bessel_integrals_binds_variants")) and \
                v.get("sha256") == v.get("sha256_now") == LQ_SHA256_07APARENT
            c["tree_lq_not_loaded"] = b is not None and not any(
                f == str(H_TREE / "AdaptiveLevin" / "levin_quadrature.py") for _n, f in (b.get("modules_own_tree") or []))
        else:
            c["lq_is_h"] = h is not None and h.get("levin_quadrature_sha256") == LQ_SHA256_07AH
            c["variant_none"] = h is not None and v is None
        checks[name] = c
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
            if not re.fullmatch(r"h-1e-[78]-\d+-\d+", uid):
                skipped.append(uid)  # e.g. an over-budget pass's record, compared apart
                continue
            if uid in seen:
                raise SystemExit(f"{uid} twice in {src}")
            seen.add(uid)
            if uid_07(uid) not in idx:
                raise SystemExit(f"{uid_07(uid)} is not in 07's census")
            pdir = run_dir / r["worker"] if k0 else run_dir
            jobs.append((uid, line, str(pdir), idx[uid_07(uid)]))
    return src, jobs, skipped


def run_pair(a, mode):
    """Modes det and instr: two run_k0.py runs, unit by unit, whole."""
    set_excluded(mode == "instr")
    ra, rb = Path(a.run_dir).resolve(), Path(a.run_dir_b).resolve()
    title = ("Gate W0 (4), determinism" if mode == "det" else "Gate W0 (3), the instrument on against off")
    hdr = header(Path(__file__).resolve(), f"levin-phase-floor 07a: {title}, {ra.name} against {rb.name}",
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
        cs_a = (da.get("calls") or []) + (da.get("other_calls") or [])
        cs_b = (db.get("calls") or []) + (db.get("other_calls") or [])
        g7 = [c.get("guard_07") for c in cs_a]
        r7 = [c.get("route_07a") for c in cs_a]
        ok = not differing and da["status"] == "done" and db["status"] == "done"
        extra = {}
        if mode == "instr":
            extra["A_route_07a_on_every_call"] = bool(cs_a) and all(isinstance(c.get("route_07a"), dict) and "error" not in c["route_07a"] for c in cs_a)
            extra["B_route_07a_on_no_call"] = all(c.get("route_07a") is None for c in cs_b) and db.get("instrument_07a") is False
            ok = ok and extra["A_route_07a_on_every_call"] and extra["B_route_07a_on_no_call"] and da.get("instrument_07a") is True
        rows.append({"unit": uid, "status": [da["status"], db["status"]], "ok": ok,
                     "differing": differing[:200], "detail": [list(x) for x in out if not x[1]][:50],
                     "excl_differing": sorted({norm(p) for p, e, _ in out if e}),
                     "n_calls": len(g7), "n_calls_windowed": sum(1 for x in g7 if isinstance(x, dict) and x.get("n_windows")),
                     "n_calls_route_changed": sum(1 for x in r7 if isinstance(x, dict) and (x.get("n_route_differs") or x.get("n_lu_differs"))),
                     "route_changes": [(c["index"], c["route_07a"].get("n_route_differs"), c["route_07a"].get("n_lu_differs"),
                                        c["route_07a"].get("own_differs"), c["route_07a"].get("aux_differs"))
                                       for c in cs_a if isinstance(c.get("route_07a"), dict) and "error" not in c["route_07a"]
                                       and (c["route_07a"].get("n_route_differs") or c["route_07a"].get("n_lu_differs"))],
                     "windows": [(c["index"], c["guard_07"].get("n_windows"), c["guard_07"].get("n_points_by_kind"))
                                 for c in cs_a if isinstance(c.get("guard_07"), dict) and c["guard_07"].get("n_windows")],
                     "wall_s": [da.get("wall_s"), db.get("wall_s")], **extra})
    wev = {**{f"A:{k}": v for k, v in worker_evidence(ra, False).items()}, **{f"B:{k}": v for k, v in worker_evidence(rb, False).items()}}
    wev_ok = bool(wev) and all(all(v for k, v in c.items() if k not in ("levin_quadrature_sha256", "lq_file", "tree")) for c in wev.values())
    n_fail = sum(1 for r in rows if not r["ok"])
    doc = {"header": hdr, "mode": mode, "excluded": sorted(CK.EXCLUDED) + ["*_s"], "n_units": len(rows),
           "n_units_identical": len(rows) - n_fail, "worker_evidence_all_hold": wev_ok, "worker_evidence": wev, "units": rows}
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(doc, indent=1, default=str))
    for r in rows:
        print(f"{r['unit']}: {'IDENTICAL' if r['ok'] else 'DIFFERS'}; status {r.get('status')}; calls {r.get('n_calls')}; "
              f"windowed {r.get('n_calls_windowed')} {r.get('windows')}; route-changed {r.get('n_calls_route_changed')} "
              f"{r.get('route_changes')}; wall {r.get('wall_s')}; differing {r['differing'][:10]}; excluded keys differing "
              f"{r.get('excl_differing')}" + (f"; on/off {r.get('A_route_07a_on_every_call')}/{r.get('B_route_07a_on_no_call')}" if mode == "instr" else ""))
    print(f"worker evidence all hold {wev_ok}")
    print(f"wrote {a.out_json} sha256 {sha256(a.out_json)}")
    return 0 if n_fail == 0 and wev_ok else 1


CLASSES = ("untouched", "aux_only", "own", "window_both", "window_07_only", "window_h_only", "no_diagnostics")


def summarise_calls(rows):
    s = {"by_class": {}, "by_class_rtol": {}, "equal_by_class": {}, "categories_by_class": {}, "units": {}}
    for r in rows:
        for c in r.get("calls") or []:
            k = c["class"]
            s["by_class"][k] = s["by_class"].get(k, 0) + 1
            kr = f"{r['rtol']}|{c['kind']}|{k}"
            s["by_class_rtol"][kr] = s["by_class_rtol"].get(kr, 0) + 1
            e = s["equal_by_class"].setdefault(k, {"equal": 0, "differing": 0})
            e["equal" if c["equal"] else "differing"] += 1
            key = "+".join(c["categories"]) or "none"
            cc = s["categories_by_class"].setdefault(k, {})
            cc[key] = cc.get(key, 0) + 1
    for reading in ("A", "B"):
        all_ = [r for r in rows if (r.get("all_w1") or {}).get(reading)]
        s["units"][reading] = {"n_all_w1": len(all_), "n_totals_equal": sum(1 for r in all_ if r["totals_equal"]),
                               "n_whole_equal": sum(1 for r in all_ if (r.get("whole") or {}).get("equal")),
                               "whole_categories": {}}
        for r in all_:
            key = "+".join((r.get("whole") or {}).get("categories") or []) or "none"
            s["units"][reading]["whole_categories"][key] = s["units"][reading]["whole_categories"].get(key, 0) + 1
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("whole", "calls", "det", "instr"))
    ap.add_argument("run_dir")
    ap.add_argument("rest", nargs="+")
    ap.add_argument("--processes", type=int, default=1)
    a = ap.parse_args()
    if a.processes > 6:
        raise SystemExit("at most six processes")
    if a.mode in ("det", "instr"):
        if len(a.rest) != 2:
            raise SystemExit(f"{a.mode} RUN_DIR_A RUN_DIR_B OUT_JSON")
        a.run_dir_b, a.out_json = a.rest
        return run_pair(a, a.mode)
    if len(a.rest) != 1:
        raise SystemExit(f"{a.mode} RUN_DIR OUT_JSON")
    a.out_json = a.rest[0]
    set_excluded(True)
    run_dir = Path(a.run_dir).resolve()
    idx_doc = json.loads(INDEX.read_text())
    if idx_doc["checkpoint_sha256"] != CHECKPOINT07_SHA256:
        raise SystemExit("the index is not of 07's checkpoint")
    src, jobs, skipped = jobs_of(run_dir, idx_doc["units"])
    hdr = header(Path(__file__).resolve(), f"levin-phase-floor 07a: {run_dir.name} against 07's census ({RUN07.name}), "
                 f"mode {a.mode}", [src, INDEX, HERE / "compare_k0.py"])
    for l in hdr:
        print(l, flush=True)
    print(f"# {len(jobs)} units to compare ({len(skipped)} other records skipped: {skipped[:10]}), {a.processes} "
          f"process(es); excluded: 07's {sorted(EXCLUDED_07)} and every key ending _s; set aside as new: "
          f"{sorted(NEW_KEYS_07A)}", flush=True)
    fn = one_whole if a.mode == "whole" else one_calls
    if a.processes > 1:
        with mp.get_context("spawn").Pool(a.processes) as pool:
            rows = list(pool.imap(fn, jobs, chunksize=8))
    else:
        rows = [fn(j) for j in jobs]
    variant = a.mode == "whole"
    wev = worker_evidence(run_dir, variant)
    wev_ok = bool(wev) and all(all(v for k, v in c.items() if k not in ("levin_quadrature_sha256", "lq_file", "tree")) for c in wev.values())
    ev_tot = {}
    for r in rows:
        for k, v in (r.get("evidence") or {}).items():
            ev_tot.setdefault(k, [0, 0])[0 if v else 1] += 1
    doc = {"header": hdr, "mode": a.mode, "run_dir": str(run_dir), "source": str(src), "source_sha256": sha256(src),
           "run07": str(RUN07), "checkpoint07_sha256": CHECKPOINT07_SHA256,
           "excluded_07": sorted(EXCLUDED_07) + ["*_s"], "set_aside_new_07a": sorted(NEW_KEYS_07A),
           "records_skipped": skipped, "n_units": len(rows),
           "statuses": {s: sum(1 for r in rows if r["status"] == s) for s in sorted({r["status"] for r in rows})},
           "unit_evidence_true_false": ev_tot, "worker_evidence_all_hold": wev_ok, "worker_evidence": wev,
           "new_keys_in_07_records": sum(r.get("new_keys_in_07_records", 0) or 0 for r in rows),
           "phase_span_keys": sum(r.get("phase_span_keys", 0) or 0 for r in rows)}
    out_p = Path(a.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if a.mode == "whole":
        n_fail = sum(1 for r in rows if not r["ok"])
        inv = {}
        for r in rows:
            for p in r.get("excl_differing", []):
                inv[p] = inv.get(p, 0) + 1
        tot = {k: sum(r.get(k, 0) or 0 for r in rows) for k in ("n_calls", "n_group_calls", "n_regions", "n_regions_returned")}
        doc.update(n_units_equal=len(rows) - n_fail, n_units_differing=n_fail, totals=tot,
                   excluded_keys_differing_units=dict(sorted(inv.items())),
                   first_differing_units=[r for r in rows if not r["ok"]][:20],
                   units=[{k: v for k, v in r.items() if k != "detail"} | ({"detail": r.get("detail")} if not r["ok"] else {})
                          for r in rows])
        out_p.write_text(json.dumps(doc, indent=1, default=str))
        for r in [r for r in rows if not r["ok"]][:20]:
            print(f"DIFFERS {r['unit']}: {r['differing'][:10]}", flush=True)
        print(f"units {len(rows)}; equal {len(rows) - n_fail}; differing {n_fail}; statuses {doc['statuses']}")
        print(f"totals {tot}; new keys in 07's records {doc['new_keys_in_07_records']}; phase_span keys {doc['phase_span_keys']}")
        print(f"unit evidence (true, false): {ev_tot}")
        print(f"worker evidence: {len(wev)} workers, all hold {wev_ok}; levin_quadrature sha256 "
              f"{sorted({str(c['levin_quadrature_sha256']) for c in wev.values()})}")
        print(f"excluded keys differing (path: units): {doc['excluded_keys_differing_units']}")
        print(f"wrote {out_p} sha256 {sha256(out_p)}")
        return 0 if n_fail == 0 and wev_ok and doc["new_keys_in_07_records"] == 0 else 1
    # calls
    calls_p = out_p.with_name(out_p.name.replace(".json", "") + ".calls.jsonl")
    with open(calls_p, "w") as fc:
        for r in rows:
            for c in r.get("calls") or []:
                fc.write(json.dumps({"unit": r["unit"], "rtol": r["rtol"], **c}, default=str) + "\n")
    summ = summarise_calls(rows)
    problems = [{k: r.get(k) for k in ("unit", "status", "status_07", "problem")} for r in rows if not r.get("ok")]
    doc.update(summary=summ, problems=problems, calls_jsonl=str(calls_p), calls_jsonl_sha256=sha256(calls_p),
               units=[{k: v for k, v in r.items() if k != "calls"} for r in rows])
    out_p.write_text(json.dumps(doc, indent=1, default=str))
    print(f"units {len(rows)}; problems {len(problems)} {problems[:5]}; statuses {doc['statuses']}")
    print(f"calls by class: {summ['by_class']}")
    print(f"equal by class: {summ['equal_by_class']}")
    print(f"differing categories by class: {json.dumps(summ['categories_by_class'])}")
    print(f"units all of W1's classes, reading A (every solve) / B (own solves): {json.dumps(summ['units'])}")
    print(f"new keys in 07's records {doc['new_keys_in_07_records']}; phase_span keys {doc['phase_span_keys']}")
    print(f"unit evidence (true, false): {ev_tot}")
    print(f"worker evidence: {len(wev)} workers, all hold {wev_ok}")
    print(f"wrote {calls_p} sha256 {sha256(calls_p)}")
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0 if not problems and wev_ok and doc["new_keys_in_07_records"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
