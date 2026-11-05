"""levin-phase-floor prompt 06g section 5, the references: the selection the captures run on. New in
06g, after 06h's phase agent's select_06h.py (kept in state/scripts_06h_phase_reference/).

The 90 calls: Z3's 30 (06c's depth20-tiling ``results.json``'s ``calls``, ids z00..z29 by
``sample_no``; 06i's 13 are among them) and the 60 drawn (out/sample/draw.json, r00..r59). Each is
matched in two censuses by the unit's ``<rtol>-<shard>-<serial>`` and the call index, with the same
notify_label and x_span (hex):
- **the g census** (the census agent's run, read only; ``refs_common.census_run_06g``): its record is
  what a capture must reproduce bit for bit (value, abserr, num_regions, max_depth, converged,
  label: x1_capture.py; the region arrays: check_captures.py). Every ``done`` record of the unit is
  read; if more than one holds the call they must agree. Records that are not ``done`` (an
  over-budget main-pass record) are counted and not used. The unit's budget is the ``done``
  record's ``budget_s``;
- **06h's census** (RUN06H, 3104c18e..., checked): v_06h and abserr_06h.
For 06i's 13 it also carries 06i's reference (its FCC res-2 reading) and 06i's m (min of abserr
06c and 06e, 06i's scores.json). A call missing from either census, or whose records disagree, stops
this script with nothing written.

Addendum 2 (2026-11-01): V8's 12 lost calls, ids v00..v11, from the orchestrator's
review/v8_sample.json (874e17db..., checked; not reselected), matched and checked in both censuses
the same way, their records also checked against the file's; a V8 call already among the 90 is
captured once under its V3 id and listed in out/sample/v8_aliases.json.

Writes out/sample/selection.json (x1_capture.py's SELECTION_JSON: unit ``g-<rest>``, index,
notify_label and the census_* fields, plus everything above). Reads only.

Usage: python -B select_refs.py   (from var/levin-phase-floor-06g/references/)
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402


def slim(c):
    """The census call record, whole (it is a dict of JSON values)."""
    return c


def main():
    run06g = RC.census_run_06g()
    ck = {"g": run06g / "checkpoint.jsonl", "06h": RC.RUN06H / "checkpoint.jsonl"}
    draw_p = RC.ROOT / "out" / "sample" / "draw.json"
    out = RC.ROOT / "out" / "sample" / "selection.json"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwritten")
    for l in header(
        Path(__file__).resolve(),
        "levin-phase-floor 06g: the references' selection (Z3's 30, the 60 drawn and addendum 2's V8 12), with both censuses' records",
        [RC.Z3_RESULTS, draw_p, ck["g"], ck["06h"], RC.I06_SCORES, RC.V8_SAMPLE],
    ):
        print(l)
    if RC.sha256(ck["06h"]) != RC.CHECKPOINT06H_SHA256:
        raise SystemExit("06h's census checkpoint is not 3104c18e...")
    print(f"g census run: {run06g}")
    entries = []
    for c in sorted(
        json.loads(RC.Z3_RESULTS.read_text())["calls"], key=lambda c: c["sample_no"]
    ):
        entries.append(
            {
                "sid": f"z{c['sample_no']:02d}",
                "source": "z3",
                "sample_no": c["sample_no"],
                "in_06i": c["sample_no"] in RC.I06_SAMPLES,
                "rest": RC.rest_of(c["unit"]),
                "index": c["index"],
                "notify_label": c["notify_label"],
                "block": c["block"],
                "rtol": c["rtol"],
            }
        )
    draw = json.loads(draw_p.read_text())
    for e in draw["drawn"]:
        entries.append(
            {
                "sid": e["sid"],
                "source": "draw",
                "sample_no": None,
                "in_06i": False,
                "rest": e["rest"],
                "index": e["index"],
                "notify_label": e["notify_label"],
                "block": None,
                "rtol": e["rtol"],
                "stratum": e["stratum"],
            }
        )
    if len(entries) != 90 or len({(e["rest"], e["index"]) for e in entries}) != 90:
        raise SystemExit("expected 90 distinct calls")
    # addendum 2: V8's 12 lost calls (review/v8_sample.json, not reselected), ids v00..v11 in the
    # file's order; a V8 call that is also one of the 90 is captured once, under the V3 id, and
    # recorded in out/sample/v8_aliases.json (it is scored in both tables, independently)
    if RC.sha256(RC.V8_SAMPLE) != RC.V8_SAMPLE_SHA256:
        raise SystemExit("review/v8_sample.json is not 874e17db...")
    v8 = json.loads(RC.V8_SAMPLE.read_text())["samples"]
    if len(v8) != 12:
        raise SystemExit("expected V8's 12 calls")
    by_key = {(e["rest"], e["index"]): e for e in entries}
    aliases = {}
    for i, s8 in enumerate(v8):
        vid = f"v{i:02d}"
        k = (RC.rest_of(s8["unit"]), s8["call_index"])
        if k in by_key:
            aliases[vid] = by_key[k]["sid"]
            by_key[k]["also_v8"] = vid
            continue
        e = {
            "sid": vid,
            "source": "v8",
            "sample_no": None,
            "in_06i": False,
            "rest": k[0],
            "index": k[1],
            "notify_label": s8["notify_label"],
            "block": s8["group"],
            "rtol": s8["rtol"],
            "v8": s8,
        }
        entries.append(e)
        by_key[k] = e
    wanted = {(e["rest"], e["index"]) for e in entries}
    found = {name: RC.scan_checkpoint(ck[name], wanted) for name in ck}
    i06 = {r["sample"]: r for r in json.loads(RC.I06_SCORES.read_text())["rows"]}
    problems = []
    sel = []
    for e in entries:
        k = (e["rest"], e["index"])
        r = dict(e)
        for name in ("g", "06h"):
            recs = found[name].get(k) or []
            done = [x for x in recs if x[1] == "done"]
            r[f"{name}_n_records"] = len(recs)
            r[f"{name}_n_records_not_done"] = len(recs) - len(done)
            if not done:
                problems.append(f"{e['sid']} {k}: no done record in the {name} census")
                continue
            if len({RC.call_key(x[3]) for x in done}) != 1:
                problems.append(f"{e['sid']} {k}: {len(done)} {name} records disagree")
                continue
            unit, status, kind, c, data = done[0]
            if c["notify_label"] != e["notify_label"]:
                problems.append(
                    f"{e['sid']} {k}: {name} label {c['notify_label']!r} != {e['notify_label']!r}"
                )
                continue
            if r.get("kind") is not None and r["kind"] != kind:
                problems.append(f"{e['sid']} {k}: kinds differ between censuses")
            r["kind"] = kind
            xs = [RC.hexof(v) for v in c["x_span"]]
            if r.get("x_span") is not None and r["x_span"] != xs:
                problems.append(f"{e['sid']} {k}: x_span differs between censuses")
            r["x_span"] = xs
            r[f"{name}_unit"] = unit
            r[f"{name}_record"] = slim(c)
            if name == "g":
                r["census_unit_budget_s"] = data.get("budget_s")
                r["census_unit_status"] = status
        if e["source"] == "draw":
            d = next(x for x in draw["drawn"] if x["sid"] == e["sid"])
            if r.get("06h_record") is not None and (
                r["06h_record"]["value"]["hex"] != d["value_06h"]
                or r["06h_record"]["abserr"]["hex"] != d["abserr_06h"]
            ):
                problems.append(f"{e['sid']}: 06h record differs from the draw's")
        if e["source"] == "v8":
            s8 = e["v8"]
            for name in ("g", "06h"):
                rec_ = r.get(f"{name}_record")
                if rec_ is not None and (
                    rec_["value"]["hex"] != s8[name]["value"]
                    or rec_["abserr"]["hex"] != s8[name]["abserr"]
                ):
                    problems.append(
                        f"{e['sid']}: {name} record differs from v8_sample.json's"
                    )
        if e["in_06i"]:
            row = i06[e["sample_no"]]
            if row["unit"] != "g-" + e["rest"] or row["index"] != e["index"]:
                problems.append(
                    f"{e['sid']}: 06i's row is {row['unit']} #{row['index']}"
                )
            r["i06"] = {
                "route": row["route"],
                "ref_30": row["ref_30"],
                "ref_hex": row["ref_hex"],
                "min_abserr": row["min_abserr"],
                "scatter": row["scatter"],
                "counts_06i": row["counts"],
                "readings": row["readings"],
            }
        if "g_record" in r:
            g = r["g_record"]
            r["unit"] = (
                "g-" + e["rest"]
            )  # x1_capture.py runs the unit at g by this name
            r["census_value"] = g["value"]["hex"]
            r["census_abserr"] = g["abserr"]["hex"]
            r["census_num_regions"] = g["num_regions"]
            r["census_max_depth"] = g["max_depth"]
            r["census_converged"] = bool(g["converged"])
            r["census"] = f"06g {ck['g']}"
        sel.append(r)
    if problems:
        for p in problems:
            print("PROBLEM", p)
        raise SystemExit(f"{len(problems)} problems; nothing written")
    out.write_text(json.dumps(sel, indent=1))
    al = RC.ROOT / "out" / "sample" / "v8_aliases.json"
    al.write_text(json.dumps(aliases, indent=1))
    print(
        f"V8 calls also among the 90 (captured once, under the V3 id): {aliases or 'none'}"
    )
    for r in sel:
        print(
            f"{r['sid']} {r['kind']:8s} {r['unit']} #{r['index']}: g value {r['census_value']} abserr {r['census_abserr']} "
            f"conv {r['census_converged']} regions {r['census_num_regions']} depth {r['census_max_depth']} | "
            f"06h value {r['06h_record']['value']['hex']} abserr {r['06h_record']['abserr']['hex']} | budget {r['census_unit_budget_s']}"
        )
    print(f"wrote {out} sha256 {RC.sha256(out)}")


if __name__ == "__main__":
    main()
