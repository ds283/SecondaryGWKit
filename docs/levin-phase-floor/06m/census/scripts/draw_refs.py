"""levin-phase-floor prompt 06g section 5, the references: draw the random sample of 60 calls. New in
06g.

The rule (prompt 06g section 5 and the references brief), applied exactly:
- **the population** is 06h's census (RUN06H, checkpoint 3104c18e..., checked), every unit record;
  each unit must have exactly one record, with status ``done`` (06h's census: 18,560 of 18,560, none
  over budget);
- **outside the 720 stationary-point rows** (06a's scan, ``refs_common.scan06a_sp``; the row is
  excluded at both rtols) **and Z3's 30 calls** (``results.json``'s ``calls``, matched by rtol, row
  and call index);
- **two kinds**: an analytic-block call is one of the unit's ``other_calls`` (its label must begin
  ``analytic``), a cosmological-group call one of its ``calls``;
- **finite abserr**: the 06h census record's ``abserr`` is finite (the population is 06h's census,
  so its record is the one read);
- **the strata**, in this order: rtol 1e-7 analytic, 1e-7 group, 1e-8 analytic, 1e-8 group; 15
  calls drawn from each uniformly, without replacement, by ``random.Random(SEED).sample`` over the
  stratum sorted by (shard, serial, call index) as integers, one generator for all four strata in
  that order;
- **the seed** is given on the command line and must already appear in the predictions file, as the
  line ``seed: <SEED>`` (the file's SHA-256 and mtime go in the output header). The draw refuses if
  its output exists.

Writes out/sample/draw.json: the header, the seed, the stratum sizes and exclusions, and the 60 calls
r00..r59 in draw order, each with its 06h census record's value, abserr, converged, num_regions,
max_depth and label. Reads only.

Usage: python -B draw_refs.py SEED PREDICTIONS_FILE   (from var/levin-phase-floor-06g/references/)
"""

import datetime
import json
import math
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402

N_PER_STRATUM = 15


def main():
    seed = int(sys.argv[1])
    pred = Path(sys.argv[2])
    text = pred.read_text()
    if f"seed: {seed}\n" not in text:
        raise SystemExit(f"the line 'seed: {seed}' is not in {pred}")
    out = RC.ROOT / "out" / "sample" / "draw.json"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwritten")
    ck = RC.RUN06H / "checkpoint.jsonl"
    hdr = header(
        Path(__file__).resolve(),
        "levin-phase-floor 06g: the references' random sample of 60 calls from 06h's census",
        [pred, RC.Z3_RESULTS, ck, RC.RUN06A / "checkpoint.jsonl"],
    )
    mtime = datetime.datetime.fromtimestamp(
        os.stat(pred).st_mtime, datetime.timezone.utc
    ).isoformat()
    hdr.append(f"# predictions file {pred} mtime {mtime}")
    for l in hdr:
        print(l)
    if RC.sha256(ck) != RC.CHECKPOINT06H_SHA256:
        raise SystemExit("06h's census checkpoint is not 3104c18e...")
    sp = RC.scan06a_sp()
    print(f"06a's stationary-point rows: {len(sp)}")
    if len(sp) != 720:
        raise SystemExit("expected 720 stationary-point rows")
    z3 = json.loads(RC.Z3_RESULTS.read_text())["calls"]
    if len(z3) != 30:
        raise SystemExit("expected Z3's 30 calls")
    z3_keys = {(RC.rest_of(c["unit"]), c["index"]) for c in z3}

    strata = {(r, k): [] for r in RC.RTOLS for k in RC.KINDS}
    counts = {
        f"{r} {k}": {
            "calls": 0,
            "in_720": 0,
            "z3": 0,
            "abserr_not_finite": 0,
            "eligible": 0,
        }
        for r in RC.RTOLS
        for k in RC.KINDS
    }
    seen_units = set()
    n_records = 0
    with open(ck) as f:
        for line in f:
            rec = json.loads(line)
            n_records += 1
            unit = rec["unit"]
            d = rec["data"]
            if unit in seen_units:
                raise SystemExit(f"{unit}: more than one record in 06h's census")
            seen_units.add(unit)
            if d["status"] != "done":
                raise SystemExit(f"{unit}: status {d['status']!r}")
            rtol = d["rtol"]
            rk = f"{d['shard']}-{d['serial']}"
            rest = RC.rest_of(unit)
            for kind, key in (("group", "calls"), ("analytic", "other_calls")):
                for c in d.get(key) or []:
                    cnt = counts[f"{rtol} {kind}"]
                    cnt["calls"] += 1
                    if kind == "analytic" and not str(c["notify_label"]).startswith(
                        "analytic"
                    ):
                        raise SystemExit(
                            f"{unit} #{c['index']}: other call labelled {c['notify_label']!r}"
                        )
                    if rk in sp:
                        cnt["in_720"] += 1
                        continue
                    if (rest, c["index"]) in z3_keys:
                        cnt["z3"] += 1
                        continue
                    e = RC.H(c["abserr"])
                    if not math.isfinite(e):
                        cnt["abserr_not_finite"] += 1
                        continue
                    cnt["eligible"] += 1
                    strata[(rtol, kind)].append(
                        {
                            "rtol": rtol,
                            "kind": kind,
                            "unit_06h": unit,
                            "rest": rest,
                            "shard": d["shard"],
                            "serial": d["serial"],
                            "index": c["index"],
                            "notify_label": c["notify_label"],
                            "x_span": [RC.hexof(v) for v in c["x_span"]],
                            "value_06h": c["value"]["hex"],
                            "abserr_06h": c["abserr"]["hex"],
                            "converged_06h": bool(c["converged"]),
                            "num_regions_06h": c["num_regions"],
                            "max_depth_06h": c["max_depth"],
                        }
                    )
    print(f"06h census records: {n_records}")
    rng = random.Random(seed)
    drawn = []
    for rtol in RC.RTOLS:
        for kind in RC.KINDS:
            pop = sorted(
                strata[(rtol, kind)],
                key=lambda e: (int(e["shard"]), int(e["serial"]), int(e["index"])),
            )
            picks = rng.sample(pop, N_PER_STRATUM)
            for e in picks:
                e = dict(e)
                e["sid"] = f"r{len(drawn):02d}"
                e["stratum"] = f"{rtol} {kind}"
                e["stratum_size"] = len(pop)
                drawn.append(e)
            print(
                f"stratum {rtol} {kind}: {len(pop)} eligible; drew "
                + ", ".join(f"{e['rest']}#{e['index']}" for e in picks)
            )
    keys = {(e["rest"], e["index"]) for e in drawn}
    if len(keys) != len(drawn):
        raise SystemExit("a call was drawn twice")
    doc = {
        "header": hdr,
        "seed": seed,
        "rule": __doc__.split("Writes out/sample/draw.json")[0],
        "n_records_06h": n_records,
        "n_stationary_point_rows": len(sp),
        "counts": counts,
        "drawn": drawn,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(json.dumps(counts, indent=1))
    for e in drawn:
        print(
            f"{e['sid']} {e['stratum']} {e['unit_06h']} #{e['index']} {e['notify_label'][-40:]!r} "
            f"value_06h {e['value_06h']} abserr_06h {e['abserr_06h']} converged {e['converged_06h']}"
        )
    print(f"wrote {out} sha256 {RC.sha256(out)}")


if __name__ == "__main__":
    main()
