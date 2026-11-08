"""levin-phase-floor 07 section 10 (Rule A): the re-run of section 2 on the amended code against run 2. New in
07's amendment.

Section 10, "What must not move": section 2.2's families at the recorded rtol (1e-9), section 2.3's 14 calls and
section 2.4's draws, re-run on the amended code by this directory's copies of run 2's scripts (rule-a/out/), are
compared with run 2's records (calibration/out/run2/):
- **every call with no phase skipped is bit-identical to run 2:** each call's record, field by field, with only
  what cannot be equal set aside -- the wall-clock times (``wall_s``, ``elapsed``, ``elapsed_s``), the module's
  SHA-256 and the replay's ``evidence`` (temporary paths, the module's file and hash), and Rule A's new key; for
  a LiouvilleGreen call under ``commit``, 06m's ``result_sha256`` and ``result_keys``, which count the result's
  keys and so differ by the new key on every call (``result_sha256_07``, every result key but the timing and the
  guard's keys, Rule A's among them, is compared instead);
- **every call with a skipped phase is listed,** with its result under both codes (value, abserr, converged,
  regions, maximum depth, window pieces, the points located);
- **a rung of Phi_w or Lambda_c that moves** is reported (section 2.2's rules, as run 2's calibrate.json records
  them, against the re-run's).
A call is "with a skipped phase" if the re-run's record of it lists one under Rule A's key. Every unit of the
re-run must pair with one of run 2's, and the reverse.

Writes rule-a/out/compare_run2.json and rule-a/out/compare_run2.txt.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prov  # noqa: E402

RUN2 = prov.REPO / "docs" / "levin-phase-floor" / "07" / "calibration" / "out" / "run2"
NEW = prov.OUT
KEY = "location_skipped_phases"
SKIPPED_LG = "location_skipped_phases_07"

VOLATILE = {
    "wall_s",
    "elapsed",
    "elapsed_s",
    "module_sha256",
    "evidence",
    KEY,
    SKIPPED_LG,
}
VOLATILE_LG_COMMIT = {"result_sha256", "result_keys"}


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def _diff(a, b, set_aside, path=""):
    """Every place where a and b differ, set_aside keys removed at every depth."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in set_aside:
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: present in only one")
                continue
            out += _diff(a[k], b[k], set_aside, f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: lengths {len(a)} and {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += _diff(x, y, set_aside, f"{path}[{i}]")
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out


def _brief(r):
    if r is None:
        return None
    return {
        "value": r.get("value_hex") or r.get("value_hex_07"),
        "abserr": r.get("abserr_hex") or r.get("abserr_hex_07"),
        "converged": r.get("converged"),
        "regions": r.get("regions", r.get("num_regions")),
        "max_depth": r.get("max_depth"),
        "window_pieces": r.get("window_pieces", r.get("num_window_pieces_07")),
        "guarded_regions": r.get("guarded_regions", r.get("num_guarded_regions_07")),
        "points": [
            {k: p.get(k) for k in ("x", "kind", "phase", "lambda")}
            for p in r.get("stationary_points", r.get("stationary_points_07")) or []
        ],
        "honest": r.get("honest"),
        "error_over_abserr": r.get("error_over_abserr"),
    }


def compare_units(old_units, new_units, key_of, part, set_aside_for):
    old = {key_of(u): u for u in old_units}
    new = {key_of(u): u for u in new_units}
    res = {
        "part": part,
        "units_run2": len(old),
        "units_rule_a": len(new),
        "only_run2": sorted(map(str, set(old) - set(new))),
        "only_rule_a": sorted(map(str, set(new) - set(old))),
        "calls": 0,
        "calls_without_skip": 0,
        "calls_without_skip_identical": 0,
        "calls_without_skip_differing": [],
        "calls_with_skip": [],
        "unit_level_differences": [],
    }
    for k in sorted(set(old) & set(new), key=str):
        o, n = old[k], new[k]
        for name, (ro, rn, skipped, extra_aside) in set_aside_for(k, o, n).items():
            res["calls"] += 1
            if skipped:
                res["calls_with_skip"].append(
                    {
                        "unit": str(k),
                        "call": name,
                        "skipped": skipped,
                        "run2": _brief(ro),
                        "rule_a": _brief(rn),
                        "identical_otherwise": not _diff(
                            ro, rn, VOLATILE | extra_aside
                        ),
                        "differences": _diff(ro, rn, VOLATILE | extra_aside)[:40],
                    }
                )
                continue
            res["calls_without_skip"] += 1
            d = _diff(ro, rn, VOLATILE | extra_aside)
            if d:
                res["calls_without_skip_differing"].append(
                    {"unit": str(k), "call": name, "differences": d[:40]}
                )
            else:
                res["calls_without_skip_identical"] += 1
    return res


def calibrate_part():
    tag = "rtol1e-09"
    old_units = _jsonl(RUN2 / f"calibrate_units_{tag}.jsonl")
    new_units = _jsonl(NEW / f"calibrate_units_{tag}.jsonl")

    def key_of(u):
        s = u["spec"]
        return (
            s["mode"],
            s["part"],
            s["label"],
            s.get("phi_w_pi"),
            s.get("lambda_c"),
            s.get("depth_max"),
        )

    def calls(k, o, n):
        ro, rn = o.get("record"), n.get("record")
        skipped = (rn or {}).get(KEY) or []
        if ro is None or rn is None:
            return {"call": ({"unit": o}, {"unit": n}, skipped, set())}
        return {"call": (ro, rn, skipped, set())}

    res = compare_units(old_units, new_units, key_of, "section 2.2", calls)
    old = json.loads((RUN2 / f"calibrate_{tag}.json").read_text())
    new = json.loads((NEW / f"calibrate_{tag}.json").read_text())
    res["rungs"] = {
        "phi_w_by_rule_pi": [old["phi_w_by_rule_pi"], new["phi_w_by_rule_pi"]],
        "lambda_c_by_rule": {
            rung: [
                old["C"].get(rung, {}).get("lambda_c_by_rule"),
                new["C"].get(rung, {}).get("lambda_c_by_rule"),
            ]
            for rung in sorted(set(old["C"]) | set(new["C"]))
        },
        "check_holds": {
            rung: [
                old["C"].get(rung, {}).get("check_holds"),
                new["C"].get(rung, {}).get("check_holds"),
            ]
            for rung in sorted(set(old["C"]) | set(new["C"]))
        },
    }
    res["rungs_move"] = res["rungs"]["phi_w_by_rule_pi"][0] != res["rungs"][
        "phi_w_by_rule_pi"
    ][1] or any(a != b for a, b in res["rungs"]["lambda_c_by_rule"].values())
    return res


def replay_part():
    old_units = _jsonl(RUN2 / "replay_units.jsonl")
    new_units = _jsonl(NEW / "replay_units.jsonl")

    def key_of(u):
        s = u["spec"]
        return (s["sid"], s["mode"], s.get("phi_w_pi"), s.get("depth_max"))

    def calls(k, o, n):
        ro, rn = o.get("record"), n.get("record")
        skipped = (rn or {}).get(KEY) or []
        if ro is None or rn is None:
            return {"call": ({"unit": o}, {"unit": n}, skipped, set())}
        return {"call": (ro, rn, skipped, set())}

    res = compare_units(old_units, new_units, key_of, "section 2.3", calls)
    old = json.loads((RUN2 / "replay.json").read_text())
    new = json.loads((NEW / "replay.json").read_text())
    res["replay_check_all_reproduce"] = [
        old.get("replay_check_all_reproduce"),
        new.get("replay_check_all_reproduce"),
    ]
    return res


def lg_part():
    old_units = _jsonl(RUN2 / "lg_seeded_draws.jsonl")[1:]
    new_units = _jsonl(NEW / "lg_seeded_draws.jsonl")[1:]

    def key_of(u):
        return (u["seed"], u["test"], u["mode"])

    unit_fields = ("outcome", "detail", "num_calls", "results", "stdout")

    def calls(k, o, n):
        out = {}
        unit_diff = _diff(
            {f: o.get(f) for f in unit_fields},
            {f: n.get(f) for f in unit_fields},
            set(),
        )
        if unit_diff:
            unit_level.append({"unit": str(k), "differences": unit_diff[:40]})
        aside = VOLATILE_LG_COMMIT if k[2] == "commit" else set()
        oc, nc = o.get("calls", []), n.get("calls", [])
        for i in range(max(len(oc), len(nc))):
            co = oc[i] if i < len(oc) else None
            cn = nc[i] if i < len(nc) else None
            skipped = (cn or {}).get(SKIPPED_LG) or []
            out[f"call {i}"] = (co, cn, skipped, aside)
        return out

    unit_level = []
    res = compare_units(old_units, new_units, key_of, "section 2.4", calls)
    res["unit_level_differences"] = unit_level
    return res


def main():
    inputs = [
        RUN2 / "calibrate_units_rtol1e-09.jsonl",
        RUN2 / "calibrate_rtol1e-09.json",
        RUN2 / "replay_units.jsonl",
        RUN2 / "replay.json",
        RUN2 / "lg_seeded_draws.jsonl",
        NEW / "calibrate_units_rtol1e-09.jsonl",
        NEW / "calibrate_rtol1e-09.json",
        NEW / "replay_units.jsonl",
        NEW / "replay.json",
        NEW / "lg_seeded_draws.jsonl",
    ]
    out = {
        "header": prov.json_header(
            __file__, "section 10 (Rule A), section 2 re-run against run 2", inputs
        ),
        "set_aside": sorted(VOLATILE),
        "set_aside_lg_commit": sorted(VOLATILE_LG_COMMIT),
        "parts": [calibrate_part(), replay_part(), lg_part()],
    }
    (NEW / "compare_run2.json").write_text(json.dumps(out, indent=1, default=str))

    txt = [f"# {line}" for line in out["header"]] + [""]
    for p in out["parts"]:
        txt += [
            f"## {p['part']}",
            "",
            f"units: run 2 {p['units_run2']}, Rule A {p['units_rule_a']}; only in run 2: "
            f"{p['only_run2'] or 'none'}; only in Rule A: {p['only_rule_a'] or 'none'}",
            f"calls compared: {p['calls']}; with no phase skipped: {p['calls_without_skip']}, of them "
            f"bit-identical to run 2: {p['calls_without_skip_identical']}, differing: "
            f"{len(p['calls_without_skip_differing'])}",
            f"calls with a skipped phase: {len(p['calls_with_skip'])}",
        ]
        if "rungs" in p:
            txt.append(
                f"rungs (run 2, Rule A): {json.dumps(p['rungs'])}; a rung moves: {p['rungs_move']}"
            )
        if "replay_check_all_reproduce" in p:
            txt.append(
                f"replay check, all reproduce (run 2, Rule A): {p['replay_check_all_reproduce']}"
            )
        for d in p["calls_without_skip_differing"]:
            txt.append(
                f"  DIFFERS, no phase skipped: {d['unit']} {d['call']}: {d['differences'][:5]}"
            )
        for d in p["unit_level_differences"]:
            txt.append(f"  unit-level difference: {d['unit']}: {d['differences'][:5]}")
        for s in p["calls_with_skip"]:
            txt.append(
                f"  skipped {s['unit']} {s['call']}: phases {s['skipped']}; identical to run 2 but for the "
                f"set-aside keys: {s['identical_otherwise']}"
            )
            txt.append(f"      run 2:  {json.dumps(s['run2'])}")
            txt.append(f"      Rule A: {json.dumps(s['rule_a'])}")
        txt.append("")
    (NEW / "compare_run2.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
