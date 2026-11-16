"""levin-phase-floor prompt 06g section 5: V8 only ("no convergence lost"), scored from the two checkpoints by
state/predictions.md (Part A the rule, Part B its readings: rule 1, the 720; rule 2, matching; rule 3, V8 and the
cause classes). New in 06g; its loaders follow 06h's analyse06h.py (scan06a_sp and the match rule are copied
from it). Reads, read only: this census's checkpoint (RUN_DIR), 06h's census checkpoint (prov.RUN06H, its
SHA-256 checked first) and 06a's run's 1e-7 checkpoint lines (the 720). Writes out/v8.md (the tables) and
out/v8_calls.json (every lost and gained call, every unmatched call, with all fields read).
Usage: python -B v8_06g.py RUN_DIR"""
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CHECKPOINT06H_SHA256, RUN06A, RUN06H, WIDER, header, sha256  # noqa: E402

RTOLS = ("1e-7", "1e-8")
CLASSES = ("node_limited", "(b), refined to the depth cap", "depth cap, without (b)", "(b), below the depth cap", "other", "unreadable")
PART_A = {"node_limited": "node_limited", "(b), refined to the depth cap": "(b) on a non-noise leaf",
          "(b), below the depth cap": "(b) on a non-noise leaf", "depth cap, without (b)": "other", "other": "other",
          "unreadable": "unreadable"}


def H(x):
    return None if x is None else float.fromhex(x["hex"] if isinstance(x, dict) else x)


def hexof(x):
    return None if x is None else (x["hex"] if isinstance(x, dict) else x)


def rowkey(uid):
    return "-".join(uid.rsplit("-", 2)[-2:])


KEEP = ("index", "notify_label", "group", "converged", "max_depth", "num_regions", "num_subregion_solves", "atol", "rtol",
        "n_at_depth_max", "depth_max", "num_untrusted_regions", "n_phase_limited_regions", "premise_checks", "premise_failed",
        "node_limited", "has_cc_leaf", "premise_06g", "checked_06g", "error_06g", "leaves")


def slim_call(kind, c):
    out = {k: c.get(k) for k in KEEP}
    out["kind"] = kind
    out["value"] = hexof(c["value"])
    out["abserr"] = hexof(c["abserr"])
    out["x_span"] = [hexof(c["x_span"][0]), hexof(c["x_span"][1])]
    for k in ("node_error_signed_sum", "node_error_abs_sum", "abserr_node_signed", "abserr_eprime_sum"):
        out[k] = hexof(c.get(k))
    ck = out.get("checked_06g")
    if isinstance(ck, dict) and "classes_all" in ck:
        out["checked_06g"] = {k: ck[k] for k in ("n_checked", "n_checked_returned", "n_b_untrusted", "n_b_untrusted_returned",
                                                 "n_noise_clause_trusted", "n_checked_then_floor_limited")}
    return out


def load(run_dir):
    """{(rtol, rowkey): unit} for both rtols."""
    out = {}
    with open(Path(run_dir) / "checkpoint.jsonl") as f:
        for line in f:
            r = json.loads(line)
            d = r["data"]
            out[(d["rtol"], rowkey(r["unit"]))] = {
                "unit": r["unit"], "status": d["status"], "row": d["row"],
                "calls": [slim_call("group", c) for c in d["calls"]] + [slim_call("analytic", c) for c in d["other_calls"]],
            }
    return out


def scan06a_sp():
    sp = set()
    with open(RUN06A / "checkpoint.jsonl") as f:
        for line in f:
            if '"rtol": "1e-7"' not in line:
                continue
            d = json.loads(line)["data"]
            if d["rtol"] != "1e-7":
                continue
            if any(s and s["n_sign_changes"] > 0 for s in (d.get("scan") or [])):
                sp.add(f"{d['shard']}-{d['serial']}")
    return sp


def match(ug, uh):
    """[(cg, ch)] by (kind, index), with equal labels and x_span bit for bit; problems (rule 2)."""
    probs = []
    bi = {(c["kind"], c["index"]): c for c in uh["calls"]}
    out = []
    for cg in ug["calls"]:
        ch = bi.get((cg["kind"], cg["index"]))
        if ch is None or ch["notify_label"] != cg["notify_label"] or ch["x_span"] != cg["x_span"]:
            probs.append({"unit": ug["unit"], "kind": cg["kind"], "index": cg["index"], "why": "no 06h call with its kind, index, label and span"})
            ch = None
        out.append((cg, ch))
    gi = {(c["kind"], c["index"]) for c in ug["calls"]}
    for ch in uh["calls"]:
        if (ch["kind"], ch["index"]) not in gi:
            probs.append({"unit": uh["unit"], "kind": ch["kind"], "index": ch["index"], "why": "in 06h only"})
            out.append((None, ch))
    return out, probs


def classify(cg):
    """state/predictions.md Part B rule 3."""
    ck = cg.get("checked_06g")
    if cg.get("error_06g") or not isinstance(ck, dict) or "n_b_untrusted" not in ck or cg.get("node_limited") is None:
        return "unreadable", None, None
    b = ck["n_b_untrusted"] >= 1
    if cg["node_limited"]:
        ending = "node_limited"
    elif (cg.get("n_at_depth_max") or 0) >= 1:
        ending = "depth cap"
    else:
        ending = "other ending"
    if ending == "node_limited":
        cls = "node_limited"
    elif ending == "depth cap":
        cls = "(b), refined to the depth cap" if b else "depth cap, without (b)"
    elif b:
        cls = "(b), below the depth cap"
    else:
        cls = "other"
    return cls, ending, b


def request(c):
    v = H(c["value"])
    if c.get("atol") is None or c.get("rtol") is None or v is None:
        return None
    return max(float(c["atol"]), float(c["rtol"]) * abs(v))


def fe(x):
    return "None" if x is None else f"{x:.3e}"


def label(c):
    return c["group"] if c["kind"] == "group" else c["notify_label"]


def main():
    run = Path(sys.argv[1])
    ck_h = sha256(RUN06H / "checkpoint.jsonl")
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06g: V8, scored", [run / "checkpoint.jsonl", WIDER / "state" / "predictions.md"]):
        print(l)
    print(f"# 06h's checkpoint {RUN06H / 'checkpoint.jsonl'} sha256 {ck_h} (must be {CHECKPOINT06H_SHA256}: {ck_h == CHECKPOINT06H_SHA256})")
    assert ck_h == CHECKPOINT06H_SHA256
    sp = scan06a_sp()
    print(f"06a's stationary-point rows: {len(sp)} (must be 720: {len(sp) == 720})")
    assert len(sp) == 720
    G = load(run)
    Hh = load(RUN06H)
    print(f"units: g {len(G)}, 06h {len(Hh)}; in one only: {sorted(set(G) ^ set(Hh))[:10]}")
    lost, gained, unmatched, unread = [], [], [], []
    scored = collections.Counter()
    for key in sorted(G):
        rtol, rk = key
        ug, uh = G[key], Hh.get(key)
        if uh is None:
            unmatched.append({"unit": ug["unit"], "why": "no 06h unit"})
            continue
        if ug["status"] != "done" or uh["status"] != "done":
            unread.append({"unit": ug["unit"], "rtol": rtol, "g": ug["status"], "06h": uh["status"]})
        pairs, probs = match(ug, uh)
        in_sp = rk in sp
        for p_ in probs:
            p_["rtol"] = rtol
            p_["stationary_point_row"] = in_sp
            unmatched.append(p_)
        for cg, ch in pairs:
            if cg is None or ch is None:
                continue
            rec = None
            if ch["converged"] and not cg["converged"]:
                cls, ending, b = classify(cg)
                rec = {"rtol": rtol, "row": rk, "row_name": ug["row"], "unit": ug["unit"], "stationary_point_row": in_sp,
                       "class": cls, "part_a_cause": PART_A[cls], "ending": ending, "b": b, "g": cg, "06h": ch,
                       "request_g": request(cg), "request_06h": request(ch)}
                lost.append(rec)
            elif not ch["converged"] and cg["converged"]:
                gained.append({"rtol": rtol, "row": rk, "row_name": ug["row"], "unit": ug["unit"], "stationary_point_row": in_sp,
                               "g": cg, "06h": ch})
            if not in_sp and ch["converged"]:
                scored[(rtol, cg["kind"])] += 1
    out_lines = []
    P = out_lines.append
    P("# V8 — no convergence lost (prompt 06g section 5), scored")
    P("")
    P(f"Scored by `scripts/v8_06g.py` on run `{run.name}` against 06h's census "
      f"`{RUN06H.name}` (checkpoint `{ck_h[:8]}…`, checked), by `state/predictions.md` (Part A V8; Part B rules 1–3). "
      "Nothing else is scored here.")
    P("")
    P("**Rule (Part A):** outside the stationary-point rows, no call that 06h reported converged is unconverged under $g$. "
      "**Holds** if true at both rtols. A failure is a finding for the user: each such call is reported with its cause "
      "((b) on a non-noise leaf, `node_limited`, or other).")
    P("")
    P(f"- Units: {len(G)} under g, {len(Hh)} under 06h; not `done` in either: {len(unread)}; unmatched calls: {len(unmatched)} "
      f"({sum(1 for u in unmatched if not u.get('stationary_point_row'))} outside the 720).")
    for rtol in RTOLS:
        P(f"- Scored at {rtol} (06h converged, outside the 720): {scored[(rtol, 'group')]} group calls, "
          f"{scored[(rtol, 'analytic')]} analytic-block calls.")
    P("")
    verdicts = {}
    for rtol in RTOLS:
        L = [x for x in lost if x["rtol"] == rtol and not x["stationary_point_row"]]
        und = any(u.get("rtol") == rtol and not u.get("stationary_point_row") for u in unmatched) or any(
            u["rtol"] == rtol for u in unread)
        verdicts[rtol] = "fails" if L else ("undecided" if und else "holds")
    overall = "fails" if "fails" in verdicts.values() else ("undecided" if "undecided" in verdicts.values() else "holds")
    P(f"**Verdict: V8 {overall.upper()}** (1e-7: {verdicts['1e-7']}; 1e-8: {verdicts['1e-8']}).")
    P("")
    P("## Lost calls, by class (outside the 720)")
    P("")
    P("| class | Part A's cause | 1e-7 calls | 1e-7 rows | 1e-8 calls | 1e-8 rows |")
    P("|---|---|---|---|---|---|")
    for cls in CLASSES:
        cells = []
        for rtol in RTOLS:
            L = [x for x in lost if x["rtol"] == rtol and not x["stationary_point_row"] and x["class"] == cls]
            cells += [str(len(L)), str(len({x["row"] for x in L}))]
        P(f"| {cls} | {PART_A[cls]} | " + " | ".join(cells) + " |")
    tot = []
    for rtol in RTOLS:
        L = [x for x in lost if x["rtol"] == rtol and not x["stationary_point_row"]]
        tot += [str(len(L)), str(len({x["row"] for x in L}))]
    P("| **total** | | " + " | ".join(f"**{t}**" for t in tot) + " |")
    P("")
    P("By Part A's three causes (rows counted once per cause; a row with calls of two causes appears under each):")
    P("")
    P("| Part A's cause | 1e-7 calls | 1e-7 rows | 1e-8 calls | 1e-8 rows |")
    P("|---|---|---|---|---|")
    for cause in ("(b) on a non-noise leaf", "node_limited", "other", "unreadable"):
        cells = []
        for rtol in RTOLS:
            L = [x for x in lost if x["rtol"] == rtol and not x["stationary_point_row"] and x["part_a_cause"] == cause]
            cells += [str(len(L)), str(len({x["row"] for x in L}))]
        P(f"| {cause} | " + " | ".join(cells) + " |")
    P("")
    for rtol in RTOLS:
        L = sorted([x for x in lost if x["rtol"] == rtol and not x["stationary_point_row"]],
                   key=lambda x: (x["row"], x["g"]["kind"], x["g"]["index"]))
        P(f"## Every lost call at {rtol}, outside the 720 ({len(L)} calls, {len({x['row'] for x in L})} rows)")
        P("")
        P("Columns: row (shard-serial, rows.json name); call index; kind; block or group; abserr 06h → g; the request "
          "max(atol, rtol|v|) under 06h → g; regions 06h → g; depth 06h → g; class. Then: n_at_depth_max, untrusted "
          "returned, floor-limited returned, (b) leaves (all checked / returned), node_limited, CC leaf, premise "
          "checks/fired under g (06h checks/failed), solves 06h → g.")
        P("")
        P("| row | idx | kind | block/group | abserr 06h → g | request 06h → g | regions | depth | class | at cap | untr. | fl.-lim. | (b) all/ret | n.l. | CC | premise g (06h) | solves |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for x in L:
            g, h = x["g"], x["06h"]
            ck = g.get("checked_06g") or {}
            pm = g.get("premise_06g") or {}
            P(f"| {x['row']} ({x['row_name']}) | {g['index']} | {g['kind']} | {label(g)} | "
              f"{H(h['abserr']):.3e} → {H(g['abserr']):.3e} | {fe(x['request_06h'])} → {fe(x['request_g'])} | "
              f"{h['num_regions']} → {g['num_regions']} | {h['max_depth']} → {g['max_depth']} | {x['class']} | "
              f"{g.get('n_at_depth_max')} | {g.get('num_untrusted_regions')} | {g.get('n_phase_limited_regions')} | "
              f"{ck.get('n_b_untrusted')}/{ck.get('n_b_untrusted_returned')} | {g.get('node_limited')} | {g.get('has_cc_leaf')} | "
              f"{pm.get('n_checks')}/{pm.get('n_fired')} ({h.get('premise_checks')}/{h.get('premise_failed')}) | "
              f"{h.get('num_subregion_solves')} → {g.get('num_subregion_solves')} |")
        P("")
        rows_by = collections.defaultdict(set)
        for x in L:
            rows_by[x["class"]].add(f"{x['row']} ({x['row_name']})")
        P(f"Rows at {rtol} by class:")
        P("")
        for cls in CLASSES:
            if rows_by[cls]:
                P(f"- **{cls}** ({len(rows_by[cls])}): " + ", ".join(sorted(rows_by[cls])))
        P("")
    P("## Calls that gained convergence (06h unconverged, g converged)")
    P("")
    P("| rtol | row | in the 720 | idx | kind | block/group | abserr 06h → g | regions | depth |")
    P("|---|---|---|---|---|---|---|---|---|")
    for x in sorted(gained, key=lambda x: (x["rtol"], x["row"], x["g"]["index"])):
        g, h = x["g"], x["06h"]
        P(f"| {x['rtol']} | {x['row']} ({x['row_name']}) | {x['stationary_point_row']} | {g['index']} | {g['kind']} | {label(g)} | "
          f"{H(h['abserr']):.3e} → {H(g['abserr']):.3e} | {h['num_regions']} → {g['num_regions']} | {h['max_depth']} → {g['max_depth']} |")
    P("")
    P("## Inside the 720 (07's; not scored)")
    P("")
    for rtol in RTOLS:
        L = [x for x in lost if x["rtol"] == rtol and x["stationary_point_row"]]
        c_ = collections.Counter(x["class"] for x in L)
        P(f"- {rtol}: {len(L)} calls lost in {len({x['row'] for x in L})} rows, by class {dict(c_)}; "
          f"gained {sum(1 for x in gained if x['rtol'] == rtol and x['stationary_point_row'])}.")
    P("")
    P(f"## Unmatched calls and units not done: {len(unmatched)} unmatched, {len(unread)} units not done")
    P("")
    for u in unmatched[:200]:
        P(f"- {u}")
    for u in unread[:200]:
        P(f"- {u}")
    text = "\n".join(out_lines) + "\n"
    (WIDER / "out" / "v8.md").write_text(text)
    (WIDER / "out" / "v8_calls.json").write_text(json.dumps({"verdicts": verdicts, "lost": lost, "gained": gained,
                                                            "unmatched": unmatched, "units_not_done": unread}, indent=1))
    print(text)


if __name__ == "__main__":
    main()
