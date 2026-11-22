"""levin-phase-floor prompt 07b extension, section 2.7.3: Tables 3 and 4 extended by the extension's variants, with
(X4) and the design/held-out split. New in the extension; it imports Part A's score_07b.py (unchanged) for the
references (``refs``) and the reference keys (``ref_key``), so every call is scored against exactly Part A's reference.

**Replays:** Part A's variants (tree = (a), b4, b8, c, d, e, f) from out/replays/<v>/, the extension's from
out/ext/replays/<v>/. A call is scored as Part A scored it: its reference's stated error at most min(abserr)/100; ratio
|v - ref|/abserr; above 1 dishonest.

**Table 4 (section 2.5 item 4, unchanged):** a variant qualifies if no scored call is converged and dishonest under it, and
each pair has r_g <= 1 or a member unconverged. Beside it, not folded in:
- **(X4)** the calls returned unconverged with |v - ref|/abserr <= 1 (scored), by set; and those unconverged and above 1;
- **the split** of the converged-and-dishonest count: the **design set**, whose leaves designed the variants (V4, W2 at
  both rtols, item 3, the pairs), and the **held-out set**, which did not (R1, R2, R3's W7, aux and own calls).
Writes OUT_JSON and OUT_MD. Usage: PYTHONPATH=<tree>:<scripts> python -B score_ext.py OUT_JSON OUT_MD VARIANT [...]
"""

import hashlib
import json
import sys
from pathlib import Path

from mpmath import mpf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import score_07b as S  # noqa: E402

ROOT = HERE.parent.parent
PART_A = ("tree", "b4", "b8", "c", "d", "e", "f")
DESIGN = ("v4", "W2", "W2~o", "item3", "pairs")
HELD_OUT = ("R1", "R2", "R2~o", "R3-W7", "R3-W7~o", "R3-aux", "R3-aux~o", "R3-own", "R3-own~o")


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rdir(v):
    return ROOT / "out" / ("replays" if v in PART_A else "ext/replays") / v


def main():
    out_j, out_m, variants = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    for p in (out_j, out_m):
        if p.exists():
            raise SystemExit(f"{p} exists")
    R = S.refs()
    table = {}
    for v in variants:
        for p in sorted(rdir(v).glob("*.json")):
            if p.name.endswith(".instrument.json"):
                continue
            d = json.loads(p.read_text())
            sid = d["sid"]
            rk = S.ref_key(sid, None, None)
            v_, e_ = S.fx(d["replay"]["value"]), S.fx(d["replay"]["abserr"])
            row = {"sid": sid, "set": d["set"], "variant": v, "value": d["replay"]["value"], "abserr": d["replay"]["abserr"],
                   "converged": d["replay"]["converged"], "regions": d["replay"]["num_regions"],
                   "max_depth": d["replay"]["max_depth"], "wall_s": d["wall_s"], "ref_key": rk,
                   "regions_sha256": d.get("regions_sha256"), "module_sha256": d["header"]["module_sha256"]}
            if rk in R:
                row["ratio"] = float(abs(mpf(v_) - R[rk]["ref"]) / e_)
                row["spread_over_abserr"] = R[rk]["spread"] / e_
            table.setdefault(sid, {})[v] = row
    pairs = {}
    for v in variants:
        for tag in ("14501", "35006"):
            a, b = table.get(f"p{tag}_7", {}).get(v), table.get(f"p{tag}_8", {}).get(v)
            if a and b:
                v7, v8, e7, e8 = S.fx(a["value"]), S.fx(b["value"]), S.fx(a["abserr"]), S.fx(b["abserr"])
                pairs.setdefault(tag, {})[v] = {"r_g": abs(v7 - v8) / e7, "r": abs(v7 - v8) / (e7 + e8),
                                               "unconverged_member": not (a["converged"] and b["converged"])}
    base = {sid: rows.get("tree") for sid, rows in table.items()}
    qual = {}
    for v in variants:
        rows = [r[v] for r in table.values() if v in r]
        scored = [r for r in rows if "ratio" in r and r["spread_over_abserr"] <= 0.01]
        bad = sorted(r["sid"] for r in scored if r["converged"] and r["ratio"] > 1)
        pair_bad = sorted(t for t, pv in pairs.items() if v in pv and pv[v]["r_g"] > 1 and not pv[v]["unconverged_member"])
        unc = [r for r in rows if not r["converged"]]
        x4 = [r for r in unc if r in scored and r["ratio"] <= 1]
        x4_bad = [r for r in unc if r in scored and r["ratio"] > 1]
        by = {}
        for r in x4:
            by[r["set"]] = by.get(r["set"], 0) + 1
        same_as_a = sum(1 for r in rows if base.get(r["sid"]) and r["regions_sha256"] == base[r["sid"]]["regions_sha256"]
                        and r["value"] == base[r["sid"]]["value"] and r["abserr"] == base[r["sid"]]["abserr"])
        qual[v] = {"n_calls": len(rows), "n_scored": len(scored), "converged_dishonest": bad, "pairs_failing": pair_bad,
                   "qualifies": not bad and not pair_bad, "unconverged": len(unc),
                   "unconverged_within_abserr": len(x4), "unconverged_within_abserr_by_set": by,
                   "unconverged_within_abserr_sids": sorted(r["sid"] for r in x4),
                   "unconverged_above_abserr_sids": sorted(r["sid"] for r in x4_bad),
                   "unconverged_not_scored": sorted(r["sid"] for r in unc if r not in scored),
                   "cd_design": sorted(s for s in bad if table[s][v]["set"] in DESIGN),
                   "cd_held_out": sorted(s for s in bad if table[s][v]["set"] in HELD_OUT),
                   "regions": sum(r["regions"] for r in rows), "wall_s": sum(r["wall_s"] for r in rows),
                   "bit_identical_to_a": same_as_a}
    hdr = {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
           "score_07b_sha256": sha256(S.__file__), "variants": variants}
    out_j.parent.mkdir(parents=True, exist_ok=True)
    out_j.write_text(json.dumps({"header": hdr, "table": table, "pairs": pairs, "qualification": qual}, indent=1))
    qa = qual.get("tree")
    md = ["## Table 4, extended (section 2.5 item 4; (X4) and the split beside it)", "",
          "| variant | qualifies | converged & dishonest (design / held-out) | pairs failing | unconverged | (X4) unconverged within abserr | unconverged above abserr | regions | wall s | identical to (a) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for v in variants:
        q = qual[v]
        rel = (lambda k: f" ({q[k] / qa[k]:.2f}x)") if qa else (lambda k: "")
        md.append(f"| {v} | {'yes' if q['qualifies'] else 'no'} | {len(q['converged_dishonest'])} ({len(q['cd_design'])} / "
                  f"{len(q['cd_held_out'])}) | {', '.join(q['pairs_failing']) or '-'} | {q['unconverged']} | "
                  f"{q['unconverged_within_abserr']} {json.dumps(q['unconverged_within_abserr_by_set'])} | "
                  f"{len(q['unconverged_above_abserr_sids'])} | {q['regions']}{rel('regions')} | {q['wall_s']:.0f}{rel('wall_s')} | "
                  f"{q['bit_identical_to_a']} / {q['n_calls']} |")
    md += ["", "### Converged and dishonest, by call", ""]
    for v in variants:
        q = qual[v]
        md.append(f"- **{v}:** design {', '.join(q['cd_design']) or 'none'}; held-out {', '.join(q['cd_held_out']) or 'none'}"
                  + (f"; unconverged above abserr {', '.join(q['unconverged_above_abserr_sids'])}" if q['unconverged_above_abserr_sids'] else ""))
    md += ["", "## Table 3, extended: honesty per call (|v - ref|/abserr; * converged and dishonest; u unconverged; n not scored)", "",
           "| call | set | " + " | ".join(variants) + " |", "|---|---|" + "---|" * len(variants)]
    for sid in sorted(table):
        rows = table[sid]
        cells = []
        for v in variants:
            r = rows.get(v)
            if not r or "ratio" not in r:
                cells.append("-" if not r else ("u" if not r["converged"] else ""))
                continue
            mark = "n" if r["spread_over_abserr"] > 0.01 else ("*" if (r["ratio"] > 1 and r["converged"]) else ("u" if not r["converged"] else ""))
            cells.append(f"{r['ratio']:.3g}{mark}")
        md.append(f"| {sid} | {next(iter(rows.values()))['set']} | " + " | ".join(cells) + " |")
    md += ["", "## Table 3, extended: regions, maximum depth, wall s per call", "",
           "| call | " + " | ".join(variants) + " |", "|---|" + "---|" * len(variants)]
    for sid in sorted(table):
        rows = table[sid]
        md.append(f"| {sid} | " + " | ".join((f"{rows[v]['regions']}/{rows[v]['max_depth']}/{rows[v]['wall_s']:.2f}" if v in rows else "-")
                                            for v in variants) + " |")
    md += ["", "## The pairs (r_g; u a member unconverged)", "", "| pair | " + " | ".join(variants) + " |", "|---|" + "---|" * len(variants)]
    for t, pv in pairs.items():
        md.append(f"| {t} | " + " | ".join((f"{pv[v]['r_g']:.3g}{' u' if pv[v]['unconverged_member'] else ''}" if v in pv else "-") for v in variants) + " |")
    out_m.write_text("\n".join(md) + "\n")
    print("\n".join(md[:4 + len(variants)]))
    print("wrote", out_j, sha256(out_j), out_m, sha256(out_m))


if __name__ == "__main__":
    main()
