"""levin-phase-floor prompt 07 section 6: three extra descriptive counts from score_07.py's output (read only): every
call with a phase Rule A skipped, by rtol and kind (scanned or not); the located points' own-window edge ratio
theta'^2/|theta''| by point kind and call kind; and the located minima per call, by label prefix. New in 07.
Usage: python -B extra_descriptive_07.py SCORE_JSON OUT_JSON"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import header, sha256  # noqa: E402


def main():
    src, out_p = Path(sys.argv[1]), Path(sys.argv[2])
    doc = json.loads(src.read_text())
    skipped = Counter()
    edges = defaultdict(list)
    minima = Counter()
    minima_calls = Counter()
    for u in doc["units"]:
        for c in u.get("calls") or []:
            if c["skipped"]:
                skipped[f"{u['rtol']}|{c['kind']}|scanned={c['scanned']}|locates={c.get('locates')}|guards={c.get('guards')}"] += 1
            for p in c.get("points") or []:
                for e in p["edge_ratio"] or []:
                    edges[f"{c['kind']}|{p['kind']}"].append(e)
            nm = sum(1 for p in c.get("points") or [] if p["kind"] == "minimum")
            if nm:
                minima[f"{u['rtol']}|{c['kind']}|{c['label'].split('group ')[-1] if c['kind'] == 'group' else c['label']}|minima={nm}"] += 1
                minima_calls[u["rtol"]] += 1
    out = {"header": header(Path(__file__).resolve(), "levin-phase-floor 07: extra descriptive counts", [src]),
           "calls_with_a_skipped_phase": dict(sorted(skipped.items())),
           "edge_ratio_by_kind": {k: {"n": len(v), "min": min(v), "median": sorted(v)[len(v) // 2], "max": max(v),
                                      "n_within_1e-3_of_4pi": sum(1 for e in v if abs(e / 12.566370614359172 - 1) <= 1e-3)}
                                  for k, v in sorted(edges.items())},
           "minima_calls": dict(sorted(minima.items())), "minima_calls_by_rtol": dict(minima_calls)}
    out_p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "header"}, indent=1))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")


if __name__ == "__main__":
    main()
