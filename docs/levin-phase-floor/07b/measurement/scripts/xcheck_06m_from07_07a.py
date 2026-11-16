"""levin-phase-floor prompt 07a section 6 (descriptive): the disagreement list re-made at h against 06m. New in 07a.
06m's census run directory was trimmed on 2026-11-10 (its TRIMMED_2026-11-10.txt; the checkpoint is gone), so 06m's
values and abserrs are taken from 07's cross-check output (var/levin-phase-floor-07/census/out/xcheck/xcheck.json, its
``pairs``: m7, m8 per pair, SHA-256 checked against 07's KEPT list), and h's from xcheck_07a.py's h-against-07 run on 07's
population (out/xcheck/h_vs_07_pop07/xcheck.json, its ``g7``, ``g8``). The classes are 07's rule
(state/xcheck_rule_20261109.md) with g read as h. Exact, by Fraction. Read only.
Usage: python -B xcheck_06m_from07_07a.py OUT_JSON
"""
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import REC07, WIDER, header, sha256  # noqa: E402

X07 = REC07 / "out" / "xcheck" / "xcheck.json"
X07_SHA256 = "533b2b0789b541aa0e19aa099340c17d90e793866edeecbe86bf924fa5543dab"  # 07's KEPT_ON_MAC_STUDIO.txt
XH = WIDER / "out" / "xcheck" / "h_vs_07_pop07" / "xcheck.json"


def F(h):
    return Fraction(float.fromhex(h))


def main():
    out_p = Path(sys.argv[1])
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07a: the disagreement list at h against 06m (06m from 07's xcheck)", [X07, XH])
    if sha256(X07) != X07_SHA256:
        raise SystemExit("07's xcheck.json has another SHA-256")
    p07 = {(p["row"], p["kind"], p["index"]): p for p in json.loads(X07.read_text())["pairs"]}
    ph = {(p["row"], p["kind"], p["index"]): p for p in json.loads(XH.read_text())["pairs"]}
    if set(p07) != set(ph):
        raise SystemExit(f"pairs differ: {len(p07)} / {len(ph)}")
    listed = []
    for k in sorted(p07):
        a, b = ph[k], p07[k]
        h7, h8, eh7, eh8 = F(a["values"]["g7"]), F(a["values"]["g8"]), F(a["abserrs"]["g7"]), F(a["abserrs"]["g8"])
        m7, m8, em7, em8 = F(b["values"]["m7"]), F(b["values"]["m8"]), F(b["abserrs"]["m7"]), F(b["abserrs"]["m8"])
        g_self, m_self = abs(h7 - h8) <= eh7, abs(m7 - m8) <= em7
        agree8 = abs(h8 - m8) <= max(eh8, em8)
        cls = ("a" if g_self and not m_self and agree8 else "b" if m_self and not g_self and agree8 else
               "c" if g_self and m_self else "d" if not g_self and not m_self else "e")
        for rt, g, m, eg, em in (("1e-7", h7, m7, eh7, em7), ("1e-8", h8, m8, eh8, em8)):
            if abs(g - m) > max(eg, em):
                listed.append({"row": k[0], "kind": k[1], "index": k[2], "rtol": rt, "class": cls,
                               "class_07": b["class"] if b["listed"][rt] else None, "group": a["group"]})
    doc = {"header": hdr, "n_pairs": len(p07), "n_listed_calls": len(listed), "by_class": dict(Counter(x["class"] for x in listed)),
           "by_class_rtol": {f"{c}|{r}": n for (c, r), n in sorted(Counter((x["class"], x["rtol"]) for x in listed).items())},
           "by_group": dict(Counter(x["group"] for x in listed)),
           "listed_at_07_too": sum(1 for x in listed if x["class_07"] is not None),
           "class_h_vs_07": {f"{x['class_07']}->{x['class']}": n for x_ in [None] for x, n in []},
           "calls": listed}
    doc["class_07_to_h"] = dict(Counter(f"{x['class_07']}->{x['class']}" for x in listed))
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: v for k, v in doc.items() if k not in ("calls", "header")}, indent=1))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")


if __name__ == "__main__":
    main()
