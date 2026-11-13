"""levin-phase-floor prompt 07, beyond section 6 (the cross-check the user asked for at stop 4): the reference cost of
each captured class (c)/(d) call by refs_projection_07.py's model, unchanged: its ``features`` of the capture, and the
coefficients alpha, beta and gamma as out/v4/refs_projection.json holds them (SHA-256 checked), per reading (QAWO r1,
r2, FCC r1, r2), one reading at a time. Only captures that reproduce 07's census are projected. New in 07. Read only.
Usage: PYTHONPATH=<tree g>:<scripts> python -B xcheck_projection_07.py CAPTURES_JSON CHECK_JSON OUT_JSON"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_projection_07 as RP  # noqa: E402
from prov import WIDER, header, sha256  # noqa: E402

PROJ = WIDER / "out" / "v4" / "refs_projection.json"
PROJ_SHA256 = "77831c1a0e5a71cbe8460d3ad6fee3bea54766a443d33b41e705af78d0d9300e"


def main():
    caps_p, chk_p, out_p = (Path(a) for a in sys.argv[1:4])
    if sha256(PROJ) != PROJ_SHA256:
        raise SystemExit("the projection file has another SHA-256")
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07: the cross-check's class (c)/(d) reference costs",
                 [caps_p, chk_p, PROJ, HERE / "refs_projection_07.py"])
    P = json.loads(PROJ.read_text())
    ab, gm = P["alpha_beta"], {k: v["gamma"] for k, v in P["gamma"].items()}
    chk = {r["sid"]: r for r in json.loads(chk_p.read_text())}
    out = []
    for e in json.loads(caps_p.read_text()):
        row = {"sid": e["sid"], "unit": e["unit"], "index": e["index"], "class": e["class"]}
        if not (e.get("capture_matches_census") and chk.get(e["sid"], {}).get("reproduces_census")):
            row["not_projected"] = "the capture does not reproduce the census"
            out.append(row)
            continue
        try:
            f = RP.features(RP.load_capture(Path(e["dir"]) / e["pickle"], e["offset"], e["length"], e["sha256"]))
        except Exception as exc:  # e.g. a call the model's Route B features do not cover
            row["not_projected"] = repr(exc)[:300]
            out.append(row)
            continue
        per = {}
        for k in ("qawo_r1", "qawo_r2", "fcc_r1", "fcc_r2"):
            D = f["D_q"] if k.startswith("qawo") else f["D_f"]
            span = f["span_q"] if k.startswith("qawo") else f["span_f"]
            per[k] = ab[k]["alpha"] * f["n_seg"] + ab[k]["beta"] * span + max(gm[k], 0.0) * D
        row.update(features=f, readings_s=per, total_s=sum(per.values()), fcc_s=per["fcc_r1"] + per["fcc_r2"],
                   qawo_each_le_1800=max(per["qawo_r1"], per["qawo_r2"]) <= 1800)
        out.append(row)
    pr = [r for r in out if "readings_s" in r]
    doc = {"header": hdr, "n_calls": len(out), "n_projected": len(pr),
           "total_all_four_s": sum(r["total_s"] for r in pr), "total_fcc_s": sum(r["fcc_s"] for r in pr),
           "total_fcc_plus_qawo_le_1800_s": sum(r["fcc_s"] + (r["readings_s"]["qawo_r1"] + r["readings_s"]["qawo_r2"] if r["qawo_each_le_1800"] else 0) for r in pr),
           "calls": out}
    out_p.write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: v for k, v in doc.items() if k not in ("header", "calls")}, indent=1))
    for r in sorted(pr, key=lambda r: r["total_s"])[:5] + sorted(pr, key=lambda r: r["total_s"])[-5:]:
        print(r["sid"], r["unit"], {k: round(v) for k, v in r["readings_s"].items()}, round(r["total_s"]))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")


if __name__ == "__main__":
    main()
