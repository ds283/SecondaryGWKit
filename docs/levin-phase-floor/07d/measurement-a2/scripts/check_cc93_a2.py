"""levin-phase-floor prompt 07d Part A' (the orchestrator's go-ahead, item 2): V93's Clenshaw-Curtis solve returns the same
CC_93, bit for bit, as item 1's harness formed it (checkrepro_a2.CallContext.cc_value, recorded as "cc93" in each leaf
record). On the first N leaves of each leaf-record directory, in sorted order: V93's ``_adaptive_levin_subregion(...,
force_direct=True)`` value against the record's cc93. New in Part A'.
Usage: PYTHONPATH=<tree>:<scripts> python -B check_cc93_a2.py OUT [N]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prov import ROOT, header  # noqa: E402


def main():
    out = Path(sys.argv[1])
    n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    if out.exists():
        raise SystemExit(f"{out} exists")
    import variant_loader as VL

    vj = json.loads((ROOT / "out" / "variants" / "variants.json").read_text())["variants"]["V93"]
    LQ = VL.install(Path(vj["path"]), vj["sha256"], "a2_V93")
    import checkrepro_a2 as CR
    import leafref_v4 as R4

    rows = []
    for d in sorted((ROOT / "out" / "leaves_a2").iterdir()):
        if not d.is_dir() or not (d.name.startswith("partA_") or d.name.startswith("nb")):
            continue
        k = 0
        for p in sorted(d.glob("*.json")):
            if k >= n_per:
                break
            doc = json.loads(p.read_text())
            spec = json.loads(Path(doc["header"]["spec"]).read_text())
            rec = R4.load_capture(spec["capture"])
            kw = dict(rec["kw"])
            kw["rtol"] = float(spec["rtol"])
            C = CR.CallContext(LQ, rec, kw)
            for L in doc["leaves"]:
                a, b = float.fromhex(L["start"]), float.fromhex(L["end"])
                v = LQ._adaptive_levin_subregion((a, b), rec["f"], C.A, id_label=C.id_label, chebyshev_order=C.N,
                                                 rtol=C.rtol, vectorize_cache=C.cache, force_direct=True)["value"]
                rows.append({"id": L["id"], "variant": float(v).hex(), "record": L["cc"]["cc93"],
                             "equal": float(v).hex() == L["cc"]["cc93"]})
                k += 1
    summ = {"n": len(rows), "equal": sum(r["equal"] for r in rows)}
    out.write_text(json.dumps({"header": header("levin-phase-floor 07d Part A': V93's CC_93 against item 1's", __file__,
                                                [Path(vj["path"])]), "summary": summ, "rows": rows}, indent=1))
    print(summ)


if __name__ == "__main__":
    main()
