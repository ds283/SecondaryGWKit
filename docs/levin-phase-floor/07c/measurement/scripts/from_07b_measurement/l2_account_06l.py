"""levin-phase-floor prompt 06l, Part L: an account of each first-population slow leaf (L-a, delta1/phi_P < 2) that
route B does not trust (L2), from the probes' own numbers. New in 06l. Read only. Descriptive, scores nothing.

Per leaf: delta1 against the node-error difference |E_P - (E_L + E_R)| and against |E_P| + |E_L + E_R| (the
triangle-inequality bound section 0.2 uses), the signs of E_P and E_L + E_R, and nu_B's terms: phi_P - |E_P| (the
parent's floor less its node error, which nu_B carries), phi^-_L + phi^-_R, |E_L + E_R|. The same for the 63 that
B trusts, as distributions. Writes out/partL_l2/account.json and prints it.
Usage: PYTHONPATH=<tree>:<scripts> python -B l2_account_06l.py
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402


def main():
    sel = json.loads((RC.ROOT / "out" / "sample" / "selection.json").read_text())
    files = [RC.ROOT / "out" / "partL" / f"{e['sid']}.json" for e in sel if e["part"] == "S"]
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 06l: L2's untrusted first-population leaves, accounted", files)
    rows = []
    for f in files:
        d = json.loads(f.read_text())
        for L in d["leaves"]:
            if L["population"] != "L-a" or not (L["d1_over_phiP"] < 2):
                continue
            E = L["E"]
            ELR = math.fsum([E["L"], E["R"]])
            ndiff = abs(math.fsum([E["P"], -E["L"], -E["R"]]))
            tri = abs(E["P"]) + abs(ELR)
            rows.append({
                "sid": d["sid"], "start": L["start_hex"], "trusted_B": L["decision"]["routeB"] == "trusted",
                "r": L["r"], "d1_over_phiP": L["d1_over_phiP"], "d1_over_nuB": L["d1_over_nuB"],
                "d1_over_node_difference": L["delta1"] / ndiff if ndiff else None,
                "d1_over_EP_plus_ELR": L["delta1"] / tri if tri else None,
                "signs_EP_ELR_opposite": (E["P"] > 0) != (ELR > 0),
                "node_difference_over_triangle_bound": ndiff / tri if tri else None,
                "nuB_terms_over_nuB": {
                    "|E_P|": abs(E["P"]) / L["nu_B_variant"],
                    "phi_P - |E_P|": (L["phi_P"] - abs(E["P"])) / L["nu_B_variant"],
                    "phi-_L + phi-_R": (L["phim"]["L"] + L["phim"]["R"]) / L["nu_B_variant"],
                    "|E_L + E_R|": abs(ELR) / L["nu_B_variant"]},
            })
    un = [r for r in rows if not r["trusted_B"]]
    tr = [r for r in rows if r["trusted_B"]]

    def rng(v):
        v = sorted(x for x in v if x is not None)
        return [v[0], v[len(v) // 2], v[-1]] if v else None
    doc = {"header": hdr, "n_first_population": len(rows), "n_untrusted": len(un), "untrusted": un,
           "summary": {grp: {k: rng([r[k] for r in rs]) for k in ("r", "d1_over_phiP", "d1_over_nuB", "d1_over_node_difference",
                                                                  "d1_over_EP_plus_ELR", "node_difference_over_triangle_bound")}
                       | {"signs_opposite": sum(r["signs_EP_ELR_opposite"] for r in rs), "n": len(rs)}
                       for grp, rs in (("untrusted", un), ("trusted", tr))}}
    out = RC.ROOT / "out" / "partL_l2" / "account.json"
    out.write_text(json.dumps(doc, indent=1))
    print("\n".join(hdr))
    print(json.dumps(doc["summary"], indent=1))
    for r in un:
        print(r["sid"], r["start"], f"r {r['r']:.4f} d1/phiP {r['d1_over_phiP']:.4f} d1/nuB {r['d1_over_nuB']:.5f} "
              f"d1/|EP-(EL+ER)| {r['d1_over_node_difference']:.5f} d1/(|EP|+|EL+ER|) {r['d1_over_EP_plus_ELR']:.5f} "
              f"opposite {r['signs_EP_ELR_opposite']} terms {({k: round(v, 5) for k, v in r['nuB_terms_over_nuB'].items()})}")
    print(f"wrote {out} sha256 {RC.sha256(out)}")


if __name__ == "__main__":
    main()
