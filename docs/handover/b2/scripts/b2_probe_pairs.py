"""
handover prompt 13 (B2), Part B: a post-run diagnostic probe, NOT part of the registered run.
Pairwise hybrids E'+i+j (terms i and j realistic, the rest exact) at the two places where the
run's interactions are largest relative to the largest single term: q-smooth 980 (pairs with
TLG) and together 1e7 (pairs with G). The pair interaction is
    I_ij = (N_E'+i+j - N_E') - A_i - A_j.
Same construction as b2_cells (apply_hybrid on Case(exact=False)), same tolerances; in-process.
Writes out/probe_pairs.txt. A record of its own SHA-256 once landed.
"""
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2_cells as C  # noqa: E402

warnings.simplefilter("ignore")


def N(shape, x, delegated_realistic):
    case, _ = C.build_case(shape, x, "R")
    exact_terms = set(C.TERMS) - set(delegated_realistic)
    C.apply_hybrid(case, exact_terms)
    out = case.run(atol=C.REF_ATOL, rtol=C.REF_RTOL)
    ref = C.reference_of(case, case.shape)
    total = float(out["total"])
    halves = abs(float(out["numeric_quad"])) + abs(float(out["WKB_Levin"]))
    declared = max(float(out["total_abserr"]), C.REF_RTOL * halves) / abs(total)
    n = C.KT.KT_NORM * total / ref["predicted"]
    return n, abs(n) * declared


lines = [f"b2_probe_pairs.py, b2_cells sha256 {C.SCRIPT_SHA256}"]
for shape, x, anchor in (("q-smooth", 980.0, "TLG"), ("together", 1e7, "G"), ("q-smooth", 1e7, "TLG")):
    base, eb = N(shape, x, [])
    A = {t: N(shape, x, [t]) for t in C.TERMS}
    lines.append(f"{shape} {x:g}: N_E'+9/8 = {base + 9 / 8:+.4e}")
    for t in C.TERMS:
        lines.append(f"   A_{t:5s} = {A[t][0] - base:+.4e} (err {A[t][1] + eb:.1e})")
    for other in C.TERMS:
        if other == anchor:
            continue
        nij, eij = N(shape, x, [anchor, other])
        I = (nij - base) - (A[anchor][0] - base) - (A[other][0] - base)
        lines.append(
            f"   pair {anchor}+{other}: N_E'+{anchor}+{other} - N_E' = {nij - base:+.4e}; "
            f"interaction I = {I:+.4e} (err {eij + A[anchor][1] + A[other][1] + 2 * eb:.1e})"
        )
text = "\n".join(lines) + "\n"
print(text)
(HERE.parent / "out" / "probe_pairs.txt").write_text(text)
