"""levin-phase-floor 06g: V2's violating rows in detail (descriptive, read only): per row and rtol, each call whose
value moved, with |v_g - v_06h| against abserr_06h + abserr_g, both abserrs, regions, depth, converged, and the
row's numeric_quad and WKB_Levin parts. New in 06g. Usage: python -B v2_detail06g.py RUN_DIR ROW..."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import RUN06H  # noqa: E402
H = lambda x: float.fromhex(x["hex"])
run, rows = Path(sys.argv[1]), sys.argv[2:]
want = {f"g-{rt}-{r}" for rt in ("1e-7", "1e-8") for r in rows}
def read(p):
    out = {}
    with open(p / "checkpoint.jsonl") as f:
        for line in f:
            u = line[10:60].split('"')[0]
            if u in want:
                out[u] = json.loads(line)["data"]
    return out
G, Hh = read(run), read(RUN06H)
for u in sorted(want):
    g, h = G[u], Hh[u]
    rg, rh = g["result"], h["result"]
    print(f"== {u}: total g {rg['total']['repr']} (abserr {rg['total_abserr']['repr']}, conv {rg['total_converged']}) 06h {rh['total']['repr']} (abserr {rh['total_abserr']['repr']}, conv {rh['total_converged']})")
    print(f"   numeric_quad g {rg['numeric_quad']['repr']} 06h {rh['numeric_quad']['repr']}; WKB_Levin g {rg['WKB_Levin']['repr']} 06h {rh['WKB_Levin']['repr']}")
    hb = {(k, c["index"]): c for k in ("calls", "other_calls") for c in h[k]}
    for k in ("calls", "other_calls"):
        for c in g[k]:
            ch = hb[(k, c["index"])]
            dv = abs(H(c["value"]) - H(ch["value"]))
            if dv == 0:
                continue
            b = H(c["abserr"]) + H(ch["abserr"])
            print(f"   {k[:5]} {c['index']} {c.get('group') or c['notify_label']}: |dv| {dv:.3e} / (e_06h + e_g) {b:.3e} = {dv/b:.3f}; abserr 06h {H(ch['abserr']):.3e} g {H(c['abserr']):.3e}; "
                  f"regions {ch['num_regions']} -> {c['num_regions']}; depth {ch['max_depth']} -> {c['max_depth']}; conv {ch['converged']} -> {c['converged']}; "
                  f"(b) leaves {(c.get('checked_06g') or {}).get('n_b_untrusted')}; untrusted returned {c.get('num_untrusted_regions')}; CC {c.get('has_cc_leaf')}")
