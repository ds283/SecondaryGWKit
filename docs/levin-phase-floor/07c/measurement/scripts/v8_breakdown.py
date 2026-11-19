"""levin-phase-floor 06g: a breakdown of out/v8_calls.json (V8's lost calls, outside the 720), read only, descriptive:
by rtol, kind and block/group; how many lost calls meet their request on abserr (budget met, unconverged only for
returned untrusted leaves at the cap); the untrusted leaves' counts; the node-limited calls in full; the gained
calls by kind. New in 06g. Usage: python -B v8_breakdown.py"""
import collections, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
d = json.loads((HERE.parent / "out" / "v8_calls.json").read_text())
H = lambda x: None if x is None else float.fromhex(x)
for rtol in ("1e-7", "1e-8"):
    L = [x for x in d["lost"] if x["rtol"] == rtol and not x["stationary_point_row"]]
    print(f"== {rtol}: {len(L)} lost calls outside the 720, {len({x['row'] for x in L})} rows")
    print("   by kind:", dict(collections.Counter(x["g"]["kind"] for x in L)))
    print("   by block/group:", dict(collections.Counter(x["g"]["group"] if x["g"]["kind"] == "group" else x["g"]["notify_label"] for x in L).most_common()))
    met = [x for x in L if x["request_g"] is not None and H(x["g"]["abserr"]) <= x["request_g"]]
    print(f"   abserr_g <= request_g (budget met; unconverged by returned untrusted leaves): {len(met)}; above the request: {len(L) - len(met)}")
    print(f"   with num_untrusted_regions >= 1: {sum(1 for x in L if (x['g'].get('num_untrusted_regions') or 0) >= 1)}")
    ab = [H(x["g"]["abserr"]) / x["request_g"] for x in L if x["request_g"]]
    ab.sort()
    if ab:
        print(f"   abserr_g / request_g: min {ab[0]:.3g} median {ab[len(ab)//2]:.3g} max {ab[-1]:.3g}")
    ah = sorted(H(x["06h"]["abserr"]) / x["request_06h"] for x in L if x["request_06h"])
    if ah:
        print(f"   abserr_06h / request_06h: min {ah[0]:.3g} median {ah[len(ah)//2]:.3g} max {ah[-1]:.3g}")
    ut = sorted((x["g"].get("num_untrusted_regions") or 0) for x in L)
    print(f"   untrusted leaves returned per call: min {ut[0]} median {ut[len(ut)//2]} max {ut[-1]}; regions g median {sorted(x['g']['num_regions'] for x in L)[len(L)//2]}, 06h median {sorted(x['06h']['num_regions'] for x in L)[len(L)//2]}")
    print("   classes:", dict(collections.Counter(x["class"] for x in L)))
    for x in L:
        if x["class"] == "node_limited":
            g, h = x["g"], x["06h"]
            print(f"   node_limited: {x['unit']} call {g['index']} {g['kind']} {g['group'] if g['kind']=='group' else g['notify_label']}: abserr 06h {H(h['abserr']):.4e} -> g {H(g['abserr']):.4e}; request {x['request_g']:.4e}; "
                  f"e' sum {H(g['abserr_eprime_sum']):.4e}, |sum E| {H(g['abserr_node_signed']):.4e}, sum|E| {H(g['node_error_abs_sum']):.4e}; regions {h['num_regions']} -> {g['num_regions']}; depth {h['max_depth']} -> {g['max_depth']}; "
                  f"(b) leaves {g['checked_06g']['n_b_untrusted']}; at cap {g['n_at_depth_max']}")
G = [x for x in d["gained"]]
print("== gained:", len(G), dict(collections.Counter((x["rtol"], x["g"]["kind"], x["stationary_point_row"]) for x in G)))
