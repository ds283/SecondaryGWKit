"""depth20-tiling steps 2 and 4: one captured call, in this (fresh) process, in one mode.

  replay  -- adaptive_levin_sincos(x_span, f, theta=theta, **kw) on the reloaded capture, at e5764a6
             as it stands: value and abserr (float.hex), num_regions, max_depth and converged
             against the census call's (the capture counts only if all equal).
  pinned  -- the same call with chebyshev_matrices swapped, in this process only, for the wrapper
             that pins x[0] := b and x[-1] := a (dcommon.pinned). First the patch is checked
             element by element against the original on several spans (the call's x_span and
             the census line's spans at depths 0, 5, 10, 15, 16, 18, 20 and their halves, at the
             orders 24 and 47): every other node and D bit-identical.

Writes OUT_JSON (every number in hex as well), and prints it.

Usage: PYTHONPATH=<tree>:<scripts> python -B runcall.py CAPTURED_JSON SAMPLE_NO MODE OUT_JSON
"""

import collections
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dprov import STATE, header, hx  # noqa: E402

import AdaptiveLevin.levin_quadrature as LQ  # noqa: E402
import dcommon as DC  # noqa: E402


def summarise(rr):
    regs = rr["regions"]
    dh = collections.Counter(r.depth for r in regs)
    return {
        "value": hx(rr["value"]), "abserr": hx(rr["abserr"]), "value_repr": repr(rr["value"]), "abserr_repr": repr(rr["abserr"]),
        "num_regions": int(rr["num_regions"]), "max_depth": int(max(r.depth for r in regs)), "converged": bool(rr["converged"]),
        "num_at_depth_20": int(dh.get(20, 0)), "num_phase_limited_regions": int(rr["num_phase_limited_regions"]),
        "num_untrusted_regions": int(rr["num_untrusted_regions"]), "num_subregion_solves": int(rr["num_subregion_solves"]),
        "num_simple_regions": int(rr["num_simple_regions"]), "abserr_resolution": hx(rr["abserr_resolution"]),
        "abserr_roundoff": hx(rr["abserr_roundoff"]), "abserr_fallback": hx(rr["abserr_fallback"]),
        "abserr_truncation": hx(rr["abserr_truncation"]), "depth_hist": {str(k): v for k, v in sorted(dh.items())},
        "elapsed_driver": rr["elapsed"],
    }


def main():
    cap_p, no, mode, out_p = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    cheap_p = STATE / "cheap_pass.json"
    hdr = header(Path(__file__).resolve(), f"depth20-tiling: sample {no}, mode {mode}", [cap_p, cheap_p, Path(LQ.__file__), Path(DC.__file__)])
    ents = [e for e in json.loads(cap_p.read_text()) if e["sample_no"] == no]
    assert len(ents) == 1
    e = ents[0]
    rec = DC.load_capture(e)
    x_span = tuple(rec["x_span"])
    kw = dict(rec["kw"])
    out = {"header": hdr, "sample_no": no, "unit": e["unit"], "index": e["index"], "notify_label": rec["notify_label"], "mode": mode,
           "x_span": [hx(x_span[0]), hx(x_span[1])],
           "kw": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v)) for k, v in kw.items()},
           "census": {"value": e["census_value"], "abserr": e["census_abserr"], "num_regions": e["census_num_regions"],
                      "max_depth": e["census_max_depth"], "converged": e["census_converged"], "call_elapsed_s": e["call_elapsed_s"]}}
    if mode == "pinned":
        line = [ln for ln in json.loads(cheap_p.read_text()) if (ln["unit"], ln["index"]) == (e["unit"], e["index"])][0]
        spans = [x_span]
        for lv in line["levels"]:
            if lv["depth"] in (0, 5, 10, 15, 16, 18, 20):
                a, b, c = (float.fromhex(lv[k]) for k in ("a", "b", "c"))
                spans += [(a, b), (a, c), (c, b)]
        chk = DC.patch_selfcheck(spans)
        out["patch_selfcheck"] = {
            "spans_x_orders": len(chk),
            "all_interior_nodes_and_D_bit_identical": all(r["interior_nodes_bit_identical"] and r["D_bit_identical"] for r in chk),
            "all_pinned_ends_exact": all(r["x0_is_b"] and r["xN_is_a"] for r in chk),
            "original_untouched": all(r["original_untouched"] for r in chk),
            "n_original_top_off": sum(not r["original_x0_was_b"] for r in chk),
            "n_original_bottom_off": sum(not r["original_xN_was_a"] for r in chk),
            "rows": chk,
        }
        if not (out["patch_selfcheck"]["all_interior_nodes_and_D_bit_identical"] and out["patch_selfcheck"]["all_pinned_ends_exact"]):
            raise SystemExit("patch self-check failed")
    t0 = time.perf_counter()
    if mode == "replay":
        rr = LQ.adaptive_levin_sincos(x_span, rec["f"], theta=rec["theta"], **kw)
    elif mode == "pinned":
        with DC.pinned():
            assert LQ.chebyshev_matrices is DC.pinned_chebyshev_matrices
            rr = LQ.adaptive_levin_sincos(x_span, rec["f"], theta=rec["theta"], **kw)
        assert LQ.chebyshev_matrices is DC._ORIG_CHEB
    else:
        raise SystemExit(mode)
    wall = time.perf_counter() - t0
    out["result"] = summarise(rr)
    out["wall_s"] = wall
    c = out["census"]
    r = out["result"]
    out["equals_census"] = (r["value"], r["abserr"], r["num_regions"], r["max_depth"], r["converged"]) == (
        c["value"], c["abserr"], c["num_regions"], c["max_depth"], c["converged"])
    out["capture_record_equals_census"] = (rec["value"], rec["abserr"], rec["num_regions"], rec["max_depth"], rec["converged"]) == (
        c["value"], c["abserr"], c["num_regions"], c["max_depth"], c["converged"])
    out_p.write_text(json.dumps(out, indent=1))
    for l in hdr:
        print(l)
    print(json.dumps({k: v for k, v in out.items() if k not in ("header", "patch_selfcheck")}, indent=1))
    if mode == "pinned":
        print("patch self-check:", json.dumps({k: v for k, v in out["patch_selfcheck"].items() if k != "rows"}))


if __name__ == "__main__":
    main()
