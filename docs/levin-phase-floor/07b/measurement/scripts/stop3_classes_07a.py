"""levin-phase-floor prompt 07a section 6, stop 3 (new in 07a; read only): from compare_07a.py's calls output, the
classes by rtol and kind; the untouched calls that differ, by the paths that differ and by their location record (skipped
at 07, at h; points and windows at h); the calls of each class whose `call` category differs; and, for reading B's extra
class (aux_only), how value, abserr and converged moved. Writes OUT_JSON and prints a summary. No verdict is scored.
Usage: python -B stop3_classes_07a.py CALLS_JSONL OUT_JSON
"""
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import header, sha256  # noqa: E402


def fr(h):
    return Fraction(float.fromhex(h))


def main():
    src, out_p = Path(sys.argv[1]), Path(sys.argv[2])
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07a: stop 3, the comparison with 07's census by class", [src])
    by = {}
    unt_loc = {"n": 0, "paths": {}, "n_points_h_nonzero": 0, "n_windows_h_nonzero": 0, "skipped_07_kinds": {},
               "skipped_h_kinds": {}, "examples": []}
    call_cat = {}
    aux = {"n": 0, "value_equal": 0, "abserr_equal": 0, "abserr_smaller_h": 0, "abserr_larger_h": 0, "conv_changes": [],
           "abserr_ratio_h_over_07": [], "value_move_over_abserr07": [], "regions_equal": 0}
    for line in open(src):
        c = json.loads(line)
        k = (c["rtol"], c["kind"], c["class"])
        by[f"{k[0]}|{k[1]}|{k[2]}"] = by.get(f"{k[0]}|{k[1]}|{k[2]}", 0) + 1
        if "call" in c["categories"]:
            call_cat[c["class"]] = call_cat.get(c["class"], 0) + 1
        if c["class"] == "untouched" and not c["equal"]:
            unt_loc["n"] += 1
            for p in sum(c["first_differing"].values(), []):
                q = p.split("]", 1)[1] if "]" in p else p
                unt_loc["paths"][q] = unt_loc["paths"].get(q, 0) + 1
            unt_loc["n_points_h_nonzero"] += bool(c["n_points_h"])
            unt_loc["n_windows_h_nonzero"] += bool(c["n_windows_h"])
            s7 = json.dumps([sorted(e.keys()) for e in (c["skipped_07"] or [])])
            sh = json.dumps([e.get("reason") for e in (c["skipped_h"] or [])])
            unt_loc["skipped_07_kinds"][s7] = unt_loc["skipped_07_kinds"].get(s7, 0) + 1
            unt_loc["skipped_h_kinds"][sh] = unt_loc["skipped_h_kinds"].get(sh, 0) + 1
            if len(unt_loc["examples"]) < 5:
                unt_loc["examples"].append({k_: c[k_] for k_ in ("unit", "kind", "index", "label", "first_differing", "skipped_07", "skipped_h")})
        if c["class"] == "aux_only":
            aux["n"] += 1
            v, a, cv = c["value"], c["abserr"], c["converged"]
            aux["value_equal"] += v[0] == v[1]
            aux["regions_equal"] += c["num_regions"][0] == c["num_regions"][1]
            ah, a7 = fr(a[0]), fr(a[1])
            aux["abserr_equal"] += ah == a7
            aux["abserr_smaller_h"] += ah < a7
            aux["abserr_larger_h"] += ah > a7
            if a7:
                aux["abserr_ratio_h_over_07"].append(float(ah / a7))
                aux["value_move_over_abserr07"].append(float(abs(fr(v[0]) - fr(v[1])) / a7))
            if cv[0] != cv[1]:
                aux["conv_changes"].append({"unit": c["unit"], "index": c["index"], "kind": c["kind"], "converged_h_07": cv})
    for k_ in ("abserr_ratio_h_over_07", "value_move_over_abserr07"):
        xs = sorted(aux[k_])
        aux[k_] = {"n": len(xs), "min": xs[0] if xs else None, "p10": xs[len(xs) // 10] if xs else None,
                   "median": xs[len(xs) // 2] if xs else None, "p90": xs[9 * len(xs) // 10] if xs else None,
                   "max": xs[-1] if xs else None}
    doc = {"header": hdr, "by_rtol_kind_class": dict(sorted(by.items())), "untouched_differing": unt_loc,
           "calls_with_call_category_differing_by_class": call_cat, "aux_only": aux}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=1))
    print(json.dumps({k_: v_ for k_, v_ in doc.items() if k_ != "header"}, indent=1)[:9000])
    print(f"wrote {out_p} sha256 {sha256(out_p)}")


if __name__ == "__main__":
    main()
