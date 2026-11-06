"""levin-phase-floor prompt 06l section 3: the census's cost projection, made before the run. New in 06l. Read only.

From K0's 100 units, run today one after the other in the same registered run (identity, then route B, six workers
each): rho = (sum of route B's unit walls) / (sum of the identity variant's), the identity variant being 06g's code.
06g's census unit walls for the same 100 units (out/k0/units.json's wall_s_06g) are reported beside them, and so is
route 1's K0 ratio as 06k recorded it, if its summary holds one.
Projections of the census wall:
- P_06k = 06h + (06g - 06h) * rho, 06k's form (06h 9,193 s, 06g 20,361 s);
- P_g = 06g * rho;
- the bracket of prompt 06l section 3: route 1's 16,758 s <= wall <= 06g's 20,361 s, since B trusts no leaf route 1
  would not.
The projection handed to the backstop is the largest of P_06k, P_g and route 1's wall, and never above 06g's wall
times 1.1 (the reasons are in the log). Writes out/census_projection.json and prints it.
Usage: PYTHONPATH=<tree>:<scripts> python -B census_projection_06l.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402

W06G, W06H, W_R1 = 20361.0, 9193.0, 16758.0


def walls(v):
    out = {}
    for line in open(RC.ROOT / "state" / "k0" / v / "units.jsonl"):
        m = json.loads(line)
        out[m["unit"]] = float(m["data"]["wall_s"])
    return out


def main():
    files = [RC.ROOT / "state" / "k0" / "identity" / "units.jsonl", RC.ROOT / "state" / "k0" / "routeB" / "units.jsonl",
             RC.ROOT / "out" / "k0" / "units.json"]
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 06l: the census's cost projection", files)
    wi, wb = walls("identity"), walls("routeB")
    units = json.loads(files[2].read_text())["units"]
    wg = {u["uid"]: float(u["wall_s_06g"]) for u in units}
    assert set(wi) == set(wb) == set(wg) and len(wi) == 100
    si, sb, sg = sum(wi.values()), sum(wb.values()), sum(wg.values())
    rho = sb / si
    by_rtol = {rt: {"identity": sum(w for u, w in wi.items() if f"-{rt}-" in u),
                    "routeB": sum(w for u, w in wb.items() if f"-{rt}-" in u),
                    "06g_census": sum(w for u, w in wg.items() if f"-{rt}-" in u)} for rt in ("1e-7", "1e-8")}
    p06k = W06H + (W06G - W06H) * rho
    pg = W06G * rho
    chosen = min(max(p06k, pg, W_R1), 1.1 * W06G)
    doc = {"header": hdr, "k0_unit_wall_sum_s": {"identity": si, "routeB": sb, "06g_census_same_units": sg},
           "k0_unit_wall_by_rtol_s": by_rtol, "rho_routeB_over_identity": rho,
           "identity_over_06g_census_same_units": si / sg,
           "P_06k_form_s": p06k, "P_g_s": pg, "bracket_s": [W_R1, W06G], "projection_s": round(chosen),
           "one_day_s": 86400.0}
    out = RC.ROOT / "out" / "census_projection.json"
    out.write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc, indent=1))
    print(f"wrote {out} sha256 {RC.sha256(out)}")


if __name__ == "__main__":
    main()
