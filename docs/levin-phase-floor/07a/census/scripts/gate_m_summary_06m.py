"""levin-phase-floor prompt 06m section 5: a small summary of out/gate_m/compare_m.json (41 MB, kept on the Mac
Studio) for the laptop's records: every key of it but the per-unit rows, plus each per-unit row's counts summed.
New in 06m. Read only. Usage: PYTHONPATH=<tree> python -B gate_m_summary_06m.py IN_JSON OUT_JSON"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import header, sha256  # noqa: E402

src, out = Path(sys.argv[1]), Path(sys.argv[2])
d = json.loads(src.read_text())
s = {k: v for k, v in d.items() if k != "units"}
s["summary_header"] = header(Path(__file__).resolve(), "levin-phase-floor 06m: Gate M's comparison, summarised", [src])
s["n_units_rows"] = len(d["units"])
s["n_units_ok"] = sum(1 for u in d["units"] if u["ok"])
s["units_by_rtol"] = {rt: sum(1 for u in d["units"] if u["unit"].split("-")[1] + "-" + u["unit"].split("-")[2] == rt) for rt in ("1e-7", "1e-8")}
out.write_text(json.dumps(s, indent=1, default=str))
print(json.dumps({k: s[k] for k in ("n_units", "n_units_equal", "n_units_differing", "statuses", "totals", "worker_evidence_all_hold", "units_by_rtol")}))
print(f"wrote {out} sha256 {sha256(out)}")
