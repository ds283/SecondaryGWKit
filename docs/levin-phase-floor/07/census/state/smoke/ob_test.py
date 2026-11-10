"""levin-phase-floor 07 census agent: a development self-test of run_census.py's over-budget pass (not scored).
Runs run_census.main() in its smoke mode (--out-dir) on one row at 1e-7 with the first budget patched to 0.5 s, so
that the unit goes over budget, and the pass's budget patched to 1,000 s. The script itself is unchanged."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_census as RCN  # noqa: E402

RCN.BUDGET_S = 0.5
RCN.OVERBUDGET_S = 1000.0
sys.argv = ["run_census.py", "--out-dir", str(Path(__file__).resolve().parent / "ob"), "--rows", "0-37232", "--rtols", "1e-7",
            "--max-workers", "1"]
RCN.main()
