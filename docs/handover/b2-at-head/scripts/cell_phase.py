"""Run one of B2's cells at HEAD with the Levin driver's T phase derivative taken from the phase it
oscillates with (`phase.theta_deriv`) instead of from `omega`: probe13's "phase" arm
(docs/handover/b2/scripts/probe13.py), applied to b2_cells' own entry point.

    PYTHONPATH=. ./venv/bin/python docs/handover/b2-at-head/scripts/cell_phase.py \
        --cell together 1e+08 R --out PATH
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "docs" / "handover" / "b2" / "scripts"))
sys.path.insert(0, str(REPO))

import ComputeTargets.phase_groups as PG  # noqa: E402
import b2_cells  # noqa: E402


def phase_deriv(self, log_z):
    return self.phase.theta_deriv(log_z, x_is_log=True, log_derivative=True)


PG._OscillatoryT.theta_deriv = phase_deriv

if __name__ == "__main__":
    b2_cells.main()
