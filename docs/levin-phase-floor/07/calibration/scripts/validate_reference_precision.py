"""levin-phase-floor 07 section 2.1 (rewritten text of 2026-11-07): the reference's precision on one case

Taken unchanged in substance from the set-aside work's copy (the tarball
var/levin-phase-floor-07/wip/07-part-a-rebased-on-06d-untracked.tar.gz,
docs/levin-phase-floor/07/calibration/scripts/validate_reference_precision.py, kept byte for byte in
docs/levin-phase-floor/07/earlier-attempts/), changed only where marked "rewrite": this paragraph; the
prompt sections it cites, which the rewrite renumbered (the old "section 9 point 2" and "point 3", the
user's rulings on the reference and on omega, are the rewritten section 2.1); and the provenance header,
now prov.py's list of lines. The text before the rewrite follows.

levin-phase-floor 07 section 2: the one case on which ``validate_reference.py`` found the
proposed and the prompt's references differing by more than 1e-29 relative -- (C), b = 1e4,
lambda = 0.25, by 4.1e-28 (1.4e-26 relative) -- recomputed by both at higher precision, to say
which of the two carried the difference.

- the proposed reference, ``contour_C``, at 60 working digits in place of 40;
- the prompt's reference made to converge, ``literal``, at 45 digits on pieces of at most pi/4 of
  phase in place of 30 digits and pi.

Writes ``out/validate_reference_precision.json``.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mpmath as mp  # noqa: E402

import reference as R  # noqa: E402
from prov import OUT, header  # noqa: E402


def main():
    case = [
        c
        for c in R.cases()
        if c["family"] == "C" and c["b"] == 1.0e4 and c["lambda"] == 0.25
    ][0]
    out = {  # rewrite: reference.py recorded as an input
        "header": header(
            Path(__file__),
            "07 section 2: reference precision",
            (HERE / "reference.py",),
        )
    }
    out["case"] = case
    contour40, _ = R.contour_C(case["b"], case["a"])
    lit30, n30 = R.literal(*R.family_C(case["b"], case["a"]))
    R.WORKING_DPS = 60
    contour60, _ = R.contour_C(case["b"], case["a"])
    R.REPORT_DPS = 45
    lit45, n45 = R.literal(*R.family_C(case["b"], case["a"]), piece_phase=mp.pi / 4)
    with mp.workdps(60):
        out["contour_40_digits"] = mp.nstr(contour40, 35)
        out["contour_60_digits"] = mp.nstr(contour60, 35)
        out["literal_30_digits_pi_pieces"] = mp.nstr(lit30, 35)
        out["literal_30_pieces"] = n30
        out["literal_45_digits_quarter_pi_pieces"] = mp.nstr(lit45, 35)
        out["literal_45_pieces"] = n45
        out["contour40_minus_literal45"] = mp.nstr(contour40 - lit45, 5)
        out["literal30_minus_literal45"] = mp.nstr(lit30 - lit45, 5)
        out["contour60_minus_literal45"] = mp.nstr(contour60 - lit45, 5)
    print(json.dumps(out, indent=1))
    path = OUT / "validate_reference_precision.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
