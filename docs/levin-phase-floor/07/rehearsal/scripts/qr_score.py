"""Score R27 and R28 on the three q-r calls against qr_reference.py's references.

levin-phase-floor 07, rehearsal agent, the user's addendum at planned stop 1. New. Reads out/qr_reference/*.json
(written by qr_reference.py at each tree) and out/captures.json (capcheck: each capture's stored record).
For each call: the two captures' integrands compared bit for bit at the 1001 sample points (x, the integrand,
theta, theta_mod_2pi, theta_deriv) and their x_span and kw; each replay reproduces its capture; then the
call's value in each store (the call's own value, unscaled, which times (1 + z_response) is the stored group,
checked by capcheck) against the reference: |v - ref|, |v - ref| / abserr in each store, and |v - ref| against
the reference's stated error. Arithmetic in mpmath at 40 digits from the doubles.

    ./venv/bin/python -I -B qr_score.py
"""

import hashlib
import json
from pathlib import Path

import mpmath as mp

R = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/rehearsal")
CALLS = [("3-182", "3-20389"), ("0-10266", "0-10036"), ("3-52", "3-20141")]  # (R28 shard-serial, R27 shard-serial)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    mp.mp.dps = 40
    print(f"# script {Path(__file__).resolve()} sha256 {sha256(__file__)}")
    caps = json.loads((R / "out" / "captures.json").read_text())["captures"]
    print(f"# input {R / 'out' / 'captures.json'} sha256 {sha256(R / 'out' / 'captures.json')}")
    out = []
    for g_id, p_id in CALLS:
        fg, fp = R / "out" / "qr_reference" / f"qmr-g-{g_id}.json", R / "out" / "qr_reference" / f"qmr-p-{p_id}.json"
        print(f"# input {fg} sha256 {sha256(fg)}")
        print(f"# input {fp} sha256 {sha256(fp)}")
        g, p = json.loads(fg.read_text()), json.loads(fp.read_text())
        cg = [c for c in caps if c["uid"] == f"g-1e-7-{g_id}"][0]
        cpp = [c for c in caps if c["uid"] == f"p-1e-7-{p_id}"][0]
        assert cg["key"] == cpp["key"]
        same_samples = g["samples"] == p["samples"]
        same_span_kw = g["x_span"] == p["x_span"] and g["kw"] == p["kw"]
        ref = mp.mpf(float.fromhex(g["reference"]["reference"]))
        err = mp.mpf(g["reference"]["stated_error"])
        row = {"key": cg["key"], "label": cg["notify_label"], "R28_uid": cg["uid"], "R27_uid": cpp["uid"],
               "integrands_bit_identical_at_1001_points": same_samples, "x_span_and_kw_equal": same_span_kw,
               "theta_deriv_zero_at_all_samples": g["samples_theta_deriv_all_zero"] and p["samples_theta_deriv_all_zero"],
               "theta_zero_at_all_samples": g["samples_theta_all_zero"] and p["samples_theta_all_zero"],
               "integrand_sign": g["samples_integrand_sign"],
               "replay_reproduces_capture": {"R28": g["replay_reproduces_capture"], "R27": p["replay_reproduces_capture"]},
               "reference": g["reference"]["reference"], "reference_repr": g["reference"]["reference_repr"],
               "reference_stated_error": g["reference"]["stated_error"],
               "reference_rounding_scale": g["reference"]["rounding_scale_eps_abs_sum_m14"],
               "reference_differences": g["reference"]["differences"], "QAGS": g["reference"]["QAGS"],
               "ray_initialized": [g["ray_initialized"], p["ray_initialized"]]}
        for tag, rec in (("R27", p["capture_record"]), ("R28", g["capture_record"])):
            v, e = mp.mpf(float.fromhex(rec["value"])), mp.mpf(float.fromhex(rec["abserr"]))
            d = abs(v - ref)
            row[tag] = {"value": rec["value"], "abserr": rec["abserr"], "converged": rec["converged"],
                        "num_regions": rec["num_regions"], "abs_v_minus_ref": mp.nstr(d, 6),
                        "abs_v_minus_ref_over_abserr": mp.nstr(d / e, 6), "honest": bool(d <= e),
                        "abs_v_minus_ref_over_ref_error": mp.nstr(d / err, 6) if err > 0 else "inf"}
        out.append(row)
        print(f"\n{row['key']} ({row['label']})")
        print(f"  integrands bit-identical at 1001 points (R27's tree vs R28's): {same_samples}; x_span and kw equal: {same_span_kw}; "
              f"theta' == 0 and theta == 0 at every sample: {row['theta_deriv_zero_at_all_samples']}, {row['theta_zero_at_all_samples']}; "
              f"integrand sign {row['integrand_sign']}")
        print(f"  replays reproduce their captures: R28 {g['replay_reproduces_capture']}, R27 {p['replay_reproduces_capture']}")
        print(f"  reference {row['reference']} ({row['reference_repr']}), stated error {row['reference_stated_error']} "
              f"(rounding scale {row['reference_rounding_scale']}); differences {row['reference_differences']}; QAGS abserr {row['QAGS']['abserr']}")
        for tag in ("R27", "R28"):
            x = row[tag]
            print(f"  {tag}: value {x['value']} abserr {x['abserr']} converged {x['converged']} regions {x['num_regions']}: "
                  f"|v - ref| {x['abs_v_minus_ref']} = {x['abs_v_minus_ref_over_abserr']} x abserr ({'honest' if x['honest'] else 'NOT honest'}); "
                  f"{x['abs_v_minus_ref_over_ref_error']} x the reference's stated error")
    dst = R / "out" / "qr_scores.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {dst} sha256 {sha256(dst)}")


if __name__ == "__main__":
    main()
