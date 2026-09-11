DEFAULT_STRING_LENGTH = 256
DEFAULT_FLOAT_PRECISION = 1e-7
DEFAULT_REDSHIFT_RELATIVE_PRECISION = 1e-7

DEFAULT_ABS_TOLERANCE = 1e-10
DEFAULT_REL_TOLERANCE = 1e-8

# absolute tolerance for the *numeric* part of the matter transfer function alone; see
# prompts/GkTk-remedial (prompt 12) and review section 12.5 of
# docs/gk-wkb-review-fable-2026-09-09.md.
#
# DEFAULT_ABS_TOLERANCE is right for the Green's function, whose |G| is enormous in these units
# (~1e10 at the hand-over in Mpc_units), so an absolute floor of 1e-10 never binds: at the
# production geometry, moving atol from 1e-10 to 1e-13 changes G by 8.5e-10 of the local
# Liouville-Green envelope and costs 0.4% more right-hand-side evaluations.
#
# It is wrong for T. The transfer function decays as 3/x^2, so |T| ~ 1.2e-2 at the numeric stop
# point and ~1e-5 deeper inside the horizon: there atol = 1e-10 is acting as a 1e-5 *relative*
# tolerance, and it, not the solver, sets the accuracy of the whole numeric region. On the exact
# radiation control (T = 1, T' = 0 five e-folds outside the horizon, production source grid and
# stop window, rtol = 1e-8), the maximum error of T relative to the local envelope is
#
#     atol = 1e-10   9.9e-6      6476 RHS evaluations
#     atol = 1e-13   2.5e-6      7403 RHS evaluations   (+14%)
#     atol = 1e-16   2.5e-6      7829 RHS evaluations   (+21%)
#
# so 1e-13 buys a factor of four for 14% more work and 1e-16 buys nothing further -- because at
# 1e-13 the remaining error is no longer the solver but the super-horizon initial condition
# T = 1, T' = 0, whose own error is 2.5e-6 of the envelope. With exact initial data the same runs
# give 1.2e-5, 3.3e-7 and 1.3e-7, which is where the 1e-13/1e-16 distinction would show. Removing
# that floor means replacing the initial condition with the series T ~ 1 - x^2/10, which is a
# specification decision and is not taken here ([00-tk-superhorizon-ic-series]).
DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13

DEFAULT_QUADRATURE_RTOL = 1e-8
# was 1e-25; tightened per prompts/source-remediation/IMPLEMENTATION_STATE.md
# [12-atol-too-loose-for-the-source-integral] -- at 1e-25, ~58% of production QuadSourceIntegral
# work items had |total|/(1+z_response) below this floor, so their quadrature "converged" before
# doing any work. rtol was confirmed not to bind (1e-8 -> 1e-11 was bit-identical on 159 items);
# atol does. 1e-32 is the exact value the source-remediation campaign measured (log 12) as
# halving the residual against the analytic oracle at an unconverged test point.
DEFAULT_QUADRATURE_ATOL = 1e-32

# value of dtheta/dz where we will switch to a Levin integration strategy
DEFAULT_LEVIN_THRESHOLD = 1.0

DEFAULT_GKSOURCE_NUMERIC_POLICY = "maximize-WKB"
DEFAULT_QUADSOURCE_NUMERIC_POLICY = "maximize-WKB"
