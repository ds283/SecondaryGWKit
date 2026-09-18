DEFAULT_STRING_LENGTH = 256
DEFAULT_FLOAT_PRECISION = 1e-7
DEFAULT_REDSHIFT_RELATIVE_PRECISION = 1e-7

# ---------------------------------------------------------------------------------------------
# THE SHARED PAIR, WHICH IS NO LONGER A TOLERANCE OF ANY COMPUTE TARGET
#
# These two were, until prompt 05a of prompts/tolerance-convergence, the accuracy parameter of
# six of the eight keyed object types at once. They are now the accuracy parameter of none of
# them: prompt 05 removed the pair from the four targets that never used it (their knob is an
# integer Gauss order), and prompt 05a below gave each of the three that did use it a constant of
# its own, measured on its own terms.
#
# What is left of DEFAULT_ABS_TOLERANCE is seven bare `fabs(a - b) < ...` float comparisons --
# ComputeTargets/GkSource.py:96, :104, :275, Quadrature/integrators/numeric_with_phase_cut.py:618,
# :737, :790 and LiouvilleGreen/WKBtools.py:83 -- plus the signature defaults of
# numeric_with_phase_cut.integrate_numeric_with_phase_cut, which production always overrides.
# That is a *different quantity* wearing a tolerance's name, and it is
# [02-shared-atol-doubles-as-a-float-comparison-epsilon], which is open and is deliberately not
# closed by moving these values. Do not retune either of them as though it were a solver
# tolerance: whatever is decided about the float-comparison epsilon, it is a decision about
# comparing two redshifts, not about integrating an ODE.
DEFAULT_ABS_TOLERANCE = 1e-10
DEFAULT_REL_TOLERANCE = 1e-8

# ---------------------------------------------------------------------------------------------
# wavenumber_exit_time: the root solve that locates horizon crossing
#
# These reach scipy's `root_scalar` (Brent) as `xtol` and `rtol`, solving
# log(k(1+z)/H) - offset = 0 in u = log(1+z) (CosmologyConcepts/wavenumber.py:_solve_horizon_exit).
# One object per wavenumber, 50 per model. Settled by the user on 2026-09-17 (README section 7 D1
# of prompts/tolerance-convergence, closed); measured by prompt 03a,
# docs/tolerance-convergence/TK-NUMERIC-AND-EXIT-TIME.md section 7.6, over three models, all fifty
# production wavenumbers and the three offsets a grid is built from (0, -5, +4), through
# `_solve_horizon_exit` and never through the datastore, on the version-2 source grid at each
# cosmology's own anchor. Reference (1e-300, 1e-14), worst drift 4.97e-14 / 7.82e-14 in u and
# 3.55e-15 from the exact 1 + z = k/(H0 e^N) on the radiation control.
#
# DEFAULT_HEXIT_REL_TOLERANCE is the *chosen* half. Brent stops at `xtol + rtol*|u|`, so at the
# largest production |u| = 38.04 an rtol of 1e-9 guarantees a displacement of 3.81e-8, which is
# the loosest setting whose guarantee clears the only floor in the tree: the
# DEFAULT_REDSHIFT_RELATIVE_PRECISION = 1e-7 at which Datastore/SQL/ObjectFactories/redshift.py
# reuses an existing redshift row, and so the displacement a grid can absorb without moving. The
# previous shared 1e-8 guarantees 3.8e-7, which is 3.8x *over* that floor. Cost: 7,009 Hubble
# evaluations against 6,963, for all 150 (k, offset) solves on all three models -- +0.7%.
#
# Two things must travel with the number. (i) The recommendation was accepted on the *guarantee*
# reading of README section 6.1: the *achieved* displacement at the old setting is 7.86e-8, which
# already clears 1e-7 by 1.3x, and on that reading the answer would have been `unchanged` (log 03a,
# deviation 4). (ii) It buys nothing on the source grid as built today: the production anchor
# _solve_horizon_exit(cosmology, 3e8/Mpc, -5) is bit-identical at (1e-10, 1e-8) and (1e-10, 1e-9)
# on both production cosmologies, so all three published grid digests are unmoved (log 05a).
DEFAULT_HEXIT_REL_TOLERANCE = 1e-9

# The absolute half is **inert, unchosen and coupled**, and README section 1.2's closing rule
# applies: the provenance of this value cannot be established from the record -- it is
# DEFAULT_ABS_TOLERANCE's 1e-10, inherited because the target shared that constant, and nothing
# says why. Measured by prompt 03a it binds at 0 of 150 (k, offset) pairs on each of the three
# models, at the old rtol and at the new one alike. What makes it matter anyway is the coupling:
# because Brent stops at `xtol + rtol*|u|`, an xtol of 1e-10 floors the pair at ~1e-10 relative
# however far rtol is tightened, taking over below rtol ~ 2.6e-12. Any future design tolerance
# below 1e-10 requires this constant to move with its partner.
DEFAULT_HEXIT_ABS_TOLERANCE = 1e-10

# ---------------------------------------------------------------------------------------------
# GkNumericIntegration: the numeric region of the tensor Green's function (DOP853)
#
# One object per (k, z_source): 29,290 / 38,105 / 58,350 per model on the version-2 source grid,
# the sector that carries the pipeline's compute cost. Settled by the user on 2026-09-17 as
# **unchanged** in value (README section 6.1 rule 4); both constants are new *names* for what the
# shared pair already supplied. Measured by prompt 03,
# docs/tolerance-convergence/GK-NUMERIC-SWEEP.md, over 50 k x 3 models x 15 (atol, rtol) cells on
# the version-2 grid at each cosmology's own anchor under BREAK_POINT_DISCONTINUITY, reference
# (1e-18, 1e-12) converged 50/50 on all three models.
#
# DEFAULT_GK_NUMERIC_REL_TOLERANCE is the *chosen* half, and what chose it is a floor that
# dominates rather than a target that is met. rtol is the whole lever here -- four decades of it
# move the maximum envelope-relative error by x13,300 -- and at 1e-8 that error is 2.6e-07. But
# the consumer's own cubic spline of the numeric G (ComputeTargets/GkSourcePolicyData.py:325-336)
# carries 1.6e-04 to 9.4e-03 of the envelope near and below the hand-over, which is **x631 to
# x37,700** the solver's error, so tightening rtol buys nothing that survives the spline
# (GK-NUMERIC-SWEEP.md sections 5.2 and 7). One decade would cost +23-25% of the evaluations of the
# largest sector in the pipeline for no change in what is delivered. The spline floor is not a
# tolerance question and has left this campaign:
# [03-numeric-g-consumer-spline-is-the-dominant-error-near-the-hand-over].
DEFAULT_GK_NUMERIC_REL_TOLERANCE = 1e-8

# The absolute half is **inert and unchosen** (README section 1.2's closing rule again): |G| is
# 2.1e+12 to 2.9e+18 in Mpc_units, so an absolute floor of 1e-10 cannot bind, and prompt 03
# measured four decades of it moving the maximum by at most 1.2%, with the corners of the matrix
# agreeing to three significant figures. It is DEFAULT_ABS_TOLERANCE's value, inherited, and the
# record does not say who chose 1e-10 or for what.
DEFAULT_GK_NUMERIC_ABS_TOLERANCE = 1e-10

# ---------------------------------------------------------------------------------------------
# TkNumericIntegration: the numeric region of the matter transfer function (DOP853)
#
# One object per wavenumber, 50 per model.
#
# DEFAULT_TK_NUMERIC_REL_TOLERANCE is a **change**, from the shared 1e-8, settled by the user on
# 2026-09-17. Measured by prompt 03a of prompts/tolerance-convergence,
# docs/tolerance-convergence/TK-NUMERIC-AND-EXIT-TIME.md sections 4.1 and 5: three models, all
# fifty production wavenumbers, **source-grid generation version 2** at each cosmology's own
# anchor, under this sector's own BREAK_POINT_ALL policy, envelope-relative error, reference
# (1e-18, 1e-12) converged 50/50 with worst drift 4.26e-11 / 5.23e-11 / 4.65e-09. Swept loose to
# tight over nine settings, 3e-11 is the *loosest* whose maximum over all three models -- 3.88e-08
# -- is at or below the floor (README section 6.1 rules 2 and 3).
#
# The floor it competes against is the T = 1, T' = 0 super-horizon initial condition, re-measured
# there at 2.39e-06 to 2.64e-06 of the envelope against the inherited 2.52e-06. The recommended
# setting sits 62x under it; the previous 1e-8 sits 141x *over* it at its worst wavenumber, leaving
# 14 of 150 runs above the 3e-6 target with a worst of 3.36e-04.
#
# Cost, at the setting and one step either side, over the whole sector on all three models (50
# objects per model), in right-hand-side evaluations: 1e-10 1,727,040 (+35.7%), **3e-11 1,774,977
# (+39.4%)**, 1e-11 1,810,617 (+42.2%), against 1,272,891 at the old 1e-8.
#
# **This is a measured setting, not a bound**, and the acceptance did not close the caveat: the
# maximum is *not monotone* in rtol. From 1e-9 down, exactly one of the 150 runs exceeds 3e-6 at
# each of 1e-9, 3e-10 and 1e-10, and it is a different run each time. 3e-11 is the loosest setting
# that clears the floor *in that sweep*; it is not a setting at which the excursion is shown to be
# impossible ([03a-tk-numeric-excursion-is-sporadic-in-rtol], open).
DEFAULT_TK_NUMERIC_REL_TOLERANCE = 3e-11

# absolute tolerance for the *numeric* part of the matter transfer function alone; see
# prompts/GkTk-remedial (prompt 12) and review section 12.5 of
# docs/gk-wkb-review-fable-2026-09-09.md. Confirmed by the user 2026-09-12 and not reopened.
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
#
# **Re-characterised by prompt 03a of prompts/tolerance-convergence (2026-09-17), value unchanged.**
# In this sector atol is *not* inert in the sense DEFAULT_GK_NUMERIC_ABS_TOLERANCE is. Across
# 1e-12 -> 1e-14 at rtol = 1e-8 it moves the median of the per-k maxima by at most 2.1x but the
# maximum by up to **205x**, by changing which wavenumber draws a bad step sequence. It is a
# step-selection knob, not the knob that sets the level: no setting of it removes the excursion,
# and rtol is what does (see DEFAULT_TK_NUMERIC_REL_TOLERANCE above).
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
