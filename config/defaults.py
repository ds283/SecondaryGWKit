DEFAULT_STRING_LENGTH = 256
DEFAULT_FLOAT_PRECISION = 1e-7
DEFAULT_REDSHIFT_RELATIVE_PRECISION = 1e-7

DEFAULT_ABS_TOLERANCE = 1e-10
DEFAULT_REL_TOLERANCE = 1e-8

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
