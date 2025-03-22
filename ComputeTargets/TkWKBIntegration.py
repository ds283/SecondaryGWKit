THETA_INDEX = 0
Q_INDEX = 0
EXPECTED_SOL_LENGTH = 1

# how large do we allow the WKB phase theta to become, before we terminate the integration and
# reset to a small value?
# we need to resolve the phase on the scale of (0, 2pi), otherwise we will compute cos(theta),
# sin(theta) and hence the transfer function incorrectly
DEFAULT_PHASE_RUN_LENGTH = 1e4

# how large do we allow omega_WKB_sq to get before switching to a "stage #2" integration?
DEFAULT_OMEGA_WKB_SQ_MAX = 1e6
