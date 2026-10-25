import os
import sys

# The temp dir every Ray instance this repository starts with `address="local"` is given
# (`ray.init(..., _temp_dir=RAY_TEMP_DIR)`): main.py, the six extract_*_data.py readers and the
# drivers under RunDrivers/. The shared rehearsal lock, `raypoolkit.rehearsal_lock`, finds an
# instance by its temp dir, so an instance started without it is invisible to the lock. The lock
# compares the string as given, so it is spelled exactly so everywhere, and on the lock's command
# line as `--ray-temp-dir /tmp/ray-sgk`. It is short because Ray's Unix sockets live under it.
RAY_TEMP_DIR = "/tmp/ray-sgk"

# Ray's token authentication, named rather than left to Ray's default. Since Ray 2.59 a new local
# instance runs in token mode when RAY_AUTH_MODE is unset, and every log then carries a warning
# saying so; with the mode set explicitly Ray logs nothing. Set here, at import, because every Ray
# start imports this module before it calls ray.init (RunDrivers/tests/test_ray_auth_mode.py), and
# Ray's own processes inherit it. An attach to an explicit host:port sends the token only in token
# mode. The token is ~/.ray/auth_token, which Ray generates if it is missing; keep it mode 600. A
# value already in the environment (for instance `disabled`) is left as it is.
#
# Ray reads the mode into its config once, when `ray` is imported, and main.py and the drivers
# import `ray` first. Without a refresh the driver keeps `disabled` while the GCS server it starts
# inherits `token` and refuses it (py313-venv 02, R7's first launch). Config.initialize("") is the
# refresh Ray's own _enable_token_auth makes when it turns the mode on itself.
RAY_AUTH_MODE = "token"
if "RAY_AUTH_MODE" not in os.environ:
    os.environ["RAY_AUTH_MODE"] = RAY_AUTH_MODE
    if "ray._raylet" in sys.modules:
        sys.modules["ray._raylet"].Config.initialize("")
