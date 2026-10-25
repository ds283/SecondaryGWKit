# The temp dir every Ray instance this repository starts with `address="local"` is given
# (`ray.init(..., _temp_dir=RAY_TEMP_DIR)`): main.py, the six extract_*_data.py readers and the
# drivers under RunDrivers/. The shared rehearsal lock, `raypoolkit.rehearsal_lock`, finds an
# instance by its temp dir, so an instance started without it is invisible to the lock. The lock
# compares the string as given, so it is spelled exactly so everywhere, and on the lock's command
# line as `--ray-temp-dir /tmp/ray-sgk`. It is short because Ray's Unix sockets live under it.
RAY_TEMP_DIR = "/tmp/ray-sgk"
