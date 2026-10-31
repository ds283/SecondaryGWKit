"""levin-phase-floor prompt 06e section 5, Z3: start run_short.sh detached (nohup sh -c, a new
session, stdin from /dev/null, output appended to out/short_driver.log), record the sh pid in
out/short_launch.txt, and exit without waiting. Unregistered (the brief: the short references run
in-session). New in 06e. Usage (from references/): python -B launch_short.py 02 08 10 13 14 22 23 27"""
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
samples = [f"{int(x):02d}" for x in sys.argv[1:]]
assert samples
cmd = " ".join(shlex.quote(a) for a in ["/bin/bash", str(HERE / "run_short.sh"), *samples]) + "; exit $?"
log = open(HERE / "out" / "short_driver.log", "ab")
child = subprocess.Popen(["nohup", "sh", "-c", cmd], cwd=str(HERE), stdin=subprocess.DEVNULL, stdout=log,
                         stderr=log, start_new_session=True, close_fds=True)
log.close()
line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} sh pid {child.pid} command {cmd} launcher pid {os.getpid()}"
with open(HERE / "out" / "short_launch.txt", "a") as f:
    f.write(line + "\n")
print(line)
