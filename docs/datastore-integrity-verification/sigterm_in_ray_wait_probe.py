# What does a SIGTERM delivered while the driver is blocked in ray.wait become?
import os, signal, threading, time, sys, traceback
import ray

ray.init(
    num_cpus=1, include_dashboard=False
)  # installs ray's sigterm_handler: sys.exit(signum)
print("handler:", signal.getsignal(signal.SIGTERM), flush=True)


@ray.remote
def slow():
    time.sleep(30)
    return 1


ref = slow.remote()
threading.Timer(3.0, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
try:
    ray.wait([ref], timeout=20)
    print("ray.wait returned normally", flush=True)
except BaseException as e:
    print(
        f"caught {type(e).__module__}.{type(e).__name__}: code={getattr(e, 'code', None)!r} args={e.args!r}",
        flush=True,
    )
