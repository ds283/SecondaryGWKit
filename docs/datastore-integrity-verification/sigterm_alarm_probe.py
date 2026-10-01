# After a SIGTERM caught in ray.wait, does the driver survive more than 3 s of cleanup?
import os, signal, threading, time
import ray

ray.init(num_cpus=1, include_dashboard=False)


@ray.remote
def slow():
    time.sleep(60)


ref = slow.remote()
threading.Timer(3.0, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
try:
    ray.wait([ref], timeout=30)
except BaseException as e:
    t0 = time.monotonic()
    print(
        f"caught {type(e).__name__} code={getattr(e,'code',None)!r}; now sleeping 8 s in cleanup",
        flush=True,
    )
    for i in range(16):
        time.sleep(0.5)
        print(f"  alive at {time.monotonic()-t0:.1f} s", flush=True)
    print("survived 8 s of cleanup", flush=True)
