import time
from math import fmod
from typing import Iterable

import ray
from ray import ObjectRef
from ray.actor import ActorHandle

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 60 * SECONDS_PER_MINUTE
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR


def _format_time(interval: float) -> str:
    interval = float(interval)
    str = ""

    if interval > SECONDS_PER_DAY:
        days = int(fmod(interval, SECONDS_PER_DAY))
        interval = interval - float(days * SECONDS_PER_DAY)
        if len(str) > 0:
            str = str + f" {days}d"
        else:
            str = f"{days}d"

    if interval > SECONDS_PER_HOUR:
        hours = int(fmod(interval, SECONDS_PER_HOUR))
        interval = interval - float(hours * SECONDS_PER_HOUR)
        if len(str) > 0:
            str = str + f" {hours}h"
        else:
            str = f"{hours}h"

    if interval > SECONDS_PER_MINUTE:
        minutes = int(fmod(interval, SECONDS_PER_MINUTE))
        interval = interval - float(minutes * SECONDS_PER_MINUTE)
        if len(str) > 0:
            str = str + f" {minutes}m"
        else:
            str = f"{minutes}m"

    if len(str) > 0:
        str = str + f" {interval:.3g}s"
    else:
        str = f"{interval:.3g}s"

    return str


class RayWorkQueue:
    def __init__(
        self,
        store: ActorHandle,
        task_list,
        task_maker,
        label_maker=None,
        create_batch_size: int = 5,
        process_batch_size: int = 1,
        max_task_queue: int = 1000,
        notify_batch_size: int = 50,
        notify_time_interval: int = 180,
        notify_min_time_interval: int = 5,
        title: str = None,
        store_results: bool = False,
    ):
        self._store = store

        self._todo = [x for x in task_list]
        self._num_total_items = len(task_list)
        self._task_maker = task_maker
        self._label_maker = label_maker

        self._create_batch_size = create_batch_size
        self._process_batch_size = process_batch_size
        self._max_task_queue = max_task_queue
        self._notify_batch_size = notify_batch_size
        self._notify_time_interval = notify_time_interval
        self._notify_min_time_interval = notify_min_time_interval

        self._inflight = {}
        self._data = {}

        self._store_results = store_results
        if store_results:
            self.results = [None for _ in range(len(task_list))]
            self._current_idx = 0

        self._num_lookup_queue = 0
        self._num_compute_queue = 0
        self._num_store_queue = 0

        self._batch = 0

        self._start_time = time.perf_counter()
        self._last_notify_time = self._start_time

        self._title = title
        if title is not None:
            print(f"\n** {title}")

    def run(self):
        while len(self._inflight) > 0 or len(self._todo) > 0:
            # if there is space in the task queue, and there are items remaining to queue,
            # then initiate new work
            # either we create a fixed batch size, or we enqueue work until the task queue is exhasuted
            if len(self._inflight) < self._max_task_queue and len(self._todo) > 0:
                count = 0
                while count < self._create_batch_size and len(self._todo) > 0:
                    # consume more tasks from the task queue and schedule their work

                    item = self._todo.pop(0)
                    ref_data = self._task_maker(item)

                    if isinstance(ref_data, Iterable):
                        if self._store_results:
                            raise RuntimeError(
                                "store_results=True is not compatible with returning multiple work items from a task maker"
                            )

                        for ref in ref_data:
                            self._inflight[ref.hex] = ref
                            self._data[ref.hex] = ("lookup", None)

                    else:
                        ref = ref_data
                        self._inflight[ref.hex] = ref

                        if self._store_results:
                            self._data[ref.hex] = ("lookup", self._current_idx)
                            self._current_idx += 1
                        else:
                            self._data[ref.hex] = ("lookup", None)

                    count += 1

            # wait for some work to complete
            done_refs, _ = ray.wait(
                list(self._inflight.values()), num_returns=self._process_batch_size
            )

            for ref in done_refs:
                ref: ObjectRef
                type, payload = self._data[ref.hex]

                if type == "lookup":
                    # result of the lookup should be a computable/storable object
                    obj = ray.get(ref)

                    if self._store_results:
                        # payload is an index into the result set
                        # we use this to store the constructed object in the right place.
                        # Later, it will be mutated in-place by the compute/store tasks
                        self.results[payload] = obj

                    if obj.available:
                        # nothing to do, object is already constructed
                        continue

                    # otherwise, schedule a compute tasks
                    compute_task: ObjectRef = (
                        obj.compute(self._label_maker(obj))
                        if self._label_maker is not None
                        else obj.compute()
                    )

                    # add this compute task to the work queue
                    self._inflight[compute_task.hex] = compute_task
                    self._data[compute_task.hex] = ("compute", obj)

                    # remove the original 'lookup' task from the work queue
                    self._inflight.pop(ref.hex, None)
                    self._data.pop(ref.hex, None)

                    self._num_compute_queue += 1
                    self._num_lookup_queue = max(self._num_lookup_queue - 1, 0)

                elif type == "compute":
                    # payload is the object that has finished computation; we want it to store the result
                    # of the computation internally, and then ubmit a store request to the Datastore service.
                    # the results will then be serialized into the database
                    payload.store()

                    store_task: ObjectRef = self._store.object_store.remote(payload)

                    # add this store task to the work queue
                    self._inflight[store_task.hex] = store_task
                    self._data[store_task.hex] = ("store", None)

                    # remove the original 'compute' task from the work queue
                    self._inflight.pop(ref.hex, None)
                    self._data.pop(ref.hex, None)

                    self._num_store_queue += 1
                    self._num_compute_queue = max(self._num_compute_queue - 1, 0)

                elif type == "store":
                    # nothing requires doing here; just remove the store task from the work queue
                    self._inflight.pop(ref.hex, None)
                    self._data.pop(ref.hex, None)

                    self._num_store_queue = max(self._num_store_queue - 1, 0)

                else:
                    raise RuntimeError(f'Unexpeccted work queue item type "{type}"')

                self._batch += 1

            now_time = time.perf_counter()
            elapsed = now_time - self._last_notify_time
            if elapsed > self._notify_min_time_interval:
                if (
                    elapsed > self._notify_time_interval
                    or self._batch > self._notify_batch_size
                ):
                    total_elapsed = now_time - self._start_time
                    num_items_remain = len(self._todo)
                    if num_items_remain == 0:
                        percent_complete = 100.0
                    else:
                        percent_complete = (
                            100.0
                            * float(self._num_total_items - num_items_remain)
                            / float(num_items_remain)
                        )
                    print(
                        f"-- {_format_time(total_elapsed)}: {len(self._todo)} work items remaining = {percent_complete:.2f}% complete"
                    )
                    print(
                        f"   inflight details: {self._num_lookup_queue} lookup, {self._num_compute_queue} compute, {self._num_store_queue} store"
                    )

                    self._batch = 0
                    self._last_notify_time = time.perf_counter()

        if self._title is not None:
            self.total_time = time.perf_counter() - self._start_time
            print(f"-- all work items complete in time {_format_time(self.total_time)}")
