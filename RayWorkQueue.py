import time
from typing import Iterable

import ray
from ray import ObjectRef
from ray.actor import ActorHandle

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 60 * SECONDS_PER_MINUTE
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR


def _format_time(interval: float) -> str:
    int_interval = int(interval)
    str = ""

    if int_interval > SECONDS_PER_DAY:
        days = int_interval // SECONDS_PER_DAY
        int_interval = int_interval - days * SECONDS_PER_DAY
        interval = interval - days * SECONDS_PER_DAY
        if len(str) > 0:
            str = str + f" {days}d"
        else:
            str = f"{days}d"

    if int_interval > SECONDS_PER_HOUR:
        hours = int_interval // SECONDS_PER_HOUR
        int_interval = int_interval - hours * SECONDS_PER_HOUR
        interval = interval - hours * SECONDS_PER_HOUR
        if len(str) > 0:
            str = str + f" {hours}h"
        else:
            str = f"{hours}h"

    if int_interval > SECONDS_PER_MINUTE:
        minutes = int_interval // SECONDS_PER_MINUTE
        int_interval = int_interval - minutes * SECONDS_PER_MINUTE
        interval = interval - minutes * SECONDS_PER_MINUTE
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
        validation_maker=None,
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
        self._validation_maker = validation_maker

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
        self._num_validation_queue = 0

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

                    # remove the original 'lookup' task from the work queue
                    self._inflight.pop(ref.hex, None)
                    self._data.pop(ref.hex, None)

                    self._num_lookup_queue = max(self._num_lookup_queue - 1, 0)

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
                    self._data[compute_task.hex] = ("compute", (payload, obj))

                    self._num_compute_queue += 1

                elif type == "compute":
                    # payload contains an index into the result set and (our local copy of) the object that has finished computation;
                    # we want it to store the result of the computation internally, and then submit a store request to the Datastore service.
                    # the results will then be serialized into the database
                    idx, obj = payload
                    obj.store()

                    store_task: ObjectRef = self._store.object_store.remote(obj)

                    # add this store task to the work queue
                    self._inflight[store_task.hex] = store_task
                    self._data[store_task.hex] = ("store", payload)

                    # remove the original 'compute' task from the work queue
                    self._inflight.pop(ref.hex, None)
                    self._data.pop(ref.hex, None)

                    self._num_store_queue += 1
                    self._num_compute_queue = max(self._num_compute_queue - 1, 0)

                elif type == "store":
                    # payload contains an index into the result set and (again, our local copy of) the object that has been freshly
                    # serialized into the datastore. We should not expect it to have its store_id available.
                    # The result of the store operation is a mutated object that has this field (and corresponding
                    # fields for any objects it contains) set, so we should replace our local copy with it
                    idx, obj = payload
                    replacement_obj = ray.get(ref)

                    # replacement object should now satisfy the available attribute
                    if not replacement_obj.available:
                        raise RuntimeError(
                            "Object returned from store service does not satisfy the available property"
                        )

                    if self._store_results:
                        self.results[idx] = replacement_obj

                    # determine whether this work queue has a validation step
                    if self._validation_maker is not None:
                        validation_task: ObjectRef = self._validation_maker(
                            replacement_obj
                        )

                        self._inflight[validation_task.hex] = validation_task
                        self._data[validation_task.hex] = (
                            "validate",
                            (idx, replacement_obj),
                        )

                        self._num_validation_queue += 1

                    # remove the original 'store' task from the work queue
                    self._inflight.pop(ref.hex, None)
                    self._data.pop(ref.hex, None)

                    self._num_store_queue = max(self._num_store_queue - 1, 0)

                elif type == "validate":
                    # payload contains an index into the result set and (still, our local copy of) the object that has been freshly
                    # validated
                    # At this stage, however, there is not much to do, except to forget the validation task and update the queue length
                    idx, obj = payload

                    result = ray.get(ref)
                    if result is not True:
                        print(
                            f"!! WARNING: {type(obj)} object with store_id={obj.store_id} did not validate after being emplaced in the datastore"
                        )

                    self._inflight.pop(ref.hex, None)
                    self._data.pop(ref.hex, None)

                    self._num_validation_queue = max(self._num_validation_queue - 1, 0)

                else:
                    raise RuntimeError(f'Unexpected work queue item type "{type}"')

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
                            / float(self._num_total_items)
                        )
                    print(
                        f"-- {_format_time(total_elapsed)}: {len(self._todo)} work items remaining = {percent_complete:.2f}% complete"
                    )
                    if self._validation_maker is not None:
                        print(
                            f"   inflight items: {self._num_lookup_queue} lookup, {self._num_compute_queue} compute, {self._num_store_queue} store, {self._num_validation_queue} validation"
                        )
                    else:
                        print(
                            f"   inflight items: {self._num_lookup_queue} lookup, {self._num_compute_queue} compute, {self._num_store_queue} store"
                        )

                    self._batch = 0
                    self._last_notify_time = time.perf_counter()

        if self._title is not None:
            self.total_time = time.perf_counter() - self._start_time
            print(f"-- all work items complete in time {_format_time(self.total_time)}")
