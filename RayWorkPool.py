import time
from datetime import datetime
from typing import Iterable

import ray
from ray import ObjectRef

from Datastore.SQL.ShardedPool import ShardedPool
from utilities import format_time

DEFAULT_CREATE_BATCH_SIZE = 5
DEFAULT_PROCESS_BATCH_SIZE = 1

DEFAULT_MAX_TASK_QUEUE = 2000
DEFAULT_NOTIFY_BATCH_SIZE = 500
DEFAULT_NOTIFY_TIME_INTERVAL = 5 * 60
DEFAULT_NOTIFY_MIN_INTERVAL = 30


def _default_compute_handler(obj, label=None) -> ObjectRef:
    if label is not None:
        return obj.compute(label=label)

    return obj.compute()


def _default_store_handler(obj, pool) -> ObjectRef:
    return pool.object_store(obj)


class RayWorkPool:
    def __init__(
        self,
        pool: ShardedPool,
        task_list,
        task_builder,
        compute_handler=_default_compute_handler,
        store_handler=_default_store_handler,
        available_handler=None,
        validation_handler=None,
        post_handler=None,
        label_builder=None,
        create_batch_size: int = DEFAULT_CREATE_BATCH_SIZE,
        process_batch_size: int = DEFAULT_PROCESS_BATCH_SIZE,
        max_task_queue: int = DEFAULT_MAX_TASK_QUEUE,
        notify_batch_size: int = DEFAULT_NOTIFY_BATCH_SIZE,
        notify_time_interval: int = DEFAULT_NOTIFY_TIME_INTERVAL,
        notify_min_time_interval: int = DEFAULT_NOTIFY_MIN_INTERVAL,
        title: str = None,
        store_results: bool = False,
    ):
        if compute_handler is not None and store_handler is None:
            raise RuntimeError(
                "If a compute maker is supplied, a store maker must also be supplied to serialize the result of the computation"
            )

        if compute_handler is None and store_handler is not None:
            raise RuntimeWarning(
                "No compute maker was supplied, but a store maker was provided. This will have no effect because there will be no compute results to store"
            )

        self._pool = pool

        # bake task_list into an explicit list (for example, so we can count how many items there are!)
        self._todo = [x for x in task_list]

        # we will pop items from the end of the list, so we need to reverse it to get the items
        # in the right order
        self._todo.reverse()

        self._num_total_items = len(self._todo)

        self._task_builder = task_builder
        self._compute_handler = compute_handler
        self._store_handler = store_handler
        self._available_handler = available_handler
        self._validation_handler = validation_handler
        self._post_handler = post_handler

        self._label_builder = label_builder

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
        self._num_available_queue = 0
        self._num_validation_queue = 0

        self._num_lookup_complete = 0
        self._num_compute_complete = 0
        self._num_store_complete = 0
        self._num_available_complete = 0
        self._num_validation_complete = 0

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

                    # pop gets the item from the end of the list
                    item = self._todo.pop()
                    ref_data = self._task_builder(item)

                    if isinstance(ref_data, Iterable):
                        if self._store_results:
                            raise RuntimeError(
                                "store_results=True is not compatible with returning multiple work items from a task builder"
                            )

                        for ref in ref_data:
                            self._inflight[ref.hex] = ref
                            self._data[ref.hex] = ("lookup", None)

                        self._num_lookup_queue += len(ref_data)

                    else:
                        ref = ref_data
                        self._inflight[ref.hex] = ref

                        if self._store_results:
                            self._data[ref.hex] = ("lookup", self._current_idx)
                            self._current_idx += 1
                        else:
                            self._data[ref.hex] = ("lookup", None)

                        self._num_lookup_queue += 1

                    count += 1

            # wait for some work to complete
            done_refs, _ = ray.wait(
                list(self._inflight.values()), num_returns=self._process_batch_size
            )

            for ref in done_refs:
                ref: ObjectRef
                type, payload = self._data[ref.hex]

                if type == "lookup":
                    # payload is an index into the result set
                    # we use this to store the constructed object in the right place.
                    # Later, it will be mutated in-place by the compute/store tasks
                    idx = payload

                    # result of the lookup should be a computable/storable object
                    obj = ray.get(ref)

                    if self._store_results:
                        self.results[idx] = obj

                    # remove the original 'lookup' task from the work queue
                    self._inflight.pop(ref.hex, None)
                    self._data.pop(ref.hex, None)

                    self._num_lookup_queue = max(self._num_lookup_queue - 1, 0)
                    self._num_lookup_complete += 1

                    if obj.available:
                        # check whether an availability handler exists
                        if self._available_handler is not None:
                            available_task: ObjectRef = self._available_handler(obj)

                            # add this task to the work queue
                            self._inflight[available_task.hex] = available_task
                            self._data[available_task.hex] = ("available", (idx, obj))

                            self._num_available_queue += 1

                        # otherwise, this is the exit point for driver flow associated with this task, so apply the
                        # postprocessing task, if one was supplied
                        elif self._post_handler is not None:
                            replacement_obj = self._post_handler(obj)
                            if replacement_obj is not None and self._store_results:
                                self.results[idx] = replacement_obj

                    else:
                        # check whether a compute handler exists
                        if self._compute_handler is not None:
                            # otherwise, schedule a compute tasks
                            label: str = (
                                self._label_builder(obj)
                                if self._label_builder is not None
                                else None
                            )
                            compute_task: ObjectRef = self._compute_handler(obj, label)

                            # add this compute task to the work queue
                            self._inflight[compute_task.hex] = compute_task
                            self._data[compute_task.hex] = ("compute", (idx, obj))

                            self._num_compute_queue += 1

                elif type == "available":
                    # payload is a pair of the target index and the constructed object
                    idx, obj = payload

                    self._inflight.pop(ref.hex, None)
                    self._data.pop(ref.hex, None)

                    self._num_available_queue = max(self._num_available_queue - 1, 0)
                    self._num_available_complete += 1

                    # this is the exit point for driver flow associated with this task, so apply the
                    # postprocessing task, if one was supplied
                    if self._post_handler is not None:
                        replacement_obj = self._post_handler(obj)
                        if replacement_obj is not None and self._store_results:
                            self.results[idx] = replacement_obj

                elif type == "compute":
                    # payload contains an index into the result set and (our local copy of) the object that has finished computation;
                    # we want it to store the result of the computation internally, and then submit a store request to the Datastore service.
                    # the results will then be serialized into the database
                    idx, obj = payload
                    obj.store()

                    # remove the original 'compute' task from the work queue
                    self._inflight.pop(ref.hex, None)
                    self._data.pop(ref.hex, None)

                    self._num_compute_queue = max(self._num_compute_queue - 1, 0)
                    self._num_compute_complete += 1

                    # is a compute handler was supplied, a store handler must have been also
                    store_task: ObjectRef = self._store_handler(obj, self._pool)

                    # add this store task to the work queue
                    self._inflight[store_task.hex] = store_task
                    self._data[store_task.hex] = ("store", payload)

                    self._num_store_queue += 1

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

                    # remove the original 'store' task from the work queue
                    self._inflight.pop(ref.hex, None)
                    self._data.pop(ref.hex, None)

                    self._num_store_queue = max(self._num_store_queue - 1, 0)
                    self._num_store_complete += 1

                    # determine whether this work queue has a validation step
                    if self._validation_handler is not None:
                        validation_task: ObjectRef = self._validation_handler(
                            replacement_obj
                        )

                        self._inflight[validation_task.hex] = validation_task
                        self._data[validation_task.hex] = (
                            "validate",
                            (idx, replacement_obj),
                        )

                        self._num_validation_queue += 1

                    # otherwise, this is the exit point for driver flow associated with this task, so apply the
                    # postprocessing task, if one was supplied
                    elif self._post_handler is not None:
                        replacement_obj = self._post_handler(obj)
                        if replacement_obj is not None and self._store_results:
                            self.results[idx] = replacement_obj

                elif type == "validate":
                    # payload contains an index into the result set and (still, our local copy of)
                    # the object that has been freshly validated
                    idx, obj = payload

                    result = ray.get(ref)
                    if result is not True:
                        print(
                            f"!! WARNING: {type(obj)} object with store_id={obj.store_id} did not validate after being emplaced in the datastore"
                        )

                    self._inflight.pop(ref.hex, None)
                    self._data.pop(ref.hex, None)

                    self._num_validation_queue = max(self._num_validation_queue - 1, 0)
                    self._num_validation_complete += 1

                    # this is the exit point for driver flow associated with this task, so apply the
                    # postprocessing task, if one was supplied
                    if self._post_handler is not None:
                        replacement_obj = self._post_handler(obj)
                        if replacement_obj is not None and self._store_results:
                            self.results[idx] = replacement_obj

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

                    now = datetime.now()
                    msg = f"   -- {now:%Y-%m-%d %H:%M:%S%z} ({format_time(total_elapsed)} running): {len(self._todo)} work items remaining = {percent_complete:.2f}% complete"
                    if percent_complete > 99.99:
                        msg += " (may be waiting for compute/store/validate tasks to finish)"
                    print(msg)

                    msg = f"      inflight items: {self._num_lookup_queue} lookup"
                    if self._compute_handler is not None:
                        msg += f", {self._num_compute_queue} compute"
                    if self._store_handler is not None:
                        msg += f", {self._num_store_queue} store"
                    if self._available_handler is not None:
                        msg += f", {self._num_available_queue} available"
                    if self._validation_handler is not None:
                        msg += f", {self._num_validation_queue} validation"
                    if self._post_handler is not None:
                        msg += f", {self._num_post_queue} post"

                    msg += f" | completed items: {self._num_lookup_complete} lookup"
                    if self._compute_handler is not None:
                        msg += f", {self._num_compute_complete} compute"
                    if self._store_handler is not None:
                        msg += f", {self._num_store_complete} store"
                    if self._available_handler is not None:
                        msg += f", {self._num_available_complete} available"
                    if self._validation_handler is not None:
                        msg += f", {self._num_validation_complete} validation"
                    if self._post_handler is not None:
                        msg += f", {self._num_post_complete} post"

                    print(msg)

                    self._batch = 0
                    self._last_notify_time = time.perf_counter()

        if self._title is not None:
            self.total_time = time.perf_counter() - self._start_time
            print(
                f"   -- all work items complete in time {format_time(self.total_time)}"
            )
