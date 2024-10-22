import time
from itertools import zip_longest
from traceback import print_tb


class WallclockTimer:
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time

        if exc_type is not None:
            print(f"type={exc_type}, value={exc_val}")
            print_tb(exc_tb)


SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 60 * SECONDS_PER_MINUTE
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR


def format_time(interval: float) -> str:
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


# grouper borrowed from itertools recipes
# https://docs.python.org/3/library/itertools.html#itertools-recipes
def grouper(iterable, n, *, incomplete="fill", fillvalue=None):
    "Collect data into non-overlapping fixed-length chunks or blocks."
    # grouper('ABCDEFG', 3, fillvalue='x') → ABC DEF Gxx
    # grouper('ABCDEFG', 3, incomplete='strict') → ABC DEF ValueError
    # grouper('ABCDEFG', 3, incomplete='ignore') → ABC DEF
    iterators = [iter(iterable)] * n
    match incomplete:
        case "fill":
            return zip_longest(*iterators, fillvalue=fillvalue)
        case "strict":
            return zip(*iterators, strict=True)
        case "ignore":
            return zip(*iterators)
        case _:
            raise ValueError("Expected fill, strict, or ignore")
