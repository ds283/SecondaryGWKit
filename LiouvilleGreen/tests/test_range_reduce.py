import unittest

from LiouvilleGreen.range_reduce_mod_2pi import range_reduce_mod_2pi


class TestRangeReduce(unittest.TestCase):

    def test_range_reduce(self):
        div_2pi, mod_2pi = range_reduce_mod_2pi(6.2948, 1.0)
