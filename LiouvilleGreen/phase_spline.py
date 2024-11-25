from math import pi, log, exp, fmod
from typing import Iterable, Tuple

from scipy.interpolate import InterpolatedUnivariateSpline

DEFAULT_CHUNK_SIZE = 200
MINIMUM_SPLINE_DATA_POINTS = 10

_twopi = 2.0 * pi


class _chunk_spline:
    """
    Contains the spline for a single chunk
    """

    def __init__(
        self,
        div_2pi_base: int,
        data: Iterable[dict[str, float]],
        x_is_log: bool = False,
    ):
        self._div_2pi_base = div_2pi_base

        # cache supplied values
        self._points = data
        self._x_sample = [p["x"] for p in data]

        # build set of y values, but rebased mod 2pi so that all values are close to zero
        # in this way, we hope to avoid loss of precision when the result of the spline is evaluated mod 2pi, as it typically
        # will be
        self._y_points = [
            (p["div_2pi"] - div_2pi_base) * _twopi + p["mod_2pi"] for p in data
        ]

        if x_is_log:
            self._log_x_points = self._x_sample
        else:
            self._log_x_points = [log(x) for x in self._x_sample]

        # find min and max values of x, so that we avoid extrapolation if we are asked to evaluate at an x position that is out-of-bounds
        self.min_log_x = min(self._log_x_points)
        self.max_log_x = max(self._log_x_points)

        self.min_x = exp(self.min_log_x)
        self.max_x = exp(self.max_log_x)

        # prefer to be 5% away from the spline boundaries so that we are free of edge-effects
        self._min_safe_log_x = 1.05 * self.min_log_x
        self._max_safe_log_x = 0.95 * self.max_log_x

        self._min_safe_x = exp(self._min_safe_log_x)
        self._max_safe_x = exp(self._max_safe_log_x)

        self._spline = InterpolatedUnivariateSpline(self._log_x_points, self._y_points)
        self._derivative = self._spline.derivative

    def _theta(self, raw_x: float, log_x: float, warn_unsafe: bool = True) -> float:
        if log_x < self.min_log_x:
            raise RuntimeError(
                f"chunk_spline: spline evaluated out-of-bounds of lower limit at log_x={log_x:.5g} (raw x={raw_x:.5g}) | Minimum allowed value is log_x={self._min_safe_log_x:.5g} (raw x={self._min_safe_x:.5g}) | Recommended safe minimum is log_x={self._min_safe_log_x:.5g} (raw x={self._min_safe_x:.5g})"
            )

        if log_x > self.max_log_x:
            raise RuntimeError(
                f"chunk_spline: spline evaluated out-of-bounds of upper limit at log_x={log_x:.5g} (raw x={raw_x:.5g}) | Maximum allowed value is log_x={self._max_safe_log_x:.5g} (raw x={self._max_safe_x:.5g}) | Recommended safe maximum is log_x={self._max_safe_log_x:.5g} (raw x={self._max_safe_x:.5g})"
            )

        if warn_unsafe:
            if log_x < self._min_safe_log_x:
                print(
                    f"## WARNING (chunk_spline): chunk spline evaluated within 5% of lower limit at log_x={log_x:.5g} (raw={raw_x:.5g}) | Recommended safe minimum is log_x={self._min_safe_log_x:.5g} (raw x={self._min_safe_x:.5g})"
                )

            if log_x > self._max_safe_log_x:
                print(
                    f"## WARNING (chunk_spline): chunk spline evaluated within 5% of upper limit at log_x={log_x:.5g} (raw={raw_x:.5g}) | Recommended safe maximum is log_x={self._max_safe_log_x:.5g} (raw x={self._max_safe_x:.5g})"
                )

        return self._spline(log_x)

    def _theta_deriv(
        self, raw_x: float, log_x: float, warn_unsafe: bool = True
    ) -> float:
        if log_x < self.min_log_x:
            raise RuntimeError(
                f"chunk_spline: spline evaluated out-of-bounds of lower limit at log_x={log_x:.5g} (raw x={raw_x:.5g}) | Minimum allowed value is log_x={self._min_safe_log_x:.5g} (raw x={self._min_safe_x:.5g}) | Recommended safe minimum is log_x={self._min_safe_log_x:.5g} (raw x={self._min_safe_x:.5g})"
            )

        if log_x > self.max_log_x:
            raise RuntimeError(
                f"chunk_spline: spline evaluated out-of-bounds of upper limit at log_x={log_x:.5g} (raw x={raw_x:.5g}) | Maximum allowed value is log_x={self._max_safe_log_x:.5g} (raw x={self._max_safe_x:.5g}) | Recommended safe maximum is log_x={self._max_safe_log_x:.5g} (raw x={self._max_safe_x:.5g})"
            )

        if warn_unsafe:
            if log_x < self._min_safe_log_x:
                print(
                    f"## WARNING (chunk_spline): chunk spline evaluated within 5% of lower limit at log_x={log_x:.5g} (raw={raw_x:.5g}) | Recommended safe minimum is log_x={self._min_safe_log_x:.5g} (raw x={self._min_safe_x:.5g})"
                )

            if log_x > self._max_safe_log_x:
                print(
                    f"## WARNING (chunk_spline): chunk spline evaluated within 5% of upper limit at log_x={log_x:.5g} (raw={raw_x:.5g}) | Recommended safe maximum is log_x={self._max_safe_log_x:.5g} (raw x={self._max_safe_x:.5g})"
                )

        return self._derivative(log_x)

    def _get_x(
        self, x: float, raw_x: float, log_x: float, x_is_log: bool
    ) -> Tuple[float, float]:
        if raw_x is not None and log_x is not None:
            return raw_x, log_x

        if raw_x is not None and log_x is None:
            raise RuntimeError(
                "chunk_spline: both log_x and raw_x are required, but only raw_x was supplied"
            )

        if raw_x is None and log_x is not None:
            raise RuntimeError(
                "chunk_spline: both log_x and raw_x are required, but only log_x was supplied"
            )

        if x is None:
            raise RuntimeError(
                "chunk_spline: a valid combination of x or (raw_x, log_x) must be supplied"
            )

        if x_is_log:
            return exp(x), x

        return x, log(x)

    def raw_theta(
        self,
        x: float = None,
        raw_x: float = None,
        log_x: float = None,
        x_is_log: bool = False,
        warn_unsafe: bool = True,
    ) -> float:
        raw_x, log_x = self._get_x(x, raw_x, log_x, x_is_log)
        theta = self._theta(raw_x=raw_x, log_x=log_x, warn_unsafe=warn_unsafe)
        return theta + self._div_2pi_base * _twopi

    def theta_mod_2pi(
        self,
        x: float = None,
        raw_x: float = None,
        log_x: float = None,
        x_is_log: bool = False,
        warn_unsafe: bool = True,
    ) -> float:
        raw_x, log_x = self._get_x(x, raw_x, log_x, x_is_log)
        theta = self._theta(raw_x=raw_x, log_x=log_x, warn_unsafe=warn_unsafe)
        return fmod(theta, _twopi)

    def theta_deriv(
        self,
        x: float = None,
        raw_x: float = None,
        log_x: float = None,
        x_is_log: bool = False,
        warn_unsafe: bool = True,
    ) -> float:
        raw_x, log_x = self._get_x(x, raw_x, log_x, x_is_log)
        return self._theta_deriv(raw_x=raw_x, log_x=log_x, warn_unsafe=warn_unsafe)


class phase_spline:
    """
    Spline the phase function for a Liouville-Green representation.
    We try to do this intelligently so that we do not lose precision when the phase is evaluated mod 2pi,
    which is eventually what will usually be done (except perhaps during Levin integration when we need
    the derivative of the phase instead)
    """

    def __init__(
        self,
        x_sample: Iterable[float],
        theta_div_2pi_sample: Iterable[int],
        theta_mod_2pi_sample: Iterable[float],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        x_is_log: bool = False,
    ):
        if len(x_sample) == 0:
            raise RuntimeError("phase_spline: empty x_sample")

        assert len(x_sample) == len(theta_div_2pi_sample)
        assert len(x_sample) == len(theta_mod_2pi_sample)

        self._min_div_2pi = min(theta_div_2pi_sample)
        self._max_div_2pi = max(theta_div_2pi_sample)

        self._chunk_size = chunk_size
        self._chunk_step = int(round(0.75 * chunk_size + 0.5, 0))

        self._spline_points = {}

        # Sift through the set of points supplied
        # The main idea is to divide the points into chunks based on their value of theta div 2pi, and build a set of splines
        # for smaller chunks. The numerical value of theta computed in each chunk is rebased so that it is close to zero.
        # This is intended to avoid loss of precision when we compute the spline mod 2pi.

        # compute the range of value of div 2pi that should be included in each chunk
        # we allow a 25% overlap for each chunk
        chunk_start = self._min_div_2pi

        while chunk_start <= self._max_div_2pi:
            chunk_end = chunk_start + self._chunk_size
            self._spline_points[(chunk_start, chunk_end)] = []
            chunk_start = chunk_start + self._chunk_step

        # sift through the supplied data points, adding them to chunks whatever chunks they fall within
        for i in range(len(x_sample)):
            x = x_sample[i]
            div_2pi = theta_div_2pi_sample[i]
            mod_2pi = theta_mod_2pi_sample[i]

            # add this point to any chunks that should contain it
            for chunk, data in self._spline_points.items():
                start, end = chunk
                if start <= div_2pi < end:
                    data.append(
                        {
                            "x": x,
                            "div_2pi": div_2pi,
                            "mod_2pi": mod_2pi,
                        }
                    )

        # check number of spline points in each chunk; if the last chunk contains too few items, we want to merge it with the previous chunk
        # otherwise, warn if any chunk contains too few points to produce a good spline

        # build list of intervals, sorted into order
        chunk_list = list(self._spline_points.keys())
        chunk_list.sort(key=lambda x: x[0])

        for i, chunk in enumerate(chunk_list):
            start, end = chunk
            data = self._spline_points[chunk]
            if len(data) < MINIMUM_SPLINE_DATA_POINTS:

                # if last point, merge
                if i == 0 or i < len(chunk_list) - 1:
                    print(
                        f"!! WARNING (phase_spline): chunk of range ({start:.5g}, {end:.5g}) has {len(data)} data points, which is below the recommended minimum to produce a good spline"
                    )

                elif i == len(chunk_list) - 1:
                    assert i > 0
                    prev_chunk = chunk_list[i - 1]
                    prev_start, prev_end = prev_chunk

                    prev_chunk_data = self._spline_points[prev_chunk]

                    new_chunk_data = prev_chunk_data.copy()
                    new_chunk_data.extend(data)

                    self._spline_points.pop(chunk)
                    self._spline_points.pop(prev_chunk)
                    self._spline_points[(prev_start, end)] = new_chunk_data

        # build spline objects
        self._chunk_list = list(self._spline_points.keys())
        self._chunk_list.sort(key=lambda x: x[0])

        self._splines = {}

        last_min_log_x = None
        last_max_log_x = None

        # check that chunks have ascending x-ranges
        for chunk in self._chunk_list:
            start, end = chunk
            data = self._spline_points[chunk]

            spline = _chunk_spline(div_2pi_base=start, data=data, x_is_log=x_is_log)

            if last_min_log_x is not None and spline.min_log_x <= last_min_log_x:
                raise RuntimeError(
                    "phase_spline: chunks are not in increasing min_x-order"
                )
            if last_max_log_x is not None and spline.max_log_x <= last_max_log_x:
                raise RuntimeError(
                    "phase_spline: chunks are not in increasing max_x-order"
                )

            self._splines[chunk] = spline
            last_min_log_x = spline.min_log_x
            last_max_log_x = spline.max_log_x

        print(
            f"## Built phase_spline objecting containing {len(self._splines)} chunks: {self._chunk_list}"
        )

    def _match_chunk(self, log_x: float):
        num_chunks = len(self._chunk_list)
        if num_chunks == 1:
            spline0 = self._splines[self._chunk_list[0]]

            if spline0.min_log_x > log_x:
                raise RuntimeError(
                    f"phase_spline: could not match log_x={log_x:.5g} to a spline chunk (only chunk has min_log_x={spline0.min_log_x:.5g}, max_log_x={spline0.max_log_x:.5g})"
                )
            if spline0.max_log_x < log_x:
                raise RuntimeError(
                    f"phase_spline: could not match log_x={log_x:.5g} to a spline chunk (only chunk has min_log_x={spline0.min_log_x:.5g}, max_log_x={spline0.max_log_x:.5g})"
                )

            return spline0

        spline0 = self._splines[self._chunk_list[0]]
        if spline0.min_log_x > log_x:
            raise RuntimeError(
                f"phase_spline: could not match log_x={log_x:.5g} to a spline chunk (first chunk already has min_log_x={spline0.min_log_x:.5g})"
            )
        splineN = self._splines[self._chunk_list[num_chunks - 1]]
        if splineN.max_log_x < log_x:
            raise RuntimeError(
                f"phase_spline: could not match log_x={log_x:.5g} to a spline chunk (last chunk already has max_log_x={splineN.max_log_x:.5g})"
            )

        # find first chunk whose min_log_x value is larger than our log_x
        # this is the first chunk that *cannot* include our point
        if splineN.min_log_x <= log_x:
            upper_bound = num_chunks
        else:
            # A is inclusive, B is exclusive (past-the-end)
            A = 0
            B = num_chunks - 1

            while A != B:
                splineA = self._splines[self._chunk_list[A]]
                splineB = self._splines[self._chunk_list[B]]
                if B == A + 1:
                    if splineA.min_log_x > log_x:
                        raise RuntimeError(
                            f"phase_spline: could not match log_x={log_x:.5g} to an upper bound chunk (A={A}, min_log_x(A)={splineA.min_log_x:.5g}, max_log_x(A)={splineA.max_log_x:.5g}, B={B}, min_log_x(B)={splineB.min_log_x:.5g}, max_log_x(B)={splineB.max_log_x:.5g})"
                        )
                    upper_bound = A
                    break
                else:
                    C = int(round((A + B) / 2, 0))
                    if C == B:
                        C = B - 1
                    elif C == A:
                        C = A + 1

                    splineC = self._splines[self._chunk_list[C]]

                    if splineC.min_log_x > log_x:
                        # C doesn't contain our log x, so it becomes the new B
                        B = C
                    else:
                        # C does contain our log x, so it becomes our new A
                        A = C

            # find last chunk whose max_log_x value is smaller than our log_x
            # this is the last chink that *cannot* include our point
            if spline0.max_log_x >= log_x:
                lower_bound = -1
            else:
                # A is exclusive this time, B is inclusive
                A = 0
                B = num_chunks - 1

            while A != B:
                splineA = self._splines[self._chunk_list[A]]
                splineB = self._splines[self._chunk_list[B]]
                if B == A + 1:
                    if splineB.max_log_x < log_x:
                        raise RuntimeError(
                            f"phase_spline: could not match log_x={log_x:.5g} to a lower bound chunk (A={A}, min_log_x(A)={splineA.min_log_x:.5g}, max_log_x(A)={splineA.max_log_x:.5g}, B={B}, min_log_x(B)={splineB.min_log_x:.5g}, max_log_x(B)={splineB.max_log_x:.5g})"
                        )
                    lower_bound = B
                    break
                else:
                    C = int(round((A + B) / 2, 0))
                    if C == B:
                        C = B - 1
                    elif C == A:
                        C = A + 1

                    splineC = self._splines[self._chunk_list[C]]

                    if splineC.min_log_x < log_x:
                        # C doesn't contain our log x, so it becomes the new A
                        A = C
                    else:
                        # C does contain our log x, so it becomes our new B
                        B = C

        # recall that both upper and lower bounds are exclusive
        # the available chunks fall strictly between the two
        assert lower_bound + 1 <= upper_bound - 1

        chunk_penalties = []
        i = lower_bound + 1
        while i < upper_bound:
            spline = self._splines[self._chunk_list[i]]

            # find relative position of log_x within this chink
            span = spline.max_log_x - spline.min_log_x
            pos = log_x - spline.min_log_x
            rel_pos = pos / span

            # penalize this chunk based on how far we are from the centre;
            metric = pow(0.5 - rel_pos, 2.0)
            chunk_penalties.append((i, metric))

        # find chunk with the smallest penalty
        chunk_penalties.sort(key=lambda x: x[1], reverse=True)
        print(
            f">> found {len(chunk_penalties)} candidate chunks for log_x={log_x:.5g} with penalties {chunk_penalties}"
        )

        return self._splines[self._chunk_list[chunk_penalties[0][0]]]

    def raw_theta(self, x: float, x_is_log: bool = False) -> float:
        if x_is_log:
            raw_x = exp(x)
            log_x = x
        else:
            raw_x = x
            log_x = log(x)

        spline = self._match_chunk(log_x)
        return spline.raw_theta(raw_x=raw_x, log_x=log_x)

    def theta_mod_2pi(self, x: float, x_is_log: bool = False) -> float:
        if x_is_log:
            raw_x = exp(x)
            log_x = x
        else:
            raw_x = x
            log_x = log(x)

        spline = self._match_chunk(log_x)
        return spline.theta_mod_2pi(raw_x=raw_x, log_x=log_x)

    def theta_deriv(self, x: float, x_is_log: bool = False) -> float:
        if x_is_log:
            raw_x = exp(x)
            log_x = x
        else:
            raw_x = x
            log_x = log(x)

        spline = self._match_chunk(log_x)
        return spline.theta_deriv(raw_x=raw_x, log_x=log_x)
