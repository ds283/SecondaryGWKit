from math import pi, log, exp, fmod
from typing import Iterable, Tuple, Optional

from numpy import sign
from scipy.interpolate import InterpolatedUnivariateSpline

from defaults import DEFAULT_FLOAT_PRECISION

DEFAULT_CHUNK_SIZE = 200
MINIMUM_SPLINE_DATA_POINTS = 10
SPLINE_TOP_BOTTOM_CUSHION = 0.001

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
        x_is_redshift: bool = False,
    ):
        self._x_is_redshift: bool = x_is_redshift
        self._div_2pi_base: int = div_2pi_base

        # cache supplied values
        self._data = data
        if self._data is None:
            raise RuntimeError(
                f"chunk_spline: supplied data is None (div_2pi_base={div_2pi_base})"
            )
        self.data_points: int = len(self._data)

        self._data.sort(key=lambda d: d["x"])
        self._x_sample = [p["x"] for p in self._data]

        if len(data) == 0:
            raise RuntimeError(
                f"chunk_spline: chunk has no data points (div_2pi_base={div_2pi_base})"
            )

        # build set of y values, but rebased mod 2pi so that all values are close to zero
        # in this way, we hope to avoid loss of precision when the result of the spline is evaluated mod 2pi, as it typically
        # will be
        self._y_points = [
            (p["div_2pi"] - div_2pi_base) * _twopi + p["mod_2pi"] for p in self._data
        ]

        if x_is_log:
            self._log_x_points = self._x_sample
        else:
            if x_is_redshift:
                self._log_x_points = [log(1.0 + x) for x in self._x_sample]
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
        self._derivative = self._spline.derivative()

    def _theta(self, raw_x: float, log_x: float, warn_unsafe: bool = True) -> float:
        # raise an error if requested value is too far outside our range
        # we allow a 1% cushion at the top and the bottom, in which we return just the top or bottom value
        if log_x < self.min_log_x * (1.0 - SPLINE_TOP_BOTTOM_CUSHION):
            raise RuntimeError(
                f"chunk_spline: spline evaluated out-of-bounds of lower limit at log_x={log_x:.5g} (raw x={raw_x:.5g}) | Minimum allowed value is log_x={self._min_safe_log_x:.5g} (raw x={self._min_safe_x:.5g}) | Recommended safe minimum is log_x={self._min_safe_log_x:.5g} (raw x={self._min_safe_x:.5g})"
            )
        elif log_x < self.min_log_x:
            log_x = self.min_log_x

        if log_x > self.max_log_x * (1.0 + SPLINE_TOP_BOTTOM_CUSHION):
            raise RuntimeError(
                f"chunk_spline: spline evaluated out-of-bounds of upper limit at log_x={log_x:.5g} (raw x={raw_x:.5g}) | Maximum allowed value is log_x={self._max_safe_log_x:.5g} (raw x={self._max_safe_x:.5g}) | Recommended safe maximum is log_x={self._max_safe_log_x:.5g} (raw x={self._max_safe_x:.5g})"
            )
        elif log_x > self.max_log_x:
            log_x = self.max_log_x

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
        # raise an error if requested value is too far outside our range
        # we allow a 1% cushion at the top and the bottom, in which we return just the top or bottom value
        if log_x < self.min_log_x * (1.0 - SPLINE_TOP_BOTTOM_CUSHION):
            raise RuntimeError(
                f"chunk_spline: spline evaluated out-of-bounds of lower limit at log_x={log_x:.5g} (raw x={raw_x:.5g}) | Minimum allowed value is log_x={self._min_safe_log_x:.5g} (raw x={self._min_safe_x:.5g}) | Recommended safe minimum is log_x={self._min_safe_log_x:.5g} (raw x={self._min_safe_x:.5g})"
            )
        elif log_x < self.min_log_x:
            log_x = self.min_log_x

        if log_x > self.max_log_x * (1.0 + SPLINE_TOP_BOTTOM_CUSHION):
            raise RuntimeError(
                f"chunk_spline: spline evaluated out-of-bounds of upper limit at log_x={log_x:.5g} (raw x={raw_x:.5g}) | Maximum allowed value is log_x={self._max_safe_log_x:.5g} (raw x={self._max_safe_x:.5g}) | Recommended safe maximum is log_x={self._max_safe_log_x:.5g} (raw x={self._max_safe_x:.5g})"
            )
        elif log_x > self.max_log_x:
            log_x = self.max_log_x

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
            if self._x_is_redshift:
                return exp(x) - 1.0, x
            return exp(x), x

        if self._x_is_redshift:
            return x, log(1.0 + x)
        return x, log(x)

    def raw_theta(
        self,
        x: Optional[float] = None,
        raw_x: Optional[float] = None,
        log_x: Optional[float] = None,
        x_is_log: bool = False,
        warn_unsafe: bool = True,
    ) -> float:
        raw_x, log_x = self._get_x(x, raw_x, log_x, x_is_log)
        theta = self._theta(raw_x=raw_x, log_x=log_x, warn_unsafe=warn_unsafe)
        return theta + self._div_2pi_base * _twopi

    def theta_mod_2pi(
        self,
        x: Optional[float] = None,
        raw_x: Optional[float] = None,
        log_x: Optional[float] = None,
        x_is_log: bool = False,
        warn_unsafe: bool = True,
    ) -> float:
        raw_x, log_x = self._get_x(x, raw_x, log_x, x_is_log)
        theta = self._theta(raw_x=raw_x, log_x=log_x, warn_unsafe=warn_unsafe)
        return fmod(theta, _twopi)

    def theta_deriv(
        self,
        x: Optional[float] = None,
        raw_x: Optional[float] = None,
        log_x: Optional[float] = None,
        x_is_log: bool = False,
        log_derivative: bool = False,
        warn_unsafe: bool = True,
    ) -> float:
        raw_x, log_x = self._get_x(x, raw_x, log_x, x_is_log)

        deriv = self._theta_deriv(raw_x=raw_x, log_x=log_x, warn_unsafe=warn_unsafe)

        if log_derivative:
            # spline is computed as a function of log(x) or log(1+z) if we are working in redshift, so derivative
            # will naturally by with respect to this
            return deriv

        # otherwise, this is not a log derivative, so we need to divide by 1/x or 1/(1+z)
        if self._x_is_redshift:
            return deriv / (1.0 + raw_x)

        return deriv / raw_x


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
        chunk_step: Optional[int] = DEFAULT_CHUNK_SIZE,
        chunk_logstep: Optional[float] = None,
        x_is_log: bool = False,
        x_is_redshift: bool = False,
        increasing: bool = True,
    ):
        if len(x_sample) == 0:
            raise RuntimeError("phase_spline: empty x_sample")

        assert len(x_sample) == len(theta_div_2pi_sample)
        assert len(x_sample) == len(theta_mod_2pi_sample)

        self._x_is_redshift: bool = x_is_redshift
        self._increasing: bool = increasing

        self._min_div_2pi: int = min(theta_div_2pi_sample)
        self._max_div_2pi: int = max(theta_div_2pi_sample)

        self._spline_points = {}

        # Sift through the set of points supplied
        # The main idea is to divide the points into chunks based on their value of theta div 2pi, and build a set of splines
        # for smaller chunks. The numerical value of theta computed in each chunk is rebased so that it is close to zero.
        # This is intended to avoid loss of precision when we compute the spline mod 2pi.

        if chunk_step is not None:
            self._build_linear_chunks(chunk_step)
        elif chunk_logstep is not None:
            self._build_log_chunks(chunk_logstep)
        else:
            raise RuntimeError(
                "phase_spline: either chunk_step or chunk_logstep must be supplied"
            )

        # sift through the supplied data points, adding them to chunks whatever chunks they fall within
        for i in range(len(x_sample)):
            x = x_sample[i]
            div_2pi = theta_div_2pi_sample[i]
            mod_2pi = theta_mod_2pi_sample[i]

            # add this point to any chunks that should contain it
            for chunk, data in self._spline_points.items():
                start, end = chunk
                # we can be inclusive at both ends. It doesn't really matter if we have a few extra data points at top or bottom.
                # Also, we need to allow start = end for the edge case where div2pi is zero for all points
                if start <= div_2pi <= end:
                    data.append(
                        {
                            "id": i,
                            "x": x,
                            "div_2pi": div_2pi,
                            "mod_2pi": mod_2pi,
                        }
                    )

        # check number of spline points in each chunk; if the last chunk contains too few items, we want to merge it with the previous chunk
        # otherwise, warn if any chunk contains too few points to produce a good spline

        issue_warning = False

        # check number of spline points in each chunk, and merge neighbouring chunks if there are too few points to produce a good spline
        finished = False
        while not finished:
            # build list of intervals, sorted into order of increasing div2pi (this works whether our div2pi values are positive or negative)
            chunk_list = list(self._spline_points.keys())

            # if phase is increasing with x, sort into ascending order, otherwise descending order (e.g. we get this with GkSource
            # where the phase is a monotone decreasing function of the source redshift
            if self._increasing:
                chunk_list.sort(key=lambda x: x[0])
            else:
                chunk_list.sort(key=lambda x: x[0], reverse=True)

            # set finished to True, so need to manually reset it to False if we merge any chunks and require another pass
            finished = True
            for i, chunk in enumerate(chunk_list):
                start, end = chunk
                data = self._spline_points[chunk]
                if len(data) < MINIMUM_SPLINE_DATA_POINTS:
                    if len(chunk_list) == 1:
                        print(
                            f"!! WARNING (phase_spline): data vector contains only {len(data)} data points, which is below the recommended minimum to produce a good spline"
                        )
                        issue_warning = True

                    elif i == 0:
                        # merge with next chunk
                        next_chunk = chunk_list[i + 1]
                        next_start, next_end = next_chunk

                        next_data = self._spline_points[next_chunk]

                        # merge data points, using dict based on point id to avoid creating duplicate points
                        # (this causes spline creation to fail)
                        _new_data = {x["id"]: x for x in next_data}
                        _new_data.update({x["id"]: x for x in data})
                        new_data = list(_new_data.values())

                        self._spline_points.pop(chunk)
                        self._spline_points.pop(next_chunk)
                        self._spline_points[(start, next_end)] = new_data
                        finished = False
                        break

                    else:
                        assert i > 0
                        prev_chunk = chunk_list[i - 1]
                        prev_start, prev_end = prev_chunk

                        prev_data = self._spline_points[prev_chunk]

                        # merge data points, using dict based on point id to avoid creating duplicate points
                        # (this causes spline creation to fail)
                        _new_data = {x["id"]: x for x in prev_data}
                        _new_data.update({x["id"]: x for x in data})
                        new_data = list(_new_data.values())

                        self._spline_points.pop(chunk)
                        self._spline_points.pop(prev_chunk)
                        self._spline_points[(prev_start, end)] = new_data
                        finished = False
                        break

        if issue_warning:
            print(
                f"## For information, the constructed chunks are (total data points={len(x_sample)}, min div2pi={self._min_div_2pi}, max div2pi={self._max_div_2pi}):"
            )
            for i, chunk in enumerate(chunk_list):
                start, end = chunk
                data = self._spline_points[chunk]
                print(f"     {i}. ({start}, {end}) contains {len(data)} data points")

        # BUILD SPLINE OBJECTS

        # build list of keys, sorted into order of increasing div2pi
        self._chunk_list = list(self._spline_points.keys())
        if self._increasing:
            self._chunk_list.sort(key=lambda x: x[0])
        else:
            self._chunk_list.sort(key=lambda x: x[0], reverse=True)

        self._splines = {}

        last_min_log_x = None
        last_max_log_x = None

        last_start = None
        last_end = None

        # check that chunks have ascending x-ranges
        for chunk in self._chunk_list:
            start, end = chunk
            data = self._spline_points[chunk]

            spline = _chunk_spline(
                div_2pi_base=start,
                data=data,
                x_is_log=x_is_log,
                x_is_redshift=x_is_redshift,
            )

            if last_min_log_x is not None and spline.min_log_x < last_min_log_x:
                raise RuntimeError(
                    f"phase_spline: chunks are not in increasing min_x-order (current chunk=({start}, {end}), min_log_x={spline.min_log_x:.5g}, max_log_x={spline.max_log_x:.5g}; last chunk=({last_start}, {last_end}), min_log_x={last_min_log_x:.5g}, max_log_x={last_max_log_x:.5g})"
                )
            if last_max_log_x is not None and spline.max_log_x < last_max_log_x:
                raise RuntimeError(
                    f"phase_spline: chunks are not in increasing max_x-order (current chunk=({start}, {end}), min_log_x={spline.min_log_x:.5g}, max_log_x={spline.max_log_x:.5g}; last chunk=({last_start}, {last_end}), min_log_x={last_min_log_x:.5g}, max_log_x={last_max_log_x:.5g})"
                )

            self._splines[chunk] = spline

            last_min_log_x = spline.min_log_x
            last_max_log_x = spline.max_log_x

            last_start = start
            last_end = end

        # print(
        #     f"## Built phase_spline object containing {len(self._splines)} chunks: (min theta dvi 2pi = {self._min_div_2pi}, max theta dvi 2pi = {self._max_div_2pi})"
        # )
        # for chunk in self._chunk_list:
        #     start, end = chunk
        #     spline = self._splines[chunk]
        #     print(f"   -- chunk: [{start}, {end}) | {spline.data_points} data points")

    def _build_linear_chunks(self, chunk_step: int):
        self._chunk_step = int(chunk_step)

        # compute the range of value of div 2pi that should be included in each chunk
        # we allow a 25% overlap for each chunk
        chunk_start = int(self._min_div_2pi)
        chunk_final = int(self._max_div_2pi)

        while chunk_start <= chunk_final:
            chunk_end = chunk_start + self._chunk_step
            self._spline_points[(chunk_start, chunk_end)] = []
            chunk_start = int(round(chunk_start + 0.75 * self._chunk_step - 0.5, 0))

    def _build_log_chunks(self, chunk_logstep: float):
        chunk_initial = int(self._min_div_2pi)
        chunk_final = int(self._max_div_2pi)

        if chunk_initial == 0 and chunk_final < 0:
            raise RuntimeError(
                "phase_spline: when using logarithmic spacing, if initial div2pi=0 then final value must be non-negative"
            )

        if chunk_final == 0 and chunk_initial > 0:
            raise RuntimeError(
                "phase_spline: when using logarithmic spacing, if final div2pi=0 then initial value must be non-positive"
            )

        if chunk_initial * chunk_final < 0:
            raise RuntimeError(
                f"phase_spline: when using logarithmic spacing, initial and final div2pi values must have the same sign (start div2pi={sign(chunk_initial)}, final div2pi={sign(chunk_final)})"
            )

        if chunk_initial == 0 and chunk_final == 0:
            self._spline_points[(0, 0)] = []
        elif chunk_initial >= 0:
            self._build_log_chunks_positive(chunk_initial, chunk_final, chunk_logstep)
        else:
            self._build_log_chunks_negative(chunk_initial, chunk_final, chunk_logstep)

    def _build_log_chunks_positive(
        self, chunk_start: int, chunk_stop: int, chunk_logstep: float
    ):
        self._chunk_logstep = float(chunk_logstep)

        assert chunk_start >= 0
        assert chunk_stop > 0

        while chunk_start <= chunk_stop:
            if chunk_start == 0:
                chunk_end = int(round(self._chunk_logstep + 1.0, 0))
            else:
                chunk_end = int(round(chunk_start * self._chunk_logstep + 1.0, 0))

            self._spline_points[(chunk_start, chunk_end)] = []

            chunk_start = int(round(0.75 * chunk_end - 0.5, 0))

    def _build_log_chunks_negative(
        self, chunk_start: int, chunk_stop: int, chunk_logstep: float
    ):
        self._chunk_logstep = float(chunk_logstep)

        assert chunk_start < 0
        assert chunk_stop <= 0

        reverse_chunk_start = -chunk_stop
        reverse_chunk_stop = -chunk_start

        while reverse_chunk_start <= reverse_chunk_stop:
            if reverse_chunk_start == 0:
                reverse_chunk_end = int(round(self._chunk_logstep + 1.0, 0))
            else:
                reverse_chunk_end = int(
                    round(reverse_chunk_start * self._chunk_logstep + 1.0, 0)
                )

            self._spline_points[(-reverse_chunk_end, -reverse_chunk_start)] = []

            reverse_chunk_start = int(round(0.75 * reverse_chunk_end - 0.5, 0))

    def _match_chunk(self, log_x: float):
        num_chunks = len(self._chunk_list)
        if num_chunks == 1:
            spline0 = self._splines[self._chunk_list[0]]

            if spline0.min_log_x > log_x * (1.0 + DEFAULT_FLOAT_PRECISION):
                raise RuntimeError(
                    f"phase_spline: could not match log_x={log_x:.5g} to a spline chunk (only chunk has min_log_x={spline0.min_log_x:.5g}, max_log_x={spline0.max_log_x:.5g})"
                )
            if spline0.max_log_x < log_x * (1.0 - DEFAULT_FLOAT_PRECISION):
                raise RuntimeError(
                    f"phase_spline: could not match log_x={log_x:.5g} to a spline chunk (only chunk has min_log_x={spline0.min_log_x:.5g}, max_log_x={spline0.max_log_x:.5g})"
                )

            return spline0, True

        spline0 = self._splines[self._chunk_list[0]]
        if spline0.min_log_x > log_x * (1.0 + DEFAULT_FLOAT_PRECISION):
            raise RuntimeError(
                f"phase_spline: could not match log_x={log_x:.5g} to a spline chunk (first chunk already has min_log_x={spline0.min_log_x:.5g}, num_chunks={num_chunks})"
            )
        splineN = self._splines[self._chunk_list[num_chunks - 1]]
        if splineN.max_log_x < log_x * (1.0 - DEFAULT_FLOAT_PRECISION):
            raise RuntimeError(
                f"phase_spline: could not match log_x={log_x:.5g} to a spline chunk (last chunk already has max_log_x={splineN.max_log_x:.5g}, num_chunks={num_chunks})"
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
                    if splineA.min_log_x > log_x * (1.0 + DEFAULT_FLOAT_PRECISION):
                        raise RuntimeError(
                            f"phase_spline: could not match log_x={log_x:.5g} to an upper bound chunk (A={A}, min_log_x(A)={splineA.min_log_x:.5g}, max_log_x(A)={splineA.max_log_x:.5g}, B={B}, min_log_x(B)={splineB.min_log_x:.5g}, max_log_x(B)={splineB.max_log_x:.5g})"
                        )
                    upper_bound = A + 1
                    break
                else:
                    C = int(round((A + B) / 2, 0))
                    if C == B:
                        C = B - 1
                    elif C == A:
                        C = A + 1

                    splineC = self._splines[self._chunk_list[C]]

                    if splineC.min_log_x > log_x * (1.0 + DEFAULT_FLOAT_PRECISION):
                        # C doesn't contain our log x, so it becomes the new B
                        B = C
                    else:
                        # C does contain our log x, so it becomes our new A
                        A = C

        # find last chunk whose max_log_x value is smaller than our log_x
        # this is the last chunk that *cannot* include our point
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
                    if splineB.max_log_x < log_x * (1.0 - DEFAULT_FLOAT_PRECISION):
                        raise RuntimeError(
                            f"phase_spline: could not match log_x={log_x:.5g} to a lower bound chunk (A={A}, min_log_x(A)={splineA.min_log_x:.5g}, max_log_x(A)={splineA.max_log_x:.5g}, B={B}, min_log_x(B)={splineB.min_log_x:.5g}, max_log_x(B)={splineB.max_log_x:.5g})"
                        )
                    lower_bound = A
                    break
                else:
                    C = int(round((A + B) / 2, 0))
                    if C == B:
                        C = B - 1
                    elif C == A:
                        C = A + 1

                    splineC = self._splines[self._chunk_list[C]]

                    if splineC.max_log_x < log_x * (1.0 - DEFAULT_FLOAT_PRECISION):
                        # C doesn't contain our log x, so it becomes the new A
                        A = C
                    else:
                        # C does contain our log x, so it becomes our new B
                        B = C

        # allowed chunks fall in (lower_bound, upper_bound)
        if (
            lower_bound + 1 > upper_bound - 1
            or lower_bound + 1 < 0
            or upper_bound > num_chunks
        ):
            print(
                f"!! ERROR (phase_spline) incompatible lower_bound and upper_bound values for log_x={log_x:.5g} (lower_bound={lower_bound}, upper_bound={upper_bound})"
            )
            for chunk in self._chunk_list:
                start, end = chunk
                spline = self._splines[chunk]
                print(
                    f"   -- chunk: [{start}, {end}) has min_log_x={spline.min_log_x:.5g}, max_log_x={spline.max_log_x:.5g})"
                )
            raise RuntimeError(
                f"phase_spline: incompatible lower_bound and upper_bound values for log_x={log_x:.5g} (lower_bound={lower_bound}, upper_bound={upper_bound})"
            )

        chunk_penalties = []
        i = lower_bound + 1
        while i < upper_bound:
            spline = self._splines[self._chunk_list[i]]

            # find relative position of log_x within this chink
            span = spline.max_log_x - spline.min_log_x
            pos = log_x - spline.min_log_x
            rel_pos = pos / span

            # penalize this chunk based on how far we are from the centre
            metric = pow(0.5 - rel_pos, 2.0)
            chunk_penalties.append((i, metric))
            i = i + 1

        # find chunk with the smallest penalty
        chunk_penalties.sort(key=lambda x: x[1])
        # print(
        #     f">> found {len(chunk_penalties)} candidate chunks for log_x={log_x:.5g} with penalties {chunk_penalties}"
        # )

        selected_chunk_index = chunk_penalties[0][0]
        return (
            self._splines[self._chunk_list[selected_chunk_index]],
            selected_chunk_index == 0 or selected_chunk_index == num_chunks - 1,
        )

    def _get_raw_log_x(self, x: float, x_is_log: bool):
        if x_is_log:
            if self._x_is_redshift:
                return exp(x) - 1.0, x
            return exp(x), x

        if self._x_is_redshift:
            return x, log(1.0 + x)
        return x, log(x)

    def raw_theta(self, x: float, x_is_log: bool = False) -> float:
        raw_x, log_x = self._get_raw_log_x(x, x_is_log)
        spline, first_or_last_chunk = self._match_chunk(log_x)
        return spline.raw_theta(raw_x=raw_x, log_x=log_x, warn_unsafe=False)

    def theta_mod_2pi(self, x: float, x_is_log: bool = False) -> float:
        raw_x, log_x = self._get_raw_log_x(x, x_is_log)
        spline, first_or_last_chunk = self._match_chunk(log_x)
        return spline.theta_mod_2pi(raw_x=raw_x, log_x=log_x, warn_unsafe=False)

    def theta_deriv(
        self, x: float, x_is_log: bool = False, log_derivative: bool = False
    ) -> float:
        raw_x, log_x = self._get_raw_log_x(x, x_is_log)
        spline, first_or_last_chunk = self._match_chunk(log_x)
        return spline.theta_deriv(
            raw_x=raw_x,
            log_x=log_x,
            log_derivative=log_derivative,
            warn_unsafe=False,
        )
