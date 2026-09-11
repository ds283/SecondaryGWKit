"""
Tests for prompt 11 of ``prompts/GkTk-remedial``: the numeric region's oscillation-resolution
diagnostic moved off the ODE right-hand side and on to the returned sample grid, the ``mode=None``
guard, and the phase-stepping extremum search.

Nothing here needs Ray or a datastore: ``numeric_with_phase_cut`` is exercised through its
undecorated ``_function`` with the ``ModelProxy`` / ``wavenumber_exit_time`` stand-ins of
``ComputeTargets/tests/wkb_reference.py`` (prompt 01), and the two production right-hand sides are
imported directly from their integration modules.

**The bit-identity constants** in :data:`GK_RAD_VALUE_SAMPLE` and friends were captured by running
the pre-change code at commit ``2ed3632`` ("Give PrimitivePhase an explicit leading-rate callable"),
which is the parent of the commit that introduces this module, on exactly the geometry
:func:`_radiation_geometry` builds. They are the returned samples, not a checksum, so a failure
says where.
"""

import io
import unittest
from contextlib import redirect_stdout
from math import pi, sqrt, hypot, log
from time import perf_counter

from scipy.integrate import solve_ivp

from ComputeTargets.GkNumericIntegration import RHS as Gk_RHS
from ComputeTargets.TkNumericIntegration import RHS as Tk_RHS
from ComputeTargets.WKB_Gk import Gk_omegaEff_sq
from ComputeTargets.WKB_Tk import Tk_omegaEff_sq
from ComputeTargets.tests.wkb_reference import (
    LambdaCDMModel,
    RadiationModel,
    horizon_exit_z,
    production_response_grid,
    production_source_grid,
    to_redshift_array,
)
from LiouvilleGreen.integration_tools import (
    DEFAULT_RELATIVE_STEP,
    find_phase_extremum,
    find_phase_minimum,
)
from Quadrature.integrators.numeric_with_phase_cut import (
    DERIV_INDEX,
    VALUE_INDEX,
    numeric_with_phase_cut,
    scan_sample_grid_for_unresolved_osc,
)
from Units import Mpc_units

UNITS = Mpc_units()

# production tolerances (main.py:2791-2792, and review §10.1's (atol, rtol) row)
PRODUCTION_ATOL = 1e-10
PRODUCTION_RTOL = 1e-8

# main.py:630, :1199 -- delta_logz is a spacing in log10(1+z)
PRODUCTION_DELTA_LOGZ = 1.0 / 100.0

# the payload keys the two object factories persist; none of them may change
EXPECTED_PAYLOAD_KEYS = {
    "data",
    "value_sample",
    "deriv_sample",
    "solver_label",
    "has_unresolved_osc",
    "unresolved_z",
    "unresolved_efolds_subh",
    "stop_deltaz_subh",
    "stop_value",
    "stop_deriv",
}

# the D2 measurement (README §7 D2): which k are swept, and how coarsely the source-redshift band
# is sampled. The full sweep (stride 1) is in the campaign log; the stride used here keeps the
# module fast while still covering both models and all three wavenumbers.
D2_K_VALUES = (1.0e5, 1.0e7, 3.0e8)
D2_LARGEST_K = 3.0e8
D2_SOURCE_STRIDE = 40


# ---------------------------------------------------------------------------------------------
# stand-ins (the pattern of ComputeTargets/tests/test_gk_wkb_phase.py)
# ---------------------------------------------------------------------------------------------


class _Wavenumber:
    def __init__(self, k: float, store_id: int, units):
        self.k = float(k)
        self.k_inv_Mpc = float(k)
        self.store_id = store_id
        self.units = units


class _KExit:
    """A ``wavenumber_exit_time`` stand-in: ``.k`` and ``.z_exit``."""

    def __init__(self, k: float, units, z_exit: float, store_id: int = 1):
        self.k = _Wavenumber(k, store_id, units)
        self.z_exit = z_exit


class _Proxy:
    """A ``ModelProxy`` stand-in: ``.get()`` and ``.units`` (for ``check_units``)."""

    def __init__(self, model, units):
        self._model = model
        self.units = units

    def get(self):
        return self._model


def _cosmology_of(model):
    return model.cosmology if model.cosmology is not None else model


def _geometry(model, k: float, efolds_suph: float = 5.0):
    """
    The production geometry of review §10: a source grid 100 per decade of z starting 5 e-folds
    outside the horizon, its 12-fold winnow as the response grid, truncated to the source redshift
    above and to ``0.85 * z_e6`` below (main.py:1172-1178), and the (z_e3, z_e6) stop window.
    """
    cosmology = _cosmology_of(model)
    z_exit = horizon_exit_z(cosmology, k, 0.0)
    z_e3 = horizon_exit_z(cosmology, k, 3.0)
    z_e6 = horizon_exit_z(cosmology, k, 6.0)
    z_source = horizon_exit_z(cosmology, k, -efolds_suph)

    source_grid = production_source_grid(z_source)
    response_grid = (
        production_response_grid(source_grid)
        .truncate(source_grid.max, keep="lower")
        .truncate(0.85 * z_e6, keep="higher-include")
    )
    return {
        "z_exit": z_exit,
        "z_e3": z_e3,
        "z_e6": z_e6,
        "source_grid": source_grid,
        "response_grid": response_grid,
    }


def _radiation_geometry():
    """The exact geometry the bit-identity constants were captured on."""
    return _geometry(RadiationModel(), 1.0e7)


def _run(
    model, k: float, geometry, z_init, z_sample, sector: str, mode="stop", **kwargs
):
    """Call the undecorated ``numeric_with_phase_cut`` for one sector."""
    is_Gk = sector == "Gk"
    payload = {}
    if mode is not None:
        payload["mode"] = mode
        payload["stop_search_window_z_begin"] = min(geometry["z_e3"], z_init.z)
        payload["stop_search_window_z_end"] = geometry["z_e6"]
    payload.update(kwargs)

    return numeric_with_phase_cut._function(
        _Proxy(model, UNITS),
        _KExit(k, UNITS, geometry["z_exit"]),
        z_init,
        z_sample,
        initial_value=0.0 if is_Gk else 1.0,
        initial_deriv=1.0 if is_Gk else 0.0,
        RHS=Gk_RHS if is_Gk else Tk_RHS,
        omega_sq=Gk_omegaEff_sq if is_Gk else Tk_omegaEff_sq,
        atol=PRODUCTION_ATOL,
        rtol=PRODUCTION_RTOL,
        delta_logz=PRODUCTION_DELTA_LOGZ,
        task_label=f"test_{sector}",
        object_label="Gr_k(z, z')" if is_Gk else "Tk(z)",
        **payload,
    )


def _quiet(fn, *args, **kwargs):
    """Run ``fn`` capturing stdout; return ``(result, captured_text)``."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = fn(*args, **kwargs)
    return result, buffer.getvalue()


# ---------------------------------------------------------------------------------------------
# bit-identity constants, captured at 2ed3632 (the parent of this module's commit)
#
# RadiationModel, k = 1e7/Mpc, source redshift 5 e-folds outside the horizon, production response
# grid, mode="stop", (atol, rtol) = (1e-10, 1e-8). 41 samples requested, 40 returned: the ODE
# terminates on the z_e6 event, and in "stop" mode the expected_values check is skipped, so the
# trailing request below z_e6 is never produced (main.py's 0.85 truncation comment now says so).
# ---------------------------------------------------------------------------------------------

# rad_Gk_1e7: z_source=1484131590.025767, n_response=41, returned=40
GK_RAD_RHS_EVALUATIONS = 12770
GK_RAD_STOP_DELTAZ_SUBH = 9575707.976271689
GK_RAD_STOP_VALUE = 220264654929.6411
GK_RAD_STOP_DERIV = -119.42124216910452
GK_RAD_VALUE_SAMPLE = (
    -341803803.65658724,
    -923013811.4046377,
    -1689218010.3929157,
    -2699279486.8741593,
    -4030762994.8338747,
    -5785837214.282248,
    -8098988513.204431,
    -11147030121.871227,
    -15161929210.074404,
    -20446866738.89425,
    -27395447898.46975,
    -36512538178.657196,
    -48431463236.095184,
    -63913395312.43267,
    -83794210610.92169,
    -108798971440.42993,
    -139050556133.53027,
    -172926246123.18857,
    -204679193470.7504,
    -220263498597.6704,
    -192618581745.1068,
    -85985084748.71297,
    100432658150.83453,
    219925432699.0291,
    -995562161.1580963,
    -199537143100.76486,
    219828328205.70648,
    -211835035053.35825,
    -34143370617.45218,
    16698543916.96437,
    -220010030366.37213,
    91840805825.89078,
    -168166862294.55396,
    -64590568899.48248,
    148280809752.75812,
    83301477652.08362,
    133190459631.06738,
    -140510043651.92438,
    36996586619.658844,
    -220062435174.30466,
)
GK_RAD_DERIV_SAMPLE = (
    1.513650252541457,
    2.6306131807096493,
    4.571753247861891,
    7.945069150922641,
    13.806777280673527,
    23.991086221353022,
    41.68125239729519,
    72.39564347545462,
    125.68188263330646,
    218.00149729555005,
    377.56191218235404,
    652.1655453032176,
    1121.198262416643,
    1911.5202078374111,
    3210.4491740187605,
    5245.814619127747,
    8131.874366385731,
    11287.049664257771,
    11701.249050084993,
    -178.4908705297638,
    -46398.75955262933,
    -153057.11776990208,
    -257145.7936440254,
    27857.565600722504,
    872679.9713808238,
    -642310.3928199792,
    -165829.79198966338,
    -1255203.5659678585,
    7865311.647842191,
    13796849.396524586,
    -1155932.128394587,
    37986400.3355037,
    46908878.51329325,
    120681742.03273974,
    162224712.83425742,
    352951085.1748269,
    -527750811.41442615,
    -886835198.3505213,
    1972904211.566817,
    148997211.03717065,
)

# rad_Tk_1e7: z_source=1484131590.025767, n_response=41, returned=40
TK_RAD_RHS_EVALUATIONS = 6611
TK_RAD_STOP_DELTAZ_SUBH = 9627866.541055543
TK_RAD_STOP_VALUE = 0.012385625763054686
TK_RAD_STOP_DERIV = -4.4629722595550977e-13
TK_RAD_VALUE_SAMPLE = (
    0.9999996896864602,
    0.9999983048358315,
    0.9999955006177992,
    0.9999904528970431,
    0.9999816041646589,
    0.9999661980764523,
    0.9999394003737516,
    0.9998928254619033,
    0.9998118830292003,
    0.9996712162032694,
    0.9994268261846038,
    0.99900210292917,
    0.998264289543817,
    0.9969829532729572,
    0.994758881975769,
    0.9909020209277906,
    0.984224374572489,
    0.9726954437609502,
    0.9528879649555801,
    0.9191471249158942,
    0.8625270409009905,
    0.7699963877267205,
    0.6257738378860896,
    0.4196743121355847,
    0.1706050944850646,
    -0.0374450605482685,
    -0.07477836786151437,
    0.02431328241190105,
    -0.0017942016882994023,
    0.005622967548641168,
    -0.007934990728463855,
    -0.004739756095497705,
    0.0005350936809764972,
    -0.0015210361706934751,
    -0.0004197627745611286,
    -0.0004365173148528718,
    -0.0002774898747581728,
    -0.00017169797070765703,
    3.12963164442033e-05,
    5.675032872868316e-05,
)
TK_RAD_DERIV_SAMPLE = (
    2.4500278194559906e-15,
    7.924867484228142e-15,
    1.949192996588316e-14,
    4.5420317897780985e-14,
    1.0446711430857315e-13,
    2.396976598863638e-13,
    5.492908662614352e-13,
    1.258511032799769e-12,
    2.8832778619813697e-12,
    6.604196808473175e-12,
    1.5131328087542823e-11,
    3.465840684497036e-11,
    7.936346617723337e-11,
    1.8166329835104175e-10,
    4.1555741566586584e-10,
    9.494700840879503e-10,
    2.1648967412663657e-09,
    4.91862929224932e-09,
    1.1105677494661891e-08,
    2.480330056010486e-08,
    5.43422854518511e-08,
    1.1506695806909421e-07,
    2.2905516264452197e-07,
    4.060112485045815e-07,
    5.678521460045803e-07,
    4.248011888108857e-07,
    -2.6237386501017316e-07,
    -3.5755317354545453e-07,
    5.092889430151063e-07,
    -4.4874019485143916e-07,
    -1.955931680635797e-07,
    -7.026420521795651e-08,
    -5.031616425779935e-07,
    1.0675385503828068e-07,
    -4.684791854815134e-07,
    -2.9382315091147814e-07,
    1.84649575761849e-07,
    -4.550082360159853e-08,
    4.948758638079408e-07,
    5.268775730819322e-08,
)
# LambdaCDMModel, k = 1e7/Mpc, same geometry: the RHS-evaluation count and the pre-change wall
# time, for the cost test. The wall times at 2ed3632 were 0.1301, 0.1326, 0.1299 s over three
# repeats on the machine the campaign was run on (review §10.2 measured 0.13 s with the per-RHS
# diagnostic against 0.09 s without).
LCDM_GK_RHS_EVALUATIONS = 12854
LCDM_GK_PRE_CHANGE_WALL_TIME = 0.1299


# ---------------------------------------------------------------------------------------------
# item 1 and item 6: the returned samples are bit-identical, in both sectors
# ---------------------------------------------------------------------------------------------


class TestBitIdentity(unittest.TestCase):
    """
    Prompt 11 §1: this is cleanup of a region review §10.1 found sound, so the returned G, G', T,
    T' samples must be **bit-identical** for the same inputs. The stop point is the one exception:
    item 2.3(2b) changes the step the extremum search takes, so the sign change is bracketed one
    step differently and ``root_scalar``'s own ``(xtol=1e-6, rtol=1e-4)`` places the root somewhere
    else inside that bracket.
    """

    def _check_stop_point(self, model, k, payload, sector):
        """The stop point is an extremum of the same sign, at value/envelope = +1."""
        z_stop = payload["stop_deltaz_subh"]
        self.assertIsNotNone(z_stop)

        value = payload["stop_value"]
        deriv = payload["stop_deriv"]

        omega_sq = Gk_omegaEff_sq if sector == "Gk" else Tk_omegaEff_sq

        # recover the redshift of the stop point: stop_deltaz_subh = k.z_exit - z_stop
        geometry = _geometry(model, k)
        z = geometry["z_exit"] - z_stop
        omega = sqrt(omega_sq(model, k, z))

        self.assertGreater(value, 0.0, f"{sector}: stop point is not a maximum")

        # the local Liouville-Green envelope of the solution at the stop point
        envelope = hypot(value, deriv / omega)
        self.assertAlmostEqual(
            value / envelope,
            1.0,
            delta=1e-6,
            msg=f"{sector}: value/envelope at the stop point is not +1",
        )

        # |derivative| is bounded by how precisely root_scalar located the root: the root is known
        # to (xtol + rtol * z) = (1e-6 + 1e-4 z), and the derivative moves at |value| omega^2 there
        root_tolerance = 1e-6 + 1e-4 * z
        self.assertLess(
            abs(deriv),
            abs(value) * omega * omega * root_tolerance,
            f"{sector}: derivative at the stop point is larger than root_scalar's own tolerance "
            f"allows",
        )

    def test_Gk_samples_are_bit_identical(self):
        model = RadiationModel()
        geometry = _radiation_geometry()
        payload, _ = _quiet(
            _run,
            model,
            1.0e7,
            geometry,
            geometry["source_grid"].max,
            geometry["response_grid"],
            "Gk",
        )

        self.assertEqual(set(payload.keys()), EXPECTED_PAYLOAD_KEYS)
        self.assertEqual(len(payload["value_sample"]), len(GK_RAD_VALUE_SAMPLE))

        for i, (got, expected) in enumerate(
            zip(payload["value_sample"], GK_RAD_VALUE_SAMPLE)
        ):
            self.assertEqual(float(got), expected, f"G sample {i} changed")
        for i, (got, expected) in enumerate(
            zip(payload["deriv_sample"], GK_RAD_DERIV_SAMPLE)
        ):
            self.assertEqual(float(got), expected, f"G' sample {i} changed")

        self._check_stop_point(model, 1.0e7, payload, "Gk")

    def test_Tk_samples_are_bit_identical(self):
        """Item 6: the TkNumericIntegration right-hand side is the same path, and its samples are
        unchanged too."""
        model = RadiationModel()
        geometry = _radiation_geometry()
        payload, _ = _quiet(
            _run,
            model,
            1.0e7,
            geometry,
            geometry["source_grid"].max,
            geometry["response_grid"],
            "Tk",
        )

        self.assertEqual(set(payload.keys()), EXPECTED_PAYLOAD_KEYS)
        self.assertEqual(len(payload["value_sample"]), len(TK_RAD_VALUE_SAMPLE))

        for i, (got, expected) in enumerate(
            zip(payload["value_sample"], TK_RAD_VALUE_SAMPLE)
        ):
            self.assertEqual(float(got), expected, f"T sample {i} changed")
        for i, (got, expected) in enumerate(
            zip(payload["deriv_sample"], TK_RAD_DERIV_SAMPLE)
        ):
            self.assertEqual(float(got), expected, f"T' sample {i} changed")

        self._check_stop_point(model, 1.0e7, payload, "Tk")


# ---------------------------------------------------------------------------------------------
# item 2: the flag semantics
# ---------------------------------------------------------------------------------------------


class TestUnresolvedOscillationFlag(unittest.TestCase):
    def test_coarse_grid_sets_the_flag(self):
        """One sample per decade is far coarser than the wavelength once the mode is inside the
        horizon: the flag fires at the first violating pair, and the warning is printed once.
        """
        model = RadiationModel()
        k = 1.0e7
        geometry = _geometry(model, k)

        # one sample per decade, starting at the top of the production search window (x = e^3)
        # and running two decades deeper. The local wavelength there is 2 pi (1+z)/x <= 0.32 z,
        # against a grid spacing of 0.9 z.
        z_top = geometry["z_e3"]
        coarse = to_redshift_array([z_top, z_top / 10.0, z_top / 100.0])

        # mode=None: this grid runs past z_e6, so there is no termination event to hit, and the
        # flag has nothing to do with the stop point in any case
        payload, printed = _quiet(
            _run,
            model,
            k,
            geometry,
            coarse.max,
            coarse,
            "Gk",
            mode=None,
        )

        self.assertTrue(payload["has_unresolved_osc"])
        self.assertIsNotNone(payload["unresolved_z"])
        self.assertIsNotNone(payload["unresolved_efolds_subh"])

        # the warning is emitted exactly once, however many pairs violate the test
        self.assertEqual(printed.count("may have developed unresolved oscillations"), 1)

        # unresolved_z is the first (largest-z) violating pair
        sampled = [float(z.z) for z in coarse][: len(payload["value_sample"])]
        first = None
        for i in range(len(sampled) - 1):
            omega_sq = Gk_omegaEff_sq(model, k, sampled[i])
            if omega_sq <= 0.0:
                continue
            if 2.0 * pi / sqrt(omega_sq) < abs(sampled[i] - sampled[i + 1]):
                first = sampled[i]
                break
        self.assertIsNotNone(first)
        self.assertEqual(payload["unresolved_z"], first)

    def test_fine_grid_leaves_the_flag_clear(self):
        """A grid finer than the wavelength everywhere sets the flag False and prints nothing."""
        model = RadiationModel()
        k = 1.0e7
        geometry = _geometry(model, k)

        # z_e3 .. z_e6 with 2000 points: at x <= e^6 = 403 the wavelength is 2 pi (1+z)/x, far
        # larger than this spacing
        z_hi = geometry["z_e3"]
        z_lo = geometry["z_e6"]
        n = 2000
        z_values = [z_hi * (z_lo / z_hi) ** (i / (n - 1)) for i in range(n)]
        fine = to_redshift_array(z_values)

        payload, printed = _quiet(
            _run,
            model,
            k,
            geometry,
            fine.max,
            fine,
            "Gk",
            mode=None,
        )

        self.assertFalse(payload["has_unresolved_osc"])
        self.assertIsNone(payload["unresolved_z"])
        self.assertIsNone(payload["unresolved_efolds_subh"])
        self.assertNotIn("may have developed unresolved oscillations", printed)

    def test_scan_skips_non_oscillatory_samples(self):
        """``omega^2 <= 0`` samples are skipped rather than raising."""
        model = RadiationModel()

        def negative_omega_sq(model_, k_, z_):
            return -1.0

        result, printed = _quiet(
            scan_sample_grid_for_unresolved_osc,
            model,
            _Wavenumber(1.0e7, 1, UNITS),
            1.0e7,
            negative_omega_sq,
            [1.0e6, 1.0e3, 1.0e1],
            "label",
        )
        self.assertFalse(result["has_unresolved_osc"])
        self.assertEqual(printed, "")

    def test_flag_is_unknown_when_no_frequency_is_supplied(self):
        """Without ``omega_sq`` the diagnostic is not requested, and the three fields are None --
        the behaviour the flag had when ``delta_logz`` was omitted."""
        model = RadiationModel()
        k = 1.0e7
        geometry = _geometry(model, k)

        payload, _ = _quiet(
            lambda: numeric_with_phase_cut._function(
                _Proxy(model, UNITS),
                _KExit(k, UNITS, geometry["z_exit"]),
                geometry["source_grid"].max,
                geometry["response_grid"],
                initial_value=0.0,
                initial_deriv=1.0,
                RHS=Gk_RHS,
                atol=PRODUCTION_ATOL,
                rtol=PRODUCTION_RTOL,
                delta_logz=PRODUCTION_DELTA_LOGZ,
                mode="stop",
                stop_search_window_z_begin=geometry["z_e3"],
                stop_search_window_z_end=geometry["z_e6"],
                task_label="test_no_omega",
                object_label="Gr_k(z, z')",
            )
        )
        self.assertIsNone(payload["has_unresolved_osc"])
        self.assertIsNone(payload["unresolved_z"])
        self.assertIsNone(payload["unresolved_efolds_subh"])


# ---------------------------------------------------------------------------------------------
# item 2.2 / README §7 D2: what the corrected test costs in printed warnings
# ---------------------------------------------------------------------------------------------


class TestD2FireRate(unittest.TestCase):
    """
    The consequence the user has to decide about (README §7 D2,
    ``[00-unresolved-osc-print-policy]``): with the test evaluated against the grid the caller
    actually supplied, how often does it fire in production?

    This is a coarse stride over the source-redshift band; the full sweep is in the campaign log.
    """

    def _sweep(self, model):
        cosmology = _cosmology_of(model)

        # the universal source grid: 5 e-folds outside the horizon for the largest k
        # (main.py:410-414), winnowed 12-fold for the response grid (main.py:424)
        source_grid = production_source_grid(
            horizon_exit_z(cosmology, D2_LARGEST_K, -5.0)
        )
        response_grid = production_response_grid(source_grid)

        report = {}
        for k in D2_K_VALUES:
            z_e3 = horizon_exit_z(cosmology, k, 3.0)
            z_e4 = horizon_exit_z(cosmology, k, 4.0)
            z_e6 = horizon_exit_z(cosmology, k, 6.0)
            z_suph_e5 = horizon_exit_z(cosmology, k, -5.0)
            geometry = {
                "z_exit": horizon_exit_z(cosmology, k, 0.0),
                "z_e3": z_e3,
                "z_e6": z_e6,
            }

            # GkNumericIntegration: one object per source redshift in the production band
            # (main.py:1170 requires z_source > z_exit_subh_e4), sampled on the response grid
            band = [z for z in source_grid if z_e4 <= z.z <= z_suph_e5][
                ::D2_SOURCE_STRIDE
            ]
            total = 0
            fired = 0
            x_fired = []
            for z_source in band:
                response = response_grid.truncate(z_source, keep="lower").truncate(
                    0.85 * z_e6, keep="higher-include"
                )
                if len(response) < 2 or response.max.z >= z_source.z:
                    continue
                payload, _ = _quiet(_run, model, k, geometry, z_source, response, "Gk")
                total += 1
                if payload["has_unresolved_osc"]:
                    fired += 1
                    z = payload["unresolved_z"]
                    x_fired.append((1.0 + z) * k / model.functions.Hubble(z))

            # TkNumericIntegration: one object per k (review §12.1), sampled on the *source* grid
            source = source_grid.truncate(z_suph_e5, keep="lower").truncate(
                0.85 * z_e6, keep="higher-include"
            )
            payload, _ = _quiet(_run, model, k, geometry, source.max, source, "Tk")

            report[k] = {
                "Gk_total": total,
                "Gk_fired": fired,
                "Gk_x_fired": x_fired,
                "Tk_fired": bool(payload["has_unresolved_osc"]),
            }
        return report

    def test_Gk_fires_on_every_object_and_Tk_on_none(self):
        for model in (RadiationModel(), LambdaCDMModel()):
            report = self._sweep(model)
            for k, row in report.items():
                self.assertGreater(row["Gk_total"], 0)

                # every Green's-function object flags: the response grid is 12x sparser than the
                # source grid, so 2 pi (1+z)/x < Delta z_grid as soon as x > ~20, and the run
                # continues to x = e^6 = 403
                self.assertEqual(
                    row["Gk_fired"],
                    row["Gk_total"],
                    f"{model.name}, k={k:.3g}: expected every Gk object to flag",
                )
                self.assertGreater(min(row["Gk_x_fired"]), 19.0)
                self.assertLess(min(row["Gk_x_fired"]), 60.0)

                # the transfer function is sampled on the source grid itself, and its effective
                # frequency carries a factor c_s = 1/sqrt(3): the trip point is at x_T ~ 270,
                # i.e. x ~ 467, and the run stops at x = e^6 = 403
                self.assertFalse(
                    row["Tk_fired"],
                    f"{model.name}, k={k:.3g}: expected no Tk object to flag",
                )


# ---------------------------------------------------------------------------------------------
# item 3: cost
# ---------------------------------------------------------------------------------------------


class TestCost(unittest.TestCase):
    def test_RHS_evaluation_count_is_unchanged(self):
        """Moving the diagnostic off the right-hand side must not change the solver's work."""
        geometry = _radiation_geometry()
        model = RadiationModel()
        payload, _ = _quiet(
            _run,
            model,
            1.0e7,
            geometry,
            geometry["source_grid"].max,
            geometry["response_grid"],
            "Gk",
        )
        self.assertEqual(payload["data"].RHS_evaluations, GK_RAD_RHS_EVALUATIONS)

        payload, _ = _quiet(
            _run,
            model,
            1.0e7,
            geometry,
            geometry["source_grid"].max,
            geometry["response_grid"],
            "Tk",
        )
        self.assertEqual(payload["data"].RHS_evaluations, TK_RAD_RHS_EVALUATIONS)

    def test_LambdaCDM_wall_time(self):
        """Review §10.2 measured 0.13 s with the per-RHS diagnostic against 0.09 s without, i.e.
        45 % of the run. The RHS count is asserted; the time is reported, not asserted, because a
        wall-clock threshold is not a property of the tree."""
        model = LambdaCDMModel()
        geometry = _geometry(model, 1.0e7)

        times = []
        for _ in range(3):
            start = perf_counter()
            payload, _ = _quiet(
                _run,
                model,
                1.0e7,
                geometry,
                geometry["source_grid"].max,
                geometry["response_grid"],
                "Gk",
            )
            times.append(perf_counter() - start)
            self.assertEqual(payload["data"].RHS_evaluations, LCDM_GK_RHS_EVALUATIONS)

        best = min(times)
        print(
            f"\n    LambdaCDMModel, k=1e7, GkNumericIntegration RHS: "
            f"{best:.4f} s (pre-change {LCDM_GK_PRE_CHANGE_WALL_TIME:.4f} s, "
            f"{100.0 * (1.0 - best / LCDM_GK_PRE_CHANGE_WALL_TIME):.1f} % faster)"
        )


# ---------------------------------------------------------------------------------------------
# item 4: the mode guard
# ---------------------------------------------------------------------------------------------


class TestModeGuard(unittest.TestCase):
    """Review §10.2: ``mode.lower()`` ran before the ``None`` test, so ``mode=None`` raised
    ``AttributeError`` instead of integrating the whole grid."""

    def _short_grid(self, model, k, geometry, points=40):
        z_hi = geometry["z_e3"]
        z_lo = geometry["z_e6"]
        return to_redshift_array(
            [z_hi * (z_lo / z_hi) ** (i / (points - 1)) for i in range(points)]
        )

    def test_mode_None_runs_to_the_end_of_the_grid(self):
        model = RadiationModel()
        k = 1.0e7
        geometry = _geometry(model, k)
        grid = self._short_grid(model, k, geometry)

        payload, _ = _quiet(_run, model, k, geometry, grid.max, grid, "Gk", mode=None)

        # not in "stop" mode, so every requested sample must be produced
        self.assertEqual(len(payload["value_sample"]), len(grid))
        self.assertIsNone(payload["stop_deltaz_subh"])
        self.assertIsNone(payload["stop_value"])
        self.assertIsNone(payload["stop_deriv"])

    def test_mode_is_case_insensitive(self):
        model = RadiationModel()
        k = 1.0e7
        geometry = _geometry(model, k)
        payload, _ = _quiet(
            _run,
            model,
            k,
            geometry,
            geometry["source_grid"].max,
            geometry["response_grid"],
            "Gk",
            mode="STOP",
        )
        self.assertIsNotNone(payload["stop_value"])

    def test_unknown_mode_raises(self):
        model = RadiationModel()
        k = 1.0e7
        geometry = _geometry(model, k)
        with self.assertRaises(ValueError):
            _quiet(
                _run,
                model,
                k,
                geometry,
                geometry["source_grid"].max,
                geometry["response_grid"],
                "Gk",
                mode="x",
            )


# ---------------------------------------------------------------------------------------------
# item 5: stepping in phase rather than in relative z
# ---------------------------------------------------------------------------------------------


def _radiation_dense_solution(k: float, z_start: float, z_stop: float):
    """
    Dense-output solution of the exact radiation Green's-function ODE,

        dG/dz = G',   dG'/dz = -2 G'/(1+z) - (k/H)^2 G,   H = (1+z)^2,

    integrated downwards from ``z_start`` with G = 0, G' = 1 (the production initial data).
    """

    def rhs(z, state):
        one_plus_z = 1.0 + z
        k_over_H = k / (one_plus_z * one_plus_z)
        return [
            state[DERIV_INDEX],
            -2.0 * state[DERIV_INDEX] / one_plus_z
            - k_over_H * k_over_H * state[VALUE_INDEX],
        ]

    sol = solve_ivp(
        rhs,
        method="DOP853",
        t_span=(z_start, z_stop),
        y0=[0.0, 1.0],
        dense_output=True,
        atol=PRODUCTION_ATOL,
        rtol=1e-12,
    )
    self_check = sol.success
    if not self_check:
        raise RuntimeError(f"radiation control solve failed: {sol.message}")
    return sol.sol


class TestPhaseSteppedExtremumSearch(unittest.TestCase):
    """
    Review §10.2: the old search stepped ``1e-3`` in relative z, which is 2 pi/(1e-3 x) samples per
    cycle -- 15 per cycle at x = 403, fewer than one at x > 6283. Inside the (z_e3, z_e6) window
    that is safe; outside it the search skips cycles silently. Stepping in phase removes the
    dependence on the width of the window. The window itself is not widened here.
    """

    K = 1.0e7

    @staticmethod
    def _omega_sq(z: float) -> float:
        """``omega^2 = (k/H)^2`` for the exact radiation control (C = 0, review §6)."""
        one_plus_z = 1.0 + z
        k_over_H = TestPhaseSteppedExtremumSearch.K / (one_plus_z * one_plus_z)
        return k_over_H * k_over_H

    @staticmethod
    def _x_of(z: float) -> float:
        return TestPhaseSteppedExtremumSearch.K / (1.0 + z)

    @staticmethod
    def _z_at_x(x: float) -> float:
        return TestPhaseSteppedExtremumSearch.K / x - 1.0

    def test_alias_is_the_same_function(self):
        self.assertIs(find_phase_minimum, find_phase_extremum)

    def test_inside_the_window_both_steps_agree(self):
        """At production x the two steps are within 3 % of each other, and they find the same
        extremum."""
        z_start = self._z_at_x(20.0)  # x = e^3, the top of the search window
        z_stop = self._z_at_x(403.0)  # x = e^6, the bottom
        sol = _radiation_dense_solution(self.K, z_start * 1.5, z_stop * 0.5)

        in_phase = find_phase_extremum(
            sol, z_start, z_stop, VALUE_INDEX, DERIV_INDEX, omega_sq=self._omega_sq
        )
        in_z = find_phase_extremum(sol, z_start, z_stop, VALUE_INDEX, DERIV_INDEX)

        self.assertGreater(in_phase["value"], 0.0)
        self.assertGreater(in_z["value"], 0.0)

        # the same extremum: much closer than the half cycle that separates neighbouring ones
        half_cycle = pi * (1.0 + in_phase["z"]) / self._x_of(in_phase["z"])
        self.assertLess(abs(in_phase["z"] - in_z["z"]), 0.2 * half_cycle)

    def test_at_x_6000_the_old_step_skips_cycles(self):
        """At x = 6e3 the old step is 0.95 of a cycle, so it steps past maxima; the phase step
        takes 16 samples per cycle and finds the first one."""
        z_start = self._z_at_x(6.0e3)
        z_stop = self._z_at_x(1.2e4)
        sol = _radiation_dense_solution(self.K, z_start * 1.2, z_stop * 0.8)

        # at the start of the search the old step covers this fraction of a cycle
        cycles_per_step = (
            DEFAULT_RELATIVE_STEP
            * z_start
            * self._x_of(z_start)
            / (2.0 * pi * (1.0 + z_start))
        )
        self.assertGreater(cycles_per_step, 0.9)

        in_phase = find_phase_extremum(
            sol, z_start, z_stop, VALUE_INDEX, DERIV_INDEX, omega_sq=self._omega_sq
        )

        # a 100x finer phase step gives the true first maximum below z_start
        reference = find_phase_extremum(
            sol,
            z_start,
            z_stop,
            VALUE_INDEX,
            DERIV_INDEX,
            omega_sq=lambda z: 1.0e4 * self._omega_sq(z),
        )

        cycle = 2.0 * pi * (1.0 + reference["z"]) / self._x_of(reference["z"])
        self.assertLess(
            abs(in_phase["z"] - reference["z"]),
            0.1 * cycle,
            "stepping in phase did not find the first maximum",
        )

        in_z = find_phase_extremum(sol, z_start, z_stop, VALUE_INDEX, DERIV_INDEX)
        self.assertGreater(
            abs(in_z["z"] - reference["z"]),
            cycle,
            "the old relative-z step was expected to skip at least one cycle here",
        )


if __name__ == "__main__":
    unittest.main()
