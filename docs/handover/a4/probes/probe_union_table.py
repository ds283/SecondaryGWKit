"""The user's construction (cs_tau on the union of numeric and WKB nodes) and a variant without the
numeric node that equals z_init: does T build at every rung, and is z_init a node (p_R)?"""
import sys
from math import sqrt
sys.path.insert(0, "/Users/ds283/Documents/Code/SecondaryGWKit/var/handover-04/scripts")
import a4_cells as C
from ComputeTargets.BackgroundModel import TAU_GAUSS_ORDER, TablePrimitive
from ComputeTargets.cumulative_table import CumulativeTable
from ComputeTargets.TkSourceFunctions import TkSourceFunctions
from ComputeTargets.tests.test_quadsource_integral import SHAPES, w_of_b, exact_Tk_inputs
from ComputeTargets.tests.test_g_phase_reanchor import _TableCsTauFixture
from ComputeTargets.tests.test_tk_source_functions import Fixture

class Union(_TableCsTauFixture):
    drop_init_node = False
    def exact_envelope_model(self):
        model = Fixture.exact_envelope_model(self)
        nodes = set(self.z_numeric) | set(self.z_numeric_below) | set(self.z_WKB)
        if self.drop_init_node:
            nodes.discard(self.crossover_z)
        nodes = sorted((float(z) for z in nodes), reverse=True)
        w = self.w
        table = CumulativeTable(nodes, lambda z: sqrt(w) / model.Hubble(z), TAU_GAUSS_ORDER, label="cs_tau").shifted(model.cs_tau(nodes[0]))
        model.functions = model.functions._replace(cs_tau=TablePrimitive(table, "cs_tau"))
        return model

class UnionOff(Union):
    drop_init_node = True

w = w_of_b(0.0)
for cls in (Union, UnionOff):
    bad = 0
    for shape_name, rungs in C.LADDER.items():
        base = [s for s in SHAPES if s.name == shape_name][0]
        for x in rungs:
            shape = C.scaled_shape(base, x)
            for label, kk in (("q", shape.q), ("r", shape.r)):
                f = cls(w, k=kk, x_max=C.fixture_x_max(x, kk, shape), drop_first_WKB_sample=True)
                try:
                    Tn, Tw = exact_Tk_inputs(f)
                    T = TkSourceFunctions(f.exact_envelope_model(), f.k_object, Tn, Tw)
                    ph = T.phase
                    j, pR = ph._leading.end_partial(ph.z_anchor)
                    msg = (f"built; {type(ph).__name__} exact_remainder={ph.exact_remainder} leading={type(ph._leading).__name__}; "
                           f"anchor {ph.z_anchor!r} == z_init {ph.z_anchor == f.crossover_z}; node index {ph._leading._table.node_index(ph.z_anchor)}; p_R={pR!r}")
                except Exception as exc:
                    bad += 1
                    msg = f"RAISED {type(exc).__name__}: {exc}"
                print(f"{cls.__name__:8s} {shape_name:9s} {x:8.0e} T{label}: {msg}", flush=True)
    print(f"== {cls.__name__}: {bad} raised")
