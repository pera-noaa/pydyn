"""Serial driver: the symmetric operator sequence of one dynamic time step (advect.F90 /
modelIntegration.F90 `do_steps`), single region, no chemistry hooks beyond callbacks, any
operator sequence (analysis/step_accuracy.py runs its own).  The run's driver -- do_steps
with the region recursion, Proces_Region, a whole step, a single region a tree of one -- is
zoom.do_steps / zoom.proces_region / zoom.step."""

from dataclasses import dataclass

import numpy as np

from .kernels import advect_x, advect_y, advect_z, convec, fluxes_from_meteo
from .redgrid import advect_x_reduced

__all__ = [
    "Model",
    "State",
]


@dataclass
class State:
    """Prognostic fields; see kernels.py for shapes.  Slopes default to zero."""

    m: np.ndarray
    rm: np.ndarray
    rxm: np.ndarray = None
    rym: np.ndarray = None
    rzm: np.ndarray = None

    def __post_init__(self):
        for name in ("rxm", "rym", "rzm"):
            if getattr(self, name) is None:
                setattr(self, name, np.zeros_like(self.rm))

    @property
    def shape(self):
        return self.m.shape

    def copy(self):
        return State(self.m.copy(), self.rm.copy(), self.rxm.copy(), self.rym.copy(), self.rzm.copy())

    def mixing_ratio(self):
        return self.rm / self.m[..., None]


class Model:
    """One dynamic step = two half-steps 'xyzvsc' and 'csvzyx' with dt = ndyn/2 each.

    Operators: x/y/z advection (kernels), 'v' convection (if `conv` inputs are supplied),
    's' sources and 'c' chemistry via optional callbacks `sources(state, dt)` and
    `chemistry(state, dt)`.  Missing pieces are no-ops, as in a transport-only run.

    `grid` is a `redgrid.RedGrid` when the region's grid is reduced, as every production
    configuration's is; x then goes through `advect_x_reduced` exactly as `replay` does it
    per event.  Left None the driver advects a uniform grid, which no production region has.

    `inverse` "crout" is convection's fastminv as the default `make -C fortran` builds it
    (without with_lapack); production builds with_lapack, which is `lapack.batched_inverse()`.
    """

    def __init__(self, bt, ndyn, limits=True, sources=None, chemistry=None, inverse="crout", grid=None):
        self.bt = np.asarray(bt, dtype=float)
        self.ndyn = ndyn
        self.limits = limits
        self.sources = sources
        self.chemistry = chemistry
        self.inverse = inverse
        self.grid = grid
        self.stats = {}

    def half_step(self, state, am, bm, cm, sequence, conv=None):
        dt = self.ndyn / 2.0
        s = state
        fields = (s.m, s.rm, s.rxm, s.rym, s.rzm)
        operators = {  # one letter of the sequence -> what it does to the state
            "x": lambda st: (advect_x_reduced(*fields, am, self.grid, self.limits, stats=st)
                             if self.grid is not None
                             else advect_x(*fields, am, self.limits, stats=st)),
            "y": lambda st: advect_y(*fields, bm, self.limits, stats=st),
            "z": lambda st: advect_z(*fields, cm, self.limits, stats=st),
            "v": lambda st: convec(*fields, dt=dt, inverse=self.inverse, **conv),
            "s": lambda st: self.sources(s, dt),
            "c": lambda st: self.chemistry(s, dt),
        }
        supplied = {"v": conv, "s": self.sources, "c": self.chemistry}
        for op in sequence:
            if op not in operators:
                raise ValueError(f"unknown operator {op!r}")
            if supplied.get(op, True) is None:
                continue  # transport-only run: the operator is a no-op
            st = {}
            operators[op](st)
            if st:
                self.stats[op] = st

    def step(self, state, pu, pv, conv=None):
        """Advance one ndyn step with time-constant meteo fluxes pu (im+1,jm,lm), pv (im,jm+1,lm) kg/s."""
        dt = self.ndyn / 2.0
        am, bm, cm = fluxes_from_meteo(pu, pv, self.bt, dt)
        self.half_step(state, am, bm, cm, "xyzvsc", conv)
        self.half_step(state, am, bm, cm, "csvzyx", conv)
        return state
