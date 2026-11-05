"""Meteo_Setup_Mass (meteo.F90) and its helpers: the surface-pressure tendency, the
Poisson correction that makes the horizontal mass fluxes consistent with it
(BalanceMassFluxes_m, grid_type_ll.F90 with_lapack path), CheckMassBalance, the linear time
interpolation of the meteo fields (TimeInterpolation 'interp*') and m2phlb1.

Shapes are the Fortran dummy-argument shapes without halos: pu (im+1, jm, lm) = pu_t(0:im),
pv (im, jm+1, lm) = pv_t(1:jm+1), pw (im, jm, lm+1) = pw_t(0:lm), sp (im, jm).

The Poisson solve (SolvePoissonFFTW) is bitwise only when it calls the same libfftw3 the
model was linked with, through the same legacy Fortran interface (`FFTWPoisson`); with
numpy's FFT (`NumpyPoisson`) it is reproducible to rounding only.  `poisson_solver` picks
FFTW when a library is found (`$PYDYN_FFTW`, `ctypes.util.find_library`, homebrew).
Everything else is transcribed operation by operation.
"""

import ctypes
import ctypes.util
import os
from datetime import datetime

import numpy as np

from .ops import convergence, seqsum

__all__ = [
    "GRAV",
    "PI",
    "FFTWPoisson",
    "NumpyPoisson",
    "check_mass_balance",
    "fill_mass_change",
    "interpol_fractions",
    "m2phlb1",
    "poisson_correction",
    "poisson_solver",
    "setup_mass",
    "time_interpolation",
]

GRAV = 9.80665  # Binas
PI = 3.14159265358979  # Binas: the literal is default real, promoted by -fdefault-real-8
MB_MAX_TRIES = 3
# meteo.F90 Meteo_Setup_Mass: (tol_rms, tol_diff) of the pre-check and of the Richardson loop
MB_TOL_PRE = (1.0e-4, 1.0e-3)
MB_TOL_TIGHT = (1.0e-7, 1.0e-6)
FFTW_FORWARD, FFTW_BACKWARD, FFTW_ESTIMATE = -1, 1, 64
FFTW_CANDIDATES = (
    "/opt/homebrew/lib/libfftw3.dylib",
    "/usr/local/lib/libfftw3.dylib",
    "/usr/lib64/libfftw3.so",
)


# ------------------------------------------------------------------ time interpolation
def interpol_fractions(t, t1, t2):
    """go_date InterpolFractions: weights (alfa1, alfa2) of t1, t2 for the instant t."""
    ds = (t2 - t1).total_seconds()
    ds1 = (t - t1).total_seconds()
    alfa2 = 0.5 if abs(ds) < np.finfo(float).tiny else ds1 / ds
    return 1.0 - alfa2, alfa2


def time_interpolation(data1, data2, alfa1, alfa2):
    return alfa1 * data1 + alfa2 * data2


def date(idate):
    """A dump header date tuple (y, m, d, H, M, S) as a datetime."""
    return datetime(*idate)


# ------------------------------------------------------------------ mass balance
def fill_mass_change(sp1, sp2, levi_db, area_m2, dt_sec):
    """FillMassChange / dt_sec: kg/s per cell that the surface-pressure change requires,
    level l taking the fraction |db(l)| of the column."""
    dm = np.abs(levi_db) * (sp2 - sp1)[..., None] / GRAV
    dm = dm * area_m2[None, :, None]
    return dm / dt_sec


def check_mass_balance(pu, pv, sp1, sp2, dt_sec, area_m2, rms_max, mav_max):
    """CheckMassBalance: (ok, rms, max) of the relative pressure-tendency mismatch between
    the column-integrated horizontal convergence and the surface-pressure change."""
    d = convergence(seqsum(pu, axis=2), seqsum(pv, axis=2)) * GRAV / area_m2[None, :]
    dmsp_dt = (sp2 - sp1) / dt_sec
    diff = (d - dmsp_dt) / ((sp1 + sp2) / 2.0)
    rms = np.sqrt(np.sum(diff**2) / diff.size)
    mav = np.max(np.abs(diff))
    return not (rms > rms_max or mav > mav_max), float(rms), float(mav)


def poisson_factor(im, jm):
    """EnsurePoissonCache: the eigenvalues of the periodic 5-point Laplacian (fac(1,1) = 1)."""
    i = np.arange(im)[:, None]
    j = np.arange(jm)[None, :]
    fac = 2.0 * (np.cos(2 * PI * i / im) + np.cos(2 * PI * j / jm) - 2.0)
    fac[0, 0] = 1.0
    return fac


class NumpyPoisson:
    """psi = inverse Laplacian of m on the doubly periodic grid, with numpy's FFT."""

    def __init__(self, im, jm):
        self.im, self.jm = im, jm
        self.fac = poisson_factor(im, jm)

    def psi(self, m):
        X = np.fft.fft2(m)
        self._divide(X)
        A = np.fft.ifft2(X) * (self.im * self.jm)  # unnormalised backward transform, as FFTW
        return A.real / float(self.im * self.jm)

    def _divide(self, X):  # X = X / fac; X(1,1) = 0: complex / real is componentwise in gfortran
        X.real[...] = X.real / self.fac
        X.imag[...] = X.imag / self.fac
        X[0, 0] = 0.0


class FFTWPoisson(NumpyPoisson):
    """The same, calling libfftw3 exactly as the Fortran does: dfftw_plan_dft_2d(im, jm) with
    FFTW_ESTIMATE on column-major complex(8) arrays, dfftw_execute_dft on the same buffers."""

    def __init__(self, im, jm, lib):
        super().__init__(im, jm)
        self.lib = ctypes.CDLL(lib)
        self.a = np.zeros((im, jm), dtype=np.complex128, order="F")
        self.x = np.zeros((im, jm), dtype=np.complex128, order="F")
        self.fwd, self.bwd = ctypes.c_int64(0), ctypes.c_int64(0)
        self._plan(self.fwd, self.a, self.x, FFTW_FORWARD)
        self._plan(self.bwd, self.x, self.a, FFTW_BACKWARD)

    def _plan(self, plan, src, dst, direction):
        args = [ctypes.c_int(v) for v in (self.im, self.jm, direction, FFTW_ESTIMATE)]
        self.lib.dfftw_plan_dft_2d_(
            ctypes.byref(plan), ctypes.byref(args[0]), ctypes.byref(args[1]), _ptr(src), _ptr(dst),
            ctypes.byref(args[2]), ctypes.byref(args[3]),
        )  # fmt: skip
        if not plan.value:
            raise RuntimeError("FFTWPoisson: dfftw_plan_dft_2d failed")

    def psi(self, m):
        self.a[...] = m
        self.lib.dfftw_execute_dft_(ctypes.byref(self.fwd), _ptr(self.a), _ptr(self.x))
        self._divide(self.x)
        self.lib.dfftw_execute_dft_(ctypes.byref(self.bwd), _ptr(self.x), _ptr(self.a))
        return self.a.real / float(self.im * self.jm)

    def __del__(self):
        for plan in (getattr(self, "fwd", None), getattr(self, "bwd", None)):
            if plan is not None and plan.value:
                self.lib.dfftw_destroy_plan_(ctypes.byref(plan))


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def fftw_library():
    """Path of a libfftw3 to call: $PYDYN_FFTW, else the first one found; None if none."""
    env = os.environ.get("PYDYN_FFTW")
    if env:
        return env
    found = ctypes.util.find_library("fftw3")
    for path in (found, *FFTW_CANDIDATES):
        if path and os.path.exists(path):
            return path
    return None


def poisson_solver(im, jm, fftw="auto"):
    """FFTWPoisson when a library is available (or `fftw` is a path), else NumpyPoisson."""
    if fftw is False:
        return NumpyPoisson(im, jm)
    lib = fftw_library() if fftw in ("auto", True) else fftw
    if lib is None:
        if fftw is True:
            raise RuntimeError("poisson_solver: no libfftw3 found (set PYDYN_FFTW)")
        return NumpyPoisson(im, jm)
    return FFTWPoisson(im, jm, lib)


def poisson_correction(m, solver=None):
    """SolvePoissonFFTW: flux corrections (u (im+1, jm), v (im, jm+1)) whose divergence is m,
    doubly periodic, gauge-fixed so that u(0,:) = 0 and v(:,0) = 0."""
    psi = (NumpyPoisson(*m.shape) if solver is None else solver).psi(m)
    # the gradient of psi across every face, the periodic wrap giving faces 0 and im (== jm)
    u = np.diff(psi, axis=0, prepend=psi[-1:], append=psi[:1])
    v = np.diff(psi, axis=1, prepend=psi[:, -1:], append=psi[:, :1])
    return u - u[0], v - v[:, :1]


def balance_mass_fluxes(pu, pv, pw, dm_dt, solver=None):
    """BalanceMassFluxes_m for a region whose parent is the single global cell (region 0):
    per level, add the Poisson correction of the residual divergence to pu and pv.  In place."""
    solver = NumpyPoisson(*dm_dt.shape[:2]) if solver is None else solver
    resid = convergence(pu, pv) + pw[..., :-1] - pw[..., 1:] - dm_dt
    for k in range(dm_dt.shape[2]):
        u, v = poisson_correction(resid[..., k], solver)
        pu[..., k] += u
        pv[..., k] += v


def setup_mass(pu, pv, pw, sp1, sp2, levi_db, area_m2, dt_sec, max_tries=MB_MAX_TRIES, solver=None):
    """The balancing part of Meteo_Setup_Mass for one meteo instant.  pu, pv are balanced in
    place; returns dict(tries, ok, pre=(rms, max), post=(rms, max), solver)."""
    solver = poisson_solver(*sp1.shape) if solver is None else solver
    dm_dt = fill_mass_change(sp1, sp2, levi_db, area_m2, dt_sec)
    ok0, rms0, mav0 = check_mass_balance(pu, pv, sp1, sp2, dt_sec, area_m2, *MB_TOL_PRE)
    if not ok0:
        raise ValueError(f"setup_mass: unbalanced input fluxes (rms {rms0:.3e}, max {mav0:.3e})")
    for tries in range(1, max_tries + 1):  # noqa: B007
        balance_mass_fluxes(pu, pv, pw, dm_dt, solver)
        ok, rms, mav = check_mass_balance(pu, pv, sp1, sp2, dt_sec, area_m2, *MB_TOL_TIGHT)
        if ok:
            break
    name = type(solver).__name__.replace("Poisson", "").lower()
    return {"tries": tries, "ok": ok, "pre": (rms0, mav0), "post": (rms, mav), "solver": name}


# ------------------------------------------------------------------ m2phlb1
def m2phlb1(m, at, bt, dxyp):
    """advect_tools m2phlb1: surface pressure from the column mass and the interface
    pressures phlb(l) = at(l) + bt(l) sp.  m (im, jm, lm); at, bt (lm+1,); dxyp (jm,)."""
    sp = seqsum(m, axis=2) * GRAV / dxyp[None, :]
    return sp, at + bt * sp[..., None]
